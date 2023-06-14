import argparse
import functools
import pathlib
from typing import Optional, Dict

import numpy as np
import time
import torch

from CARTO import ROOT_DIR
from CARTO.Decoder import utils as arti_utils
from CARTO.lib import partnet_mobility, vis_utils
from CARTO.simnet.lib import camera
from CARTO.simnet.lib.net import common
from CARTO.simnet.lib.net.dataset import (
    PanopticOutputs,
    extract_left_numpy_img,
    panoptic_collate,
)
from CARTO.simnet.lib.net.panoptic_trainer import PanopticModel
from CARTO.simnet.lib.net.post_processing import orochi_outputs

from CARTO.lib import rename_unpickler


class CARTO:
    def __init__(
        self,
        model_id: str,
        checkpoint_id: Optional[str] = None,
        inference_config_file: pathlib.Path = (
            pathlib.Path(__file__).parent / "inference_config.txt"
        ),
        vis_dir: Optional[pathlib.Path] = None,
        lod_start: int = 4,
        lod_end: int = 7,
    ):
        output_dir = pathlib.Path(
            ROOT_DIR / ".." / "datasets" / "encoder" / "runs" / f"{model_id}"
        )
        if checkpoint_id is None:
            checkpoint_path = common.get_latest_checkpoint_file(
                output_dir / "checkpoints"
            )
        else:
            checkpoint_path = output_dir / "checkpoints" / checkpoint_id

        print(f"Loading checkpoint from {checkpoint_path = }")
        torch_checkpoint = torch.load(checkpoint_path, pickle_module=rename_unpickler)

        hparams = argparse.Namespace(**torch_checkpoint["hyper_parameters"])

        hparams.checkpoint = checkpoint_path

        # Load Inference specific parameters?
        parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
        common.add_train_args(parser, enforce_required=False)
        print(f"Loading inference config from {inference_config_file}")

        # Overwrite/Add inference specific config to normal config
        self.hparams, remain = parser.parse_known_args(
            args=[f"@{str(inference_config_file)}"], namespace=hparams
        )
        print(f"Found these unbounded {remain = } parameters")

        if hparams.decoder_training_id == "":
            hparams.decoder_training_id = "506fd5f4-ff54-4413-8e13-6f066f182dfb"
            print(
                f"[WARNING] Decoder Training ID not defined, using default: {hparams.decoder_training_id}"
            )
        (
            self.decoder,
            self.joint_decoder,
            self.shape_embedding,
            self.joint_embedding,
            self.additional_output,
        ) = arti_utils.load_full_decoder(hparams.decoder_training_id)

        self.decoder_wrapper = functools.partial(
            self.decoder.get_ply_meshes,
            distance_threshold=1e-2,
            lod_start=lod_start,
            lod_current=lod_end,
            estimate_normals=False,
            chunk_size=5e5,  # TODO Make this editable to allow different GPUs more easily
        )

        self.model = PanopticModel(hparams, 0, None, None)
        self.model.cuda()
        self.model.eval()

        self.camera = camera.ZED2Camera1080p()

        self.model_dir = output_dir

        #### TMP HACK!! REMOVE ####
        self.inference_config_file = inference_config_file

    def __call__(self, sample: PanopticOutputs, is_target: bool = False):
        stereos_imgs, panoptic_targets = panoptic_collate(
            [sample], rgbd=self.hparams.model_rgbd
        )
        stereos_imgs = stereos_imgs.cuda()
        rgb_img = extract_left_numpy_img(stereos_imgs[0])

        with torch.no_grad():
            model_predictions: PanopticOutputs = self.model(stereos_imgs)

        return CARTOPrediction(
            model_predictions,
            rgb_img,
            self.decoder_wrapper,
            self.joint_decoder,
            self.camera,
            panoptic_targets,
        )


class CARTOPrediction:
    def __init__(
        self,
        model_predictions,
        rgb_img,
        decoder,
        joint_decoder,
        camera,
        panoptic_targets,
        is_target: bool = False,
    ):
        self.model_predictions = model_predictions
        self.rgb_img = rgb_img

        self.decoder = decoder
        self.joint_decoder = joint_decoder
        self.camera = camera
        self.panoptic_targets = panoptic_targets
        self.is_target = is_target

    def set_vis_dir(self, dir: pathlib.Path):
        dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = dir

    def compute_embs_and_poses(self):
        (
            self._abs_pose_outputs,
            _,
            _,
            self._latent_embeddings_shape,
            _,
            self._peaks_pred,
            self._latent_embeddings_arti,
            _,
        ) = self.model_predictions.graspable_objects_obbs[0].compute_embs_and_poses(
            is_target=self.is_target
        )

    def get_canonical_objects(
        self,
        ply: bool = False,
        timing_counter: Dict[str, float] = {
            "detection": [],
            "optimization": [],
            "reconstruction": [],
        },
    ):
        if hasattr(self, "canonical_points") and not ply:
            return self.canonical_points

        # Extract from feature maps
        start_extraction_inference = time.time()
        self.compute_embs_and_poses()
        detection_time = time.time() - start_extraction_inference
        if len(timing_counter["detection"]) > 0:
            timing_counter["detection"][-1] += detection_time  # Add to latest
        else:
            timing_counter["detection"] += [detection_time]

        # Reconstruct the objects
        start_reconstruction = time.time()
        canonical_ply_objects = self.decoder(
            self._latent_embeddings_shape,
            self._latent_embeddings_arti,
        )
        timing_counter["reconstruction"] += [
            ((time.time() - start_reconstruction) / len(self._latent_embeddings_arti))
            if len(self._latent_embeddings_arti) > 0
            else 0.0
        ]

        self.canonical_points = [
            np.array(ply_object.points) for ply_object in canonical_ply_objects
        ]
        if ply:
            return canonical_ply_objects
        else:
            return self.canonical_points

    def get_camera_objects(self):
        if not hasattr(self, "canonical_points") or not hasattr(
            self, "abs_pose_outputs"
        ):
            self.get_canonical_objects()

        self.points_2d = []
        self.obb_2d = []

        for pose_output, canonical_pc in zip(
            self._abs_pose_outputs, self.canonical_points
        ):
            points_homo = camera.convert_points_to_homopoints(canonical_pc.T)
            points_homo = partnet_mobility.TO_SIMNET_FRAME.matrix @ points_homo
            canonical_pc = camera.convert_homopoints_to_points(points_homo).T
            pc_trans, rotated_box = camera.transform_pc(
                pose_output, canonical_pc, _CAMERA=self.camera, use_camera_RT=False
            )

            points_mesh = camera.convert_points_to_homopoints(pc_trans.T)
            points_2d_mesh = vis_utils.project(self.camera.K_matrix, points_mesh)
            self.points_2d.append(points_2d_mesh.T)

            points_obb = camera.convert_points_to_homopoints(rotated_box.T)
            points_2d_obb = vis_utils.project(self.camera.K_matrix, points_obb)
            self.obb_2d.append(points_2d_obb.T)

        return self.points_2d, self.obb_2d

    def get_peaks(self):
        if not hasattr(self, "peaks_pred"):
            self.compute_embs_and_poses()
        return self._peaks_pred

    def get_poses(self):
        if not hasattr(self, "abs_pose_outputs"):
            self.compute_embs_and_poses()
        return self._abs_pose_outputs

    def get_joint_state_and_type(
        self,
        timing_counter: Dict[str, float] = {
            "detection": [],
            "optimization": [],
            "reconstruction": [],
        },
    ):
        if not hasattr(self, "latent_embeddings_arti") or not hasattr(
            self, "abs_pose_outputs"
        ):
            self.get_canonical_objects(timing_counter=timing_counter)

        start_reconstruction = time.time()
        if len(self._latent_embeddings_arti) > 0:
            joint_decoder_results = self.joint_decoder(
                torch.Tensor(self._latent_embeddings_arti).cuda()
            )
            all_joint_states_pred = np.array(
                joint_decoder_results["state"].cpu().detach().numpy(), dtype=float
            )
            all_joint_types_pred = arti_utils.get_joint_type_batch(
                joint_decoder_results["state"].cpu().detach()
            )
        else:
            all_joint_states_pred = []
            all_joint_types_pred = []
        timing_counter["reconstruction"][-1] += time.time() - start_reconstruction

        return all_joint_states_pred, all_joint_types_pred

    ######## Save Image Functions ########
    def save_rgb(self, name: str = "rgb"):
        vis_utils.save_image(self.rgb_img / 255, self.vis_dir / f"{name}.png")

    def save_segmentation(self, name: str = "seg"):
        seg_vis_img = self.model_predictions.room_segmentation[0].get_visualization_img(
            self.rgb_img, is_target=self.is_target
        )
        vis_utils.save_image(seg_vis_img, self.vis_dir / f"{name}.png")

    def save_depth(self, name: str = "depth"):
        disp_img = self.model_predictions.depth[0].get_visualization_img(
            self.rgb_img, is_target=self.is_target
        )
        vis_utils.save_image(disp_img, self.vis_dir / f"{name}.png")

    def save_bbox(self, name: str = "bbox"):
        vis_img = orochi_outputs.visualize_img(
            self.model_predictions,
            self.rgb_img,
            camera_model=self.camera,
            class_list=None,
            is_target=self.is_target,
        )
        vis_utils.save_image(vis_img, self.vis_dir / f"{name}.png")

    def save_heatmap(self, name: str = "heatmap"):
        heatmap = orochi_outputs.visualize_heatmap(self.model_predictions, self.rgb_img)
        vis_utils.save_image(heatmap, self.vis_dir / f"{name}.png")

    def save_poses(self, name: str = "poses"):
        poses = orochi_outputs.visualize_img(
            self.model_predictions,
            self.rgb_img,
            self.camera,
            [],
            poses=True,  # Draws 9D coordinate systems and not boxes
            is_target=self.is_target,
        )
        vis_utils.save_image(poses, self.vis_dir / f"{name}.png")

    def save_2d_points(self, name: str = "points"):
        if not hasattr(self, "points_2d"):
            self.get_camera_objects()
        points_2d = self.points_2d
        point_overlay_img = vis_utils.overlay_projected_points(
            self.rgb_img.copy() / 255.0, points_2d
        )
        vis_utils.save_image(point_overlay_img, self.vis_dir / f"{name}.png")

    def save_pred_obb(self, name: str = "obb"):
        if not hasattr(self, "obb_2d"):
            self.get_camera_objects()

        axes = []
        for pose_output in self._abs_pose_outputs:
            xyz_axis = (
                0.3 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            )
            sRT = pose_output.camera_T_object @ pose_output.scale_matrix
            transformed_axes = camera.transform_coordinates_3d(xyz_axis, sRT)
            axes.append(
                camera.calculate_2d_projections(
                    transformed_axes, self.camera.K_matrix[:3, :3]
                )
            )

        colors_box = [
            (234.0 / 255.0, 237.0 / 255.0, 63.0 / 255.0)
        ]  # Neon Yellow # TODO Maybe make this iteratable?

        box_overlay_img = self.rgb_img.copy() / 255
        for object_obb, axis in zip(self.obb_2d, axes):
            box_overlay_img = vis_utils.draw_bboxes_glow(
                box_overlay_img, object_obb, axis, colors_box[0]
            )
        vis_utils.save_image(box_overlay_img, self.vis_dir / f"{name}.png")
