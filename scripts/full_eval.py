import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import dataclasses
import pathlib
import sys

import matplotlib.pyplot as plt

# torch.cuda.set_device(1)
# Ensure mesh_to_sdf is imported first
import mesh_to_sdf
import numpy as np
import open3d as o3d
import pytorch3d
import scipy
import seaborn as sns
import torch
import tqdm

from CARTO import ROOT_DIR

from CARTO.Decoder import utils as arti_utils
from CARTO.lib.eval_utils import compute_and_plot_mAP
from CARTO.simnet.lib import camera
from CARTO.simnet.lib.net.dataset import Dataset, PanopticOutputs
from CARTO.Encoder.inference import CARTO, CARTOPrediction

sns.set()

import copy
import pathlib
import random
import sys
import time

import numpy as np
import open3d as o3d
import tyro

from CARTO.asdf_eval import ASDF_Evaluator
from CARTO.Decoder import config
from CARTO.lib.eval_utils import get_gt_pc, get_ids_from_seg
from CARTO.lib import rename_unpickler
from CARTO.lib import partnet_mobility


### SOME HARD_CODED_CONFIGS
@dataclasses.dataclass
class FullEvalConfig:
    MAX_COUNT: int = 1000
    METHOD: str = "ours"
    GT_ASDF: bool = True
    START_IDX: int = 0
    END_IDX: int = 10000000
    MODEL_ID: str = "14i8yfym"
    seed: int = 12345
    write: bool = True
    start_lod: int = 4
    end_lod: int = 8
    n_points: int = int(1e6)


def main(cfg: FullEvalConfig):
    carto = CARTO(cfg.MODEL_ID)

    RUN_NAME = f"eval/{cfg.METHOD}"
    if cfg.METHOD == "asdf":
        RUN_NAME += "_" + ("gt" if cfg.GT_ASDF else "no_gt")

    out_dir = carto.model_dir / RUN_NAME
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / "checkpoint.txt", mode="w") as f:
        f.write(f"{carto.hparams.checkpoint = }")

    _CAMERA = camera.ZED2Camera1080p()

    if cfg.METHOD == "asdf":
        asdf_evalutor = ASDF_Evaluator(_CAMERA)
    elif cfg.METHOD != "ours":
        assert False, "Unknown Method"

    dataset_location = carto.hparams.test_path

    timing_counter = {
        "detection": [],
        "optimization": [],
        "reconstruction": [],
    }

    REAL_DATA = False
    # REAL_DATA = True
    if REAL_DATA:
        dataset_location = "TODO"

    dataset = Dataset(dataset_location, carto.hparams)
    print(f"{len(dataset)} samples @ {dataset_location}")

    all_categories = [
        "BG",
        "Dishwasher",
        "Knife",
        "Laptop",
        "Microwave",
        "Oven",
        "Refrigerator",
        "Stapler",
        "StorageFurniture",
        "Table",
        "WashingMachine",
    ]
    categories = [
        "Dishwasher",
        "Laptop",
        "Microwave",
        "Oven",
        "Refrigerator",
        "Table",
        "WashingMachine",
    ]

    categories_to_id = {}
    for i in range(len(all_categories)):
        categories_to_id[all_categories[i]] = i

    categories_for_mean = []
    for i in range(len(categories)):
        categories_for_mean.append(categories_to_id[categories[i]])

    print("categories_for_mean", categories_for_mean)

    categories = [
        "BG",
        "Dishwasher",
        "Knife",
        "Laptop",
        "Microwave",
        "Oven",
        "Refrigerator",
        "Stapler",
        "StorageFurniture",
        "Table",
        "WashingMachine",
    ]

    categories_to_id = {}
    for i in range(len(categories)):
        categories_to_id[categories[i]] = i

    pred_results = []
    for sample_id in tqdm.tqdm(range(cfg.START_IDX, min(len(dataset), cfg.END_IDX))):
        sample: PanopticOutputs = dataset[sample_id]

        # Skip empty
        if len(sample.val_data[0].articulated_object_states) == 0:
            continue

        start_simnet_inference = time.time()
        if REAL_DATA:
            sample.stereo_imgs[0] = sample.stereo_imgs[0][:, ::2, ::2]

        # TODO Use this below
        carto_prediction: CARTOPrediction = carto(sample)

        timing_counter["detection"] += [time.time() - start_simnet_inference]

        # Overwrite with "GT"-labels
        # model_predictions = panoptic_targets
        # is_target = True

        if cfg.METHOD == "ours":
            # TODO Add timing again?
            # Get absolut psoe outputs, shape and articulation embeddings
            all_points_canonical_pred = carto_prediction.get_canonical_objects(
                timing_counter=timing_counter
            )
            peaks_pred = carto_prediction.get_peaks()
            abs_pose_outputs = carto_prediction.get_poses()

            (
                all_joint_states_pred,
                all_joint_types_pred,
            ) = carto_prediction.get_joint_state_and_type(timing_counter)
            print(timing_counter)
        elif cfg.METHOD == "asdf":
            (
                all_points_canonical_pred,
                abs_pose_outputs,
                all_joint_states_pred,
                all_joint_types_pred,
                peaks_pred,
            ) = asdf_evalutor.process_sample(
                carto_prediction.model_predictions,
                carto_prediction.panoptic_targets,
                use_gt_values=cfg.GT_ASDF,
                timing_counter=timing_counter,
            )
            print(timing_counter)
        else:
            assert False, "Unkown Method"

        # if len(all_pcds) > 0:
        #   o3d.io.write_point_cloud(f"{sample_id}_pred.ply", all_pcds[0])

        # Run evaluation
        all_points_canonical_gt = []
        peaks_gt = []
        all_joint_states_gt = []
        all_joint_types_gt = []
        class_ids_gt = []

        (
            abs_pose_outputs_target,
            img,
            img_nms,
            latent_embeddings_shape_gt,
            scores_out,
            peaks_target,
            latent_embeddings_arti_gt,
            indices_arti,
        ) = carto_prediction.panoptic_targets.graspable_objects_obbs[
            0
        ].compute_embs_and_poses(
            is_target=True
        )

        for arti_state in sample.val_data[0].articulated_object_states:
            gt_pc, gt_cls_id = get_gt_pc(
                arti_state.object_id,
                arti_state.global_scale,
                arti_state.raw_joint_config,
                cfg.n_points,
            )

            all_points_canonical_gt.append(np.copy(gt_pc))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_pc)
            # o3d.io.write_point_cloud(f"{sample_id}_gt.ply", pcd)

            # Weird issue with pose..
            pose = copy.deepcopy(arti_state.pose)
            pose.camera_T_object[1, :] *= -1
            pose.camera_T_object[2, :] *= -1

            ## Match peaks to poses
            for pose_gt_target, target_peak in zip(
                abs_pose_outputs_target, peaks_target
            ):
                if (
                    np.linalg.norm(
                        pose_gt_target.camera_T_object - pose.camera_T_object
                    )
                    > 1e-3
                ):
                    continue

                peaks_gt.append(target_peak)

            class_ids_gt.append(gt_cls_id)

            joint_def = arti_utils.get_joint_def_with_canonical_limits(
                arti_state.object_id, arti_state.global_scale
            )
            zero_joint_dict = arti_utils.get_zero_joint_dict(
                arti_state.raw_joint_config, joint_def, arti_state.global_scale
            )

            joint_id = list(zero_joint_dict.keys())[0]

            all_joint_states_gt.append(zero_joint_dict[joint_id])
            all_joint_types_gt.append(joint_def[joint_id]["type"])

        assert len(peaks_gt) == len(
            peaks_target
        ), "Not all peaks copied/too many copied"

        ####
        print("Matching")
        print(f"{peaks_gt}")
        print(f"{peaks_pred}")

        chamfer_results = []
        joint_type_results = []
        joint_state_results = []
        class_ids = []

        if len(peaks_pred) > 0:
            cost_matrix = np.linalg.norm(
                np.array(peaks_gt)[:, None, :] - np.array(peaks_pred), axis=-1
            )
            print(cost_matrix)
            gt_idx, pred_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

            for gt_idx, pred_idx in zip(gt_idx, pred_idx):
                class_ids.append(class_ids_gt[gt_idx])
                gt_points = all_points_canonical_gt[gt_idx]
                pred_points = all_points_canonical_pred[pred_idx]

                cd, _ = pytorch3d.loss.chamfer_distance(
                    torch.Tensor(gt_points).unsqueeze(0).cuda(),
                    torch.Tensor(pred_points).unsqueeze(0).cuda(),
                )

                # print(f"{float(cd) * 1000}")
                # print(all_joint_types_pred[pred_idx])
                # print(all_joint_types_gt[gt_idx])
                # print(float(all_joint_states_pred[pred_idx]))
                # print(floatall_joint_states_gt[gt_idx]))

                chamfer_results.append(float(cd))
                joint_type_results.append(
                    all_joint_types_gt[gt_idx] == all_joint_types_pred[pred_idx]
                )
                joint_state_results.append(
                    np.abs(
                        all_joint_states_pred[pred_idx] - all_joint_states_gt[gt_idx]
                    )
                )

        misdetections = len(peaks_gt) - len(peaks_pred)

        result = {}
        result["chamfer_gt_class_ids"] = np.array(class_ids)
        result["chamfer_results"] = chamfer_results
        result["joint_type_results"] = joint_type_results
        result["joint_state_results"] = joint_state_results
        result["misdetections"] = misdetections

        scores_out = np.zeros((len(all_points_canonical_pred),))

        # Do the simnet/partnet mobility transform
        all_pcs = []
        for idx in range(len(all_points_canonical_pred)):
            pred_pc = all_points_canonical_pred[idx]
            canonical_partnet_mobility_transform = partnet_mobility.TO_SIMNET_FRAME
            points_homo = camera.convert_points_to_homopoints(pred_pc.T)
            points_homo = canonical_partnet_mobility_transform.matrix @ points_homo
            pred_pc = camera.convert_homopoints_to_points(points_homo).T
            all_pcs.append(pred_pc)

        # Do the simnet/partnet mobility transform + get correct GT poses (verified this from vis script/notebook)

        all_points_gt = []
        abs_pose_targets = []
        class_ids_gt = []
        for arti_state in sample.val_data[0].articulated_object_states:
            n_samples = all_pcs[0].shape[0] if len(all_pcs) > 0 else 2048
            gt_pc, gt_cls_id = get_gt_pc(
                arti_state.object_id,
                arti_state.global_scale,
                arti_state.raw_joint_config,
                n_samples,
            )
            pose = copy.deepcopy(arti_state.pose)
            pose.camera_T_object[1, :] *= -1
            pose.camera_T_object[2, :] *= -1
            abs_pose_targets.append(pose)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_pc)
            # o3d.io.write_point_cloud(f"{sample_id}_gt.ply", pcd)

            # Do the correct transform thing
            canonical_partnet_mobility_transform = partnet_mobility.TO_SIMNET_FRAME
            points_homo = camera.convert_points_to_homopoints(gt_pc.T)
            points_homo = canonical_partnet_mobility_transform.matrix @ points_homo
            pc_uni = camera.convert_homopoints_to_points(points_homo).T
            all_points_gt.append(pc_uni)
            class_ids_gt.append(gt_cls_id)

        # Get the list of GT sizes and poses
        gt_sRT = np.zeros((len(all_points_gt), 4, 4), dtype=float)
        gt_size = np.zeros((len(all_points_gt), 6), dtype=float)  # New size

        for k in range(len(abs_pose_targets)):
            gt_pc = np.array(all_points_gt[k])
            gt_size_ = np.concatenate(
                (gt_pc.min(axis=0), gt_pc.max(axis=0))
            )  # New size

            gt_size[k] = gt_size_
            gt_sRT[k] = copy.deepcopy(
                abs_pose_targets[k].camera_T_object @ abs_pose_targets[k].scale_matrix
            )

        # Get predicted class ids
        seg_output = carto_prediction.model_predictions.room_segmentation[0]
        class_ids_predicted = get_ids_from_seg(
            seg_output, carto_prediction.get_peaks(), is_target=False  # ?
        )

        # Get the list of predicted sizes and poses
        f_sRT = np.zeros((len(all_pcs), 4, 4), dtype=float)
        f_size = np.zeros((len(all_pcs), 6), dtype=float)  # New size

        for k in range(len(all_pcs)):
            pred_pc = np.array(all_pcs[k])
            pred_size = np.concatenate(
                (pred_pc.min(axis=0), pred_pc.max(axis=0))
            )  # New size

            f_size[k] = pred_size
            f_sRT[k] = copy.deepcopy(
                abs_pose_outputs[k].camera_T_object @ abs_pose_outputs[k].scale_matrix
            )

        # Add to results
        result["gt_class_ids"] = np.array(class_ids_gt)
        result["gt_RTs"] = gt_sRT
        result["gt_scales"] = gt_size

        result["pred_class_ids"] = np.array(class_ids_predicted)
        result["pred_scores"] = np.array(scores_out)
        result["pred_RTs"] = f_sRT
        result["pred_scales"] = f_size

        pred_results.append(result)

        print(str(out_dir / "single_results" / f"{sample_id}.npz"))
        print(result)

        if len(pred_results) == cfg.MAX_COUNT:
            print(
                f"Skipped {((sample_id +1)- len(pred_results)) / (sample_id + 1)} empty samples"
            )
            break

        if cfg.write:
            (out_dir / "single_results").mkdir(exist_ok=True, parents=True)
            np.savez(
                str(out_dir / "single_results" / f"{sample_id}.npz"),
                result,
            )

        # print(timing_counter)

    print("Timing")
    # print(timing_counter)
    for key, values in timing_counter.items():
        print(f"{key}\t{np.nanmean(values)}")

    if cfg.write:
        compute_and_plot_mAP(out_dir, pred_results, nocs_results=None)


if __name__ == "__main__":
    cfg: FullEvalConfig = tyro.parse(FullEvalConfig)
    torch.random.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    main(cfg)
