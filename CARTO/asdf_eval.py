# Ensure mesh_to_sdf is imported first
import copy
import dataclasses
import pathlib
import sys
import time
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import mesh_to_sdf
import numpy as np
import open3d as o3d
import seaborn as sns
import torch
import trimesh
from CARTO.lib import partnet_mobility
from CARTO.simnet.lib.net.post_processing import obb_outputs, pose_outputs

from CARTO.simnet.lib import camera

sns.set()

FIG_DPI = 400

## Adding ASDF to our search path
# For code release: should or maybe could be a submodule
ASDF_BASE_PATH = "external_libs/A-SDF/"
sys.path.append(ASDF_BASE_PATH)

import asdf
import helpers_pc as asdf_helpers_pc
from skimage.transform import rescale, resize

categories = [
    "background",
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


@dataclasses.dataclass
class ASDF_Preloaded:
    decoder: "ASDF_Decoder"
    specs: "ASDF_specs"


def get_masks_out(
    seg_output, is_target: bool = False, detections=None, median_mask_filter=21
):
    if is_target:
        category_seg_output = np.ascontiguousarray(seg_output.seg_pred)
    else:
        category_seg_output = np.ascontiguousarray(seg_output.seg_pred)
        category_seg_output = np.argmax(category_seg_output[0], axis=0)

    # In GT we might have thin rendering issues due to simnet data generation
    category_seg_output = cv2.medianBlur(
        category_seg_output[..., None].astype(np.uint8), median_mask_filter
    )

    obj_ids = np.unique(category_seg_output)
    obj_ids = obj_ids[1:]
    print(obj_ids)
    masks_target = category_seg_output == obj_ids[:, None, None]
    contour_centers = []
    act_contours = []
    for m in range(len(obj_ids)):
        contours, _ = cv2.findContours(
            masks_target[m].astype(np.uint8).copy(),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_TC89_L1,
        )

        for c in range(len(contours)):
            if cv2.contourArea(contours[c]) < 650:
                continue
            act_contours.append(contours[c])
            moments = cv2.moments(contours[c])
            if moments["m00"] == 0.0 or moments["m00"] == 0.0:
                continue
            try:
                contour_centers.append(
                    (
                        int(moments["m01"] / moments["m00"]),
                        int(moments["m10"] / moments["m00"]),
                    )
                )
            except:
                print("Encounterd something bad")
                continue

    contour_centers = np.array(contour_centers)
    mask_out = []
    class_ids = []
    class_ids_name = []
    class_centers = []
    for l in range(len(act_contours)):
        out = np.zeros_like(seg_output.seg_pred)
        viz = act_contours[l]
        idx = act_contours[l]
        center = contour_centers[l]
        temp = np.copy(contour_centers)
        temp[l] = 1000
        distance = np.linalg.norm(temp - center, axis=1)
        closest_index = np.argmin(distance)
        if distance[closest_index] < 30:
            if cv2.contourArea(act_contours[l]) < cv2.contourArea(
                act_contours[closest_index]
            ):
                continue
        out = np.zeros_like(category_seg_output)
        class_id_from_mask = category_seg_output[center[0], center[1]]

        cv2.drawContours(out, [idx], -1, 255, cv2.FILLED, 1)

        # Before we continue, check if peaks are in the mask
        merged_masks = False
        if detections is not None:
            in_mask_detections = np.array(
                [
                    detection
                    for detection in detections
                    if out[detection[0], detection[1]] == 255.0
                ]
            )
            if in_mask_detections.shape[0] > 1:
                merged_masks = True

        if not merged_masks:
            class_ids.append(np.int(class_id_from_mask))
            class_ids_name.append(class_id_from_mask)
            class_centers.append((center[0], center[1]))
            mask_out.append(out)
        else:
            masks_out, centers = split_mask_by_euclidean_distance(
                out, in_mask_detections
            )
            mask_out.extend(masks_out)
            class_ids_name.extend([class_id_from_mask] * len(masks_out))
            class_centers.extend(centers)

    mask_out = np.array(mask_out)
    assert len(mask_out) == len(class_ids_name)
    assert len(mask_out) == len(class_centers)
    return mask_out, class_ids_name, np.array(class_centers)


def split_mask_by_euclidean_distance(mask, in_mask_detections):
    W = np.moveaxis(np.indices(mask.shape), 0, -1)
    print(W.shape)
    distance_field_per_pixel = (
        np.ones((mask.shape[0], mask.shape[1], in_mask_detections.shape[0])) * np.inf
    )
    for i, detection in enumerate(in_mask_detections):
        distance_field_per_pixel[..., i] = np.linalg.norm(W - detection, axis=-1)

    detection_association = np.argmin(distance_field_per_pixel, axis=-1)

    masks_out = []
    centers = []

    for i, detection in enumerate(in_mask_detections):
        print(np.count_nonzero(detection_association == i))
        new_mask = np.logical_and(mask == 255, detection_association == i)

        new_mask = new_mask * 255.0
        contours, _ = cv2.findContours(
            new_mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
        )

        for c in range(len(contours)):
            moments = cv2.moments(contours[c])
            if moments["m00"] == 0.0 or moments["m00"] == 0.0:
                continue
            try:
                centers.append(
                    (
                        int(moments["m01"] / moments["m00"]),
                        int(moments["m10"] / moments["m00"]),
                    )
                )
                # Only add once
                break
            except:
                print("Encounterd something bad")
                continue

        masks_out.append(new_mask)

    return masks_out, centers


class ASDF_Evaluator:
    def __init__(
        self, _CAMERA, normal_scale: float = 0.25, final_samples_amount: int = 20000
    ):
        ## Preload ASDF
        asdf_experiment_base_dir = pathlib.Path(ASDF_BASE_PATH)
        categories_to_load = [
            "Dishwasher",
            "Laptop",
            "Microwave",
            "Oven",
            "Refrigerator",
            "Table",
            "WashingMachine",
        ]
        self.preloaded_asdf_dict: Dict[str, ASDF_Preloaded] = {}
        for category in categories_to_load:
            asdf_experiment_dir = (
                asdf_experiment_base_dir / "examples" / "CARTO" / category
            )
            decoder, specs = asdf.load_from_experiment(
                asdf_experiment_dir, base_path_ours="", checkpoint=1000
            )
            self.preloaded_asdf_dict[category] = ASDF_Preloaded(
                decoder=decoder.cpu(), specs=specs
            )

        self._CAMERA = _CAMERA
        self.normal_scale = normal_scale
        self.final_samples_amount = final_samples_amount

    def process_sample(
        self,
        model_predictions,
        panoptic_targets,
        use_gt_values: bool = False,
        use_gt_disp: bool = True,
        sample_idx: True = 0,
        write_output=False,
        timing_counter={"detection": [], "optimization": [], "reconstruction": []},
    ):
        scale = self.normal_scale

        if use_gt_disp:
            disp = panoptic_targets.depth[0].get_prediction(is_target=True)
        else:
            disp = model_predictions.depth[0].get_prediction(is_target=False)

        if use_gt_values:
            model_predictions = panoptic_targets

        depth = camera.disp_to_depth(
            disp, self._CAMERA.K_matrix, self._CAMERA.stereo_baseline
        )

        start_normal_calculation = time.time()
        points_all = camera.convert_homopoints_to_points(
            self._CAMERA.deproject_depth_image(depth, use_RT=False)
        ).T
        points_all = points_all.reshape(depth.shape[0], depth.shape[1], 3)

        H, W = depth.shape
        surface_normals_all = asdf_helpers_pc.surface_normal(
            points_all[:: int(1 / scale), :: int(1 / scale), :],
            int(scale * H),
            int(scale * W),
        )
        surface_normals_all = resize(surface_normals_all, (H, W), anti_aliasing=True)
        timing_counter["detection"][-1] += time.time() - start_normal_calculation

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_all.reshape(-1, 3))
        pcd.normals = o3d.utility.Vector3dVector(surface_normals_all.reshape(-1, 3))
        o3d.io.write_point_cloud(f"{sample_idx}_reprojected_pointcloud_full.ply", pcd)

        model_predictions.graspable_objects_obbs[0].convert_to_numpy_from_torch()
        # Get Detections to filter two instances
        heatmap = model_predictions.graspable_objects_obbs[0].heatmap[0]

        start_peak_detection = time.time()
        detections, _, _ = obb_outputs.compute_peaks(heatmap)
        detections = detections.astype(int)

        model_predictions.room_segmentation[0].convert_to_numpy_from_torch()
        mask_out, class_ids, class_centers = get_masks_out(
            model_predictions.room_segmentation[0],
            is_target=use_gt_values,
            detections=detections,
        )
        print(class_centers)
        timing_counter["detection"][-1] += time.time() - start_peak_detection

        if len(class_centers) == 0:
            return [], [], [], [], []

        if write_output:
            plt.figure()
            # plt.imshow(model_predictions.room_segmentation[0].seg_pred)
            for idx in range(mask_out.shape[0]):
                plt.figure()
                plt.imshow(mask_out[idx, ...])
                plt.scatter(class_centers[idx][1], class_centers[idx][0])
                plt.scatter(detections[:, 1], detections[:, 0])
                # plt.savefig()

        abs_pose_field = np.copy(
            model_predictions.graspable_objects_obbs[0].abs_pose_field
        )[0]
        # This is always batched?
        # if not use_gt_values:
        #   abs_pose_field = abs_pose_field[0]
        abs_pose_outputs = pose_outputs.extract_abs_pose_from_peaks(
            np.copy(class_centers), abs_pose_field
        )

        plt.close("all")

        # Starting Main Loop

        all_points_2d = []
        all_points_3d_canonical = []
        all_abs_poses = []
        all_joint_states = []
        all_joint_types = []
        all_class_centers = []

        for idx, detection in enumerate(class_centers):
            # Class centers: C x 2
            instance_id = class_ids[idx]
            if instance_id == 0:
                continue
            abs_pose = abs_pose_outputs[idx]
            instance_mask = mask_out[idx] == 255.0
            category = categories[instance_id]
            # print(f"Found {category = }")
            all_class_centers.append(detection)

            # All Points
            points = points_all[instance_mask]
            normals = surface_normals_all[instance_mask]

            if write_output:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.normals = o3d.utility.Vector3dVector(normals)
                o3d.io.write_point_cloud(
                    f"{sample_idx}_{idx+1}_reprojected_pointcloud.ply", pcd
                )

                # points_2d_repro = viz_utils.project(_CAMERA.K_matrix, camera.convert_points_to_homopoints(points.T))
                # print(points_2d_repro)
                # img = viz_utils.overlay_projected_points(rgb_img / 255, points_2d_repro[None, ...])
                # img = _CAMERA.project_points_to_depth_img(camera.convert_points_to_homopoints(points.T))
                # plt.figure()
                # plt.imshow(img)

            start_optimiziation = time.time()

            points_homo = camera.convert_points_to_homopoints(
                points.T
            )  # In camera frame
            points_trans_homo = np.linalg.inv(abs_pose.camera_T_object) @ points_homo
            points_trans_homo = np.linalg.inv(abs_pose.scale_matrix) @ points_trans_homo
            points_trans_homo = (
                np.linalg.inv(partnet_mobility.TO_SIMNET_FRAME.matrix)
                @ points_trans_homo
            )
            points_trans = camera.convert_homopoints_to_points(
                points_trans_homo
            )  # In Object Centric Frame

            # Transform normals
            normals_trans = np.linalg.inv(abs_pose.camera_T_object[:3, :3]) @ normals.T
            normals_trans = (
                np.linalg.inv(partnet_mobility.TO_SIMNET_FRAME.matrix[:3, :3])
                @ normals_trans
            )

            pos, neg = asdf_helpers_pc.generate_SDF(points_trans.T, normals_trans.T)

            if write_output:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_trans.T)
                pcd.normals = o3d.utility.Vector3dVector(normals_trans.T)
                o3d.io.write_point_cloud(
                    f"{sample_idx}_{idx+1}_reprojected_pointcloud_canonical.ply", pcd
                )

            # Get ASDF-decoder
            decoder = self.preloaded_asdf_dict[category].decoder.cuda()
            specs = self.preloaded_asdf_dict[category].specs

            # Stack according to A-SDF convention

            data_sdf = [[torch.Tensor(pos), torch.Tensor(neg)], torch.Tensor(0), -1]

            ## Call to A-SDF
            err, joint_state, lat_vec, atc_vec = asdf.asdf_reconstruct.reconstruct(
                decoder,
                int(800),  # Iterations
                specs["CodeLength"],
                data_sdf,
                specs["ClampingDistance"],
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                articulation=specs["Articulation"],
                specs=specs,
                infer_with_gt_atc=False,
                num_atc_parts=specs["NumAtcParts"],
                do_sup_with_part=specs["TrainWithParts"],
                return_atc_value=True,  # Instead of returning the error we return the actual value
            )

            # print(f"{err = }")
            # print(f"{joint_state = }")

            timing_counter["optimization"] += [time.time() - start_optimiziation]

            file_name = (
                f"{sample_idx}_{idx+1}_asdf_mesh" if write_output else "tmp_mesh"
            )

            start_reconstruction = time.time()

            asdf.mesh.create_mesh(
                decoder,
                lat_vec,
                file_name,
                N=256,
                max_batch=int(2**18),
                atc_vec=atc_vec,
                do_sup_with_part=specs["TrainWithParts"],
                specs=specs,
            )
            decoder.cpu()

            # Resample to get PC
            reconstruction = trimesh.load(file_name + ".ply")
            points_canonical = reconstruction.sample(self.final_samples_amount)

            timing_counter["reconstruction"] += [time.time() - start_reconstruction]

            # Leaving this here as this could be useful for someone
            # points_mesh = camera.convert_points_to_homopoints(points_canonical.T)
            # points_mesh = partnet_mobility.TO_SIMNET_FRAME.matrix @ points_mesh
            # points_mesh = abs_pose.scale_matrix @ points_mesh
            # points_mesh = abs_pose.camera_T_object @ points_mesh
            # points_2d_mesh = viz_utils.project(self._CAMERA.K_matrix, points_mesh)
            # points_2d_mesh = points_2d_mesh.T
            # all_points_2d.append(points_2d_mesh)

            joint_type = "prismatic" if category == "Table" else "revolute"

            all_points_3d_canonical.append(np.copy(points_canonical))
            all_abs_poses.append(copy.deepcopy(abs_pose))
            all_joint_states.append(
                joint_state / 180 * np.pi
            )  # Convert to rad for comparison
            all_joint_types.append(joint_type)

        return (
            all_points_3d_canonical,
            all_abs_poses,
            all_joint_states,
            all_joint_types,
            all_class_centers,
        )
