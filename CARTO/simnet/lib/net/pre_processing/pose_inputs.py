import numpy as np
from scipy.stats import multivariate_normal

from CARTO.simnet.lib.net.post_processing import epnp
from CARTO.simnet.lib.label import Pose
from CARTO.simnet.lib import datapoint

_HEATMAP_THRESHOLD = 0.3
_DOWNSCALE_VALUE = 8
_PEAK_CONCENTRATION = 0.8


def compute_network_targets(poses, masks, camera_model):
    heatmaps = compute_heatmaps_from_masks(masks)
    vertex_target = compute_vertex_field(poses, heatmaps, camera_model)
    z_centroid = compute_z_centroid_field(poses, heatmaps)
    return datapoint.Pose(
        heat_map=np.max(heatmaps, axis=0),
        vertex_target=vertex_target,
        z_centroid=z_centroid,
    )


def compute_heatmaps_from_masks(masks):
    heatmaps = [compute_heatmap_from_mask(mask) for mask in masks]
    return heatmaps


def compute_heatmap_from_mask(mask):
    if np.sum(mask) == 0:
        raise ValueError("Mask is empty")
    coords = np.indices(mask.shape)
    coords = coords.reshape([2, -1]).T
    mask_f = mask.flatten()
    indices = coords[np.where(mask_f > 0)]
    mean_value = np.floor(np.average(indices, axis=0))
    cov = np.cov((indices - mean_value).T)
    cov = cov * _PEAK_CONCENTRATION
    multi_var = multivariate_normal(mean=mean_value, cov=cov)
    density = multi_var.pdf(coords)
    heat_map = np.zeros(mask.shape)
    heat_map[coords[:, 0], coords[:, 1]] = density
    return heat_map / np.max(heat_map)


def compute_vertex_field(poses, heatmaps, camera_model):
    H, W = heatmaps[0].shape[0], heatmaps[0].shape[1]
    # Compute the projected box pixels.
    boxes = []
    for pose in poses:
        pose_no_rot = Pose(
            camera_T_object=pose.camera_T_no_rot_object, scale_matrix=pose.scale_matrix
        )
        boxes.append(epnp.project_pose_onto_image(pose_no_rot, camera_model))
    # For each vertex compute the displacement field.
    disp_fields = []
    vertex_target = np.zeros(
        [len(poses), int(H / _DOWNSCALE_VALUE), int(W / _DOWNSCALE_VALUE), 16]
    )
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    for i in range(8):
        vertex_points = []
        coords = np.indices([H, W])
        coords = coords.transpose((1, 2, 0))
        for box_idx, bbox, heatmap in zip(range(len(boxes)), boxes, heatmaps):
            disp_field = np.zeros([H, W, 2])
            vertex_point = np.array([bbox[i][0], bbox[i][1]])
            mask = heatmap_indices == box_idx
            disp_field[mask] = coords[mask] - vertex_point
            # Normalize by height and width
            disp_field[mask, 0] = 1.0 - (disp_field[mask, 0] + H) / (2 * H)
            disp_field[mask, 1] = 1.0 - (disp_field[mask, 1] + W) / (2 * W)
            vertex_target[box_idx, :, :, (2 * i) : 2 * i + 2] = disp_field[
                ::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE
            ]
    return np.max(vertex_target, axis=0)


def compute_z_centroid_field(poses, heatmaps):
    z_centroid_target = np.zeros(
        [len(poses), heatmaps[0].shape[0], heatmaps[0].shape[1]]
    )
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    for pose, heat_map, ii in zip(poses, heatmaps, range(len(heatmaps))):
        mask = heatmap_indices == ii
        z_centroid_target[ii, mask] = pose.camera_T_object[2, 3]
    # Normalize z_centroid by 1. and multiply by 10 to avoid tensorrt float precision issues.
    return np.sum(z_centroid_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]
