import numpy as np
from scipy.stats import multivariate_normal

from CARTO.simnet.lib import datapoint

_DOWNSCALE_VALUE = 1
_PEAK_CONCENTRATION = 0.8


def compute_network_targets(boxes, masks, height, width):
    if len(boxes) == 0:
        return datapoint.Box(
            heat_map=np.zeros([height, width]),
            vertex_target=np.zeros([height, width, 4]),
        )
    heatmaps = compute_heatmaps_from_masks(masks)
    vertex_target = compute_vertex_field(boxes, heatmaps)
    return datapoint.Box(heat_map=np.max(heatmaps, axis=0), vertex_target=vertex_target)


def compute_network_targets_from_detections(
    detections, occ_threshold, min_height, truncation_level, height, width
):
    detections_marked, masks = mark_ignore_in_box_detections(detections)
    ignore_mask = np.zeros([height, width])
    for detection, mask in zip(detections, masks):
        if not detetion.ignore:
            boxes.append(detection)
            masks.append(masks)
        else:
            ignore_mask[mask] = 1.0
    if len(boxes) == 0:
        return datapoint.Box(
            heat_map=np.zeros([height, width]),
            vertex_target=np.zeros([height, width, 4]),
        )
    heatmaps = compute_heatmaps_from_masks(masks)
    vertex_target = compute_vertex_field(boxes, heatmaps)
    return datapoint.Box(
        heat_map=np.max(heatmaps, axis=0),
        vertex_target=vertex_target,
        ignore_mask=ignore_mask,
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


def compute_vertex_field(bboxes, heatmaps):
    H, W = heatmaps[0].shape[0], heatmaps[0].shape[1]
    # For each vertex compute the displacement field.
    disp_fields = []
    vertex_target = np.zeros(
        [len(bboxes), int(H / _DOWNSCALE_VALUE), int(W / _DOWNSCALE_VALUE), 4]
    )
    heatmap_indices = np.argmax(np.array(heatmaps), axis=0)
    for i in range(2):
        vertex_points = []
        coords = np.indices([H, W])
        coords = coords.transpose((1, 2, 0))
        for box_idx, bbox, heatmap in zip(range(len(bboxes)), bboxes, heatmaps):
            disp_field = np.zeros([H, W, 2])
            vertex_point = np.array([bbox[i][0], bbox[i][1]])
            mask = heatmap_indices == box_idx
            disp_field[mask] = coords[mask] - vertex_point
            # Normalize by height and width
            disp_field[mask, 0] = 1.0 - (disp_field[mask, 0] + H) / (2 * H)
            disp_field[mask, 1] = 1.0 - (disp_field[mask, 1] + W) / (2 * W)
            vertex_target[box_idx, :, :, (2 * i) : (2 * i) + 2] = disp_field[
                ::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE
            ]
    return np.max(vertex_target, axis=0)
