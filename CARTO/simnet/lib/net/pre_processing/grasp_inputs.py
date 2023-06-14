import numpy as np
from scipy.stats import multivariate_normal

from CARTO.simnet.lib.net.pre_processing import pose_inputs
from CARTO.simnet.lib.non_convex_grasper import _NUM_GRASPS_PER_OBJECT
from CARTO.simnet.lib import datapoint

_HEATMAP_THRESHOLD = 0.3
_DOWNSCALE_VALUE = 8
_PEAK_CONCENTRATION = 0.8


def compute_network_targets(grasps, masks, height, width):
    assert len(grasps) == len(masks)
    if len(grasps) == 0:
        height_d = int(height / _DOWNSCALE_VALUE)
        width_d = int(width / _DOWNSCALE_VALUE)
        return datapoint.Grasps(
            heat_map=np.zeros([height, width]),
            grasp_success_target=np.zeros([height_d, width_d, _NUM_GRASPS_PER_OBJECT]),
        )
    heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
    grasp_success_target = compute_grasp_target(grasps, heatmaps)
    return datapoint.Grasps(
        heat_map=np.max(heatmaps, axis=0),
        grasp_success_target=grasp_success_target,
    )


def compute_grasp_target(grasps_per_objects, heat_maps, threshold=0.3):
    grasp_target = np.zeros(
        [
            len(grasps_per_objects),
            heat_maps[0].shape[0],
            heat_maps[0].shape[1],
            _NUM_GRASPS_PER_OBJECT,
        ]
    )
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for grasps_per_object, heat_map, ii in zip(
        grasps_per_objects, heat_maps, range(len(heat_maps))
    ):
        grasp_values = np.zeros(_NUM_GRASPS_PER_OBJECT)
        mask = heatmap_indices == ii
        for jj, grasp in enumerate(grasps_per_object):
            grasp_values[jj] = grasp.success
        grasp_target[ii, mask] = grasp_values
    return np.sum(grasp_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]
