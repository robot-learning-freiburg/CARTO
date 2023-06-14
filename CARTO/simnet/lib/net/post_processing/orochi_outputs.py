import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn
from torch.nn import functional as F

from CARTO.simnet.lib import color_stuff
from CARTO.simnet.lib import datapoint
from CARTO.simnet.lib.net.dataset import PanopticOutputs


def visualize_img(
    panoptic_outputs: PanopticOutputs,
    c_img,
    camera_model,
    class_list,
    poses=False,
    prune_distance=False,
    is_target=False,
):
    c_img = np.copy(c_img)

    c_img = panoptic_outputs.room_segmentation[0].get_visualization_img(
        c_img, is_target=is_target
    )

    if len(panoptic_outputs.handhold_obbs) > 0:
        c_img = panoptic_outputs.handhold_obbs[0].get_visualization_img(
            0, c_img, camera_model=camera_model, poses=poses
        )
    if len(panoptic_outputs.cabinet_door_obbs) > 0:
        c_img = panoptic_outputs.cabinet_door_obbs[0].get_visualization_img(
            0, c_img, camera_model=camera_model, class_list=[], poses=poses
        )
    if len(panoptic_outputs.graspable_objects_obbs) > 0:
        c_img = panoptic_outputs.graspable_objects_obbs[0].get_visualization_img(
            0,
            c_img,
            camera_model=camera_model,
            class_list=class_list,
            prune_distance=prune_distance,
            poses=poses,
        )

    return c_img


def visualize_heatmap(panoptic_outputs: PanopticOutputs, c_img):
    if len(panoptic_outputs.graspable_objects_obbs) > 0:
        # print(panoptic_outputs.graspable_objects_obbs[0].heatmap.shape)
        # print(np.max(panoptic_outputs.graspable_objects_obbs[0]))
        # print(np.min(panoptic_outputs.graspable_objects_obbs[0].heatmap.shape))
        heatmap = cv2.applyColorMap(
            (
                np.clip(
                    panoptic_outputs.graspable_objects_obbs[0].heatmap[0, ...], 0.0, 1.0
                )
                * 255.0
            ).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        gray = cv2.cvtColor(c_img.copy(), cv2.COLOR_RGB2GRAY).astype(np.uint8)
        gray_full = np.zeros_like(heatmap)
        gray_full[..., 0] = gray
        gray_full[..., 1] = gray
        gray_full[..., 2] = gray
        return cv2.addWeighted(gray_full, 0.9, heatmap.astype(np.uint8), 0.4, 0)
