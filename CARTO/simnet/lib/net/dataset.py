# Copyright 2019 Toyota Research Institute.  All rights reserved.

import dataclasses
import os
import random
import pathlib
from typing import List, Tuple, Any

import cv2
import numpy as np
import torch
import IPython
import torch.nn.functional as F
from torch.utils.data import Dataset

from CARTO.simnet.lib import datapoint
from CARTO.simnet.lib.net.post_processing.segmentation_outputs import SegmentationOutput
from CARTO.simnet.lib.net.post_processing.depth_outputs import DepthOutput
from CARTO.simnet.lib.net.post_processing.pose_outputs import PoseOutput
from CARTO.simnet.lib.net.post_processing.obb_outputs import OBBOutput


def extract_left_numpy_img(anaglyph):
    anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
    anaglyph_np = anaglyph_np.transpose((1, 2, 0))
    left_img = anaglyph_np[..., 0:3] * 255.0
    return left_img


def extract_right_numpy_img(anaglyph):
    anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
    anaglyph_np = anaglyph_np.transpose((1, 2, 0))
    left_img = anaglyph_np[..., 3:6] * 255.0
    return left_img


def create_anaglyph(stereo_dp):
    height, width, _ = stereo_dp.left_color.shape
    image = np.zeros([height, width, 6], dtype=np.uint8)
    cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(stereo_dp.right_color, stereo_dp.right_color, 0, 255, cv2.NORM_MINMAX)
    image[..., 0:3] = stereo_dp.left_color
    image[..., 3:6] = stereo_dp.right_color
    image = image * 1.0 / 255.0
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).float()


# Struct for Panoptic Outputs
@dataclasses.dataclass
class PanopticOutputs:
    depth: list = dataclasses.field(default_factory=list)
    small_depth: list = dataclasses.field(default_factory=list)
    room_segmentation: List[SegmentationOutput] = dataclasses.field(
        default_factory=list
    )
    cabinet_door_obbs: List[OBBOutput] = dataclasses.field(default_factory=list)
    handhold_obbs: List[OBBOutput] = dataclasses.field(default_factory=list)
    graspable_objects_obbs: List[OBBOutput] = dataclasses.field(default_factory=list)
    grasp_quality_scores: list = dataclasses.field(default_factory=list)
    val_data: List[datapoint.ValData] = dataclasses.field(default_factory=list)
    stereo_imgs: list = dataclasses.field(default_factory=list)


def to_list(target):
    if target is None:
        return []
    target.convert_to_torch_from_numpy()
    return [target]


class Dataset(Dataset):
    def __init__(
        self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None
    ):
        super().__init__()

        if datapoint_dataset is None:
            datapoint_dataset = datapoint.make_dataset(dataset_uri)

        self.datapoint_handles = datapoint_dataset.list()
        # No need to shuffle, already shufled based on random uids
        self.hparams = hparams

        if preprocess_image_func is None:
            self.preprocces_image_func = create_anaglyph
        else:
            self.preprocces_image_func = preprocess_image_func

    def __len__(self):
        return len(self.datapoint_handles)

    def __getitem__(self, idx):
        dp: datapoint.Panoptic = self.datapoint_handles[idx].read()

        # Process image
        anaglyph = self.preprocces_image_func(dp.stereo)
        if dp.val_data.scene_name == "unlabeled_data":
            return PanopticOutputs(
                depth=[]
                if dp.depth is None
                else [DepthOutput(torch.Tensor(dp.depth), self.hparams)],
                room_segmentation=[],
                cabinet_door_obbs=[],
                handhold_obbs=[],
                graspable_objects_obbs=[],
                grasp_quality_scores=[],
                small_depth=[],
                val_data=[dp.val_data],
                stereo_imgs=[anaglyph],
                language=[],
            )

        # Segmenation targets
        segmentation_target = to_list(SegmentationOutput(dp.segmentation, self.hparams))

        # Ground truth disparity
        depth_target = to_list(DepthOutput(dp.depth, self.hparams))

        # OBB output heads
        if dp.cabinet_door_obb:
            cabinet_door_obb_target = OBBOutput(
                dp.cabinet_door_obb.heat_map,
                dp.cabinet_door_obb.vertex_target,
                dp.cabinet_door_obb.z_centroid,
                dp.cabinet_door_obb.cov_matrices,
                self.hparams,
                class_field=dp.cabinet_door_obb.classes,
            )
        else:
            cabinet_door_obb_target = None
        cabinet_door_obb_target = to_list(cabinet_door_obb_target)

        if dp.handhold_obb:
            handhold_obb_target = OBBOutput(
                dp.handhold_obb.heat_map,
                dp.handhold_obb.vertex_target,
                dp.handhold_obb.z_centroid,
                dp.handhold_obb.cov_matrices,
                self.hparams,
            )
        else:
            handhold_obb_target = None
        handhold_obb_target = to_list(handhold_obb_target)

        if dp.graspable_objects_obb:
            graspable_objects_obb_target = OBBOutput(
                dp.graspable_objects_obb.heat_map,
                dp.graspable_objects_obb.vertex_target,
                dp.graspable_objects_obb.z_centroid,
                dp.graspable_objects_obb.cov_matrices,
                self.hparams,
                class_field=dp.graspable_objects_obb.classes,
                shape_emb=dp.graspable_objects_obb.shape_emb,
                arti_emb=dp.graspable_objects_obb.arti_emb,
                abs_pose_field=dp.graspable_objects_obb.abs_pose,
            )
        else:
            graspable_objects_obb_target = None
        graspable_objects_obb_target = to_list(graspable_objects_obb_target)

        # Grasp quality
        # grasp_quality_scores_target = GraspOutput(
        #    dp.grasps.heat_map, dp.grasps.grasp_success_target, self.hparams
        # )
        # Convert targets to pytorch
        # grasp_quality_scores_target.convert_to_torch_from_numpy()

        # Add the language input to panoptic outputs
        return PanopticOutputs(
            depth=depth_target,
            room_segmentation=segmentation_target,
            cabinet_door_obbs=cabinet_door_obb_target,
            handhold_obbs=handhold_obb_target,
            graspable_objects_obbs=graspable_objects_obb_target,
            grasp_quality_scores=[],
            small_depth=[],
            val_data=[dp.val_data],
            stereo_imgs=[anaglyph],
        )


def panoptic_collate(batch, rgbd=False) -> Tuple[torch.Tensor, Any, PanopticOutputs]:
    # list of elements per patch
    # Each element is a tuple of (stereo,imgs)
    panoptic_targets = PanopticOutputs()
    stereo_images_list = []

    for ii in range(len(batch)):
        panoptic_targets.depth.extend(batch[ii].depth)
        panoptic_targets.room_segmentation.extend(batch[ii].room_segmentation)
        panoptic_targets.cabinet_door_obbs.extend(batch[ii].cabinet_door_obbs)
        panoptic_targets.handhold_obbs.extend(batch[ii].handhold_obbs)
        panoptic_targets.graspable_objects_obbs.extend(batch[ii].graspable_objects_obbs)
        panoptic_targets.grasp_quality_scores.extend(batch[ii].grasp_quality_scores)
        panoptic_targets.val_data.extend(batch[ii].val_data)
        stereo_images_list.extend(batch[ii].stereo_imgs)

    stereo_images_torch = torch.stack(stereo_images_list)
    if rgbd:
        stereo_images_torch = torch.cat(
            (
                stereo_images_torch[:, :3, ...],
                torch.stack(
                    [po_target.depth_pred for po_target in panoptic_targets.depth]
                ).unsqueeze(1),
            ),
            dim=1,
        )

    return stereo_images_torch, panoptic_targets
