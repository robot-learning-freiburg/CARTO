import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn
from torch.nn import functional as F

from CARTO.simnet.lib import color_stuff
from CARTO.simnet.lib import datapoint

# Panoptic Segmentation Colors


class SegmentationOutput:
    def __init__(self, seg_pred, hparams):
        self.seg_pred = seg_pred
        self.is_numpy = False
        self.hparams = hparams

    # Converters for torch to numpy
    def convert_to_numpy_from_torch(self):
        self.seg_pred = np.ascontiguousarray(self.seg_pred.float().cpu().numpy())
        self.is_numpy = True

    def convert_to_torch_from_numpy(self):
        self.seg_pred = torch.from_numpy(np.ascontiguousarray(self.seg_pred)).long()
        self.is_numpy = False

    def get_visualization_img(self, left_image, is_target=False):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        if is_target:
            seg_mask = self.seg_pred
            max_number = int(seg_mask.max()) + 1
        else:
            seg_mask = np.argmax(self.seg_pred, axis=1)[0]
            assert self.seg_pred.ndim == 4  # 1 x L x H x W
            max_number = int(self.seg_pred.shape[1])
        return draw_segmentation_mask(left_image, seg_mask, num_classes=max_number)

    def get_visualization_img_with_categories(
        self, left_image, detections, class_list, is_target=False
    ):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        if is_target:
            seg_mask_predictions = self.seg_pred
        else:
            seg_mask_predictions = np.argmax(self.seg_pred[0], axis=0)

        return draw_segmentation_mask_with_categories(
            left_image, seg_mask_predictions, detections, class_list
        )

    def get_prediction(self):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        return self.seg_pred[0]

    def compute_loss(self, seg_targets, log, name):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        seg_target_stacked = []
        for seg_target in seg_targets:
            seg_target_stacked.append(seg_target.seg_pred)
        seg_target_batch = torch.stack(seg_target_stacked)
        seg_target_batch = seg_target_batch.to(torch.device("cuda:0"))
        if len(seg_target_batch.shape) == 4:
            seg_target_batch = torch.argmax(seg_target_batch, dim=1)
        seg_loss = F.cross_entropy(
            self.seg_pred, seg_target_batch, reduction="mean", ignore_index=-100
        )
        log[name] = seg_loss.item()
        return self.hparams.loss_seg_mult * seg_loss


def draw_segmentation_mask(color_img, seg_mask, num_classes=7):
    assert len(seg_mask.shape) == 2
    seg_mask = seg_mask.astype(np.uint8)
    # TODO(mike.laskey) Replace this with a set list.
    if num_classes == 7:
        colors = color_stuff.get_panoptic_colors()
    else:
        colors = color_stuff.get_colors(num_classes)

    color_img = color_img_to_gray(color_img)
    for ii, color in zip(range(num_classes), colors):
        if ii == 0:  # ignore background class
            continue

        colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
        colored_mask[seg_mask == ii, :] = color
        color_img = cv2.addWeighted(
            color_img.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
        )
    return cv2.cvtColor(color_img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def color_img_to_gray(image):
    gray_scale_img = np.zeros(image.shape)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        gray_scale_img[:, :, i] = img
    gray_scale_img[:, :, i] = img
    return gray_scale_img


def draw_segmentation_mask_with_categories(color_img, seg_mask, detections, class_list):
    assert len(seg_mask.shape) == 2
    seg_mask = seg_mask.astype(np.int)
    seg_mask_vis = draw_segmentation_mask(
        color_img, seg_mask, num_classes=len(class_list)
    )
    for detection in detections:
        pixel_x = int(detection[0])
        pixel_y = int(detection[1])

        category_id = seg_mask[pixel_x, pixel_y]
        category = class_list[category_id]

        if category.name == "background":
            color = (255, 0, 0)  # dark blue
        else:
            color = (255, 128, 128)  # light blue

        seg_mask_vis = cv2.putText(
            seg_mask_vis,
            category.name,
            (pixel_y, pixel_x),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
    return seg_mask_vis
