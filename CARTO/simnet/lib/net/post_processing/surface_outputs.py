import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn

from CARTO.simnet.lib import datapoint
from torch.nn import functional as F
from CARTO.simnet.lib.net import losses

_masked_l1_loss = losses.MaskedL1Loss()
_MAX_DISP = 128


class SurfaceOutput:
    def __init__(self, surface_pred, hparams):
        self.surface_pred = surface_pred
        self.is_numpy = False
        self.loss = nn.SmoothL1Loss(reduction="none")
        self.hparams = hparams

    # Converters for torch to numpy
    def convert_to_numpy_from_torch(self):
        self.surface_pred = np.ascontiguousarray(self.surface_pred.cpu().numpy())
        self.surface_pred.transpose((1, 2, 0))
        self.is_numpy = True

    def convert_to_torch_from_numpy(self):
        self.surface_pred.transpose((2, 0, 1))
        self.surface_pred = torch.from_numpy(
            np.ascontiguousarray(self.surface_pred)
        ).float()
        self.is_numpy = False

    def get_visualization_img(self, left_img_np):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()

        surface = self.surface_pred[0]
        downscale_factor = int(left_img_np.shape[0] / disp.shape[0])
        left_img = left_img_np[::downscale_factor, ::downscale_factor]
        viz_img = np.zeros([left_img.shape[0] * 2, left_img.shape[1], 3])
        viz_img[0 : left_img.shape[0], :, :] = left_img
        viz_img[left_img.shape[0] : left_img.shape[0] + disp.shape[0], :, :] = surface
        return viz_img

    def compute_loss(self, surface_targets, log):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        surface_target_stacked = []
        for surface_target in surface_targets:
            surface_target_stacked.append(surface_target.surface_pred)
        surface_target_batch = torch.stack(surface_target_stacked)
        surface_target_batch = surface_target_batch.to(torch.device("cuda:0"))
        mask = torch.sum(surface_target_batch, axis=1) > 0
        surface_loss = self.loss(surface_target_batch, self.surface_pred, mask)
        log["surface"] = surface_loss
        return self.hparams.loss_surface_mult * surface_loss
