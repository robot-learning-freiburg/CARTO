import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn

from CARTO.simnet.lib import datapoint
from torch.nn import functional as F
from CARTO.simnet.lib.net import losses

_mse_loss = losses.MSELoss()


class DepthOutput:
    def __init__(self, depth_pred, hparams):
        self.depth_pred = depth_pred
        self.is_numpy = False
        self.disp_loss = DisparityLoss(hparams.max_disparity, False)
        self.loss = nn.SmoothL1Loss(reduction="none")
        self.hparams = hparams

    # Converters for torch to numpy
    def convert_to_numpy_from_torch(self):
        self.depth_pred = np.ascontiguousarray(self.depth_pred.float().cpu().numpy())
        self.is_numpy = True
        return self.depth_pred

    def convert_to_torch_from_numpy(self):
        self.depth_pred[self.depth_pred > self.hparams.max_disparity] = (
            self.hparams.max_disparity - 1
        )
        self.depth_pred = torch.from_numpy(
            np.ascontiguousarray(self.depth_pred)
        ).float()
        self.is_numpy = False
        return self.depth_pred

    def get_prediction(self, is_target: bool = False):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        if is_target:
            return self.depth_pred
        else:
            return self.depth_pred[0]

    def get_visualization_img(
        self, left_img_np, corner_scale=1, raw_disp=True, is_target: bool = False
    ):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()

        if is_target:
            disp = self.depth_pred
        else:
            disp = self.depth_pred[0]

        if raw_disp:
            return disp_map_visualize(disp, self.hparams.max_disparity)
        disp_scaled = disp[::corner_scale, ::corner_scale]
        left_img_np[
            : disp_scaled.shape[0], -disp_scaled.shape[1] :
        ] = disp_map_visualize(disp_scaled, self.hparams.max_disparity)
        return left_img_np

    def compute_loss(self, depth_targets, log, name):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        depth_target_stacked = []
        for depth_target in depth_targets:
            depth_target_stacked.append(depth_target.depth_pred)
        depth_target_batch = torch.stack(depth_target_stacked)
        depth_target_batch = depth_target_batch.to(torch.device("cuda:0"))
        depth_loss = self.disp_loss(self.depth_pred, depth_target_batch)
        log[name] = depth_loss.item()
        return self.hparams.loss_depth_mult * depth_loss


class DisparityLoss(nn.Module):
    """Smooth L1-loss for disparity with check for valid ground truth"""

    def __init__(self, max_disparity, stdmean_scaled):
        super().__init__()

        self.max_disparity = max_disparity
        self.stdmean_scaled = stdmean_scaled
        self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, disparity, disparity_gt, right=False, low_range_div=None):
        # Scale ground truth disparity based on output scale.
        scale_factor = disparity_gt.shape[2] // disparity.shape[2]
        disparity_gt = downsample_disparity(disparity_gt, scale_factor)
        max_disparity = self.max_disparity / scale_factor
        if low_range_div is not None:
            max_disparity /= low_range_div

        # with torch.no_grad():
        #    valid_mask = get_disparity_valid_mask(disparity_gt, max_disparity, right)

        batch_size, _, _ = disparity.shape
        loss = torch.zeros(1, dtype=disparity.dtype, device=disparity.device)

        # Not all batch elements may have ground truth for disparity, so we compute the loss for each batch element
        # individually.
        valid_count = 0
        for batch_idx in range(batch_size):
            if torch.sum(disparity_gt[batch_idx, :, :]) < 1e-3:
                continue

            single_loss = self.loss(
                disparity[batch_idx, :, :], disparity_gt[batch_idx, :, :]
            )
            valid_count += 1

            if self.stdmean_scaled:
                # Scale loss by standard deviation and mean of ground truth to reduce influence of very high
                # disparities.
                gt_std, gt_mean = torch.std_mean(disparity_gt[batch_idx, :, :])
                loss += torch.mean(single_loss) / (gt_mean + 2.0 * gt_std)
            else:
                # Scale loss by scale factor due to difference of expected magnitude of disparity at different scales.
                loss += torch.mean(single_loss) * scale_factor
        # Avoid potential divide by 0.
        if valid_count > 0:
            return loss / batch_size
        else:
            return loss


def downsample_disparity(disparity, factor):
    """Downsample disparity using a min-pool operation

    Input can be either a Numpy array or Torch tensor.
    """
    with torch.no_grad():
        # Convert input to tensor at the appropriate number of dimensions if needed.
        is_numpy = type(disparity) == np.ndarray
        if is_numpy:
            disparity = torch.from_numpy(disparity)
        new_dims = 4 - len(disparity.shape)
        for i in range(new_dims):
            disparity = disparity.unsqueeze(0)

        disparity = F.max_pool2d(disparity, kernel_size=factor, stride=factor) / factor

        # Convert output disparity back into same format and number of dimensions as input.
        for i in range(new_dims):
            disparity = disparity.squeeze(0)
        if is_numpy:
            disparity = disparity.numpy()
        return disparity


def get_disparity_valid_mask(disparity, max_disparity, right=False):
    """Generate mask where disparity is valid based on the given max_disparity"""
    IGNORE_EDGE = False
    result = torch.logical_and(disparity > 1e-3, disparity < (max_disparity - 1 - 1e-3))
    if IGNORE_EDGE:
        width = disparity.shape[-1]
        edge_mask = (
            torch.arange(width, dtype=disparity.dtype, device=disparity.device) - 1
        )
        if right:
            edge_mask = torch.flip(edge_mask, (0,))
        edge_mask = edge_mask.expand_as(disparity)
        valid_edge = disparity < edge_mask
        result = torch.logical_and(result, valid_edge)
    return result


def turbo_vis(heatmap, normalize=False, uint8_output=False):
    assert len(heatmap.shape) == 2
    if normalize:
        heatmap = heatmap.astype(np.float32)
        heatmap -= np.min(heatmap)
        heatmap /= np.max(heatmap)
    assert heatmap.dtype != np.uint8

    x = heatmap
    x = x.clip(0, 1)
    a = (x * 255).astype(int)
    b = (a + 1).clip(max=255)
    f = x * 255.0 - a
    turbo_map = datapoint.TURBO_COLORMAP_DATA_NP[::-1]
    pseudo_color = turbo_map[a] + (turbo_map[b] - turbo_map[a]) * f[..., np.newaxis]
    pseudo_color[heatmap < 0.0] = 0.0
    pseudo_color[heatmap > 1.0] = 1.0
    if uint8_output:
        pseudo_color = (pseudo_color * 255).astype(np.uint8)
    return pseudo_color


def disp_map_visualize(x, max_disp):
    assert len(x.shape) == 2
    x = x.astype(np.float64)
    valid = (x < max_disp) & np.isfinite(x)
    if valid.sum() == 0:
        return np.zeros_like(x).astype(np.uint8)
    x -= np.min(x[valid])
    x /= np.max(x[valid])
    x = 1.0 - x
    x[~valid] = 0.0
    x = turbo_vis(x)
    x = (x * 255).astype(np.uint8)
    return x[:, :, ::-1]
