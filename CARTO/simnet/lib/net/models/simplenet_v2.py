from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from CARTO.simnet.lib.net.models.hdrn_beta import (
    SimpleTransitionBlock,
    BasicResidualBlock,
    hdrn_beta_narrow,
    hdrn_beta_base,
    hdrn_beta_narrow3g,
    hdrn_beta_base3g,
    ACTIVATION,
)

# STEREO_BACKBONE = partial(hdrn_beta_narrow, sd_rate=0.1)
STEREO_BACKBONE = partial(hdrn_beta_narrow3g, sd_rate=0.1)
# BACKBONE = partial(hdrn_beta_base, sd_rate=0.1)
BACKBONE = partial(hdrn_beta_base3g, sd_rate=0.1)

# The new stereo is better but there is no fast variant right now. When then is true, the old stereo cost volume
# processing is used (with new backbone).
BAD_BUT_FAST_STEREO = True
# When using new stereo, you can use this to make it faster at some loss of accuracy.
CV_DISPARITY_STRIDE = 1
DISPARITY_AUX_CHANNELS = 32


@torch.jit.script
def strided_grouped_cost_volume(
    left, right, num_disparities: int, groups: int, stride: int
):
    batch_size, channels, height, width = left.shape
    channels_per_group = channels // groups

    left = left.reshape(batch_size, groups, channels_per_group, height, width)
    right = right.reshape(batch_size, groups, channels_per_group, height, width)

    output = torch.zeros(
        (batch_size, channels_per_group, num_disparities // stride + 1, height, width),
        dtype=left.dtype,
        device=left.device,
    )

    for i in range(0, num_disparities + 1, stride):
        output[:, :, i // stride, :, i:] = torch.mean(
            left[:, :, :, :, i:] * right[:, :, :, :, : width - i], dim=1
        )

    return output


class DynamicCostVolume(nn.Module):
    def forward(self, left, right, num_disparities: int, groups: int, stride: int):
        return strided_grouped_cost_volume(left, right, num_disparities, groups, stride)


@torch.jit.script
def strided_soft_argmin(input, stride: int):
    _, channels, _, _ = input.shape

    softmin = F.softmin(input, dim=1)
    index_tensor = (
        torch.arange(0, channels, dtype=softmin.dtype, device=softmin.device).view(
            1, channels, 1, 1
        )
        * stride
    )
    output = torch.sum(softmin * index_tensor, dim=1, keepdim=True)
    return output


class DynamicSoftArgmin(nn.Module):
    def forward(self, input, stride: int):
        return strided_soft_argmin(input, stride)


class SimpleHead(nn.Module):
    def __init__(self, in_channels, out_channels, output_upsample=None):
        super().__init__()

        idxs = []
        convs = []
        for idx, channels in enumerate(in_channels):
            idxs.append(idx)
            convs.append(nn.Conv2d(channels, out_channels, kernel_size=1, bias=True))
        self.idxs = idxs[::-1]
        self.convs = nn.ModuleList(convs[::-1])

        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        if output_upsample is not None:
            self.upsample_out = nn.Upsample(
                scale_factor=output_upsample, mode="bilinear", align_corners=False
            )
        else:
            self.upsample_out = None

    def forward(self, group_outputs: List[torch.Tensor]):
        outputs = None
        for idx, module in enumerate(self.convs):
            current = module(group_outputs[self.idxs[idx]])
            if outputs is None:
                outputs = current
            else:
                outputs = self.upsample2(outputs) + current

        if self.upsample_out is not None:
            outputs = self.upsample_out(outputs)
        return outputs


class BasicCostVolumeResidualBlock3d(nn.Module):
    def __init__(
        self,
        channels,
        *,
        add_preact,
        add_last_norm,
        norm=nn.BatchNorm3d,
        activation=ACTIVATION
    ):
        super().__init__()
        if add_preact:
            self.preact_norm = norm(channels)
        else:
            self.preact_norm = None
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm(channels)
        self.activation = activation
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm_last = norm(channels) if add_last_norm else None

    def forward(self, inputs):
        if self.preact_norm is not None:
            outputs = self.activation(self.preact_norm(inputs))
        else:
            outputs = inputs
        outputs = self.activation(self.norm1(self.conv1(outputs)))
        outputs = self.conv2(outputs)
        outputs = inputs + outputs

        if self.norm_last is not None:
            outputs = self.activation(self.norm_last(outputs))
        return outputs


class ProcessCostVolume(nn.Module):
    """Process cost volume and prepare for soft argmin operation"""

    def __init__(self, in_channels):
        super().__init__()

        activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.block3d_1 = BasicCostVolumeResidualBlock3d(
            in_channels, add_preact=False, add_last_norm=False, activation=activation
        )
        self.block3d_2 = BasicCostVolumeResidualBlock3d(
            in_channels, add_preact=True, add_last_norm=False, activation=activation
        )
        self.block3d_3 = BasicCostVolumeResidualBlock3d(
            in_channels, add_preact=True, add_last_norm=False, activation=activation
        )
        self.block3d_4 = BasicCostVolumeResidualBlock3d(
            in_channels, add_preact=True, add_last_norm=True, activation=activation
        )
        self.conv_out = nn.Conv3d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, inputs):
        outputs = self.block3d_1(inputs)
        outputs = self.block3d_2(outputs)
        outputs = self.block3d_3(outputs)
        outputs = self.block3d_4(outputs)
        outputs = self.conv_out(outputs)
        outputs = torch.flatten(outputs, 1, 2)

        return outputs


class ProcessCostVolumeBadButFast(nn.Module):
    """Process cost volume and prepare for soft argmin operation"""

    def __init__(self, num_disparities):
        super().__init__()

        activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.block2d_1 = BasicResidualBlock(
            num_disparities,
            add_preact=False,
            add_last_norm=False,
            activation=activation,
        )
        self.block2d_2 = BasicResidualBlock(
            num_disparities, add_preact=True, add_last_norm=False, activation=activation
        )
        self.block2d_3 = BasicResidualBlock(
            num_disparities, add_preact=True, add_last_norm=False, activation=activation
        )
        self.block2d_4 = BasicResidualBlock(
            num_disparities, add_preact=True, add_last_norm=True, activation=activation
        )
        self.conv_out = nn.Conv2d(
            num_disparities, num_disparities, kernel_size=1, bias=True
        )

    def forward(self, inputs):
        outputs = torch.flatten(inputs, 1, 2)
        outputs = self.block2d_1(outputs)
        outputs = self.block2d_2(outputs)
        outputs = self.block2d_3(outputs)
        outputs = self.block2d_4(outputs)
        outputs = self.conv_out(outputs)

        return outputs


class HdrnBetaStereo(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.num_disparities = hparams.max_disparity
        self.internal_scale = hparams.cost_volume_downsample_factor
        self.internal_num_disparities = self.num_disparities // self.internal_scale
        assert self.internal_scale in [4, 8, 16]
        if self.internal_scale == 4:
            self.offset = 0
        elif self.internal_scale == 8:
            self.offset = 1
        elif self.internal_scale == 16:
            self.offset = 2
        else:
            assert False

        self.features = STEREO_BACKBONE()

        if BAD_BUT_FAST_STEREO:
            self.groups = 32
            score_features = self.groups
            self.process_cost_volume = ProcessCostVolumeBadButFast(
                self.internal_num_disparities + 1
            )
            self.stride = 1
        else:
            cv_features = 8
            self.groups = 8
            score_features = self.groups * cv_features
            self.process_cost_volume = ProcessCostVolume(cv_features)
            self.stride = CV_DISPARITY_STRIDE

        self.score_features = SimpleHead(
            self.features.get_output_channels()[self.offset :], score_features
        )
        self.cost_volume = DynamicCostVolume()
        self.soft_argmin = DynamicSoftArgmin()

    def forward(self, left_image, right_image):
        left_features = self.features(left_image)
        right_features = self.features(right_image)

        left_score = self.score_features(left_features[self.offset :])
        right_score = self.score_features(right_features[self.offset :])

        cost_volume = self.cost_volume(
            left_score,
            right_score,
            num_disparities=self.internal_num_disparities,
            groups=self.groups,
            stride=self.stride,
        )
        cost_volume = self.process_cost_volume(cost_volume)

        disparity_small = self.soft_argmin(cost_volume, stride=self.stride)

        return disparity_small


class StereoBackbone(nn.Module):
    def __init__(self, hparams, aux_channels=None):
        super().__init__()

        self.stereo = HdrnBetaStereo(hparams)

        self.disparity_features = nn.Sequential(
            SimpleTransitionBlock(1, DISPARITY_AUX_CHANNELS, stride=1),
            BasicResidualBlock(
                DISPARITY_AUX_CHANNELS, add_preact=False, add_last_norm=True
            ),
        )
        aux_channels = DISPARITY_AUX_CHANNELS + aux_channels

        self.features = BACKBONE(aux_channels=aux_channels)

    def forward(self, stacked_img, language_features=None):
        left = stacked_img[:, 0:3]
        right = stacked_img[:, 3:6]
        small_disparity = self.stereo(left, right)

        disparity_features = self.disparity_features(small_disparity)
        aux_inputs = torch.cat([disparity_features, language_features], dim=1)

        all_features = self.features(left, aux_inputs)

        outputs = {}
        for idx, features in enumerate(all_features):
            outputs["p{}".format(idx + 2)] = features
        outputs["small_disp"] = small_disparity
        return outputs
