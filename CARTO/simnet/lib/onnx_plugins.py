"""Plugins that can be used in an ONNX model."""
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from CARTO.simnet.lib.net.models.layers.cost_volume import (
    dot_product_cost_volume,
    DotProductCostVolume,
)
from CARTO.simnet.lib.net.models.layers.soft_argmin import SoftArgmin, soft_argmin

# TODO(krishnashankar): Arguments of functions in modules below
# differ from those of the base class(es) they inherit from, and
# pylint complains. For now, disable here and consider disabling
# globally.

# pylint: disable=arguments-differ
# pylint: disable=protected-access


@torch.autograd.function.traceable
class ExportableUpsampleFunction(torch.autograd.Function):
    """Upsample function that can be traced for ONNX export."""

    @staticmethod
    def symbolic(g, inputs, scale_factor):
        assert scale_factor == 2, "Only 2x upsample implemented"
        return g.op(
            "TRT_PluginV2",
            inputs,
            version_s="0.0.1",
            namespace_s="",
            data_s="",
            name_s="UpsampleBilinearEvenSquare",
        )

    @staticmethod
    def forward(ctx, inputs, scale_factor):
        return F.interpolate(
            inputs, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

    @staticmethod
    def backward(_):
        raise RuntimeError("Backward not implemented")


class ExportableUpsample(nn.Module):
    """Upsample module that can be used in an ONNX model."""

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return ExportableUpsampleFunction.apply(inputs, self.scale_factor)


class UpsampleWithConvTranspose(nn.Module):
    """Upsample model implemented with transposed convolution."""

    def __init__(self, scale_factor):
        super(UpsampleWithConvTranspose, self).__init__()
        self.weights = None
        self.scale_factor = utils._pair(scale_factor)

        def check_scale_factor(scale_factor):
            assert scale_factor == 1 or scale_factor % 2 == 0

        check_scale_factor(self.scale_factor[0])
        check_scale_factor(self.scale_factor[1])

    def get_kernel_size(self, factor):
        return 2 * factor - factor % 2

    def bilinear_upsample_kernel(self, size):
        """Get a transpoed convolution kernel that implemented upsampling for the
        given size."""

        def get_factor_and_center(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            return factor, center

        factor_h, center_h = get_factor_and_center(size[0])
        factor_w, center_w = get_factor_and_center(size[1])
        og = np.ogrid[: size[0], : size[1]]
        return (1 - abs((og[0] - center_h) / factor_h)) * (
            1 - abs((og[1] - center_w) / factor_w)
        )

    def bilinear_upsample_weights(self, factor, nchannels):
        """Get transposed convolution weights for upsampling."""
        filter_size_h = self.get_kernel_size(factor[0])
        filter_size_w = self.get_kernel_size(factor[1])

        weights = np.zeros(
            (filter_size_h, filter_size_w, nchannels, nchannels), dtype=np.float32
        )

        kernel = self.bilinear_upsample_kernel((filter_size_h, filter_size_w))

        for c in range(nchannels):
            weights[:, :, c, c] = kernel

        return weights

    def forward(self, inputs):
        in_channels = inputs.shape[1]
        if self.weights is None:
            weights = self.bilinear_upsample_weights(self.scale_factor, in_channels)
            # Order weights to be compatible with pytorch (in_channels, out_channels, height, width).
            self.weights = (
                torch.from_numpy(weights.transpose(2, 3, 0, 1))
                .to(inputs.device)
                .type(inputs.dtype)
            )
        output = torch.nn.functional.conv_transpose2d(
            inputs,
            self.weights,
            stride=self.scale_factor,
            padding=(self.scale_factor[0] // 2, self.scale_factor[1] // 2),
        )
        return output


@torch.autograd.function.traceable
class ExportableDotProductCostVolumeFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, left, right, num_disparities, is_right):
        assert not is_right
        serialized_data = struct.pack("<iiiii", num_disparities, 0, 0, 0, 0)
        return g.op(
            "TRT_PluginV2",
            left,
            right,
            version_s="0.0.1",
            namespace_s="",
            data_s=serialized_data,
            name_s="DotProductCostVolume",
        )

    @staticmethod
    def forward(ctx, left, right, num_disparities, is_right):
        return dot_product_cost_volume(left, right, num_disparities, is_right)


class ExportableDotProductCostVolume(nn.Module):
    def __init__(self, num_disparities, is_right):
        super().__init__()
        self.num_disparities = num_disparities
        self.is_right = is_right

    def forward(self, left, right):
        return ExportableDotProductCostVolumeFunction.apply(
            left, right, self.num_disparities, self.is_right
        )


@torch.autograd.function.traceable
class ExportableSoftArgminFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op(
            "TRT_PluginV2",
            input,
            version_s="0.0.1",
            namespace_s="",
            data_s="",
            name_s="SoftArgmin",
        )

    @staticmethod
    def forward(ctx, input):
        return soft_argmin(input)


class ExportableSoftArgmin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ExportableSoftArgminFunction.apply(input)


def fix_module_train(module):
    """Replace all modules in the given module with ONNX-compatible modules."""
    for child_module_name, child_module in module.named_children():
        if isinstance(child_module, nn.Upsample):
            scale_factor = int(child_module.scale_factor)
            # TensorRT plugin can only load 2x upsample from ONNX currently, so
            # otherwise use transposed convolution.
            if False and scale_factor == 2:
                module._modules[child_module_name] = ExportableUpsample(scale_factor)
            else:
                module._modules[child_module_name] = UpsampleWithConvTranspose(
                    scale_factor
                )
        elif len(list(child_module.children())) > 0:
            fix_module_train(child_module)


def fix_module_onnx(module):
    """Replace all modules in the given module with ONNX-compatible modules."""
    for child_module_name, child_module in module.named_children():
        if isinstance(child_module, nn.Upsample):
            scale_factor = int(child_module.scale_factor)
            # TensorRT plugin can only load 2x upsample from ONNX currently, so
            # otherwise use transposed convolution.
            if False and scale_factor == 2:
                module._modules[child_module_name] = ExportableUpsample(scale_factor)
            else:
                module._modules[child_module_name] = UpsampleWithConvTranspose(
                    scale_factor
                )
        elif isinstance(child_module, DotProductCostVolume):
            num_disparities = child_module.num_disparities
            is_right = child_module.is_right
            module._modules[child_module_name] = ExportableDotProductCostVolume(
                num_disparities, is_right
            )
        elif isinstance(child_module, SoftArgmin):
            module._modules[child_module_name] = ExportableSoftArgmin()
        elif len(list(child_module.children())) > 0:
            fix_module_onnx(child_module)
