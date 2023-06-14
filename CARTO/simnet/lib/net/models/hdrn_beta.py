from typing import Optional

import torch
import torch.nn as nn

from CARTO.simnet.lib.net.models.layers.stochastic_depth import StochasticDepth

# Using ReLU saves a bunch of vram at the cost of some quality. If you don't care about vram, use SiLU.
# ACTIVATION = nn.ReLU(inplace=True)
ACTIVATION = nn.SiLU(inplace=True)


def get_sd_rates(sd_rate, num_blocks):
    total_blocks = sum(num_blocks)
    if sd_rate is not None and sd_rate > 0.0:
        step = sd_rate / (total_blocks - 1)
    else:
        step = None

    result = []
    count = 0
    for block_size in num_blocks:
        subresult = []
        for _ in range(block_size):
            if step is not None:
                subresult.append(count * step)
            else:
                subresult.append(None)
            count += 1
        result.append(subresult)
    return result


class SimpleTransitionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        *,
        norm=nn.BatchNorm2d,
        activation=ACTIVATION
    ):
        assert stride in (1, 2, 4)
        assert not (in_channels == out_channels and stride == 1)
        super().__init__()

        if stride == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride, bias=False
            )
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, inputs):
        return self.activation(self.norm(self.conv(inputs)))


class BasicResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        dilation_rate=1,
        add_preact,
        add_last_norm,
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=None
    ):
        super().__init__()
        if add_preact:
            self.preact_norm = norm(channels)
        else:
            self.preact_norm = None
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            bias=False,
        )
        self.norm1 = norm(channels)
        self.activation = activation
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            bias=False,
        )
        self.norm_last = norm(channels) if add_last_norm else None

        if stochastic_depth_rate is not None and stochastic_depth_rate > 0.0:
            self.stochastic_depth = StochasticDepth(stochastic_depth_rate, mode="row")
        else:
            self.stochastic_depth = None

    def forward(self, inputs):
        if self.preact_norm is not None:
            outputs = self.activation(self.preact_norm(inputs))
        else:
            outputs = inputs
        outputs = self.activation(self.norm1(self.conv1(outputs)))
        outputs = self.conv2(outputs)
        if self.stochastic_depth is not None:
            outputs = self.stochastic_depth(outputs)
        outputs += inputs

        if self.norm_last is not None:
            outputs = self.activation(self.norm_last(outputs))
        return outputs


def resnet_group(
    block_func,
    channels,
    num_blocks,
    *,
    dilation_rates=[1],
    norm=nn.BatchNorm2d,
    activation=ACTIVATION,
    stochastic_depth_rates=[None]
):
    assert len(dilation_rates) > 0
    assert len(stochastic_depth_rates) > 0

    residual_blocks = [
        block_func(
            channels,
            dilation_rate=dilation_rates[idx % len(dilation_rates)],
            stochastic_depth_rate=stochastic_depth_rates[
                idx % len(stochastic_depth_rates)
            ],
            add_preact=idx > 0,
            add_last_norm=idx == num_blocks - 1,
            norm=norm,
            activation=activation,
        )
        for idx in range(num_blocks)
    ]
    return nn.Sequential(*residual_blocks)


class HybridDilatedResNetBeta(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        dilation_rates,
        norm,
        activation,
        stochastic_depth_rate,
        aux_channels,
    ):
        super().__init__()
        assert len(num_blocks) == 3 or len(num_blocks) == 4

        self.num_channels = num_channels

        stochastic_depth_rates = get_sd_rates(stochastic_depth_rate, num_blocks)

        self.stem = SimpleTransitionBlock(
            3, num_channels[0], stride=4, norm=norm, activation=activation
        )

        if aux_channels is not None:
            self.aux_merge = nn.Sequential(
                BasicResidualBlock(
                    num_channels[0] + aux_channels,
                    add_preact=False,
                    add_last_norm=False,
                    norm=norm,
                    activation=activation,
                ),
                BasicResidualBlock(
                    num_channels[0] + aux_channels,
                    add_preact=True,
                    add_last_norm=True,
                    norm=norm,
                    activation=activation,
                ),
                SimpleTransitionBlock(
                    num_channels[0] + aux_channels, num_channels[0], stride=1
                ),
            )
        else:
            self.aux_merge = None

        self.group1 = resnet_group(
            BasicResidualBlock,
            num_channels[0],
            num_blocks[0],
            stochastic_depth_rates=stochastic_depth_rates[0],
            norm=norm,
            activation=activation,
        )

        self.transition2 = SimpleTransitionBlock(
            num_channels[0], num_channels[1], stride=2, norm=norm, activation=activation
        )
        self.group2 = resnet_group(
            BasicResidualBlock,
            num_channels[1],
            num_blocks[1],
            stochastic_depth_rates=stochastic_depth_rates[1],
            norm=norm,
            activation=activation,
        )

        self.transition3 = SimpleTransitionBlock(
            num_channels[1], num_channels[2], stride=2, norm=norm, activation=activation
        )
        self.group3 = resnet_group(
            BasicResidualBlock,
            num_channels[2],
            num_blocks[2],
            dilation_rates=dilation_rates,
            stochastic_depth_rates=stochastic_depth_rates[2],
            norm=norm,
            activation=activation,
        )

        if len(num_blocks) == 4:
            assert len(num_channels) == len(num_blocks)
            self.transition4 = SimpleTransitionBlock(
                num_channels[2],
                num_channels[3],
                stride=2,
                norm=norm,
                activation=activation,
            )
            self.group4 = resnet_group(
                BasicResidualBlock,
                num_channels[3],
                num_blocks[3],
                stochastic_depth_rates=stochastic_depth_rates[3],
                norm=norm,
                activation=activation,
            )
        else:
            self.transition4 = None
            self.group4 = None

    def get_output_channels(self):
        return self.num_channels

    def forward(self, inputs, aux_inputs: Optional[torch.Tensor] = None):
        outputs = self.stem(inputs)
        if aux_inputs is not None:
            assert self.aux_merge is not None
            outputs = torch.cat([outputs, aux_inputs], dim=1)
            outputs = self.aux_merge(outputs)
        group1_outputs = self.group1(outputs)

        outputs = self.transition2(group1_outputs)
        group2_outputs = self.group2(outputs)

        outputs = self.transition3(group2_outputs)
        group3_outputs = self.group3(outputs)

        if self.group4 is not None:
            outputs = self.transition4(group3_outputs)
            group4_outputs = self.group4(outputs)

            return [group1_outputs, group2_outputs, group3_outputs, group4_outputs]
        else:
            return [group1_outputs, group2_outputs, group3_outputs]


def hdrn_beta_base(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8, 3],
        num_channels=[64, 128, 256, 512],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_narrow(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8, 3],
        num_channels=[32, 64, 128, 256],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_wide(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8, 3],
        num_channels=[128, 256, 512, 1024],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_mid(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8, 3],
        num_channels=[64, 128, 160, 256],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_base3g(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8],
        num_channels=[64, 128, 256],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_narrow3g(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8],
        num_channels=[32, 64, 128],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )


def hdrn_beta_wide3g(sd_rate=0.1, aux_channels=None):
    return HybridDilatedResNetBeta(
        num_blocks=[3, 4, 8],
        num_channels=[128, 256, 512],
        dilation_rates=[1, 2, 5, 9],
        norm=nn.BatchNorm2d,
        activation=ACTIVATION,
        stochastic_depth_rate=sd_rate,
        aux_channels=aux_channels,
    )
