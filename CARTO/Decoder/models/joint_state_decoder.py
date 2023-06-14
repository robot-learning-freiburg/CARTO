from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from CARTO.Decoder import config


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, weight_normalizer=lambda x: x):
        super(ClassificationHead, self).__init__()

        lin_state = weight_normalizer(nn.Linear(in_dim, 1))  # Continous prediction
        lin_type = weight_normalizer(
            nn.Linear(in_dim, 2, bias=False)
        )  # 0: revolute, 1: prismatic
        setattr(self, "lin_state", lin_state)
        setattr(self, "lin_type", lin_type)

    def forward(self, input):
        lin_state = getattr(self, "lin_state")
        lin_type = getattr(self, "lin_type")

        state_pred = lin_state(input)
        type_pred = torch.sigmoid(lin_type(input))
        return {"state": state_pred, "type": type_pred}


class ZeroOneHead(nn.Module):
    def __init__(self, in_dim, weight_normalizer=lambda x: x):
        super(ZeroOneHead, self).__init__()
        lin_module = weight_normalizer(nn.Linear(in_dim, 1))
        setattr(self, "lin_module", lin_module)

    def forward(self, input):
        lin_module = getattr(self, "lin_module")
        pred = torch.sigmoid(lin_module(input))
        return {"state": pred}


class JointStateDecoder(nn.Module):
    def __init__(
        self,
        cfg: config.JointStateDecoderModelConfig,
        joint_config_latent_code_dim: int = 16,
    ):
        super(JointStateDecoder, self).__init__()
        self.joint_config_latent_code_dim = joint_config_latent_code_dim

        dims = [joint_config_latent_code_dim] + cfg.dims
        self.num_layers = len(dims)

        weight_normalizer = config.get_weight_normalizer(cfg.weight_normalizer)

        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            linear_layer = weight_normalizer(nn.Linear(dims[layer], out_dim))
            # linear_layer = nn.utils.weight_norm(linear_layer)
            setattr(self, "lin" + str(layer), linear_layer)

        if cfg.output_head == config.JointDecoderOutputHeadStyle.CLASSIFICATION:
            out_head_class = ClassificationHead
        elif cfg.output_head == config.JointDecoderOutputHeadStyle.ZERO_ONE_HEAD:
            out_head_class = ZeroOneHead
        else:
            raise ModuleNotFoundError(f"Unknown output head {cfg.output_head}")

        setattr(
            self,
            "output_head",
            out_head_class(dims[-1], weight_normalizer=weight_normalizer),
        )

        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, input):
        assert (
            input.size()[-1] == self.joint_config_latent_code_dim
        ), f"{input.size()[-1]} == {self.joint_config_latent_code_dim}"

        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            x = lin(x)
            x = self.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        output_head = getattr(self, "output_head")
        return output_head(x)
