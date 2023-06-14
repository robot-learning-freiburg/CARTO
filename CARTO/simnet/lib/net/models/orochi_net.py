import collections
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from CARTO.simnet.lib.net.models import simplenet
from CARTO.simnet.lib.net.dataset import PanopticOutputs
from CARTO.simnet.lib.net.post_processing import (
    segmentation_outputs,
    depth_outputs,
    pose_outputs,
    obb_outputs,
)

MODEL_SEM_SEG_HEAD_IN_FEATURES = ["p2", "p3", "p4"]
MODEL_POSE_HEAD_IN_FEATURES = ["p3", "p4"]

MODEL_SEM_SEG_HEAD_IGNORE_VALUE = 255
MODEL_SEM_SEG_HEAD_COMMON_STRIDE = 4
MODEL_POSE_HEAD_COMMON_STRIDE = 8
MODEL_SEM_SEG_HEAD_LOSS_WEIGHT = 1.0


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
          norm (nn.Module, optional): a normalization layer
          activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:],
                    self.padding,
                    self.dilation,
                    self.kernel_size,
                    self.stride,
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    """
    Args:
      norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
        or a callable that takes a channel number and returns
        the normalization layer as a nn.Module.

    Returns:
      nn.Module or None: the normalization layer
    """
    if out_channels == 32:
        N = 16
    else:
        N = 32
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": torch.nn.BatchNorm2d,
            # "SyncBN": NaiveSyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(N, channels),
            # "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, input_shape, num_classes, model_norm="BN", num_filters_scale=4):
        super().__init__()
        MODEL_SEM_SEG_HEAD_NORM = model_norm
        MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale

        self.in_features = MODEL_SEM_SEG_HEAD_IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
        conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
        self.common_stride = MODEL_SEM_SEG_HEAD_COMMON_STRIDE
        norm = MODEL_SEM_SEG_HEAD_NORM
        self.bilinear_upsample = nn.Upsample(
            scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=partial(F.relu, inplace=True),
                )
                c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(
            conv_dims, num_classes, kernel_size=1, stride=1, padding=0
        )
        c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        """
        Returns:
          In training, returns (None, dict of losses)
          In inference, returns (predictions, {})
        """
        x = self.layers(features)
        x = self.bilinear_upsample(x)
        return x

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](F.relu(features[f]))
            else:
                x = x - -self.scale_heads[i](F.relu(features[f]))
        x = self.predictor(x)
        return x

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        return loss


class PoseFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, input_shape, num_classes, model_norm="BN", num_filters_scale=4):
        super().__init__()
        MODEL_SEM_SEG_HEAD_NORM = model_norm
        MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale
        self.in_features = MODEL_POSE_HEAD_IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
        conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
        self.common_stride = MODEL_POSE_HEAD_COMMON_STRIDE
        norm = MODEL_SEM_SEG_HEAD_NORM

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=partial(F.relu, inplace=True),
                )
                c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        )
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(
            conv_dims, num_classes, kernel_size=1, stride=1, padding=0
        )
        c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        """
        Returns:
          In training, returns (None, dict of losses)
          In inference, returns (predictions, {})
        """
        x = self.layers(features)
        return x

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](F.relu(features[f]))
            else:
                x = x - -self.scale_heads[i](F.relu(features[f]))
        x = self.predictor(x)
        return x


class ShapeSpec(
    collections.namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])
):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.

    Attributes:
      channels:
      height:
      width:
      stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


def res_fpn(hparams):
    return PanopticNet(hparams)


class DepthHead(nn.Module):
    def __init__(self, backbone_output_shape_4x, backbone_output_shape_8x, hparams):
        super().__init__()
        self.head = SemSegFPNHead(
            backbone_output_shape_4x,
            num_classes=1,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        self.hparams = hparams

    def forward(self, features):
        depth_pred = self.head(features)
        depth_pred = depth_pred.squeeze(dim=1)
        return depth_outputs.DepthOutput(depth_pred, self.hparams)


class SegmentationHead(nn.Module):
    def __init__(
        self, backbone_output_shape_4x, backbone_output_shape_8x, num_classes, hparams
    ):
        super().__init__()
        self.head = SemSegFPNHead(
            backbone_output_shape_4x,
            num_classes=num_classes,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        self.hparams = hparams

    def forward(self, features):
        pred = self.head(features)
        return segmentation_outputs.SegmentationOutput(pred, self.hparams)


class OBBHead(nn.Module):
    def __init__(
        self,
        backbone_output_shape_4x,
        backbone_output_shape_8x,
        hparams,
        num_object_classes=0,
        use_latent_emb=False,
        use_abs_pose=False,
    ):
        super().__init__()
        self.hparams = hparams
        self.heatmap_head = SemSegFPNHead(
            backbone_output_shape_4x,
            num_classes=1,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )

        self.vertex_head = PoseFPNHead(
            backbone_output_shape_8x,
            num_classes=16,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )

        self.z_centroid_head = PoseFPNHead(
            backbone_output_shape_8x,
            num_classes=1,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )

        self.rotation_head = PoseFPNHead(
            backbone_output_shape_8x,
            num_classes=6,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        if num_object_classes > 0:
            self.class_head = PoseFPNHead(
                backbone_output_shape_8x,
                num_classes=num_object_classes,
                model_norm=hparams.model_norm,
                num_filters_scale=hparams.num_filters_scale,
            )
        else:
            self.class_head = None

        self.latent_shape_embedding_head = (
            PoseFPNHead(
                backbone_output_shape_8x,
                num_classes=hparams.shape_embedding_size,
                model_norm=hparams.model_norm,
                num_filters_scale=hparams.num_filters_scale,
            )
            if use_latent_emb
            else None
        )

        self.latent_arti_embedding_head = (
            PoseFPNHead(
                backbone_output_shape_8x,
                num_classes=hparams.joint_embedding_size,
                model_norm=hparams.model_norm,
                num_filters_scale=hparams.num_filters_scale,
            )
            if use_latent_emb
            else None
        )

        self.abs_pose_head = (
            PoseFPNHead(
                backbone_output_shape_8x,
                num_classes=13,  # in that order: 9 for rot, 3 for trans,1 for scale --> 9+3+1=13
                model_norm=hparams.model_norm,
                num_filters_scale=hparams.num_filters_scale,
            )
            if use_abs_pose
            else None
        )

    def forward(self, features):
        z_centroid_output = self.z_centroid_head(features).squeeze(dim=1)
        heatmap_output = self.heatmap_head(features).squeeze(dim=1)
        vertex_output = self.vertex_head(features)
        rotation_output = self.rotation_head(features)

        classes_ouptut = (
            self.class_head(features) if self.class_head is not None else None
        )

        latent_shape_output = (
            self.latent_shape_embedding_head(features)
            if self.latent_shape_embedding_head is not None
            else None
        )

        latent_arti_output = (
            self.latent_arti_embedding_head(features)
            if self.latent_arti_embedding_head is not None
            else None
        )

        abs_pose_output = (
            self.abs_pose_head(features) if self.abs_pose_head is not None else None
        )

        return obb_outputs.OBBOutput(
            heatmap_output,
            vertex_output,
            z_centroid_output,
            rotation_output,
            self.hparams,
            class_field=classes_ouptut,
            shape_emb=latent_shape_output,
            arti_emb=latent_arti_output,
            abs_pose_field=abs_pose_output,
        )


class PanopticNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams.model_rgbd:
            self.backbone = simplenet.PrecomputedStereoBackbone(
                hparams
            )  # in_channels is not used
        else:
            self.backbone = simplenet.StereoBackbone(hparams, in_dim=64)

        # ResFPN used p2,p3,p4,p5 (32 channels)
        # DRN uses only p2,p3,p4 (no need for p5 since dilation increases striding naturally)
        backbone_output_shape_4x = {
            "p2": ShapeSpec(channels=64, height=None, width=None, stride=4),
            "p3": ShapeSpec(channels=64, height=None, width=None, stride=8),
            "p4": ShapeSpec(channels=64, height=None, width=None, stride=16),
        }

        backbone_output_shape_8x = {
            "p3": ShapeSpec(channels=64, height=None, width=None, stride=8),
            "p4": ShapeSpec(channels=64, height=None, width=None, stride=16),
        }

        # Add segmentation head.
        # TODO(mike.laskey) Move number classes to hparams
        self.seg_head = SegmentationHead(
            backbone_output_shape_4x,
            backbone_output_shape_8x,
            7
            + len(
                hparams.object_categories
            ),  # TODO could be (1 + categories) for including bg?
            hparams,
        )
        # Add pose heads.
        # TODO Nick: door and handholds are removed completely currently
        self.cabinet_door_obb_head = OBBHead(
            backbone_output_shape_4x,
            backbone_output_shape_8x,
            hparams,
            num_object_classes=2,
        )
        self.handholds_obb_head = OBBHead(
            backbone_output_shape_4x, backbone_output_shape_8x, hparams
        )
        self.graspable_objects_obb_head = OBBHead(
            backbone_output_shape_4x,
            backbone_output_shape_8x,
            hparams,
            num_object_classes=len(hparams.object_categories),
            use_latent_emb=True,
            use_abs_pose=True,
        )
        # Add depth head.
        self.depth_head = DepthHead(
            backbone_output_shape_4x, backbone_output_shape_8x, hparams
        )

    def forward(self, image, language=None):
        features = self.backbone(image)
        if not self.hparams.model_rgbd:
            small_disp_output = features["small_disp"]
            small_disp_output = small_disp_output.squeeze(dim=1)
            small_depth_output = depth_outputs.DepthOutput(
                small_disp_output, self.hparams
            )
            small_depth = [small_depth_output]
            depth = [self.depth_head(features)]
        else:
            small_depth = []
            depth = []

        return PanopticOutputs(
            small_depth=small_depth,
            depth=depth,
            room_segmentation=[self.seg_head(features)],
            # cabinet_door_obbs=[self.cabinet_door_obb_head(features)],
            # handhold_obbs=[self.handholds_obb_head(features)],
            # TODO Nick: Remove completely
            cabinet_door_obbs=[],
            handhold_obbs=[],
            graspable_objects_obbs=[self.graspable_objects_obb_head(features)],
            # grasp_quality_scores=[self.grasp_quality_head(features)],
        )
