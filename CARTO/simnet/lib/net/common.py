import cv2
import pathlib
from importlib.machinery import SourceFileLoader
import numpy as np
import torch
import wandb
import re
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate
import functools

from CARTO.simnet.lib.net.init.default_init import default_init
from CARTO.simnet.lib.net.dataset import Dataset, panoptic_collate
from CARTO.simnet.lib.onnx_plugins import fix_module_train

from CARTO.lib import rename_unpickler


def add_dataset_args(parser, prefix, required=False):
    group = parser.add_argument_group("{}_dataset".format(prefix))
    group.add_argument("--{}_path".format(prefix), type=str, required=required)
    # group.add_argument("--{}_fraction".format(prefix), type=str, default=None)
    group.add_argument("--{}_batch_size".format(prefix), default=16, type=int)
    group.add_argument("--{}_num_workers".format(prefix), default=7, type=int)
    # group.add_argument("--{}_random_crop".format(prefix), default=None, type=int, nargs=2)


def add_train_args(parser, enforce_required: bool = True):
    parser.add_argument("--max_steps", type=int, required=True and enforce_required)
    parser.add_argument("--output", type=str, required=True and enforce_required)

    _DEFAULT_SEED = 12345
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        nargs="?",
        help=f"When --seed is not used, we will use {_DEFAULT_SEED} as the default seed; if just --seed, we will not seed (i.e. random); if --seed NUMBER, we will seed with NUMBER",
    )

    add_dataset_args(parser, "train")
    add_dataset_args(parser, "val")
    add_dataset_args(parser, "test", required=False)

    # For finetuning, allow running validation more often
    parser.add_argument("--val_check_interval", default=1.0, type=float)

    # Limit number of batches to run for val and test.
    parser.add_argument("--limit_test_batches", default=1000, type=int)
    parser.add_argument("--limit_val_batches", default=1000, type=int)

    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument("--optim_type", default="sgd", type=str)
    optim_group.add_argument("--optim_learning_rate", default=0.02, type=float)
    optim_group.add_argument("--optim_momentum", default=0.9, type=float)
    optim_group.add_argument("--optim_weight_decay", default=1e-4, type=float)
    optim_group.add_argument("--optim_poly_exp", default=0.9, type=float)
    optim_group.add_argument("--optim_warmup_epochs", default=None, type=int)
    parser.add_argument("--model_file", type=str, required=True and enforce_required)
    parser.add_argument("--model_name", type=str, required=True and enforce_required)
    parser.add_argument("--model_rgbd", type=bool, default=False)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--skip_val_metrics", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_false")
    # Ignore Mask Search.
    parser.add_argument("--min_height", default=0.0, type=float)
    parser.add_argument("--min_occlusion", default=0.0, type=float)
    parser.add_argument("--min_truncation", default=0.0, type=float)
    # Backbone configs
    parser.add_argument("--model_norm", default="BN", type=str)
    parser.add_argument("--num_filters_scale", default=4, type=int)

    # Loss weights
    parser.add_argument(
        "--skip_loading_class_heads", default=False, action="store_true"
    )
    parser.add_argument("--frozen_stereo_checkpoint", default=None, type=str)
    parser.add_argument("--frozen_stereo_torchscript", default=None, type=str)
    parser.add_argument("--backbone_checkpoint", default=None, type=str)
    parser.add_argument("--stereo_backbone_checkpoint", default=None, type=str)
    parser.add_argument("--loss_seg_mult", default=1.0, type=float)
    parser.add_argument("--loss_depth_mult", default=1.0, type=float)
    parser.add_argument("--loss_heatmap_mult", default=100.0, type=float)
    parser.add_argument("--loss_vertex_mult", default=0.1, type=float)
    parser.add_argument("--loss_z_centroid_mult", default=0.1, type=float)
    parser.add_argument("--loss_rotation_mult", default=0.1, type=float)
    parser.add_argument("--loss_keypoint_mult", default=0.1, type=float)
    parser.add_argument("--loss_latent_emb_mult", default=0.1, type=float)
    parser.add_argument("--loss_abs_pose_mult", default=0.1, type=float)
    parser.add_argument("--loss_grasp_mult", default=0.0, type=float)
    # Stereo Stem Args
    parser.add_argument(
        "--loss_disparity_stdmean_scaled",
        action="store_true",
        help="If true, the loss will be scaled based on the standard deviation and mean of the "
        "ground truth disparities",
    )
    parser.add_argument("--cost_volume_downsample_factor", default=4, type=int)
    parser.add_argument("--max_disparity", default=90, type=int)
    parser.add_argument(
        "--fe_features",
        default=16,
        type=int,
        help="Number of output features in feature extraction stage",
    )
    parser.add_argument(
        "--fe_internal_features",
        default=32,
        type=int,
        help="Number of features in the first block of the feature extraction",
    )
    # keypoint head args
    parser.add_argument("--num_keypoints", default=1, type=int)

    parser.add_argument("--decoder_training_id", default="", type=str)
    parser.add_argument("--shape_embedding_size", default=32, type=int)
    parser.add_argument("--joint_embedding_size", default=16, type=int)


def get_config_value(hparams, prefix, key):
    full_key = "{}_{}".format(prefix, key)
    if hasattr(hparams, full_key):
        return getattr(hparams, full_key)
    else:
        return None


def get_loader(hparams, prefix, preprocess_func=None, datapoint_dataset=None):
    datasets = []
    path = get_config_value(hparams, prefix, "path")
    datasets.append(
        Dataset(
            path,
            hparams,
            preprocess_image_func=preprocess_func,
            datapoint_dataset=datapoint_dataset,
        )
    )
    batch_size = get_config_value(hparams, prefix, "batch_size")

    panoptic_collate_ = functools.partial(panoptic_collate, rgbd=hparams.model_rgbd)

    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        collate_fn=panoptic_collate_,
        num_workers=get_config_value(hparams, prefix, "num_workers"),
        pin_memory=not hparams.no_pin_memory,
        drop_last=True,
        shuffle=prefix == "train",
    )


def prune_state_dict(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key[6:]] = state_dict.pop(key)
    return state_dict


def keep_only_stereo_weights(state_dict):
    pruned_state_dict = {}
    for key in list(state_dict.keys()):
        if "stereo" in key:
            pruned_state_dict[key] = state_dict[key]
    return pruned_state_dict


def get_latest_checkpoint_file(checkpoint_path: pathlib.Path) -> pathlib.Path:
    all_ckpt_files = checkpoint_path.glob("*")
    current_highest = None
    current_highest_epoch = 0
    current_highest_step = 0
    print(f"Found following checkpoints")
    for ckpt_file in all_ckpt_files:
        ckpt_dict = re.match(
            r"\bepoch\b=(?P<epoch>\d+)-\bstep\b=(?P<step>\d+)", ckpt_file.stem
        ).groupdict()
        print(ckpt_file.stem)
        if int(ckpt_dict["epoch"]) < current_highest_epoch:
            continue
        elif (
            int(ckpt_dict["epoch"]) == current_highest_epoch
            and int(ckpt_dict["step"]) < current_highest_step
        ):
            continue
        current_highest = ckpt_file
        current_highest_epoch = int(ckpt_dict["epoch"])
        current_highest_step = int(ckpt_dict["step"])
    print(f"Returning {current_highest}")
    return current_highest


def get_model(hparams):
    model_path = (pathlib.Path(__file__).parent / hparams.model_file).resolve()
    print("Using model class from:", model_path)
    net_module = SourceFileLoader(hparams.model_name, str(model_path)).load_module()
    net_attr = getattr(net_module, hparams.model_name)

    # If we are running inference we need to get object_categories from the checkpoint
    checkpoint = None
    if not hasattr(hparams, "object_categories"):
        print("Getting object categories from checkpoint:", hparams.checkpoint)
        if checkpoint is None:
            print(hparams.checkpoint)
            checkpoint = torch.load(hparams.checkpoint, map_location="cpu")
        hparams.object_categories = checkpoint["hyper_parameters"]["object_categories"]

    model = net_attr(hparams)
    model.apply(default_init)

    # For large models use imagenet weights.
    # This speeds up training and can give a +2 mAP score on car detections
    if hparams.num_filters_scale == 1:
        model.load_imagenet_weights()
    # If we are exporting to the robot, using a TensorRT compatible version of the net.
    # Note only nets trained for the robot use batch norm, so we use that.
    if hparams.model_norm == "BN":
        fix_module_train(model)

    if hparams.frozen_stereo_checkpoint is not None:
        print(
            "Restoring stereo weights from checkpoint:",
            hparams.frozen_stereo_checkpoint,
        )
        state_dict = torch.load(hparams.frozen_stereo_checkpoint, map_location="cpu")[
            "state_dict"
        ]
        state_dict = prune_state_dict(state_dict)
        state_dict = keep_only_stereo_weights(state_dict)
        model.load_state_dict(state_dict, strict=False)

    if hparams.backbone_checkpoint is not None:
        state_dict = torch.load(hparams.backbone_checkpoint, map_location="cpu")[
            "state_dict"
        ]
        keys = sorted(state_dict.keys())
        for key in keys:
            if key.startswith("model.") and "pretrain" not in key:
                new_key = key[6:]
                state_dict[new_key] = state_dict[key]
            del state_dict[key]
        model.backbone.features.load_state_dict(state_dict, strict=False)

    if hparams.stereo_backbone_checkpoint is not None:
        state_dict = torch.load(hparams.stereo_backbone_checkpoint, map_location="cpu")[
            "state_dict"
        ]
        keys = sorted(state_dict.keys())
        for key in keys:
            if key.startswith("model.") and "pretrain" not in key:
                new_key = key[6:]
                state_dict[new_key] = state_dict[key]
            del state_dict[key]
        model.backbone.stereo.features.load_state_dict(state_dict, strict=False)

    if hparams.checkpoint is not None:
        print("Restoring from checkpoint:", hparams.checkpoint)
        if checkpoint is None:
            checkpoint = torch.load(
                hparams.checkpoint, map_location="cpu", pickle_module=rename_unpickler
            )
        state_dict = checkpoint["state_dict"]
        state_dict = prune_state_dict(state_dict)
        if hparams.skip_loading_class_heads:
            for key in list(state_dict.keys()):
                prefix, _, _ = key.partition(".")
                if prefix.endswith("_head"):
                    print(
                        "Ignoring output head from checkpoint (not good for inference):",
                        key,
                    )
                    del state_dict[key]

        # remove any weights that have a shape mismatch
        current_state_dict = model.state_dict()
        for k in current_state_dict:
            if k not in state_dict:
                continue
            if state_dict[k].size() == current_state_dict[k].size():
                continue
            print(
                f"Ignoring checkpoint (size mismatch) {k!r}:",
                state_dict[k].size(),
                current_state_dict[k].size(),
            )
            state_dict[k] = current_state_dict[k]

        model.load_state_dict(state_dict, strict=False)

    return model
