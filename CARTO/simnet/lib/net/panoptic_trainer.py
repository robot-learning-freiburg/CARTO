import os
import copy
import time

os.environ["PYTHONHASHSEED"] = str(1)

import argparse
from importlib.machinery import SourceFileLoader
import sys

from typing import Tuple, Any

import random

random.seed(123456)
import cv2
import numpy as np

np.random.seed(123456)
import torch

torch.manual_seed(123456)

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from CARTO.simnet.lib.net import common
from CARTO.simnet.lib import datapoint
from CARTO.simnet.lib.datapoint import Panoptic
from CARTO.simnet.lib.net.dataset import PanopticOutputs
from CARTO.simnet.lib.net.dataset import extract_left_numpy_img
from CARTO.simnet.lib.net.post_processing.eval3d import Eval3d
from CARTO.simnet.lib.net.post_processing import orochi_outputs
from CARTO.simnet.lib.net.functions.learning_rate import (
    lambda_learning_rate_poly,
    lambda_warmup,
)

_GPU_TO_USE = 0


class StereoWrapper(object):
    def __init__(self, path):
        self.stereo = torch.jit.load(path, map_location="cpu")


class PanopticModel(pl.LightningModule):
    def __init__(
        self, hparams, epochs=None, val_eval_metrics=None, test_eval_metrics=None
    ):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.epochs = epochs

        self.model = common.get_model(hparams)
        self.val_eval_metrics = val_eval_metrics
        self.test_eval_metrics = test_eval_metrics

        if hparams.frozen_stereo_torchscript is not None:
            self.stereo_wrapper = StereoWrapper(hparams.frozen_stereo_torchscript)
        else:
            self.stereo_wrapper = None

        self.category_names = [c.name for c in hparams.object_categories]

    def forward(self, image, language=None):
        if self.stereo_wrapper is None:
            return self.model(image, language)
        else:
            _, channels, _, _ = image.shape
            # There are two options for running the model with pre-computed stereo.
            # You can either pass in a 4-channel tensor with is left+disparity or a
            # 6-channel tensor which is stacked left+right. In the stacked left+right
            # case, the stereo torchscript will be run on the left+right and then
            # concated with left to form the expected 4-channel input tensor.
            assert channels == 4 or channels == 6, "Channels must be 4 or 6"
            if channels == 6:
                with torch.no_grad():
                    left = image[:, 0:3, :, :]
                    right = image[:, 3:6, :, :]
                    self.stereo_wrapper.stereo.to(self.device)
                    self.stereo_wrapper.stereo.eval()
                    stereo_outputs, _ = self.stereo_wrapper.stereo(left, right)
                    disparity = stereo_outputs["disparity"]
                    image = torch.cat([image[:, 0:3, :, :], disparity], dim=1)
            return self.model(image, language)

    def compute_loss(
        self,
        batch: Tuple[Any, Any, PanopticOutputs],
        batch_idx,
        panoptic_outputs: PanopticOutputs,
        log,
        prefix,
    ):
        image, language, panoptic_targets = batch

        input_height, input_width = image.shape[-2:]

        loss = torch.zeros(1, dtype=image.dtype, device=image.device)

        # Compute Segmentation loss
        if len(panoptic_outputs.room_segmentation) > 0:
            loss = loss + panoptic_outputs.room_segmentation[0].compute_loss(
                panoptic_targets.room_segmentation,
                log,
                f"{prefix}_detailed/loss/segmentation",
            )
        # Compute OBBs loss
        if (
            len(panoptic_outputs.cabinet_door_obbs) > 0
            and len(panoptic_targets.cabinet_door_obbs) > 0
        ):
            loss = loss + panoptic_outputs.cabinet_door_obbs[0].compute_loss(
                panoptic_targets.cabinet_door_obbs,
                log,
                f"{prefix}_detailed/loss/door_obb",
            )
        if (
            len(panoptic_outputs.handhold_obbs) > 0
            and len(panoptic_targets.handhold_obbs) > 0
        ):
            loss = loss + panoptic_outputs.handhold_obbs[0].compute_loss(
                panoptic_targets.handhold_obbs,
                log,
                f"{prefix}_detailed/loss/handhold_obb",
            )
        if (
            len(panoptic_outputs.graspable_objects_obbs) > 0
            and len(panoptic_targets.graspable_objects_obbs) > 0
        ):
            loss = loss + panoptic_outputs.graspable_objects_obbs[0].compute_loss(
                panoptic_targets.graspable_objects_obbs,
                log,
                f"{prefix}_detailed/loss/objects_obb",
            )
        if (
            len(panoptic_outputs.small_depth) > 0
            and len(panoptic_targets.small_depth) > 0
        ):
            loss = loss + panoptic_outputs.small_depth[0].compute_loss(
                panoptic_targets.depth, log, f"{prefix}_detailed/loss/small_disp"
            )
        if len(panoptic_outputs.depth) > 0 and len(panoptic_targets.depth) > 0:
            loss = loss + panoptic_outputs.depth[0].compute_loss(
                panoptic_targets.depth, log, f"{prefix}_detailed/loss/disp"
            )

        # Compute grasp quality loss
        # loss = loss + panoptic_outputs.grasp_quality_scores[0].compute_loss(
        #    panoptic_targets.grasp_quality_scores, log, f'{prefix}_detailed/loss/grasp_quality'
        # )
        # Compute door state

        log[f"{prefix}/loss/total"] = loss.item()

        return loss

    def compute_metrics(
        self,
        batch: Tuple[Any, Any, PanopticOutputs],
        batch_idx,
        panoptic_outputs: PanopticOutputs,
        log_prefix,
    ):
        # tkollar: Batch size currently has to be == 1.
        assert log_prefix == "val" or log_prefix == "test"

        image, language, panoptic_targets = batch

        with torch.no_grad():
            # Compute mAP score
            try:
                # These processors operate on batches.
                if log_prefix == "val":
                    self.val_eval_metrics.process_sample(
                        language, panoptic_outputs, panoptic_targets
                    )
                elif log_prefix == "test":
                    self.test_eval_metrics.process_sample(
                        language, panoptic_outputs, panoptic_targets
                    )
            except np.linalg.LinAlgError as e:
                # When training at half precision, this can happen sometimes at the beginning.
                print("Couldn't process sample due to error:", e)

    def log_images(
        self,
        batch: Tuple[Any, Any, PanopticOutputs],
        batch_idx,
        panoptic_outputs: PanopticOutputs,
        prefix,
    ):
        image, language, panoptic_targets = batch

        with torch.no_grad():
            llog = {}
            left_image_np = extract_left_numpy_img(image[0])
            logger = self.logger.experiment
            try:
                orochi_vis = orochi_outputs.visualize_img(
                    panoptic_outputs,
                    np.copy(left_image_np),
                    self.val_eval_metrics.camera_model,
                    self.category_names,
                    poses=False,
                )
                llog[f"{prefix}/orochi_box"] = (orochi_vis, prefix)
                orochi_vis = orochi_outputs.visualize_img(
                    panoptic_outputs,
                    np.copy(left_image_np),
                    self.val_eval_metrics.camera_model,
                    self.category_names,
                    poses=True,  # Draws 9D coordinate systems and not boxes
                )
                llog[f"{prefix}/orochi_pose"] = (orochi_vis, prefix)

                if len(panoptic_outputs.graspable_objects_obbs) > 0:
                    heatmap_img = orochi_outputs.visualize_heatmap(
                        panoptic_outputs, np.copy(left_image_np)
                    )
                    llog[f"{prefix}/orochi_heatmap"] = (heatmap_img, prefix)

            except (np.linalg.LinAlgError, cv2.error) as e:
                # When training at half precision, this can happen sometimes at the beginning.
                print("Couldn't render box image due to error:", e)

            if panoptic_outputs.depth:
                depth_vis = panoptic_outputs.depth[0].get_visualization_img(
                    np.copy(left_image_np)
                )
                llog[f"{prefix}/disparity"] = (depth_vis, prefix)

            if panoptic_outputs.small_depth:
                small_depth_vis = panoptic_outputs.small_depth[0].get_visualization_img(
                    np.copy(left_image_np)
                )
                llog[f"{prefix}/svcn"] = (small_depth_vis, prefix)
            # TODO Nick: Add visualization for
            # Reconstructed objects?

        logger = self.logger.experiment
        if "tensorboard" in str(type(logger)):
            for key, value in llog.items():
                image_tensor = torch.from_numpy(value[0].copy())
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.permute((2, 0, 1))
                else:
                    image_tensor = image_tensor.unsqueeze(0)
                logger.add_image(key, image_tensor, self.global_step)
        else:
            wandb_log = {}
            for key, value in llog.items():
                wandb_log[key] = wandb.Image(value[0], caption=value[1])
            logger.log(wandb_log)

    def training_step(self, batch, batch_idx):
        image, language, _ = batch
        outputs = self.forward(image, language)

        log = {}
        loss = self.compute_loss(batch, batch_idx, outputs, log, "train")

        if (batch_idx % 200) == 0:
            prefix = "train"
            self.log_images(batch, batch_idx, outputs, prefix)

        for key, value in log.items():
            self.log(key, value)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate_step(batch, batch_idx, "test")

    def evaluate_step(self, batch, batch_idx, log_prefix):
        image, language, panoptic_targets = batch
        outputs = self.forward(image, language)

        log = {}
        # Test is unlabeled real data.
        scene_name = panoptic_targets.val_data[0].scene_name
        if scene_name == "labeled_data":
            start_time = time.time()
            self.compute_metrics(batch, batch_idx, outputs, log_prefix)
        # else: # Why? --> Record images in any case!
        # prefix = log_prefix + f'/{batch_idx}'
        prefix = log_prefix
        start_time = time.time()
        self.log_images(batch, batch_idx, outputs, prefix)
        for key, value in log.items():
            self.log(key, value)
        return 0.0

    def validation_epoch_end(self, outputs):
        return self.evaluate_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.evaluate_epoch_end(outputs, "test")

    def evaluate_epoch_end(self, outputs, log_prefix):
        assert log_prefix == "val" or log_prefix == "test"
        self.trainer.checkpoint_callback.save_best_only = False
        log = {}

        if log_prefix == "val":
            self.val_eval_metrics.process_all_dataset(log)
            self.val_eval_metrics.reset()
        elif log_prefix == "test":
            self.test_eval_metrics.process_all_dataset(log)
            self.test_eval_metrics.reset()

        for key, value in log.items():
            self.log(key, value)
        return {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.optim_learning_rate
        )
        lr_lambda = lambda_learning_rate_poly(self.epochs, self.hparams.optim_poly_exp)
        if (
            self.hparams.optim_warmup_epochs is not None
            and self.hparams.optim_warmup_epochs > 0
        ):
            lr_lambda = lambda_warmup(self.hparams.optim_warmup_epochs, 0.2, lr_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
