import os

os.environ["PYTHONHASHSEED"] = str(1)

# Allow the sentence tokenizer to be run in parallel.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import json
import pathlib
import random
import sys
from importlib.machinery import SourceFileLoader

import cv2
import IPython

# To ensure mesh_to_sdf is imported before pyrender
import mesh_to_sdf
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from typing import Optional
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from CARTO.app.panoptic_tidying import tidy_classes
from CARTO.lib import camera, datapoint
from CARTO.lib.datapoint import Panoptic
from CARTO.lib.net import common
from CARTO.lib.net.data_module import DataModule
from CARTO.lib.net.dataset import PanopticOutputs
from CARTO.lib.net.panoptic_trainer import PanopticModel
from CARTO.lib.net.post_processing.eval3d import Eval3d
from CARTO.lib.shapenet_utils import NOCS_CATEGORIES
from CARTO.lib import partnet_mobility

# ./runner.sh simnet/app/panoptic_category_reconstruction/net_train.py @simnet/app/panoptic_category_reconstruction/net_config_overfit.txt

_GPU_TO_USE = 0


def set_seed(seed: Optional[int]):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EvalMethod:
    def __init__(self, hparams, log_prefix="val"):
        assert log_prefix == "val" or log_prefix == "test"

        self.objects_eval_3d = Eval3d()
        self.doors_eval_3d = Eval3d()
        self.handholds_eval_3d = Eval3d()
        self.camera_model = camera.ZED2Camera1080p()
        self.log_prefix = log_prefix

    def process_sample(
        self, panoptic_outputs: PanopticOutputs, panoptic_targets: Panoptic
    ):
        batch_size = len(panoptic_targets.val_data)

        for i in range(batch_size):
            val_data = panoptic_targets.val_data[i]
            if val_data.scene_name == "unlabeled_data":
                continue

            ## Compute detections
            if len(panoptic_outputs.cabinet_door_obbs) > 0:
                door_detections = panoptic_outputs.cabinet_door_obbs[0].get_detections(
                    i,
                    camera_model=self.camera_model,
                    class_list=val_data.door_class_ids,
                )
                self.doors_eval_3d.process_sample(
                    door_detections, val_data.door_detections, val_data.scene_name
                )

            if len(panoptic_outputs.graspable_objects_obbs) > 0:
                objects_detections = panoptic_outputs.graspable_objects_obbs[
                    0
                ].get_detections(
                    i,
                    camera_model=self.camera_model,
                    class_list=val_data.object_class_ids,
                )
                self.objects_eval_3d.process_sample(
                    objects_detections, val_data.object_detections, val_data.scene_name
                )

            if len(panoptic_outputs.handhold_obbs) > 0:
                handhold_detections = panoptic_outputs.handhold_obbs[0].get_detections(
                    i, camera_model=self.camera_model
                )
                self.handholds_eval_3d.process_sample(
                    handhold_detections,
                    val_data.handhold_detections,
                    val_data.scene_name,
                )

    def process_all_dataset(self, log):
        log[
            self.log_prefix + "/objects 3Dmap"
        ] = self.objects_eval_3d.process_all_3D_dataset()
        log[
            self.log_prefix + "/cabinet 3Dmap"
        ] = self.doors_eval_3d.process_all_3D_dataset()
        log[
            self.log_prefix + "/handhold 3Dmap"
        ] = self.handholds_eval_3d.process_all_3D_dataset()
        log[
            self.log_prefix + "/object_class_accuracy"
        ] = self.objects_eval_3d.process_category_accuracy()
        log[
            self.log_prefix + "/door_class_accuracy"
        ] = self.doors_eval_3d.process_category_accuracy()

    def reset(self):
        self.objects_eval_3d = Eval3d()
        self.doors_eval_3d = Eval3d()
        self.handholds_eval_3d = Eval3d()


if __name__ == "__main__":
    print("WARNING -- This was not tested for the code release -- WARNING")
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    common.add_train_args(parser)
    hparams = parser.parse_args()
    set_seed(hparams.seed)
    categories = [
        "Dishwasher",
        "Knife",
        "Laptop",
        "Microwave",
        "Oven",
        "Refrigerator",
        "Stapler",
        "StorageFurniture",
        "Table",
        "WashingMachine",
    ]
    hparams.object_categories = [
        partnet_mobility.partnet_mobility_db[object_cat] for object_cat in categories
    ]

    train_ds = datapoint.make_dataset(hparams.train_path)
    samples_per_epoch = len(train_ds.list())
    samples_per_step = hparams.train_batch_size
    steps = hparams.max_steps
    # max to allow overfitting for a single example
    steps_per_epoch = max(samples_per_epoch // samples_per_step, 1)
    epochs = int(np.ceil(steps / steps_per_epoch))
    actual_steps = epochs * steps_per_epoch
    print(f"{epochs = } {samples_per_epoch = } {actual_steps = }")
    model = PanopticModel(
        hparams, epochs, EvalMethod(hparams, "val"), EvalMethod(hparams, "test")
    )
    data_module = DataModule(hparams, train_ds)
    model_checkpoint = ModelCheckpoint(
        # save_top_k=-1, # -1 Saves all models --> deactivate to save some space
        every_n_epochs=1,
        mode="max",  # Does not do anything as we do not have monitor= set (--> saves latest)
    )
    if hparams.wandb_name is not None:
        logger = loggers.WandbLogger(name=hparams.wandb_name, project="arti2real")
    else:
        logger = loggers.TensorBoardLogger(save_dir=hparams.output)
    # Mixed precision training uses 16-bit precision floats, otherwise use 32-bit floats.
    precision = 16 if hparams.use_amp else 32

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=[_GPU_TO_USE],
        callbacks=[model_checkpoint],
        val_check_interval=hparams.val_check_interval,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        logger=logger,
        default_root_dir=hparams.output,
        precision=precision,
    )
    trainer.fit(model, data_module)
    if hparams.test_path is not None:
        trainer.test(model, data_module)
