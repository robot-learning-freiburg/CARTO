import copy
import json
import pathlib
from collections import namedtuple
from typing import Any, Tuple, Union

import numpy as np
import open3d as o3d
import torch

from CARTO import ROOT_DIR
from CARTO.simnet.lib import datapoint
from CARTO.simnet.lib.net.dataset import Dataset, PanopticOutputs, create_anaglyph
from CARTO.simnet.lib.net.post_processing.depth_outputs import DepthOutput

MISSING_LABELS_ID = [
    "YgMop2xGR9QQg3tvtHV8LE",
    "7mcbjCWqCUbcmuDRys7pKb",
    "ftGQw2kjUAGi4EEKm2sath",
    "8GBGdsC7F636882P4rmEC8",
    "JLnmWMvp6pN6CuSLLWwoab",
    "kYpjYEACcccqbJZX7HrpXG",
    "9wRKEke2FaHzNg42rWDisS",
    "QLctV2dhgTprak4d9HCwzY",
    "S7Ty7vSH6YcyBuDpCmThtD",
]

depth_hparams = namedtuple("depth_hparams", ["max_disparity"])


def convert_labels(old_labels):
    labels = copy.deepcopy(old_labels)
    labels["id"] = labels["filename"].split(".")[0]
    for object_idx in range(len(labels["objects"])):
        object_dict = labels["objects"][object_idx]

        center = np.array(
            [
                object_dict["centroid"]["x"],
                object_dict["centroid"]["y"],
                object_dict["centroid"]["z"],
            ]
        )
        zyx_array = np.array(
            [
                object_dict["rotations"]["z"],
                object_dict["rotations"]["y"],
                object_dict["rotations"]["x"],
            ]
        )
        zyx_array = zyx_array / 180 * np.pi
        R = o3d.geometry.get_rotation_matrix_from_zyx(zyx_array)
        extent = np.array(
            [
                object_dict["dimensions"]["length"],
                object_dict["dimensions"]["width"],
                object_dict["dimensions"]["height"],
            ]
        )
        object_dict["center"] = center
        object_dict["rotation"] = R
        object_dict["extent"] = extent

        del object_dict["centroid"]
        del object_dict["rotations"]
        del object_dict["dimensions"]

        labels["objects"][object_idx] = object_dict

    del labels["folder"]
    del labels["filename"]
    del labels["path"]
    return labels


class RealDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path],
        load_pc: bool = False,
        skip_without_labels=True,
    ):
        self.dataset_path = pathlib.Path(dataset_path)
        simnet_dataset = datapoint.make_dataset(str(self.dataset_path / "data"))
        self.datapoint_handles = simnet_dataset.list()
        if skip_without_labels:
            self.datapoint_handles = list(
                filter(lambda x: x.uid not in MISSING_LABELS_ID, self.datapoint_handles)
            )
        self.load_pc = load_pc

        self.hparams = depth_hparams(max_disparity=180)

    def __len__(self):
        return len(self.datapoint_handles)

    def __getitem__(self, idx) -> Tuple[PanopticOutputs, Any, Any]:
        # TODO Update the Any!
        local_handle: datapoint.LocalReadHandle = self.datapoint_handles[idx]

        dp: datapoint.Panoptic = local_handle.read()
        anaglyph = create_anaglyph(dp.stereo)

        panoptic_out = PanopticOutputs(
            depth=[]
            if dp.depth is None
            else [DepthOutput(torch.Tensor(dp.depth), self.hparams)],
            room_segmentation=[],
            cabinet_door_obbs=[],
            handhold_obbs=[],
            graspable_objects_obbs=[],
            grasp_quality_scores=[],
            small_depth=[],
            val_data=[dp.val_data],
            stereo_imgs=[anaglyph],
        )

        # Load labels
        with (
            ROOT_DIR / ".." / self.dataset_path / "labels" / f"{local_handle.uid}.json"
        ).open() as label_file:
            labels = json.load(label_file)
        labels = convert_labels(labels)

        if not self.load_pc:
            return panoptic_out, labels

        pointcloud_loc = (
            ROOT_DIR
            / ".."
            / self.dataset_path
            / "pointclouds"
            / f"{local_handle.uid}.ply"
        )

        pc = o3d.io.read_point_cloud(str(pointcloud_loc))

        return panoptic_out, labels, pc
