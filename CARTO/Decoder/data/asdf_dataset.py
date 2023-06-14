import pathlib
import torch
import sys
import json
import numpy as np
import trimesh
import logging

from CARTO.Decoder.data.dataset import DataPoint

from typing import List, Dict

## Adding ASDF to our search path
# For code release: should or maybe could be a submodule
ASDF_BASE_PATH = "external_libs/A-SDF"
sys.path.append(ASDF_BASE_PATH)

try:
    from asdf.data import SDFSamples
except Exception as e:
    logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level


class ASDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        all_file_ids: List[pathlib.Path],
        subsample_amount: int = 12500,
        load_ram: bool = True,
        train: bool = True,
        load_gt: bool = False,
    ):
        ### Create A-SDF datasets..

        all_file_ids = [
            f"{file_ids}_{'train' if train else 'test'}.json"
            for file_ids in all_file_ids
        ]
        self.all_file_splits = []
        self.dataset_categories = []

        for file_ids in all_file_ids:
            with open(pathlib.Path(ASDF_BASE_PATH) / file_ids, "r") as f:
                json_split = json.load(f)
            self.all_file_splits.append(json_split)
            self.dataset_categories.append(list(json_split["shape2motion"].keys())[0])

        self.ASDF_datasets = []
        self.stops = [0]

        for file_split in self.all_file_splits:
            asdf_set = SDFSamples(
                pathlib.Path(ASDF_BASE_PATH) / "data",
                file_split,
                subsample_amount,
                load_ram=load_ram,
                articulation=True,
            )

            self.ASDF_datasets.append(asdf_set)
            self.stops.append(self.stops[-1] + len(asdf_set))

        self.stops = np.array(self.stops)
        self.load_gt = load_gt

    def __len__(self) -> int:
        return self.stops[-1]

    def __getitem__(self, idx: int) -> DataPoint:
        dataset_idx = len(self.stops) - np.count_nonzero(idx < self.stops) - 1
        category = self.dataset_categories[dataset_idx]

        # dataset_idx =
        if category == "laptop":
            limits = [-1.5708, 0.0]  # Upper limit does not matter
        else:
            limits = [0.0, 0.0]

        local_idx = idx - self.stops[dataset_idx]
        asdf_data = self.ASDF_datasets[dataset_idx][local_idx]
        (tensor, joint_state, instance_id), i = asdf_data
        points = tensor[:, :3]
        sdf = tensor[:, 3]
        parts = tensor[:, 4]

        datapoint = DataPoint(
            object_id=f"{category}_{instance_id}",
            joint_config_id=str(idx),
            joint_config={"joint": float(joint_state / 180 * np.pi)},
            points=points.float().cpu(),
            sdf_values=sdf.float().cpu(),
        )
        datapoint.joint_def = {
            "joint": {
                "type": "revolute",  # All ASDFs objects are revolute
                "limit": limits,
            }
        }
        if not self.load_gt:
            return datapoint

        corresponding_split = self.all_file_splits[dataset_idx]
        instance_name = f"{corresponding_split['shape2motion'][self.dataset_categories[dataset_idx]][local_idx]}"

        ground_truth_samples_filename = (
            pathlib.Path(ASDF_BASE_PATH)
            / "data"
            / "SurfaceSamples"
            / "shape2motion"
            / category
            / (instance_name + ".obj")
        )
        normalization_params_filename = (
            pathlib.Path(ASDF_BASE_PATH)
            / "data"
            / "NormalizationParameters"
            / "shape2motion"
            / category
            / (instance_name + ".npz")
        )

        gt_mesh = trimesh.load(ground_truth_samples_filename)
        gt_points = gt_mesh.vertices

        # Apply the inverse normalization
        normalization_params = np.load(normalization_params_filename)
        offset = normalization_params["offset"]
        scale = normalization_params["scale"]
        gt_points = (gt_points + offset) * scale
        datapoint.full_pc = np.copy(gt_points)

        return datapoint

    # Same as for our-SDF
    @staticmethod
    def collate_fn(datapoints: List[DataPoint]) -> Dict:
        return {
            "object_id": [datapoint.object_id for datapoint in datapoints],
            "joint_config_id": [
                str(datapoint.joint_config_id) for datapoint in datapoints
            ],
            "joint_config": [datapoint.joint_config for datapoint in datapoints],
            "zero_joint_config": [
                datapoint.zero_joint_config for datapoint in datapoints
            ],
            "joint_definition": [datapoint.joint_def for datapoint in datapoints],
            "sdf": torch.stack(
                [torch.FloatTensor(datapoint.sdf_values) for datapoint in datapoints]
            ),
            "points": torch.stack(
                [torch.FloatTensor(datapoint.points) for datapoint in datapoints]
            ),
        }
