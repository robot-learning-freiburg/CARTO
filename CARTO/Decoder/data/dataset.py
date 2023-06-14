from typing import Dict, List, Optional, Union
import torch
import pathlib
from CARTO.simnet.lib.datapoint import decompress_datapoint, compress_datapoint
import dataclasses
import numpy as np
import urdfpy
import yaml
import logging
import copy

import functools
from CARTO.Decoder import utils, config

SPLITS_DIR = config.BASE_DIR / "split_files"


class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """

    def __init__(self, fget):
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value


@dataclasses.dataclass
class DataPoint:
    object_id: str
    joint_config_id: str
    joint_config: Dict[
        str, float
    ]  # TODO: Can I actually save and load all the information for the joint with that?
    scale: float = 1.0
    # Legacy SDF
    sdf_values: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    # New PC
    full_pc: Optional[np.ndarray] = None
    full_normals: Optional[np.ndarray] = None

    def __eq__(self, __o: "DataPoint") -> bool:
        return (
            self.object_id == __o.object_id
            and self.joint_config_id == __o.joint_config_id
        )

    @lazy_property
    def joint_def(self):
        return utils.get_joint_def_with_canonical_limits(
            self.object_id, prismatic_scale=self.scale
        )

    @lazy_property
    def zero_joint_config(self):
        return utils.get_zero_joint_dict(
            self.joint_config, self.joint_def, prismatic_scale=self.scale
        )

    @staticmethod
    def from_meta(old_datapoint: "DataPoint"):
        datapoint = DataPoint(
            object_id=old_datapoint.object_id,
            joint_config_id=old_datapoint.joint_config_id,
            joint_config=copy.deepcopy(old_datapoint.joint_config),
            scale=old_datapoint.scale,
        )
        return datapoint


class Rescaler3D:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x: DataPoint):
        # Rescale SDF
        if x.sdf_values is not None:
            x.sdf_values /= self.scale
        if x.points is not None:
            x.points /= self.scale

        # Rescale PC
        if x.full_pc is not None:
            x.full_pc /= self.scale

        # Multiply scales for prismatic
        x.scale *= self.scale
        return x


def get_dataset_split_dict(
    dataset_folder: pathlib.Path,
    split_name: Union[str, List[str]] = "",
    file_name="ids.yaml",
):
    if split_name == "":
        all_files = list(dataset_folder.glob("*.zstd"))
        split_dicts = {"train": all_files, "val": all_files, "test": []}
    elif isinstance(split_name, str):
        split_name = [split_name]

    if len(split_name) > 0:
        split_dicts = utils.AccumulatorDict()
        for split_ in split_name:
            with open(SPLITS_DIR / split_ / file_name, "r") as file:
                split_dicts_local = yaml.load(file, yaml.Loader)
            split_dicts.increment_dict(split_dicts_local)

    if len(split_dicts["val"]) == 0:
        split_dicts["val"] = split_dicts["train"]

    # We only loaded the object ids and not the full
    if file_name == "object_ids.yaml":
        object_split_dicts = copy.deepcopy(split_dicts)
        for split_name, split_object_ids in object_split_dicts.items():
            split_dicts[split_name] = []
            for object_id in split_object_ids:
                all_files = list(dataset_folder.glob(f"{object_id}_*.zstd"))
                split_dicts[split_name].extend(all_files)
    return split_dicts


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_ids: List[pathlib.Path],
        cache_only_meta=False,
        rescaler: Rescaler3D = Rescaler3D(),
    ):
        self.file_ids: List[pathlib.Path] = file_ids
        if cache_only_meta:
            self.pre_loaded_datapoints = [None] * len(self.file_ids)
        self.cache_only_meta = cache_only_meta
        self.rescaler = rescaler

    def __len__(self) -> int:
        return len(self.file_ids)
        # return 2

    def __getitem__(self, idx: int) -> DataPoint:
        if self.cache_only_meta and self.pre_loaded_datapoints[idx]:
            return self.pre_loaded_datapoints[idx]

        file_path = self.file_ids[idx]
        with open(file_path, "rb") as fh:
            buf = fh.read()
            datapoint: DataPoint = decompress_datapoint(buf)
            datapoint = self.rescaler(datapoint)

        if self.cache_only_meta:
            datapoint.points = None
            datapoint.sdf_values = None
            datapoint.full_normals = None
            datapoint.full_pc = None
            self.pre_loaded_datapoints[idx] = datapoint
        return datapoint

    @staticmethod
    def collate_meta(datapoints: List[DataPoint]) -> Dict:
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
        }

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


class SDFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_ids: List[pathlib.Path],
        cache_dir_name: str = "sdf_cache",
        subsample: int = 16000,
        cache_in_ram: bool = True,
        rescaler: Rescaler3D = Rescaler3D(),
    ):
        self.file_ids: List[pathlib.Path] = file_ids
        self.cache_dir_name = cache_dir_name
        self.subsample = subsample
        self.cache_in_ram = cache_in_ram
        self.rescaler = rescaler
        # Preload all the points, to allow multi processing to distribute them, useful when num_workers > 0,
        # otherwise lazy caching could also work
        if self.cache_in_ram:
            self.cache_list = [
                self.load_file(idx, rescaler=self.rescaler) for idx in range(len(self))
            ]

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> DataPoint:
        if self.cache_in_ram:
            datapoint = self.cache_list[idx]
        else:
            datapoint = self.load_file(idx, rescaler=self.rescaler)
        assert datapoint is not None

        # Randomly subsample equal pos/neg points for SDF
        N = datapoint.sdf_values.shape[0]
        if N < self.subsample:
            return datapoint

        sdf_equal, points_equal = utils.equal_sdf_split(
            datapoint.sdf_values, datapoint.points, self.subsample
        )

        if self.cache_in_ram:
            # We create a new datapoint otherwise we overwrite the one saved in the list
            datapoint = DataPoint.from_meta(datapoint)
        datapoint.sdf_values = sdf_equal
        datapoint.points = points_equal
        return datapoint

    def load_file(self, idx, rescaler: Rescaler3D):
        """
        Loads an SDF file and caches it correctly
        """
        file_path = self.file_ids[idx]
        cache_dir = file_path.parent / self.cache_dir_name
        cache_files = list(cache_dir.glob(f"{file_path.stem}.zstd"))
        assert (
            len(cache_files) < 2
        ), "Same file can't be cached twice, something is wrong"
        # Use cached file
        datapoint: DataPoint = None
        if len(cache_files) == 1:  # and False:
            # Load cached
            try:
                with open(str(cache_files[0]), "rb") as fh:
                    buf = fh.read()
                    datapoint: DataPoint = decompress_datapoint(buf)
            except EOFError as e:
                print(f"Couldn't load model {cache_files[0]}\n\t{e}")
        else:
            # Load original
            with open(file_path, "rb") as fh:
                buf = fh.read()
                datapoint: DataPoint = decompress_datapoint(buf)
            # Throw out normals/pc
            datapoint.full_normals = None
            datapoint.full_pc = None
            datapoint = rescaler(datapoint)
            # Cache file
            buf = compress_datapoint(datapoint)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / f"{file_path.stem}.zstd", "wb") as fh:
                fh.write(buf)
        return datapoint

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


class iROADDataset(torch.utils.data.Dataset):
    SPECIAL_META_KEYS = [
        "object_id",
        "joint_config_id",
        "joint_config",
        "zero_joint_config",
        "joint_definition",
    ]

    def __init__(
        self,
        file_ids: List[pathlib.Path],
        test: bool = False,
        cache_dir_name: str = "iroad_cache",
        lods: int = 7,
        current_lod: int = 7,
        rescaler: Rescaler3D = Rescaler3D(),
        ignore_iou: bool = True,
        cache_in_ram: bool = False,
    ):
        self.file_ids: List[pathlib.Path] = file_ids
        self.cache_dir_name = cache_dir_name
        self.octgrid = grid.OctGrid(subdiv=1)
        self.lods = lods
        self.current_lod = current_lod
        self.rescaler = rescaler
        self.test = test
        self.ignore_iou = ignore_iou

        self.cache_in_ram = cache_in_ram
        if self.cache_in_ram:
            self.cache = [self.__loaditem__(idx) for idx in range(len(self))]

    def __len__(self) -> int:
        return len(self.file_ids)
        # return 1

    def __loaditem__(self, idx) -> Dict:
        file_path = self.file_ids[idx]
        cache_dir = file_path.parent / self.cache_dir_name
        # Look up cache
        # print(file_path.stem)
        cache_files = list(cache_dir.glob(f"{file_path.stem}.npy"))
        assert (
            len(cache_files) < 2
        ), "Same file can't be cached twice, something is wrong"
        # Use cached file
        model = None
        if len(cache_files) == 1:  # and False:
            try:
                model = np.load(str(cache_files[0]), allow_pickle=True).item()
            except EOFError as e:
                print(f"Couldn't load model {cache_files[0]}\n\t{e}")

            # If we previously created data with lower lod as we are currently sampling
            if self.current_lod not in model:
                model = None
        if not model:
            # Generate new file
            with open(file_path, "rb") as fh:
                buf = fh.read()
                datapoint: DataPoint = decompress_datapoint(buf)
                datapoint = self.rescaler(
                    datapoint
                )  # Ensures joint_def and zero_joint_config are rescaled aswell!
            model = _get_model_annotations(
                self.octgrid,
                datapoint.full_pc,
                datapoint.full_normals,
                idx,
                lods=self.lods,
            )

            model["object_id"] = datapoint.object_id
            model["joint_config_id"] = str(datapoint.joint_config_id)
            model["joint_config"] = datapoint.joint_config
            model["zero_joint_config"] = datapoint.zero_joint_config
            model["joint_definition"] = datapoint.joint_def

            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / f"{file_path.stem}.npy", model)
        return model

    def __getitem__(self, idx: int) -> Dict:
        if self.cache_in_ram:
            model = self.cache[idx]
        else:
            model = self.__loaditem__(idx)

        # Done with getting `model`-variable
        # Visualize with open3d
        # pcd_sdf_vis = o3d.geometry.PointCloud()
        # pcd_sdf_vis.points = o3d.utility.Vector3dVector(
        #     model[self.lods]['xyz'][model[self.lods]['occ']] - (
        #         np.expand_dims(model[self.lods]['sdf'][model[self.lods]['occ']], axis=1) *
        #         model[self.lods]['nrm'][model[self.lods]['occ']]
        #     )
        # )
        # pcd_sdf_vis.normals = o3d.utility.Vector3dVector(
        #     model[self.lods]['nrm'][model[self.lods]['occ']]
        # )
        # pcd_sdf_vis.colors = o3d.utility.Vector3dVector(
        #     (model[self.lods]['nrm'][model[self.lods]['occ']] + 1) / 2
        # )
        # o3d.visualization.draw_geometries([pcd_sdf_vis])

        # TODO Copy from here + Make it work
        # https://github.com/TRI-ML/iROAD/blob/e3e8441ff1bd5e3091740630a3234e26811ce8ed/data/data.py#L174
        output = utils.copy_dict_keys(model, self.SPECIAL_META_KEYS)
        output[0] = {}
        output[0]["ids"] = np.int(np.zeros([1]))
        output[0]["occ"] = np.ones([1]).astype(bool)
        output[0]["xyz"] = np.zeros([1, 3])

        lod_ids_local_all = []
        lod_ids_local_all.append(np.arange(model[1]["xyz"].shape[0]))

        if self.test:
            output["pcd"] = model["pcd"]
            output["nrm"] = model["nrm"]
            if not self.ignore_iou:
                output["xyz_iou"] = model["xyz_iou"]
                output["occ_iou"] = model["occ_iou"]
        else:
            # Randomly sample N points from each level
            for lod in range(1, self.current_lod + 1):
                n_points = min(((2**lod) ** 3), 2**10)
                lod_ids_global = np.arange(model[lod]["xyz"].shape[0])  # LoD point ids
                lod_ids_global_filt = lod_ids_global[
                    model[lod]["occ"]
                ]  # LoD ids after occ filtering
                lod_ids_local = lod_ids_local_all[
                    -1
                ]  # LoD local point ids (after random selection from prev. level)
                lod_occ_local = model[lod]["occ"][lod_ids_local]  # LoD occlusions
                if len(np.arange(lod_ids_local.shape[0])[lod_occ_local]) > 0:
                    lod_ids_local_rnd = np.random.choice(
                        np.arange(lod_ids_local.shape[0])[lod_occ_local],
                        size=n_points,
                        replace=True,
                    )  # sample N local ids from LoD
                else:
                    return None
                    # if lod == 1:
                #     lod_ids_local_rnd = np.random.choice(np.arange(lod_ids_local.shape[0])[lod_occ_local], size=n_points, replace=False)  # sample N local ids from LoD
                lod_ids_local_rnd_filt = lod_ids_local[
                    lod_ids_local_rnd
                ]  # LoD local ids after occ filtering

                # Compute ids of the next level
                lod_next_ids = np.zeros((lod_ids_global.shape[0], 8)).astype(int)
                lod_next_ids[lod_ids_global_filt] = np.arange(
                    lod_ids_global_filt.shape[0] * 8
                ).reshape(-1, 8)
                lod_next_ids_rnd = lod_next_ids[lod_ids_local_rnd_filt].flatten()
                lod_ids_local_all.append(lod_next_ids_rnd)

                # Form output for the current level
                output[lod] = {}
                output[lod]["ids"] = lod_ids_local_rnd
                output[lod]["occ"] = lod_occ_local
                output[lod]["xyz"] = model[lod]["xyz"][lod_ids_local][lod_ids_local_rnd]
                output[lod]["sdf"] = np.expand_dims(
                    model[lod]["sdf"][lod_ids_local][lod_ids_local_rnd], axis=-1
                )
                output[lod]["nrm"] = model[lod]["nrm"][lod_ids_local][lod_ids_local_rnd]

        return output

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None

        batch_only_meta = []
        for i, batch_item in enumerate(batch):
            reduced_batch_item, only_meta_batch = utils.remove_dict_keys(
                batch_item, iROADDataset.SPECIAL_META_KEYS
            )
            batch[i] = reduced_batch_item
            batch_only_meta.append(only_meta_batch)

        collated_batch = torch.utils.data.dataloader.default_collate(batch)

        for meta_key in iROADDataset.SPECIAL_META_KEYS:
            collated_batch[meta_key] = []
        for batch_item in batch_only_meta:
            for meta_key in iROADDataset.SPECIAL_META_KEYS:
                collated_batch[meta_key].append(batch_item[meta_key])
        return collated_batch
