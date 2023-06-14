from collections import defaultdict
from curses import meta
from typing import Any, Dict, List, Union, Callable

import pathlib

import numpy as np

## TODO: Add later for loading into SimNet generator etc.
import urdfpy
from CARTO.lib import transform
import copy
from CARTO.lib.compression import read_compressed_json, extract_compressed_tarfile

# Assumes structure like this
# CARTO/
#     CARTO/
#         lib/
#             partnet_mobility.py
#     datasets/
#         partnet-mobility-v0/

CARTO_PARENT = pathlib.Path(__file__).parent.resolve() / ".." / ".."


class PartNetMobilityV0:
    def __init__(
        self: "PartNetMobilityV0",
        root_path: Union[str, pathlib.Path] = CARTO_PARENT
        / "datasets/partnet-mobility-v0",
    ):
        self.root_path = root_path
        self.index: List[Dict[Any, Any]] = read_compressed_json(
            self.root_path / "index.json.zst"
        )
        self.id_index: Dict[str, Any] = {meta["model_id"]: meta for meta in self.index}
        self._cache: Dict[Any, Any] = {}
        self._filter: Callable[[Dict[Any, Any]], bool] = lambda _: True

    def get_object_meta(self, id_: str) -> str:
        assert self.exists(id_)
        return self.id_index[id_]

    def get_object(self, id_, disable_cache: bool = False):
        if disable_cache:
            return self._get_sample(id_)

        if id_ not in self._cache:
            # logging.info(f'PartnetMobilityV0: Fetching sample {id_}')
            self._cache[id_] = self._get_sample(id_)
        return self._cache[id_]

    def sample(self, weighted=False, category_sublist=[]) -> str:
        ...

    def set_filter(self, filter):
        self._filter = filter

    def reset_filter(self):
        self._filter = lambda _: True

    def exists(self, id_: str) -> bool:
        return id_ in self.id_index

    @property
    def index_list(self):
        return [
            index
            for index in self.id_index.keys()
            if self._filter(self.get_object_meta(index))
        ]

    def __len__(self) -> int:
        return len(self.index_list)

    def _get_sample(self, id_):
        local_sample_path = self.root_path / "unpacked" / id_
        if local_sample_path.exists():
            return local_sample_path
        local_tarfile_path = (self.root_path / "tarfiles" / id_).with_suffix(".tar.zst")
        extract_compressed_tarfile(local_tarfile_path, local_sample_path)
        return local_sample_path


PartNetMobilityV0DB = PartNetMobilityV0()

default_joint_type_excluded_category = defaultdict(
    lambda: [], {"Stapler": ["stapler_body"], "Toilet": ["lever"]}
)
default_joint_type_excluded_instances = defaultdict(lambda: [], {})

# Assuming Single DoF!
category_joint_limits = {
    "Laptop": np.array([0.0, np.pi]),
    "Toilet": np.array([0.0, np.pi / 2]),
    "Microwave": np.array([0.0, np.pi]),
    "Refrigerator": np.array([0.0, np.pi]),
    "Box": np.array([0.0, 3 / 2 * np.pi]),
    "Dishwasher": np.array([0.0, np.pi / 2]),
    "Oven": np.array([0.0, np.pi / 2]),
    "StorageFurniture": np.array([0.0, np.pi]),
    "WashingMachine": np.array([0, np.pi]),
    "Stapler": np.array([0, 2 / 3 * np.pi]),
    "Table": np.array([0.0, np.pi]),
    "Knife": np.array([0, np.pi]),
}

# Overrides category limits
instance_joint_limits = {
    # All StorageFurniture
    "123994ddd751ef58c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "1715965e2e1e33e1c59350d819542ec7": np.array([0.0, np.pi / 2]),
    "1aeff9aef2610ee1c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "2cf613433894022dc59350d819542ec7": np.array([0.0, np.pi / 2]),
    "39d9512e482ef253c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "3a0492e3892bb29f12de5317fe5b354f": np.array([0.0, 2 / 3 * np.pi]),
    "429638b99d096c89c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "45c91d543ef3c1a829a50a2b3c5e5b6": np.array([0.0, np.pi / 2]),
    "55d1668a7d33f8ebc0b2397831029b54": np.array([0.0, 2 / 3 * np.pi]),
    "56031b004d5306954da5feafe6f1c8fc": np.array([0.0, np.pi / 2]),
    "56f7c9a029b6b40d12de5317fe5b354f": np.array([0.0, 2 / 3 * np.pi]),
    "779cda069c3b6338824662341ce2b233": np.array([0.0, np.pi / 2]),
    "786566b66299405a4da5feafe6f1c8fc": np.array([0.0, np.pi / 2]),
    "902a342783278a9d824662341ce2b233": np.array([0.0, np.pi / 2]),
    "920b27c2d3c83a0c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "9a7263ce1cb720d7c59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "aee149ff795d5a7ac59350d819542ec7": np.array([0.0, 2 / 3 * np.pi]),
    "b4e35d962847fb8e86d53ab0fe94e911": np.array([0.0, 2 / 3 * np.pi]),
    # All WashingMachine
    "ae8441235538d4415d85df7c37878bb6": np.array([0.0, np.pi / 2]),
    "u999c818c-3575-446f-93c2-c994abe0936f": np.array([0.0, np.pi / 2]),
    "uc5f2e411-5f8e-4b10-8393-75ce97e6e869": np.array([0.0, np.pi / 2]),
    "ud7bfe8da-a840-4efa-92d4-3d437469f827": np.array([0.0, np.pi / 2]),
    # Microwave opening down
    "95bc6fb98624ea3229d75ea275a1cb4e": np.array([0.0, np.pi / 2]),
}

INSTANCE_EXCLUSION_LIST = [
    # dense microwave 7292
    "b9f1eeea355194c19941e769880462e7",
    # weird oven 101971
    "d8e0d47be86833ffe8cd55fcf3e9137c",
    # weird oven 101930
    "a18d795628d848a5589577ccf07b31e7",
    # All StoragFurniture with revolute and hole at the top and not fixed!
    "24da7fd5e33513814da5feafe6f1c8fc",
    "29ce2b045bc6111912de5317fe5b354f",
    "2a3a3bf0878f1fa4c59350d819542ec7",
    "315f29b2492f66a9c59350d819542ec7",
    "36bfa6f0a5897be786d53ab0fe94e911",
    "39db396f57ec698cc59350d819542ec7",
    "3b577e2c7bbb8ca4c59350d819542ec7",
    "3c2a50e5907b0fb64da5feafe6f1c8fc",
    "3cfebf4fdfa1384da5feafe6f1c8fc",
    "3d2870c83ad35dfe86d53ab0fe94e911",
    "40f90a159f03321cc59350d819542ec7",
    "45c91d543ef3c1a829a50a2b3c5e5b6",
    "55bfa46d7b39f4dcc59350d819542ec7",
    "56031b004d5306954da5feafe6f1c8fc",
    "779cda069c3b6338824662341ce2b233",
    "786566b66299405a4da5feafe6f1c8fc",
    "7aca460dbd4ef77712de5317fe5b354f",
    "8415b7cd04f981d94692707833167ca3",
    "902a342783278a9d824662341ce2b233",
    "a234f5b48a26fe1d12de5317fe5b354f",
    "bceca165030250f94da5feafe6f1c8fc",
]

# In SimNet all PartNetMobility Objects are rotated around the XY-Axis, this makes them front facing in SimNet!
TO_SIMNET_FRAME = transform.Transform.from_aa(axis=transform.Y_AXIS, angle_deg=90.0)
TO_SIMNET_FRAME.apply_transform(
    transform.Transform.from_aa(axis=transform.X_AXIS, angle_deg=-90)
)


def get_joint_name_exclusion_list(meta_data):
    return (
        default_joint_type_excluded_category[meta_data["model_cat"]]
        + default_joint_type_excluded_instances[meta_data["model_id"]]
    )


def get_canonical_joint_limits(meta_data, joint_id):
    instance_limits = np.array(meta_data["joints"][joint_id]["limit"])
    if meta_data["model_id"] in instance_joint_limits:
        inst_limits = instance_joint_limits[meta_data["model_id"]]
        return inst_limits + instance_limits[0]
    elif (
        meta_data["model_cat"] in category_joint_limits
        and meta_data["joints"][joint_id]["type"] == "revolute"
    ):
        cat_limits = category_joint_limits[meta_data["model_cat"]]
        return cat_limits + instance_limits[0]
    else:
        return instance_limits


def get_joint_dict(joint: urdfpy.Joint) -> Dict[str, Any]:
    """
    Given urdfpy.Joint returns a dict with relevant info:
    - id
    - parent
    - type
    - limit
    """
    # TODO Nick: We might wanna make this a datacontainer for typing?
    joint_dict = {
        "id": joint.name,
        "parent": joint.parent,
        "type": joint.joint_type,
        "limit": (joint.limit.lower, joint.limit.upper)
        if joint.limit is not None
        else (0.0, 0.0),
    }
    return joint_dict


def is_joint_relevant(
    joint_dict,
    excluded_joint_names=[],
    no_limit_ok=False,
    min_prismatic=0.1,
    min_revolute=0.1,
):
    if joint_dict["parent"] == "base":
        return False

    # Parse base joint types
    if joint_dict["type"] in ["fixed", "planar", "floating"]:
        return False

    # Parse three moving part joint types
    if joint_dict["type"] == "continuous" and not no_limit_ok:
        return False

    joint_range = np.abs(joint_dict["limit"][0] - joint_dict["limit"][1])
    if joint_dict["type"] == "revolute" and joint_range < min_revolute:
        return False
    if joint_dict["type"] == "prismatic" and joint_range < min_prismatic:
        return False

    if joint_dict["sem_name"] in excluded_joint_names:
        return False

    return True


def get_filter_function(
    category_list=["all"],
    max_unique_parents=2,
    max_joints=np.inf,
    allowed_joints=["prismatic", "revolute"],
    **kwargs
):
    def relevant_joint(joint_dict, excluded_joint_names=[]):
        return is_joint_relevant(
            joint_dict, excluded_joint_names=excluded_joint_names, **kwargs
        )

    def filter(meta_data):
        if meta_data["model_cat"] not in category_list and "all" not in category_list:
            return False

        if meta_data["model_id"] in INSTANCE_EXCLUSION_LIST:
            return False

        excluded_joint_names = get_joint_name_exclusion_list(meta_data)
        relevant_joints = [
            joint
            for joint in meta_data["joints"].values()
            if relevant_joint(joint, excluded_joint_names)
        ]

        if len(relevant_joints) < 1:
            return False

        if len(relevant_joints) > max_joints:
            return False

        for joint in relevant_joints:
            if joint["type"] not in allowed_joints:
                return False

        unique_parents = [joint["parent"] for joint in relevant_joints]
        if (
            len(set(unique_parents)) > max_unique_parents
        ):  # Only load two level part trees
            return False
        return True

    return filter, relevant_joint


def get_instance_filter(instance_id_list: List[str]):
    def filter(meta_data):
        return meta_data["model_id"] in instance_id_list

    return filter
