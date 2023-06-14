from typing import List
import tyro
import dataclasses
import pathlib
import enum
import re
import operator
from CARTO.Decoder import config, utils
from CARTO.Decoder.data import dataset
from CARTO.simnet.lib.datasets import PartNetMobilityV0DB
import copy
import math
import tqdm

import random
import numpy as np
import itertools
import yaml

# A bunch of objects that are very unique and thus should only be in the train
ONLY_TRAIN = [
    "ae8441235538d4415d85df7c37878bb6",
    "uc96ccf13-0360-49f7-885b-0efdae244728",
    "857147b940e582c9c05f7b017030de8",
    "9f4eb0d734a2b7a4ab610b0c94236463",
    "3b5d7b7fb89b178d50711e66b0db6ed",
    "95bc6fb98624ea3229d75ea275a1cb4e",
    "ae8441235538d4415d85df7c37878bb6",
    "c16cba81-714d-4b1a-94cd-7a148af83db0",
    "d4f4b5bf712a96b13679ccb6aaef8b00",
    "b7bb5f021d40ae59f7b4b759a0c02e4a",
]


def get_partnet_id_from_file_id(file_id):
    file_id = str(file_id)
    split_results = re.split("[/_.]", file_id)
    assert split_results[-1] == "zstd"
    assert split_results[-2] == "pickle"

    joint_config_id = split_results[-3]
    object_id = split_results[-4]
    return object_id


class SplitLevel(enum.Enum):
    # We will pick a random set of in-training instance ids to test on
    Instance = enum.auto()
    # For each category in the dataset we will randomly split the instances in train and test
    Category = enum.auto()
    # We will completly exclude a training category from the test set
    InterCategory = enum.auto()


@dataclasses.dataclass
class SplitGenArgs:
    dataset_path: str
    split_level: SplitLevel
    test_categories: List[str] = config.default_field([])
    # Instance level: of total files, Category level: of total instances per category, InterCategory: ignored
    test_ratio: float = 0.2
    # Instance level: of total files, Category level: of total instances per category, InterCategory: of total instances per category
    val_ratio: float = 0.1
    seed: int = 123
    suffix: str = ""
    prefix: str = ""


def get_split_indices(N, test_ratio, val_ratio):
    cut_off_1 = math.floor(N * (1 - test_ratio - val_ratio))
    cut_off_2 = math.floor(N * (1 - test_ratio))
    return cut_off_1, cut_off_2


def main(args: SplitGenArgs):
    dataset_path = pathlib.Path(args.dataset_path)
    all_file_ids = list(dataset_path.glob("*.zstd"))

    folder_string = args.prefix
    folder_string += args.split_level._name_
    if len(args.test_categories) > 0:
        folder_string += "_" + "_".join(args.test_categories)
    folder_string += args.suffix
    split_file_folder = dataset_path / "splits" / folder_string
    split_file_folder.mkdir(exist_ok=True, parents=True)

    # file_ids_to_object_cat = {}
    object_cat_to_file_ids = utils.AccumulatorDict()
    object_id_to_file_ids = utils.AccumulatorDict()
    object_cat_to_object_ids = utils.AccumulatorDict(accumulator=operator.or_)

    proc_dataset = dataset.SimpleDataset(all_file_ids, cache_only_meta=True)
    # for file_id in all_file_ids:
    for idx, datapoint in tqdm.tqdm(enumerate(proc_dataset)):
        # object_id = get_partnet_id_from_file_id(file_id)
        object_id = datapoint.object_id
        model_cat = PartNetMobilityV0DB.get_object_meta(object_id)["model_cat"]

        joint_id = list(datapoint.joint_config)[0]
        joint_type = datapoint.joint_def[joint_id]["type"]

        model_cat = f"{model_cat}_{joint_type}"

        file_id = all_file_ids[idx]
        object_cat_to_file_ids.increment(model_cat, [file_id])
        object_id_to_file_ids.increment(object_id, [file_id])
        object_cat_to_object_ids.increment(model_cat, {object_id})
        # object_cat_to_object_ids.increment(model_cat, set(object_id))

    split_dict = utils.AccumulatorDict()

    if args.split_level == SplitLevel.Instance:
        for file_ids in object_id_to_file_ids.values():
            cut_off_1, cut_off_2 = get_split_indices(
                len(file_ids), args.test_ratio, args.val_ratio
            )

            random.shuffle(file_ids)
            train_files = file_ids[:cut_off_1]
            val_files = file_ids[cut_off_1:cut_off_2]
            test_files = file_ids[cut_off_2:]

            split_dict.increment("train", train_files)
            split_dict.increment("val", val_files)
            split_dict.increment("test", test_files)
    elif args.split_level == SplitLevel.Category:
        all_object_ids = {"train": [], "val": [], "test": []}
        for category, object_ids in object_cat_to_object_ids.items():
            object_ids = list(object_ids)
            cut_off_1, cut_off_2 = get_split_indices(
                len(object_ids), args.test_ratio, args.val_ratio
            )

            re_shuffle = True
            re_shuffle_count = 0
            while re_shuffle:
                random.shuffle(object_ids)
                train_ids = object_ids[:cut_off_1]
                val_ids = object_ids[cut_off_1:cut_off_2]
                test_ids = object_ids[cut_off_2:]

                re_shuffle = any(val_id in ONLY_TRAIN for val_id in val_ids) or any(
                    test_id in ONLY_TRAIN for test_id in test_ids
                )
                re_shuffle_count += 1
            print(f"Re-shuffled: {re_shuffle_count}")

            all_object_ids["train"].extend(train_ids)
            all_object_ids["val"].extend(val_ids)
            all_object_ids["test"].extend(test_ids)

            split_dict.increment(
                "train",
                list(
                    itertools.chain(*[object_id_to_file_ids[id_] for id_ in train_ids])
                ),
            )
            split_dict.increment(
                "val",
                list(itertools.chain(*[object_id_to_file_ids[id_] for id_ in val_ids])),
            )
            split_dict.increment(
                "test",
                list(
                    itertools.chain(*[object_id_to_file_ids[id_] for id_ in test_ids])
                ),
            )

            category_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
            category_split_dict = {
                key: list(
                    itertools.chain(*[object_id_to_file_ids[id_] for id_ in vals])
                )
                for key, vals in category_ids.items()
            }

            category_folder = split_file_folder / category
            category_folder.mkdir(exist_ok=True, parents=True)

            with open(category_folder / "ids.yaml", "w") as f:
                yaml.dump(category_split_dict, f)
            with open(category_folder / "object_ids.yaml", "w") as f:
                yaml.dump(category_ids, f)

    elif args.split_level == SplitLevel.InterCategory:
        for category, object_ids in object_cat_to_object_ids.items():
            if category in args.test_categories:
                split_dict.increment(
                    "test",
                    list(
                        itertools.chain(
                            *[object_id_to_file_ids[id_] for id_ in object_ids]
                        )
                    ),
                )
            else:
                object_ids = list(object_ids)
                N = len(object_ids)
                cut_off = math.floor(N * (1 - args.val_ratio))
                train_ids = object_ids[:cut_off]
                val_ids = object_ids[cut_off:]
                split_dict.increment(
                    "train",
                    list(
                        itertools.chain(
                            *[object_id_to_file_ids[id_] for id_ in train_ids]
                        )
                    ),
                )
                split_dict.increment(
                    "val",
                    list(
                        itertools.chain(
                            *[object_id_to_file_ids[id_] for id_ in val_ids]
                        )
                    ),
                )

    print("--- Split Statistics ---")
    print("-      File Splits     -")
    for name, files in split_dict.items():
        print(f"- {name}: {len(files)}")

    with open(split_file_folder / "config.yaml", "w") as file:
        file.write(tyro.to_yaml(args))
    with open(split_file_folder / "ids.yaml", "w") as file:
        yaml.dump(split_dict, file)

    if args.split_level == SplitLevel.Category:
        print("-  Object ID Splits  -")
        for set_name, object_ids in all_object_ids.items():
            print(f"- {set_name}: {len(object_ids)}")
        with open(split_file_folder / "object_ids.yaml", "w") as file:
            yaml.dump(all_object_ids, file)


if __name__ == "__main__":
    args = tyro.parse(SplitGenArgs)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
