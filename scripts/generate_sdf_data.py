import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import multiprocessing
from concurrent import futures
import pathlib
import tqdm
import functools
import gc
import traceback
import logging

from typing import Dict, Any, Callable, List
import itertools

# import pyrender
import trimesh
import urdfpy
import numpy as np

import uuid

import tyro

import dataclasses

from CARTO.lib import partnet_mobility
from CARTO.lib.partnet_mobility import PartNetMobilityV0DB, PartNetMobilityV0
from CARTO.simnet.lib.datapoint import compress_datapoint, decompress_datapoint

from CARTO.Decoder import utils, config
from CARTO.Decoder.data import dataset


def get_datapoint(
    object_id,
    urdf_object,
    cfg: config.GenerationConfig,
    joint_config={},
    transform=np.eye,
    debug_print: bool = False,
):
    object_trimesh, _, scale = utils.object_to_trimesh(
        urdf_object,
        joint_config=joint_config,
        base_transform=transform,
        origin_frame=cfg.origin_frame,
    )
    points, sdf = utils.object_to_sdf(
        object_trimesh, number_samples=cfg.sdf_number_samples
    )

    pc_points, pc_normals = utils.object_to_point_cloud(
        object_trimesh, number_samples=cfg.pc_number_samples
    )

    datapoint_id = uuid.uuid4()
    datapoint = dataset.DataPoint(
        object_id=object_id,
        joint_config_id=str(datapoint_id),
        scale=scale,
        sdf_values=sdf,
        points=points,
        joint_config=joint_config,
        full_pc=pc_points,
        full_normals=pc_normals,
    )
    gc.collect()
    return datapoint


def save_to_disk(path: pathlib.Path, datapoint: dataset.DataPoint):
    buf = compress_datapoint(datapoint)
    file_path = (
        path / f"{datapoint.object_id}_{str(datapoint.joint_config_id)}.pickle.zstd"
    )
    with open(file_path, "wb") as fh:
        fh.write(buf)
    gc.collect()


def save_to_disk_future(path: pathlib.Path, future_):
    datapoint = future_.result()
    save_to_disk(path, datapoint)


def process_object_id(
    object_id: str,
    save_path: str,
    cfg: config.GenerationConfig,
    joint_filter: Callable[[Dict[str, Any]], bool] = lambda _: True,
    parallel_executor=None,
) -> List[futures.Future]:
    object_meta = PartNetMobilityV0DB.get_object_meta(object_id)
    object_path = PartNetMobilityV0DB.get_object(object_id)

    urdf_object = urdfpy.URDF.load(str(object_path / "mobility.urdf"))

    joints_of_interest: List[str] = []
    for joint_id, joint in object_meta["joints"].items():
        if not joint_filter(
            joint, partnet_mobility.get_joint_name_exclusion_list(object_meta)
        ):
            continue
        joints_of_interest.append(joint_id)

    # assert len(joints_of_interest) == 1
    if len(joints_of_interest) > 1:
        print(f"Skipping object with {len(joints_of_interest)} joints of interest")
        return []

    joint_configs_to_render = []
    for joint_steps_state in itertools.product(
        *[range(cfg.num_configs)] * len(joints_of_interest)
    ):
        # print(f"{joint_steps_state = }")
        joint_config = {}
        for joint_id, joint_step in zip(joints_of_interest, joint_steps_state):
            limits = partnet_mobility.get_canonical_joint_limits(object_meta, joint_id)
            joint_range = limits[1] - limits[0]
            # print(f"{configs_per_joint}, {joint_range}, {joint_step}")
            joint_config[joint_id] = limits[0] + (
                (joint_range * joint_step / (cfg.num_configs - 1))
                if cfg.num_configs > 1
                else joint_range / 2
            )

        # print(f"{joint_config = }")
        joint_configs_to_render.append(joint_config)

    canonical_transform = object_meta["canonical_transformation"]

    # results = []
    if parallel_executor is not None:
        all_futures = []
        for joint_config in joint_configs_to_render:
            future = parallel_executor.submit(
                get_datapoint,
                object_id,
                urdf_object,
                cfg,
                joint_config=joint_config,
                transform=canonical_transform,
            )
            save_to_disk_ = functools.partial(save_to_disk_future, save_path)
            future.add_done_callback(save_to_disk_)

            # This causes futures not to be deleted properly when done
            # unless the full list is destroyed?
            all_futures.append(future)
        return all_futures
    else:
        for joint_config in tqdm.tqdm(joint_configs_to_render):
            datapoint = get_datapoint(
                object_id,
                urdf_object,
                cfg,
                joint_config=joint_config,
                transform=canonical_transform,
            )
            save_to_disk(save_path, datapoint)
            # results.append({"joint_config": joint_config, "points": points, "sdf": sdf})
        return []


def main(args: config.GenerationConfig):
    object_filter, joint_filter = partnet_mobility.get_filter_function(
        category_list=args.categories,
        max_unique_parents=args.max_unique_parents,
        no_limit_ok=args.no_limit_ok,
        min_prismatic=args.min_prismatic,
        min_revolute=args.min_revolute,
        max_joints=args.max_joints,
        allowed_joints=args.allowed_joints,
    )
    partnet_mobility_db = PartNetMobilityV0()

    if args.id_file:
        object_ids: List[str] = open(args.id_file, "r").read().splitlines()
        object_filter = partnet_mobility.get_instance_filter(object_ids)

    partnet_mobility_db.set_filter(object_filter)
    print(f"Length of filtered dataset: {len(partnet_mobility_db)}")

    suffix = f"_{args.suffix}" if args.suffix != "" else ""
    prefix = f"{args.prefix}_" if args.prefix != "" else ""

    save_path: pathlib.Path = config.BASE_DIR / "generated_data"
    if args.id_file:
        args.id_file
        save_path /= (
            f"{prefix}{pathlib.Path(args.id_file).stem}_{args.num_configs}{suffix}"
        )
    else:
        save_path /= f"{prefix}{'_'.join(args.categories)}_{args.num_configs}{suffix}"
    save_path.mkdir(exist_ok=True, parents=True)

    def save_cfg():
        with open(save_path / "config.yaml", "w") as file:
            file.write(tyro.to_yaml(args))

    save_cfg()

    mp_context = multiprocessing.get_context("forkserver")
    # mp_context = multiprocessing.get_context("spawn")

    # TODO Nick: Maybe replace this with a queue?
    # for object_id in tqdm.tqdm(PartNetMobilityV0DB.index_list[:3]):
    # for object_id in PartNetMobilityV0DB.index_list[55:]:
    for object_id in tqdm.tqdm(partnet_mobility_db.index_list):
        if args.parallel:
            divider = 1
            retry = True
            while retry:
                retry = False

                with futures.ProcessPoolExecutor(
                    max_workers=(args.max_workers // divider), mp_context=mp_context
                ) as parallel_executor:
                    try:
                        all_futures = process_object_id(
                            object_id,
                            save_path,
                            args,
                            joint_filter=joint_filter,
                            parallel_executor=parallel_executor,
                        )

                        # This seem not to free the memory correctly!
                        # TODO Check with gc in thread
                        with tqdm.tqdm(total=len(all_futures), leave=False) as pbar:
                            for future in futures.as_completed(all_futures):
                                pbar.update(1)
                                datapoint: dataset.DataPoint = future.result()
                                # This gets rescaling in unit-sphere
                                # args.max_extent = float(
                                #     max(np.max(np.linalg.norm(datapoint.full_pc, axis=-1)), args.max_extent)
                                # )
                                # This gets rescaling in unit-cube
                                args.max_extent = float(
                                    max(
                                        np.max(np.abs(datapoint.full_pc)),
                                        args.max_extent,
                                    )
                                )
                                # As our objects are bounding box aligned, they should be roughly equal --> doesn't really matter
                                save_cfg()

                        parallel_executor.shutdown(wait=True)
                        gc.collect()
                    except Exception as e:
                        # We failed in generating
                        logging.error(traceback.format_exc())
                        print(f"Error processing {object_id}")
                        divider *= 2
                        if divider > args.max_workers:
                            retry = False
                            continue
                        print(
                            f"As memory overflowd, Re-trying with less workers --> dividing (//) workers by {divider = }"
                        )
                        retry = True
                        # Delete previously saved
                        already_saved = save_path.glob(f"{object_id}_*")
                        for file_ in already_saved:
                            file_.unlink()
        else:
            process_object_id(
                object_id,
                save_path,
                args,
                joint_filter=joint_filter,
                parallel_executor=None,
            )


if __name__ == "__main__":
    args: config.GenerationConfig = tyro.parse(config.GenerationConfig)
    main(args)
