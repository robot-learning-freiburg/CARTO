import multiprocessing
from concurrent import futures
import pathlib

import tqdm
import functools
import gc

from typing import Dict, Any, Callable, List
import itertools

# import pyrender
import trimesh
import urdfpy

from CARTO.simnet.lib import partnet_mobility
from CARTO.simnet.lib.datasets import PartNetMobilityV0DB
from CARTO.simnet.lib.datapoint import compress_datapoint, decompress_datapoint

import uuid
from CARTO.Decoder import utils, config
from CARTO.Decoder.data import dataset
import open3d as o3d
import numpy as np


def process_object_id(
    object_id: str,
    joint_filter: Callable[[Dict[str, Any]], bool] = lambda _: True,
    joint_offset: float = 0.0,
):
    # object_id = "187d79cd04b2bdfddf3a1b0d597ce76e"

    object_path = PartNetMobilityV0DB.get_object(object_id)
    object_meta = PartNetMobilityV0DB.get_object_meta(object_id)

    joints_of_interest: List[str] = []
    # Artifact from preprocessing
    for joint_id, joint in object_meta["joints"].items():
        if not joint_filter(
            joint, partnet_mobility.get_joint_name_exclusion_list(object_meta)
        ):
            continue
        joints_of_interest.append(joint_id)

    joint_config = {}
    for joint_id, joint in object_meta["joints"].items():
        joint_config[joint_id] = joint["limit"][0] + (
            joint_offset if joint_id in joints_of_interest else 0.0
        )

    canonical_transform = np.array(
        PartNetMobilityV0DB.get_object_meta(object_id)["canonical_transformation"]
    )
    urdf_object = urdfpy.URDF.load(str(object_path / "mobility.urdf"))
    trimesh_object, _, _ = utils.object_to_trimesh(
        urdf_object, joint_config=joint_config, base_transform=canonical_transform
    )
    # points, sdf = utils.object_to_sdf(trimesh_object)
    # points = points[sdf <= 0]
    points, _ = utils.object_to_point_cloud(trimesh_object, number_samples=100000)
    color = utils.get_random_color()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.points = o3d.utility.Vector3dVector(s_pc.points)
    pcd.paint_uniform_color(color)
    return pcd


def main():
    object_filter, joint_filter = partnet_mobility.get_filter_function(
        # category_list=["Microwave", "Laptop"],
        category_list=["Laptop"],
        # category_list=["Microwave"],
        # category_list=["WashingMachine"],
        max_unique_parents=1,
        no_limit_ok=False,
        min_prismatic=0.1,
        min_revolute=0.1,
    )
    PartNetMobilityV0DB.set_filter(object_filter)
    print(f"Length of filtered dataset: {len(PartNetMobilityV0DB)}")

    pcds = []
    for object_id in tqdm.tqdm(PartNetMobilityV0DB.index_list):
        pcd: o3d.geometry.PointCloud = process_object_id(
            object_id, joint_filter=joint_filter, joint_offset=1.5
        )
        pcds.append(pcd)
        pcd_local = [pcd]
        o3d.visualization.draw_geometries(pcd_local)

    o3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    main()
