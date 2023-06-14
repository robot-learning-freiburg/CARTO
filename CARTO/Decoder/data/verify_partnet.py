import os

# Otherwise we can't use pyrender
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ['DISPLAY'] = ':1'

import multiprocessing
from concurrent import futures
import pathlib

import tqdm
import functools
import gc
import yaml

import numpy as np

from typing import Dict, Any, Callable, List
import itertools
import open3d as o3d

# import pyrender
import trimesh
import urdfpy

from CARTO.simnet.lib import partnet_mobility
from CARTO.simnet.lib.datasets import PartNetMobilityV0, PartNetMobilityV0DB
from CARTO.simnet.lib.datapoint import compress_datapoint, decompress_datapoint
from CARTO.Decoder.visualizing import offscreen

import uuid
from CARTO.Decoder import utils, config
from CARTO.Decoder.data import dataset
import pyrender


def process_object_id(
    object_id: str,
    joint_filter: Callable[[Dict[str, Any]], bool] = lambda _: True,
    joint_offset: float = 0.0,
):
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
        if joint_id in joints_of_interest:
            limits = partnet_mobility.get_canonical_joint_limits(object_meta, joint_id)
            # limits = np.array(object_meta["joints"][joint_id]["limit"])
            # joint_config[joint_id] = limits[0]
            # joint_config[joint_id] = limits[1]
            joint_config[joint_id] = limits[0] + joint_offset
        else:
            joint_config[joint_id] = 0.0

    canonical_transform = np.array(
        PartNetMobilityV0DB.get_object_meta(object_id)["canonical_transformation"]
    )

    # if PartNetMobilityV0DB.get_object_meta(object_id)["model_cat"] == "Scissors":
    #   canonical_transform = trimesh.transformations.rotation_matrix(
    #       np.pi / 2, np.array([0., 0.0, -1.])
    #   )
    # print(canonical_transform)
    # canonical_transform = trimesh.transformations.random_rotation_matrix()

    urdf_object = urdfpy.URDF.load(str(object_path / "mobility.urdf"))

    # return utils.object_to_trimesh(urdf_object, joint_config, base_transform=canonical_transform)
    if len(joints_of_interest) == 1:
        # print(object_meta["joints"][joints_of_interest[0]])
        obj_trimesh, _, _ = utils.object_to_trimesh(
            urdf_object,
            joint_config=joint_config,
            base_transform=canonical_transform,
            origin_frame=config.ObjectOrigin.CLOSED_STATE
            # origin_frame=config.ObjectOrigin.PARTNETMOBILITY
        )
        return obj_trimesh
    else:
        return None


def main():
    object_filter, joint_filter = partnet_mobility.get_filter_function(
        # category_list=["Box"],
        # category_list=["Scissors"],
        # category_list=["Pliers"],
        # category_list=["Stapler"],
        # category_list=["Knife"],
        # category_list=["Dishwasher"],
        # category_list=["Microwave"],
        # category_list=["Oven"],
        # category_list=["Table"],
        # category_list=["WashingMachine"],
        # category_list=["Refrigerator"],
        category_list=["StorageFurniture"],
        # category_list=["Laptop"],
        # category_list=["Toilet"],
        # category_list=["Microwave", "Scissors"],
        # category_list=["Pliers", "Scissors", "Stapler"],
        # category_list=["Pliers", "Scissors"],
        # category_list=["Microwave", "Fridge", "Toilet", "WashingMachine", "Dishwasher", "Oven"],
        # category_list=[
        #     "Box", "Dishwasher", "Door", "Laptop", "Microwave", "Oven", "Refrigerator", "Safe",
        #     "StorageFurniture", "Table", "Toilet", "TrashCan", "WashingMachine", "Window", "Stapler"
        # ],
        # category_list=[
        #     "Dishwasher", "Laptop", "Microwave", "Oven", "Refrigerator", "StorageFurniture", "Table",
        #     "WashingMachine", "Stapler"
        # ],
        max_unique_parents=2,
        max_joints=1,
        no_limit_ok=False,
        min_prismatic=0.1,
        min_revolute=0.1,
        allowed_joints=["revolute"],
        # allowed_joints=["prismatic"]
    )
    partnet_mobility_db = PartNetMobilityV0()
    partnet_mobility_db.set_filter(object_filter)
    print(f"Length of filtered dataset: {len(partnet_mobility_db)}")
    # exit(0)
    joint_offset = 0.5
    # joint_offset = 3.14159

    scene = pyrender.Scene()
    added_to_scene = 0
    object_ids = partnet_mobility_db.index_list 
    for id_ in object_ids:
        print(id_)

    pcds = []

    # Hardcode some
    object_ids = ["187d79cd04b2bdfddf3a1b0d597ce76e"]

    for object_id in tqdm.tqdm(object_ids):
        # for object_id in tqdm.tqdm(object_ids[:1]):
        trimesh_scene = process_object_id(
            object_id, joint_filter=joint_filter, joint_offset=joint_offset
        )
        if trimesh_scene is None:
            continue
        trimesh_single: trimesh.Trimesh = trimesh_scene.dump(concatenate=True)

        o3d_mesh: o3d.geometry.TriangleMesh = trimesh_single.as_open3d
        o3d_mesh.paint_uniform_color([1, 0.706, 0])
        o3d_mesh.compute_vertex_normals()
        pcds.append(o3d_mesh)
        # scene.add(pyrender.Mesh.from_trimesh(trimesh_single))
        added_to_scene += 1
        # print(object_id)
        # Single Scene
        # scene_local = offscreen.get_default_scene()
        # scene_local.add(pyrender.Mesh.from_trimesh(trimesh_single))
        # pyrender.Viewer(scene_local, use_raymond_lighting=True, show_world_axis=True)
        # pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)
        print(f"{object_id}")
        o3d.visualization.draw_geometries([o3d_mesh]
    print(f"Objects in scene {added_to_scene}")
    # pyrender.Viewer(scene, use_raymond_lighting=True, show_world_axis=True)

    if True:
        points = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float,
        )
        # points /= 2.
        lines = np.array(
            [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        )
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(line_set)
    pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(pcds)
    # o3d.visualization.enable_indirect_light()


if __name__ == "__main__":
    # TODO Use new tyro feature to parse function header?
    main()
