"""Partnet + annotations to scene graph functions."""
import json
import copy

import numpy as np
import trimesh
import IPython

from CARTO.simnet.lib import sg
from CARTO.simnet.lib import pose_sampler
from CARTO.simnet.lib import transform
from CARTO.simnet.lib import datasets
from CARTO.simnet.lib import primitive


def load_points_from_file(points_file_path):
    """Returns the homogenous decimated vertices (Nx4)"""
    with open(points_file_path) as f:
        rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows]
    points = [[float(point) for point in coords] for coords in coords_set]
    points = np.array(points)
    # Turn to homogenous coordinates
    ones = np.ones([points.shape[0], 1])
    return np.concatenate((points, ones), axis=1)


def load_labels_from_file(points_file_path):
    """Returns a list of integer labels corresponding to each part"""
    with open(points_file_path) as f:
        rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows]
    labels = [[int(point) for point in coords] for coords in coords_set]
    return labels


def load_trimesh_parts(file_names, file_path):
    meshes = []
    for file_name in file_names:
        meshes.append(trimesh.load(file_path / "objs" / (file_name + ".obj")))
    return meshes


def create_parameterized_scene_graph(partnet_node, file_path):
    node = sg.Node()
    node.meta.partnet_id = partnet_node["id"]
    if "objs" in partnet_node:
        node.meshes = load_trimesh_parts(partnet_node["objs"], file_path)
        node.meta.is_part = True
    if "children" not in partnet_node:
        return node
    for partnet_child_node in partnet_node["children"]:
        child_node = create_parameterized_scene_graph(partnet_child_node, file_path)
        node.add_child(child_node)
    return node


def get_orphan(tree_correction, partnet_node):
    if "children" in partnet_node:
        for child in partnet_node["children"]:
            if int(child["id"]) == int(tree_correction["child"]):
                orphan = copy.deepcopy(child)
                partnet_node["children"].remove(child)
                return orphan
            orphan = get_orphan(tree_correction, child)
            if orphan is not None:
                return orphan
    return None


def place_orphan_with_parent(tree_correction, partnet_node, orphan):
    if int(partnet_node["id"]) == int(tree_correction["parent"]):
        if "children" in partnet_node:
            partnet_node["children"].append(orphan)
        else:
            partnet_node["children"] = [orphan]
    if "children" in partnet_node:
        for child in partnet_node["children"]:
            place_orphan_with_parent(tree_correction, child, orphan)


def apply_corrections(parent_child_corrections, partnet_node):
    for parent_child_correction in parent_child_corrections:
        orphan = get_orphan(parent_child_correction, partnet_node)
        assert orphan is not None
        place_orphan_with_parent(parent_child_correction, partnet_node, orphan)


def add_shelves_to_part_tree(root_node, nominal_root_node, shelf_list):
    for shelf in shelf_list:
        if root_node.meta.partnet_id == int(shelf["node"]):
            root_node.meta.is_shelf = True
            bounds = nominal_root_node.concatenated_mesh.bounds
            root_node.meta.nominal_concatenated_mesh_bounds = bounds
    for i in range(len(root_node.children)):
        add_shelves_to_part_tree(
            root_node.children[i], nominal_root_node.children[i], shelf_list
        )


def load_part_tree(class_name, mesh_id, nominal=False, meta_data=None, raw=False):
    """Convert partnet + mmt annotations to scene graph."""
    mmt_annotation = datasets.MMTAnnotationsDB.get_sample_meta(class_name)
    labeled_partnet_data = mmt_annotation["partnet_data"][class_name]
    if class_name in ["Car", "Guitar", "Airplane", "Motorbike"]:
        file_path = datasets.AdobeV0DB.get_sample(mesh_id)
    elif class_name in ["Can", "Camera"]:
        file_path = datasets.NOCSV0DB.get_sample(mesh_id)
    elif class_name in ["Marker"]:
        file_path = None
    else:
        file_path = datasets.PartnetV0DB.get_sample(mesh_id)

    if mesh_id in labeled_partnet_data["Objects"]:
        parent_child_corrections = labeled_partnet_data["Objects"][mesh_id][
            "parent_child_corrections"
        ]
        shelf_labels = labeled_partnet_data["Objects"][mesh_id]["shelves"]
    else:
        articulation_labels = []
        parent_child_corrections = []
        shelf_labels = []

    if file_path is not None:
        with open(file_path / "result.json") as fh:
            partnet_root_node = json.load(fh)[0]
            apply_corrections(parent_child_corrections, partnet_root_node)
        root_node = create_parameterized_scene_graph(partnet_root_node, file_path)
    else:
        root_node = primitive.make_plane()
    meshes = root_node.get_all_meshes()
    concatenated_mesh = trimesh.util.concatenate(meshes[0], meshes[1:])
    if meta_data is None:
        meta_data = mmt_annotation["mesh_meta_data"]
        assert "parameterized_transform" in meta_data
    transform.center_mesh(root_node)
    if not raw:
        if "parameterized_transform" in meta_data:
            pt = meta_data["parameterized_transform"]
            if "scale_range" in pt:
                new_parent = sg.Node()
                if "height_limit" in pt:
                    scale_value = np.random.uniform(
                        pt["scale_range"][0], pt["scale_range"][1]
                    )
                    scale_value = max(pt["height_limit"][1], scale_value)
                    scale_value = min(pt["height_limit"][0], scale_value)
                    transform.apply_absolute_scale_value(root_node, scale_value)
                else:
                    scale_value = np.random.uniform(
                        pt["scale_range"][0], pt["scale_range"][1]
                    )
                    transform.apply_absolute_scale_value(root_node, scale_value)

        if "initial_transform" in meta_data:
            it = meta_data["initial_transform"]
            new_parent = sg.Node()
            if "scale_matrix" in it:
                initial_rotate = transform.Transform.from_aa(
                    np.array(it["rotation_axis"]), it["angle"]
                ).matrix
                scale_transform = transform.apply_scale_matrix(
                    root_node, it["scale_matrix"] @ initial_rotate
                )
            else:
                new_parent = sg.Node()
                new_parent.transform = transform.Transform.from_aa(
                    np.array(it["rotation_axis"]), it["angle"]
                )
                # Apply transform to mesh and throw away
                new_parent.add_child(root_node)
                root_node = new_parent

    root_node.meta.is_appliance = meta_data["is_appliance"]
    root_node.meta.is_object = True
    anno_id = None
    if file_path is not None:
        with open(file_path / "meta.json") as fh:
            anno_id = json.load(fh)["anno_id"]
    else:
        anno_id = "0000"
    assert anno_id is not None
    if isinstance(anno_id, int):
        anno_id = "0000"
    root_node.meta.shapenet_object_id = mesh_id + "_" + anno_id
    root_node.meta.class_id = class_name
    return root_node
