import copy
import uuid

import numpy as np
import trimesh
import pyrender
import IPython
from typing import List

from CARTO.simnet.lib import transform

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


class Metadata:
    def __init__(self):
        self.is_object = False
        self.is_mesh_visible = True


class Node:
    """Implicitly defines a scene graph by recursivly iterating over children."""

    def __init__(self, name="", category_name="", camera=None):
        self.id = uuid.uuid4()
        self.name = name
        self.category_name = category_name
        self.object_id = ""
        self.object_joint_configuration = {}
        self.raw_object_joint_configuration = {}
        self.children: List["Node"] = []
        self.meta = Metadata()
        self.transform = transform.Transform()
        self.texture = None
        self.scene_size = None
        self.meshes = []
        self.camera = camera
        self.light = None
        self.point_cloud = None
        self._parent = None
        self._recursive_concatenated_mesh = None
        self._recursive_concatenated_decimated_mesh_vertices = None

    def assign_new_id(self):
        self.id = uuid.uuid4()

    def assign_new_id_recursive(self):
        def _func(node):
            node.id = uuid.uuid4()

        self.map_mutating(_func)

    def make_resize_mesh_node(self, new_width, new_height, new_depth):
        width, height, depth = self.bbox_lengths()
        scale_node = Node()
        scale_node.transform.matrix[0, 0] = new_width / width
        scale_node.transform.matrix[1, 1] = new_height / height
        scale_node.transform.matrix[2, 2] = new_depth / depth
        self.insert_parent(scale_node)
        scale_node.apply_transforms_to_meshes()
        return scale_node

    def swap_with_parent(self):
        # This is basically a tree rotation (e.g. Red-Black tree balancing)
        parent = self.parent
        gparent_exists = not parent.is_root
        if gparent_exists:
            gparent = parent.parent

        # Calculate new transforms
        if gparent_exists:
            gparent_T_self = self.get_transform_matrix_to_ancestor(gparent)
        else:
            gparent_T_self = transform.Transform().matrix
        self_T_parent = np.linalg.inv(self.get_transform_matrix_to_ancestor(parent))

        # remove existing links
        if gparent_exists:
            gparent._replace_child(parent, None)
        parent._replace_child(self, None)

        # Add new links
        parent._parent = None
        self._parent = None
        if gparent_exists:
            gparent.add_child(self)
        self.add_child(parent)

        # Update transforms
        self.transform.matrix = gparent_T_self
        parent.transform.matrix = self_T_parent

        return parent

    def apply_transform(self, transform):
        self.transform.apply_transform(transform)
        return self

    def bbox(self):
        sg_deepcopy = self.deepcopy()
        sg_deepcopy.apply_transforms_to_meshes()
        return sg_deepcopy.concatenated_mesh.bounds  # [min,max] x [x,y,z]

    def centroid(self):
        sg_deepcopy = self.deepcopy()
        sg_deepcopy.apply_transforms_to_meshes()
        return transform.compute_trimesh_centroid(sg_deepcopy.concatenated_mesh)

    def bbox_lengths(self):
        return self.bbox()[1, :] - self.bbox()[0, :]

    def sample_child_in_bbox(self, ratio=1.0):
        # shrink the bbox
        bbox = self.bbox()
        bbox_center = np.mean(bbox, axis=0, keepdims=True)
        bbox_inset = ratio * bbox + (1 - ratio) * bbox_center
        translation = [
            np.random.uniform(*bbox_inset[:, 0]),
            np.random.uniform(*bbox_inset[:, 1]),
            np.random.uniform(*bbox_inset[:, 2]),
        ]
        return self.add_child(
            Node(name="bbox_sample").apply_transform(
                transform.Transform.from_aa(translation=translation)
            )
        )

    def make_child_at_corner(self, right=False, up=False, front=False):
        # TODO: add semantic frame config per node, for now assume shapenet standard
        bbox = self.bbox()
        x = bbox[1, 0] if right else bbox[0, 0]
        y = bbox[1, 1] if up else bbox[0, 1]
        z = bbox[1, 2] if front else bbox[0, 2]
        name = (
            f"corner["
            f'{"R" if right else "L"}'
            f'{"U" if up else "D"}'
            f'{"F" if front else "B"}]'
        )
        return self.add_child(
            Node(name=name).apply_transform(
                transform.Transform.from_aa(translation=[x, y, z])
            )
        )

    def make_parent_at_corner(self, *args, **kwargs):
        return self.make_child_at_corner(*args, **kwargs).swap_with_parent().parent

    def clear_node(self):
        for child in self.children:
            child._parent = None
        self.children = []
        self.meshes = []
        self.meta = Metadata()
        self.decimated_meshes_vertices = []
        self.texture = None
        self._recursive_concatenated_mesh = None
        self._recursive_concatenated_decimated_mesh_vertices = None
        self.camera = None

    def find_by_id(self, id_, assert_if_missing=True):
        if self.id == id_:
            return self
        for child in self.children:
            node = child.find_by_id(id_, assert_if_missing=False)
            if node is not None:
                return node
        if assert_if_missing:
            assert False
        return None

    def find_by_name(self, name_, assert_if_missing=True):
        """
        Returns the first node with `name` containing
        """
        if self.name == name_:
            return self
        for child in self.children:
            node = child.find_by_name(name_, assert_if_missing=False)
            if node is not None:
                return node
        if assert_if_missing:
            assert False
        return None

    def assert_unity_scale(self):
        root = self.get_root()
        root_T_self = self.get_transform_matrix_to_ancestor(root)
        assert np.isclose(np.linalg.det(root_T_self[0:3, 0:3]), 1.0)

    def add_child(self, child_node, assert_if_already_set=True):
        self.children.append(child_node)
        child_node._set_parent(
            self, assert_if_already_set=assert_if_already_set
        )  # pylint: disable=protected-access
        return child_node

    def make_child_at_centroid(self):
        child = Node()
        centroid = transform.compute_trimesh_centroid(self.concatenated_mesh)
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=centroid)
        )
        self.add_child(child)
        return child

    def make_parent_at_centroid(self):
        return self.make_child_at_centroid().swap_with_parent().parent

    def make_child_at_surface_centroid(self):
        child = Node()
        bounds = self.concatenated_mesh.bounds
        centroid = copy.deepcopy(self.concatenated_mesh.centroid)
        centroid[1] = bounds[1][1]
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=centroid)
        )
        self.add_child(child)
        return child

    def make_child_at_edge_centroid(self):
        child = Node()
        bounds = self.concatenated_mesh.bounds
        centroid = copy.deepcopy(self.concatenated_mesh.centroid)
        centroid[1] = bounds[1][1]
        centroid[2] = bounds[1][2]
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=centroid)
        )
        self.add_child(child)
        return child

    def make_child_at_surface_edge(self):
        child = Node()
        bounds = self.concatenated_mesh.bounds
        centroid = copy.deepcopy(self.concatenated_mesh.centroid)
        centroid[1] = bounds[0][1]
        centroid[2] = bounds[1][2]
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=centroid)
        )
        self.add_child(child)
        return child

    def make_child_at_origin(self):
        child = Node()
        centroid = np.zeros(3)
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=centroid)
        )
        self.add_child(child)
        return child

    def make_child_at_grounded_centroid(self):
        child = Node()
        grounded_centroid = np.zeros(3)
        grounded_centroid[1] = -self.transform.matrix[1, 3]
        child.transform.apply_transform(
            transform.Transform.from_aa(translation=grounded_centroid)
        )
        self.add_child(child)
        return child

    def _set_parent(self, node, assert_if_already_set=False):
        if self._parent is not None and assert_if_already_set:
            assert False
        self._parent = node

    @property
    def is_root(self):
        return self._parent is None

    @property
    def parent(self):
        assert self._parent is not None
        return self._parent

    def sample_textures(self, texture_types, probabilities=None):
        def _func(node):
            if len(node.meshes) > 0 and node.texture is None:
                node.texture = np.random.choice(texture_types, p=probabilities)

        self.map_mutating(_func)

    def apply_textures(self):
        def _func(node):
            if node.texture is not None:
                node.texture.sample_texture(node)

        self.map_mutating(_func)

    @property
    def global_point_cloud(self, n_points=2048):
        """
        TODO Nick: Not sure if that works, returns pointcloud in the global frame?
        """
        return trimesh.sample.sample_surface(self.concatenated_mesh, n_points)[0]

    def gt_pointcloud(self):
        sg_deepcopy = self.deepcopy()
        sg_deepcopy.apply_transforms_to_meshes()
        sampled_pc = trimesh.sample.sample_surface(sg_deepcopy.concatenated_mesh, 2048)
        # obj_node.point_cloud = sampled_pc[0]
        return sampled_pc[0]  # [min,max] x [x,y,z]

    def insert_parent(self, new_parent):
        if not self.is_root:
            old_parent = self.parent
            # pylint: disable=protected-access
            old_parent._replace_child(self, new_parent)
            # pylint: enable=protected-access
        self._parent = None
        new_parent.add_child(self)
        return self

    def _replace_child(self, old_child, new_child):
        found = 0
        found_idx = None
        for idx, child in enumerate(self.children):
            if child == old_child:
                found_idx = idx
                found += 1
        if found != 1:
            raise ValueError(f"child not found 1 times {found}")
        if new_child is not None:
            self.children[found_idx] = new_child
            new_child._set_parent(self)  # pylint: disable=protected-access
        else:
            del self.children[found_idx]

    def debug(self, level=0, title=""):
        if title:
            print(f"--- Scene Graph: {title} ---")
        print(f'[{level}] {" "*level}{self!r}')
        for child in self.children:
            child.debug(level=level + 1)

    def __repr__(self):
        if self.name:
            return (
                f"Node(name={self.name}, "
                f"transform={self.transform}, num_meshes={len(self.meshes)})"
            )
        return f"Node(transform={self.transform}, num_meshes={len(self.meshes)})"

    def map_mutating(self, func, stop_nodes=None):
        if stop_nodes is None:
            stop_nodes = []
        self._recursive_concatenated_mesh = None
        if self in stop_nodes:
            return
        for child in self.children:
            child.map_mutating(func, stop_nodes=stop_nodes)
        func(self)

    def flatmap(self, func, results=None):
        if results is None:
            results = []
        for child in self.children:
            child.flatmap(func, results=results)
        results.extend(func(self))
        return results

    def flatmap_w_stop_node(self, func, stop_node, results=None):
        if results is None:
            results = []
        if self is stop_node:
            return results
        for child in self.children:
            child.flatmap_w_stop_node(func, stop_node, results=results)
        results.extend(func(self))
        return results

    def lookat(self, target_node, up_vec=Y_AXIS, up_node=None):
        """Updates self's rotation to target node's origin."""
        if up_node is None:
            # up_node defaults to root node if nothing is specified
            up_node = self.get_root()

        def normalize(vec):
            return vec / np.linalg.norm(vec)

        up_vec = normalize(up_vec)

        root_node = self.get_root()
        assert root_node == target_node.get_root()
        assert root_node == up_node.get_root()
        root_T_self = self.get_transform_matrix_to_ancestor(root_node)
        root_T_target = target_node.get_transform_matrix_to_ancestor(root_node)
        root_T_up = up_node.get_transform_matrix_to_ancestor(root_node)
        up_vec_self = root_T_self[:3, :3].T @ root_T_up[:3, :3] @ up_vec
        target_self = np.linalg.inv(root_T_self) @ root_T_target
        target_self = target_self[:3, 3]

        z_axis = -1.0 * normalize(target_self)
        x_axis = normalize(np.cross(up_vec_self, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        rotation_matrix = np.stack(
            [
                x_axis,
                y_axis,
                z_axis,
            ],
            axis=0,
        ).T
        self.transform.rotation = rotation_matrix
        assert self.transform.is_SE3()

    def get_root(self):
        if self.is_root:
            return self
        else:
            return self.parent.get_root()

    def get_all_meshes(self):
        return self.flatmap(lambda n: n.meshes)

    def get_all_visible_meshes(self):
        def _func(node):
            if node.meta.is_mesh_visible:
                return node.meshes
            return []

        return self.flatmap(_func)

    def get_all_meshes_w_stop_node(self, stop_node):
        return self.flatmap_w_stop_node(lambda n: n.meshes, stop_node)

    def get_all_object_ids(self):
        def _func(node):
            if node.meta.is_object:
                return [node.id]
            return []

        return self.flatmap(_func)

    def get_all_objects(self):
        def _func(node):
            if node.meta.is_object:
                return [node]
            return []

        return self.flatmap(_func)

    def get_all_nodes(self):
        def _func(node):
            return [node]

        return self.flatmap(_func)

    def get_camera_node(self):
        def _func(node):
            if node.camera is None:
                return []
            return [node]

        cameras = self.flatmap(_func)
        assert len(cameras) == 1
        return cameras[0]

    def get_all_nodes_by_name(self, name):
        obj_nodes = self.get_all_nodes()
        obj_nodes_by_name = []
        for obj_node in obj_nodes:
            if obj_node.name != name:
                continue
            obj_nodes_by_name.append(obj_node)
        return obj_nodes_by_name

    def get_all_light_nodes(self):
        def _func(node):
            if node.light is None:
                return []
            return [node]

        return self.flatmap(_func)

    def apply_transforms_to_meshes(self, accumlated_transform_matrix=None):
        self._recursive_concatenated_mesh = None
        if accumlated_transform_matrix is None:
            accumlated_transform_matrix = np.eye(4)

        if not self.transform.is_concrete:
            print("name : ", self.name)
            raise ValueError("Can only apply transforms to concrete transforms")
        accumlated_transform_matrix = np.matmul(
            accumlated_transform_matrix, self.transform.matrix
        )

        for mesh in self.meshes:
            mesh.apply_transform(accumlated_transform_matrix)

        self.transform = transform.Transform()

        for child_node in self.children:
            child_node.apply_transforms_to_meshes(
                accumlated_transform_matrix=accumlated_transform_matrix
            )

    def collapse_into_objects(self):
        def _func(node):
            if node.meta.is_object:
                node.concat_into_node()

        self.map_mutating(_func)

    def concat_into_node(self):
        self.meshes = [self.concatenated_mesh]
        self.children = []

    def set_scene_size(self, scene_size):
        self.scene_size = scene_size

    @property
    def concatenated_mesh(self):
        if self._recursive_concatenated_mesh is None:
            meshes = self.get_all_meshes()
            assert len(meshes) > 0
            if len(meshes) > 1:
                self._recursive_concatenated_mesh = trimesh.util.concatenate(
                    meshes[0], meshes[1:]
                )
            else:
                self._recursive_concatenated_mesh = meshes[0]
        return self._recursive_concatenated_mesh

    def get_transform_matrix_to_ancestor(self: "Node", ancestor_node: "Node"):
        if ancestor_node == self:
            if np.any(np.isnan(self.transform.matrix)):
                print("1.mine", self)
                print("1.mine.matrix", self.transform.matrix)
                assert False
            return self.transform.matrix
        partial = (
            self._parent.get_transform_matrix_to_ancestor(ancestor_node)
            @ self.transform.matrix
        )
        if np.any(np.isnan(partial)):
            print("2.parent", self._parent)
            print(
                "2.parent result",
                self._parent.get_transform_matrix_to_ancestor(ancestor_node),
            )
            print("2.mine", self)
            print("2.mine.transform", self.transform.matrix)
            assert False
        return partial

    def deepcopy(self):
        return copy.deepcopy(self)

    def debug_viewer(self, camera_node=None, show=True, fast=True):
        scene = pyrender.Scene()
        new_node_for_meshes = self.deepcopy()
        new_node_for_meshes.apply_transforms_to_meshes()

        if fast:
            trimesh_scene = trimesh.Scene()
            for mesh in new_node_for_meshes.get_all_visible_meshes():
                trimesh_scene.add_geometry(mesh)
            trimesh_single = trimesh_scene.dump(concatenate=True)
            scene.add(pyrender.Mesh.from_trimesh(trimesh_single))
        else:
            for mesh in new_node_for_meshes.get_all_visible_meshes():
                scene.add(pyrender.Mesh.from_trimesh(mesh))

        if camera_node is not None:
            root_T_camera = camera_node.get_transform_matrix_to_ancestor(self)
            scene.add(camera_node.camera.get_pyrender_camera(), pose=root_T_camera)
            height = camera_node.camera.height
            width = camera_node.camera.width
        else:
            height = 512
            width = 512
        for light_node in self.get_all_light_nodes():
            root_T_light = light_node.get_transform_matrix_to_ancestor(self)
            scene.add(light_node.light.get_pyrender_light(), pose=root_T_light)
        if show:
            pyrender.Viewer(
                scene,
                viewport_width=width,
                viewport_height=height,
                use_raymond_lighting=True,
            )
            return None
        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        # Make up simple light node for color:
        scene.add(pyrender.DirectionalLight())
        color, depth = r.render(
            scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
        )
        r.delete()
        return color, depth
