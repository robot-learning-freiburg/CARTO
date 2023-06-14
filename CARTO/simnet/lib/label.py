"""Normalized coordinate labels."""
import dataclasses
import typing

import numpy as np
import trimesh
import cv2
import copy
import pyrender

from CARTO.simnet.lib import transform, camera, color_stuff, sg, primitive


# @dataclasses.dataclass
# class TargetLabel:
#     obj_node: sg.Node
#     instance_mask: np.ndarray
#     target_name: str
#     category_name: str
#     camera_T_object: np.ndarray
#     occlusion_score: float = 0.0
#     is_fully_in_image: bool = False
#     image_occlusion_score: float = 0.0


@dataclasses.dataclass
class Pose:
    camera_T_object: np.ndarray
    scale_matrix: np.ndarray


@dataclasses.dataclass
class BoundingBox:
    category_name: str
    bounding_box: np.ndarray


@dataclasses.dataclass
class OBB:
    camera_T_no_rot_object: np.ndarray  # Used to regress scale in a canonical frame.
    scale_matrix: np.ndarray
    cov_matrix: np.ndarray
    voxelized_mesh_in_camera_frame: typing.Any
    camera_T_object: np.ndarray
    category_name: str = None


# Wraps a particular keypoint class in an image.
# If the image contains N shirts, and each shirt contains two keypoints
# (e.g. sleeves), the shape of PIXELS is (2N, 2), with one pixel coordinate
# pair for each sleeve on each shirt.
@dataclasses.dataclass
class Keypoint:
    pixels: np.ndarray


@dataclasses.dataclass
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclasses.dataclass
class Label:
    def __init__(self, scene_graph, target_names, target_categories=None):
        if target_categories is None:
            target_categories = []
        self.target_labels: typing.List[TargetLabel] = []
        self.scene_graph = scene_graph
        self.camera_model = scene_graph.get_camera_node().camera
        all_objects = populate_target_objects(scene_graph)

        # print(f"{target_names = }")
        # print(f"{target_categories = }")
        self.target_categories = target_categories
        for target_name in target_names:
            for obj in all_objects:
                # print(f"{obj.target_name = }")
                # print(f"{target_name = }")
                if obj.target_name != target_name:
                    continue
                self.target_labels.append(obj)

        # print("======================")

        self.target_names = target_names
        self.camera_model = scene_graph.get_camera_node().camera

    def target_label_iterator(self, occlusion_score=0.0, target_name=""):
        for target_label in self.target_labels:
            if target_name != "" and target_label.target_name != target_name:
                continue
            print(
                f"{target_label.occlusion_score = }, {target_label.image_occlusion_score = }"
            )
            if (
                target_label.occlusion_score <= occlusion_score
                or target_label.image_occlusion_score <= occlusion_score
            ):
                continue
            yield target_label

    def get_segmentation_mask(self):
        seg_mask = np.zeros([self.camera_model.height, self.camera_model.width])
        for target_label in self.target_labels:
            class_num = 1
            for target_name in self.target_names:
                if target_name != target_label.target_name:
                    class_num += 1
                else:
                    break
            seg_mask[target_label.instance_mask > 0] = class_num
        return seg_mask

    def get_object_category_segmentation_mask(self):
        seg_mask = np.zeros([self.camera_model.height, self.camera_model.width])
        for class_num, target_category in enumerate(self.target_categories):
            for target_label in self.target_labels:
                # print(
                #     f"[get_object_category_segmentation_mask] {target_category.name} == {target_label.category_name}"
                # )
                if target_category.name == target_label.category_name:
                    # print(f"{class_num = }")
                    # Zero for background!! See segmentation_outputs.draw_segmentation_mask
                    seg_mask[target_label.instance_mask > 0] = class_num + 1

        return seg_mask

    def get_instance_masks(self, occlusion_score=0.0):
        masks = []
        occ_scores = []
        for target_label in self.target_label_iterator(occlusion_score=occlusion_score):
            masks.append(target_label.instance_mask)
            occ_scores.append(target_label.occlusion_score)
        return masks, occ_scores

    def get_boxes(self, occlusion_score=0.0):
        boxes = []
        for target_label in self.target_label_iterator(occlusion_score=occlusion_score):
            scales = target_label.obj_node.bbox_lengths()
            scale_matrix = np.eye(4)
            scale_matrix[0:3, 0:3] = np.diag(scales)
            boxes.append(
                BoundingBox(
                    bounding_box=camera.get_2d_bbox_of_9D_box(
                        self.camera_model, target_label.camera_T_object, scale_matrix
                    ),
                    category_name=target_label.category_name,
                )
            )
        return boxes

    def get_object_ids_and_config(self, target_name, occlusion_score=0.6):
        object_ids = []
        joint_configs = []
        raw_joint_configs = []
        for target_label in self.target_label_iterator(
            occlusion_score=occlusion_score, target_name=target_name
        ):
            object_ids.append(target_label.obj_node.object_id)
            joint_configs.append(target_label.obj_node.object_joint_configuration)
            raw_joint_configs.append(
                target_label.obj_node.raw_object_joint_configuration
            )
            # TODO target_label.obj_node.scale_factor?
        return object_ids, joint_configs, raw_joint_configs

    def get_oriented_bounding_boxes(
        self, target_name, occlusion_score=0.6, voxel_amount=128
    ):
        def compute_scale_from_points(points):
            assert points.shape[1] == 3
            x_bound = np.max(points[:, 0], axis=0) - np.min(points[:, 0], axis=0) + 5e-3
            y_bound = np.max(points[:, 1], axis=0) - np.min(points[:, 1], axis=0) + 5e-3
            z_bound = np.max(points[:, 2], axis=0) - np.min(points[:, 2], axis=0) + 5e-3
            scale_matrix = np.diag([x_bound, y_bound, z_bound, 1.0])
            return scale_matrix

        def compute_pca_box(camera_T_homopoints):
            points = camera.convert_homopoints_to_points(camera_T_homopoints).T
            assert points.shape[1] == 3
            # Compute translation
            translation = np.average(points, axis=0)
            # Zero mean the points
            centered_points = points - translation
            # Compute the covariance matrix
            cov_matrix = np.cov(centered_points.T)
            assert cov_matrix.shape[0] == 3
            U, D, Vh = np.linalg.svd(cov_matrix, full_matrices=True)
            d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
            if d:
                D[-1] = -D[-1]
                U[:, -1] = -U[:, -1]
            # Rotation from world to points.
            rotation = U
            ## Overwrite to make axis-aligned?
            # rotation = np.eye(3)

            # align points along principle axis
            aligned_points = np.linalg.inv(rotation) @ centered_points.T
            # Ccompute scale matrix
            scale_matrix = compute_scale_from_points(aligned_points.T)
            camera_T_object = np.eye(4)
            camera_T_object[0:3, 0:3] = rotation
            camera_T_object[0:3, 3] = translation

            return camera_T_object, scale_matrix, cov_matrix

        obbs = []
        masks = []
        for target_label in self.target_label_iterator(
            occlusion_score=occlusion_score, target_name=target_name
        ):
            masks.append(target_label.instance_mask)
            mesh_in_root = copy.deepcopy(target_label.obj_node.concatenated_mesh)
            root_T_camera = (
                self.scene_graph.get_camera_node().get_transform_matrix_to_ancestor(
                    self.scene_graph
                )
            )
            # root_T_mesh = target_label.obj_node.get_transform_matrix_to_ancestor(self.scene_graph)
            mesh_in_root.apply_transform(np.linalg.inv(root_T_camera))
            # mesh_in_root.apply_transform(np.linalg.inv(root_T_mesh))  # Mesh centric view

            print(f"{mesh_in_root.extents.max() / voxel_amount = }")
            voxelized_mesh_in_camera_frame = mesh_in_root.voxelized(
                pitch=mesh_in_root.extents.max() / voxel_amount,
                # method="binvox" # binvox does not work yet on EC2
                # method="ray"
                method="subdivide",
            )
            camera_T_points = voxelized_mesh_in_camera_frame.points
            camera_T_homopoints = camera.convert_points_to_homopoints(camera_T_points.T)

            camera_T_object, scale_matrix, cov_matrix = compute_pca_box(
                camera_T_homopoints
            )

            # camera_T_object = np.linalg.inv(root_T_camera) @ root_T_mesh @ camera_T_object
            # cov_matrix = root_T_mesh[:3, :3] @ cov_matrix

            bounding_box = camera.get_2d_bbox_of_9D_box(
                self.camera_model, camera_T_object, scale_matrix
            )
            camera_T_no_rot_object = transform.Transform.from_aa(
                axis=transform.X_AXIS, angle_deg=45.0
            ).matrix
            camera_T_no_rot_object[0:3, 3] = camera_T_object[0:3, 3]
            obbs.append(
                OBB(
                    camera_T_no_rot_object=camera_T_no_rot_object,
                    scale_matrix=scale_matrix,
                    cov_matrix=cov_matrix,
                    camera_T_object=camera_T_object,
                    voxelized_mesh_in_camera_frame=voxelized_mesh_in_camera_frame,
                    category_name=target_label.category_name,
                )
            )
        return obbs, masks

    # def get_door_state(self, occlusion_score=0.6):
    #     door_states = []
    #     for target_label in self.target_label_iterator(
    #         occlusion_score=occlusion_score, target_name="cabinet_door"
    #     ):
    #         door_states.append(target_label.obj_node.is_open)
    #     return door_states

    # def label_door_state(self, door_obbs, occlusion_score=0.6):
    #     door_states = self.get_door_state(occlusion_score=occlusion_score)

    #     assert len(door_states) == len(door_obbs)
    #     for door_obb, door_state in zip(door_obbs, door_states):
    #         if door_state:
    #             door_obb.category_name = "open"
    #         else:
    #             door_obb.category_name = "close"
    #         if door_obb.category_name not in DOOR_STATES:
    #             raise ValueError("Invalid Door States")

    # def get_grasp_labels(self, obbs, masks, robot_T_camera, max_range=10.0):
    #     grasps_per_obb = []
    #     masks_per_obb = []
    #     for obj_obb, mask in zip(obbs, masks):
    #         grasps_per_obb.append(score_grasps(obj_obb, robot_T_camera))
    #         masks_per_obb.append(mask)
    #     return grasps_per_obb, masks_per_obb

    # def get_keypoints(self, target_name, occlusion_score=0.6):
    #     keypoints = (
    #         []
    #     )  # contains keypoint wrappers for each key vertex, grouped by key vertex group.
    #     all_obj_keypoints = []  # all object keypoints, grouped by object
    #     all_obj_valid = []  # all object keypoints, grouped by object
    #     root_T_camera = (
    #         self.scene_graph.get_camera_node().get_transform_matrix_to_ancestor(
    #             self.scene_graph
    #         )
    #     )
    #     camera_model = self.scene_graph.get_camera_node().camera
    #     # For each cloth object, find the pixel location of all key vertices in the camera plane
    #     for target_label in self.target_label_iterator(
    #         occlusion_score=occlusion_score, target_name=target_name
    #     ):
    #         meshes = target_label.obj_node.meshes
    #         if len(meshes) != 1:
    #             raise ValueError("Target label must consist of a single mesh.")
    #         obj_mesh = meshes[0]
    #         obj_keypoints = []  # keeps track of pixel locations of keypoints
    #         valid_masks = []
    #         all_obj_keypoints.append(obj_keypoints)
    #         all_obj_valid.append(valid_masks)
    #         # iterate over the classes of keypoints
    #         for key_vertex_group in target_label.obj_node.key_vertices:
    #             root_T_points = []
    #             # iterate over each key vertex in the vertex group (e.g. left sleeve, right sleeve)
    #             for vertex_id in key_vertex_group:
    #                 root_T_points.append(np.copy(obj_mesh.vertices[vertex_id]))
    #             root_T_points = np.vstack(root_T_points)
    #             root_T_homopoints = camera.convert_points_to_homopoints(root_T_points.T)
    #             camera_T_homopoints = np.linalg.inv(root_T_camera) @ root_T_homopoints
    #             pixels, pixel_depths, valid_mask = camera.project_points(
    #                 camera_T_homopoints.T,
    #                 camera_model.proj_matrix,
    #                 (camera_model.height, camera_model.width),
    #             )
    #             obj_keypoints.append(pixels)
    #             valid_masks.append(valid_mask)

    #     # right now all_obj_keypoints has 3 levels: object -> vertex group -> vertex
    #     num_objects = len(all_obj_keypoints)  # number of objects to label
    #     # iterate over each object
    #     for i in range(num_objects):
    #         # iterate over vertex groups
    #         for j in range(len(all_obj_keypoints[0])):
    #             if i == 0:
    #                 keypoints.append(
    #                     []
    #                 )  # add empty list for each keypoint group if first object

    #             for k in range(len(all_obj_keypoints[i][j])):
    #                 if i == 0:
    #                     keypoints[j].append(
    #                         []
    #                     )  # add empty list for each keypoint in group if first object
    #                 cur_kp = keypoints[j][k]  # group index j, keypoint index k
    #                 px = all_obj_keypoints[i][j][
    #                     k
    #                 ]  # find pixel for keypoint k of group j of object i
    #                 valid = all_obj_valid[i][j][k]  # check if above pixel is valid
    #                 if valid:
    #                     cur_kp.append(px)  # add to wrapper for keypoint
    #                 if i == num_objects - 1:
    #                     # if last object, combine the keypoints from all objects together and replace
    #                     kp = Keypoint(pixels=cur_kp)
    #                     keypoints[j][k] = kp
    #     # keypoints is now a list of lists of Keypoint wrappers where the outer
    #     # list is over vertex groups and the inner list is a keypoint wrapper
    #     # for each keypoint in the vertex group.
    #     return keypoints

    # TODO Nick: Potentially refactor these functions!

    def get_poses(self, target_name, occlusion_score=0.6):
        """
        TODO Nick: seems like it returns a list of poses for all objects in the scene that are not occluded
        """
        poses = []
        x_index = []
        shapes_names = []
        sym_shapes = ["bottle", "can", "bowl"]  # For Shapenet
        for target_label in self.target_label_iterator(
            occlusion_score=occlusion_score, target_name=target_name
        ):
            # center = [np.average(indices) for indices in np.where(target_label.instance_mask>0)]
            # x_index.append(center[1])
            root_T_object = target_label.obj_node.get_transform_matrix_to_ancestor(
                self.scene_graph
            )
            root_T_camera = (
                self.scene_graph.get_camera_node().get_transform_matrix_to_ancestor(
                    self.scene_graph
                )
            )

            # print(f"{target_label.obj_node = }")
            # print(f"{root_T_camera = }")
            # print(f"{root_T_object = }")

            root_camera_T_object = np.linalg.inv(root_T_camera) @ root_T_object
            scale_factor = target_label.obj_node.scale_factor
            scale_matrix = np.eye(4)
            scale_mat = scale_factor * np.eye(3, dtype=float)
            scale_matrix[0:3, 0:3] = scale_mat

            poses.append(
                transform.Pose(
                    camera_T_object=root_camera_T_object, scale_matrix=scale_matrix
                )
            )
            shapes_names.append(target_label.category_name)
        # print("x_index poses", x_index)
        # poses = [x for y, x in sorted(zip(x_index, poses), key=lambda pair: pair[0])]
        return poses, shapes_names

    def get_pointclouds(self, target_name, occlusion_score=0.6):
        """
        TODO Nick: seems like it returns a list of pc for all objects in the scene that are not occluded
        """
        point_clouds = []
        x_index = []
        for target_label in self.target_label_iterator(
            occlusion_score=occlusion_score, target_name=target_name
        ):
            # center = [np.average(indices) for indices in np.where(target_label.instance_mask>0)]
            # x_index.append(center[1])
            point_cloud = target_label.obj_node.point_cloud
            point_clouds.append(point_cloud)
        # point_clouds = [x for y, x in sorted(zip(x_index, point_clouds)), key=lambda pair: pair[0])]
        return point_clouds

    def get_rotated_pointclouds(
        self,
        target_name,
        _CAMERA=None,
        occlusion_score=0.6,
    ):
        """
        Nick: This could be broken, needs to be checked
        """
        point_clouds = []
        for target_label in self.target_label_iterator(
            occlusion_score=occlusion_score, target_name=target_name
        ):
            mesh_in_root = target_label.obj_node.concatenated_mesh
            point_cloud = trimesh.sample.sample_surface(mesh_in_root, 2048)

            # ------------------- #
            root_T_homopoints = camera.convert_points_to_homopoints(point_cloud[0].T)
            root_T_camera = (
                self.scene_graph.get_camera_node().get_transform_matrix_to_ancestor(
                    self.scene_graph
                )
            )
            morphed_pc_homopoints = (
                _CAMERA.RT_matrix @ np.linalg.inv(root_T_camera) @ root_T_homopoints
            )
            morphed_pc_homopoints = camera.convert_homopoints_to_points(
                morphed_pc_homopoints
            ).T
            # ------------------- #
            # Replace above with?
            # tranform_pc(identity_pose, point_cloud, _CAMERA=_CAMERA, use_camera_RT=False)

            # print("point cloud in get", point_cloud[0].shape)
            # o3d.visualization.draw_geometries([point_cloud[0]])
            # point_cloud = target_label.obj_node.gt_pointcloud()
            # print("point cloud in get", point_cloud.shape)
            # print("point cloud shape", point_cloud[0].shape)
            point_clouds.append(morphed_pc_homopoints)
        return point_clouds

    # def get_gt_pointclouds(self, target_name, _CAMERA, occlusion_score):
    #   """
    #   TODO Nick: This could be broken, needs to be checked !!
    #   """
    #   poses = []
    #   scales = self.get_current_scales(target_name)
    #   pointclouds = self.get_pointclouds(target_name)
    #   poses_old = self.get_poses(target_name)
    #   obbs, _ = self.get_oriented_bounding_boxes(target_name)

    #   print("len obb pose scales", len(obbs), len(poses_old), len(scales))
    #   for obb, pose, scale in zip(obbs, poses_old, scales):
    #     poses.append(
    #         transform.Pose(camera_T_object=pose.camera_T_object, scale_matrix=pose.scale_matrix)
    #     )
    #   pc_points = []
    #   box_points = []
    #   images = []
    #   for pose, pc in zip(poses, pointclouds):
    #     morphed_pc_homopoints, morphed_box_points = camera.tranform_pc(
    #         pose, pc, use_camera_RT=False
    #     )  # TODO Nick: Maybe this should be true
    #     image = _CAMERA.project_points_to_depth_img(morphed_pc_homopoints)

    #     pc_points.append(morphed_pc_homopoints)
    #     box_points.append(morphed_box_points)
    #     images.append(image)
    #   return pc_points, box_points, images


# Visualization of Label Code
def draw_pixelwise_mask(color_img, seg_mask, colors=None):
    num_classes = int(np.max(seg_mask)) + 1
    if num_classes == 0:
        return color_img
    if colors == None:
        colors = get_unique_colors(num_classes)
    colored_mask = np.zeros([seg_mask.shape[0], seg_mask.shape[1], 3])
    for ii, color in zip(range(num_classes), colors):
        colored_mask[seg_mask == ii, :] = color
    color_img = cv2.addWeighted(
        color_img.astype(np.uint8), 0.9, colored_mask.astype(np.uint8), 0.4, 0
    )
    return color_img


def draw_absolute_pose(color_img, poses, camera_model=None):
    """
    **Warning**: Poses are not in the center but rather at the corner
    """

    def draw_9dof_cad_img(detection_img, c_img):
        flat_detection_img = detection_img.reshape([-1, 3])
        flat_c_img = c_img.reshape([-1, 3])
        valid_indices = np.where(np.sum(flat_detection_img, axis=1) < (255 * 2))
        flat_c_img[valid_indices] = flat_detection_img[valid_indices][:, ::-1]
        return flat_c_img.reshape(c_img.shape)

    root = sg.Node()
    camera_node = sg.Node()
    if camera_model is None:
        camera_node.camera = camera.Camera()
    else:
        camera_node.camera = camera_model
    camera_transform = sg.Node()
    pyrender_T_opengl = transform.Transform.from_aa(
        axis=transform.X_AXIS, angle_deg=180.0
    ).matrix
    if len(poses) == 0:
        return color_img
    colors = color_stuff.get_colors(len(poses))
    camera_transform.add_child(camera_node)
    root.add_child(camera_transform)
    for pose, color in zip(poses, colors):
        camera_T_object = pyrender_T_opengl @ pose.camera_T_object
        # pose_node = primitive.make_9DOF_pose(pose.scale_matrix, color=color)
        pose_node = primitive.make_coordinate_frame()
        scale_node = sg.Node()
        scale_node.transform = transform.Transform(pose.scale_matrix)
        scale_node.add_child(pose_node)
        transform_node = sg.Node()
        transform_node.transform = transform.Transform(pose.camera_T_object)
        transform_node.add_child(scale_node)
        root.add_child(transform_node)
    detection_img, _ = root.debug_viewer(camera_node=camera_node, show=False)
    img = draw_9dof_cad_img(detection_img, color_img)
    return img


def draw_boxes(color_img, boxes):
    for box in boxes:
        cv2.rectangle(
            color_img,
            (int(box[0][1]), int(box[0][0])),
            (int(box[1][1]), int(box[1][0])),
            (0, 255, 0),
            2,
        )
    return color_img


### CODE TO GENERATE INSTNACE MASKS ####
def populate_target_objects(scene_graph):
    """
    Parses the scene graph and returns the object ids of
    all un-occluded objects and the object masks.
    Args:
        scene_graph: a scene graph data structure
    Returns:
        a list of object ids , a list of masks
    """
    object_data: typing.List[TargetLabel] = get_object_masks(
        scene_graph, scene_graph.get_all_objects()
    )
    target_data = []
    for obj_datum in object_data:
        image_occlusion, occlusion_score = is_object_unoccluded(obj_datum, scene_graph)
        obj_datum.is_fully_in_image = image_occlusion == 1.0
        obj_datum.occlusion_score = occlusion_score
        obj_datum.image_occlusion_score = image_occlusion
        target_data.append(obj_datum)

    return target_data


def get_object_masks(scene_graph, object_nodes):
    """
    Given a list of trimesh objects and a camera pose this
    returns the object masks
    Args:
       scene_graph: the overall scene graph
    Returns:
       A list of binary object masks
    """
    camera_node = scene_graph.get_camera_node()
    object_meshes = []
    root_T_objects = []
    for obj_node in object_nodes:
        object_meshes.append(obj_node.concatenated_mesh)
        root_T_objects.append(obj_node.get_transform_matrix_to_ancestor(scene_graph))
    object_meshes_flat = copy.deepcopy(object_meshes)
    # Get unique colors.
    colors = get_unique_colors(len(object_nodes))
    # Assign colors.
    for color, root_T_object, mesh_ in zip(colors, root_T_objects, object_meshes_flat):
        mesh_.apply_transform(root_T_object)
        color_visual = trimesh.visual.ColorVisuals(vertex_colors=color)
        mesh_.visual = color_visual
    color_img, depth_img = render_flat_scene(
        object_meshes_flat, camera_node, scene_graph
    )
    object_data = []
    root_T_camera = camera_node.get_transform_matrix_to_ancestor(scene_graph)
    for color, object_node, root_T_object in zip(colors, object_nodes, root_T_objects):
        mask = np.all(color_img == color, axis=-1).astype(np.uint8)
        object_data.append(
            TargetLabel(
                instance_mask=mask,
                obj_node=object_node,
                target_name=object_node.name,
                category_name=object_node.category_name,
                camera_T_object=np.linalg.inv(root_T_camera) @ root_T_object,
            )
        )
    return object_data


def render_flat_scene(objects, camera_node, scene_graph):
    """Renders a flat scene (or scene with no lighting effects) of the given
    objects
      Args:
         objects: a list of trimesh objects
         camera_node : The camera node of the scene
         scene_graph : The overall scene graph
      Returns:
          A rendered color img
    """

    root_T_camera = camera_node.get_transform_matrix_to_ancestor(scene_graph)
    scene = pyrender.Scene()
    scene.add(camera_node.camera.get_pyrender_camera(), pose=root_T_camera)
    height = camera_node.camera.height
    width = camera_node.camera.width
    for obj in objects:
        scene.add(pyrender.Mesh.from_trimesh(obj))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    # Define render flags for flat rendering.
    render_flags = (
        pyrender.constants.RenderFlags.FLAT
        | pyrender.constants.RenderFlags.SKIP_CULL_FACES
        | pyrender.constants.RenderFlags.TRI_DISABLE_ANTIALIASING
    )

    color_img, depth_img = renderer.render(scene, flags=render_flags)
    renderer.delete()
    return color_img, depth_img


def is_object_fully_in_image(obj_node, camera_node, root):
    """
    Calculates the visible percentage of the object in the image
    """
    root_T_object = obj_node.get_transform_matrix_to_ancestor(root)
    root_T_camera = camera_node.get_transform_matrix_to_ancestor(root)
    obj_mesh = obj_node.concatenated_mesh
    camera_T_root = np.linalg.inv(root_T_camera)
    obj_mesh.apply_transform(root_T_object)
    homovertices = camera.convert_points_to_homopoints(obj_mesh.vertices.T)
    homovertices = camera_T_root @ homovertices
    homopixels = camera_node.camera.project(homovertices)
    pixels = camera.convert_homopixels_to_pixels(homopixels).T

    # if np.min(pixels[:, 0]) < 0 or np.min(pixels[:, 1]) < 0:
    #   return False
    height = camera_node.camera.height
    width = camera_node.camera.width
    # if np.max(pixels[:, 0]) >= width or np.max(pixels[:, 1]) >= height:
    #   return False
    # return True

    return np.count_nonzero(
        np.logical_and.reduce(
            (
                pixels[:, 0] > 0,
                pixels[:, 1] > 0,
                pixels[:, 0] < width,
                pixels[:, 1] < height,
            )
        )
    ) / (pixels.shape[0])


def is_object_unoccluded(target_label, scene_graph):
    """
    Given an object and a mask from  the current scene, this determines if the
    object is occluded by looking at the percentage it is covered.
    Args:
       obj: A trimesh object
       mask: A binary mask
       camera_node: The camera node of the scene
       scene_graph: The scene graph of the world
    Returns
        True if unoccluded, False otherwise
    """
    camera_node = scene_graph.get_camera_node()
    object_data = get_object_masks(scene_graph, [target_label.obj_node])
    true_object_mask = object_data[0].instance_mask
    mask_intersection = true_object_mask * target_label.instance_mask
    if float(np.sum(true_object_mask)) == 0:
        return False, 0.0
    percent_overlap = np.sum(mask_intersection) / float(np.sum(true_object_mask))
    image_occlusion = is_object_fully_in_image(
        target_label.obj_node, camera_node, scene_graph
    )
    return image_occlusion, percent_overlap


def get_colors(num_colors):
    assert num_colors > 0

    colors = list(
        color_stuff.Color("red").range_to(color_stuff.Color("purple"), num_colors)
    )
    color_rgb = 255 * np.array([np.array(a.get_rgb()) for a in colors])
    color_rgb = [a.astype(np.int) for a in color_rgb]
    return color_rgb


def get_unique_colors(num_colors):
    """
    Gives a the specified number of unique colors
    Args:
       num_colors: an int specifying the number of colors
    Returs:
       A list of  rgb colors in the range of (0,255)
    """
    color_rgb = get_colors(num_colors)

    if num_colors != len(np.unique(color_rgb, axis=0)):
        raise ValueError("Colors returned are not unique.")

    return color_rgb


if __name__ == "__main__":
    main()
