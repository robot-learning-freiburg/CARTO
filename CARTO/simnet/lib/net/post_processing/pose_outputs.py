import numpy as np
import cv2
import copy
import IPython
import torch
import torch.nn as nn

from skimage.feature import peak_local_max

from CARTO.simnet.lib import color_stuff
from CARTO.simnet.lib import label
from CARTO.simnet.lib import camera
from CARTO.simnet.lib import transform
from CARTO.simnet.lib.net.post_processing.epnp import optimize_for_9D
from CARTO.simnet.lib.net.post_processing import epnp
from CARTO.simnet.lib.net.post_processing import nms
from CARTO.simnet.lib.net import losses
import roma

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()


class PoseOutput:
    def __init__(self, heatmap, vertex_field, z_centroid_field, hparams):
        self.heatmap = heatmap
        self.vertex_field = vertex_field
        self.z_centroid_field = z_centroid_field
        self.is_numpy = False
        self.hparams = hparams

    # Converters for torch to numpy
    def convert_to_numpy_from_torch(self):
        self.heatmap = np.ascontiguousarray(self.heatmap.float().cpu().numpy())
        self.vertex_field = np.ascontiguousarray(
            self.vertex_field.float().cpu().numpy()
        )
        self.vertex_field = self.vertex_field.transpose((0, 2, 3, 1))
        self.vertex_field = self.vertex_field / 100.0
        self.z_centroid_field = np.ascontiguousarray(
            self.z_centroid_field.float().cpu().numpy()
        )
        self.z_centroid_field = self.z_centroid_field / 100.0 + 1.0
        self.is_numpy = True

    def convert_to_torch_from_numpy(self):
        self.vertex_field = self.vertex_field.transpose((2, 0, 1))
        self.vertex_field = 100.0 * self.vertex_field
        self.vertex_field = torch.from_numpy(
            np.ascontiguousarray(self.vertex_field)
        ).float()
        self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
        # Normalize z_centroid by 1.
        self.z_centroid_field = 100.0 * (self.z_centroid_field - 1.0)
        self.z_centroid_field = torch.from_numpy(
            np.ascontiguousarray(self.z_centroid_field)
        ).float()
        self.is_numpy = False

    def get_detections(self):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()

        poses, scores = compute_9D_poses(
            np.copy(self.heatmap[0]),
            np.copy(self.vertex_field[0]),
            np.copy(self.z_centroid_field[0]),
        )

        detections = []
        for pose, score in zip(poses, scores):
            bbox = epnp.get_2d_bbox_of_9D_box(pose.camera_T_object, pose.scale_matrix)
            detections.append(
                eval3d.Detection(
                    camera_T_object=pose.camera_T_object,
                    bbox=bbox,
                    score=score,
                    scale_matrix=pose.scale_matrix,
                )
            )

        detections = nms.run(detections)

        return detections

    def get_visualization_img(self, left_img):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        return draw_pose_from_outputs(
            self.heatmap[0], self.vertex_field[0], self.z_centroid_field[0], left_img
        )

    def compute_loss(self, pose_targets, log, prefix):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        vertex_target = torch.stack(
            [pose_target.vertex_field for pose_target in pose_targets]
        )
        z_centroid_field_target = torch.stack(
            [pose_target.z_centroid_field for pose_target in pose_targets]
        )
        heatmap_target = torch.stack(
            [pose_target.heatmap for pose_target in pose_targets]
        )
        # Move to GPU
        heatmap_target = heatmap_target.to(torch.device("cuda:0"))
        vertex_target = vertex_target.to(torch.device("cuda:0"))
        z_centroid_field_target = z_centroid_field_target.to(torch.device("cuda:0"))

        vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
        log[f"{prefix}/vertex_loss"] = vertex_loss.item()
        z_centroid_loss = _mask_l1_loss(
            z_centroid_field_target, self.z_centroid_field, heatmap_target
        )
        log[f"{prefix}/z_centroid"] = z_centroid_loss.item()

        heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
        log[f"{prefix}/heatmap"] = heatmap_loss.item()
        return (
            self.hparams.loss_vertex_mult * vertex_loss
            + self.hparams.loss_heatmap_mult * heatmap_loss
            + self.hparams.loss_z_centroid_mult * z_centroid_loss
        )


def find_nearest(peaks, value):
    """
    Sorts the peaks by euclidean distance to value
    """
    newList = np.linalg.norm(peaks - value, axis=1)
    return peaks[np.argsort(newList)]


def extract_peaks_from_centroid(centroid_heatmap, min_distance=5, min_confidence=0.3):
    peaks = peak_local_max(
        centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence
    )
    return peaks


def extract_peaks_from_centroid_sorted(
    centroid_heatmap, min_distance=5, min_confidence=0.3, point=[0, 0]
):
    peaks = extract_peaks_from_centroid(
        centroid_heatmap, min_distance=min_distance, min_confidence=min_confidence
    )
    peaks = find_nearest(peaks, point)
    return peaks


def _topk(scores, K=15, threshold=0.7):
    B, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(B, height * width), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # print("topk scores", topk_scores)
    ind = topk_scores > threshold
    topk_ys = topk_ys[ind]
    topk_xs = topk_xs[ind]
    return topk_scores, topk_inds, topk_ys, topk_xs


def nms_heatmap(heat, kernel=7):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def extract_peaks_from_centroid_nms(
    centroid_heatmap, point=[0, 0], kernel=7, min_confidence: float = 0.3
):
    centroid_heatmap = torch.tensor(centroid_heatmap).unsqueeze(0)
    centroid_heatmap = nms_heatmap(centroid_heatmap, kernel=kernel)
    topk_scores, topk_inds, topk_ys, topk_xs = _topk(
        centroid_heatmap, threshold=min_confidence
    )
    peaks = np.array([topk_ys.numpy(), topk_xs.numpy()])
    peaks = np.transpose(peaks)
    peaks = find_nearest(peaks, point)
    return peaks, topk_scores, topk_inds, topk_ys, topk_xs


def extract_vertices_from_peaks(peaks, vertex_fields, c_img, scale_factor=8):
    assert peaks.shape[1] == 2
    assert vertex_fields.shape[2] == 16
    height, width = c_img.shape[0:2]
    vertex_fields = vertex_fields
    vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (2 * height) - height
    vertex_fields[:, :, 1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 * width) - width
    bboxes = []
    for ii in range(peaks.shape[0]):
        bbox = get_bbox_from_vertex(
            vertex_fields, peaks[ii, :], scale_factor=scale_factor
        )
        bboxes.append(bbox)
    return bboxes


def extract_z_centroid_from_peaks(peaks, z_centroid_output, scale_factor=8):
    assert peaks.shape[1] == 2
    z_centroids = []
    for ii in range(peaks.shape[0]):
        index = np.zeros([2])
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)
        index = index.astype(np.int)
        z_centroids.append(z_centroid_output[index[0], index[1]])
    return z_centroids


def extract_cov_matrices_from_peaks(peaks, cov_matrices_output, scale_factor=8):
    assert peaks.shape[1] == 2
    cov_matrices = []
    for ii in range(peaks.shape[0]):
        index = np.zeros([2])
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)
        index = index.astype(np.int)
        cov_mat_values = cov_matrices_output[index[0], index[1], :]
        cov_matrix = np.array(
            [
                [cov_mat_values[0], cov_mat_values[3], cov_mat_values[4]],
                [cov_mat_values[3], cov_mat_values[1], cov_mat_values[5]],
                [cov_mat_values[4], cov_mat_values[5], cov_mat_values[2]],
            ]
        )
        cov_matrices.append(cov_matrix)
    return cov_matrices


def extract_latent_emb_from_peaks(peaks, latent_emb_output, scale_factor=8):
    assert peaks.shape[1] == 2
    latent_embeddings = []
    indices = []
    # print("cov matrix", cov_matrices_output.shape)
    for ii in range(peaks.shape[0]):
        index = np.zeros([2])
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)
        index = index.astype(np.int)
        # print("indices: ", ii, "  :", index[0]*scale_factor, index[1]*scale_factor, "/n")
        latent_emb = latent_emb_output[index[0], index[1], :]
        latent_embeddings.append(latent_emb)
        # indices.append(index)
        indices.append(index * scale_factor)
    return latent_embeddings, indices


def extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=8):
    assert peaks.shape[1] == 2
    abs_poses = []
    scales = []
    for ii in range(peaks.shape[0]):
        index = np.zeros([2])
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)
        index = index.astype(np.int)
        # print(f"{abs_pose_output.shape = }")
        # print(f"{index = }")
        abs_pose_values = abs_pose_output[index[0], index[1], :]
        # print(abs_pose_values)
        rotation_matrix = np.array(
            [
                [abs_pose_values[0], abs_pose_values[1], abs_pose_values[2]],
                [abs_pose_values[3], abs_pose_values[4], abs_pose_values[5]],
                [abs_pose_values[6], abs_pose_values[7], abs_pose_values[8]],
            ]
        )

        # TODO Investigate if that helps?
        # print(f"before {rotation_matrix = }")
        rotation_matrix = roma.special_procrustes(torch.Tensor(rotation_matrix)).numpy()
        # print(f"after {rotation_matrix = }")
        translation_vector = np.array(
            [abs_pose_values[9], abs_pose_values[10], abs_pose_values[11]]
        )

        transformation_mat = np.eye(4)
        transformation_mat[:3, :3] = rotation_matrix
        transformation_mat[:3, 3] = translation_vector

        scale = abs_pose_values[12]
        scale_matrix = np.eye(4)
        scale_mat = scale * np.eye(3, dtype=float)
        scale_matrix[0:3, 0:3] = scale_mat
        scales.append(scale_matrix)

        abs_poses.append(
            transform.Pose(
                camera_T_object=transformation_mat, scale_matrix=scale_matrix
            )
        )
    return abs_poses


def get_bbox_from_vertex(vertex_fields, index, scale_factor=8):
    assert index.shape[0] == 2
    index[0] = int(index[0] / scale_factor)
    index[1] = int(index[1] / scale_factor)
    bbox = vertex_fields[index[0], index[1], :]
    bbox = bbox.reshape([8, 2])
    bbox = scale_factor * (index) - bbox
    return bbox


def draw_peaks(centroid_target, peaks):
    centroid_target = np.clip(centroid_target, 0.0, 1.0) * 255.0
    color = (0, 0, 255)
    height, width = centroid_target.shape
    # Make a 3 Channel image.
    c_img = np.zeros([centroid_target.shape[0], centroid_target.shape[1], 3])
    c_img[:, :, 1] = centroid_target
    for ii in range(peaks.shape[0]):
        point = (int(peaks[ii, 1]), int(peaks[ii, 0]))
        c_img = cv2.circle(c_img, point, 8, color, -1)
    return cv2.resize(c_img, (width, height))


def draw_pose_from_outputs(heatmap_output, vertex_output, z_centroid_output, c_img):
    poses, _ = compute_9D_poses(
        np.copy(heatmap_output), np.copy(vertex_output), np.copy(z_centroid_output)
    )
    return draw_9dof_cv2_boxes(c_img, poses)


def draw_pose_9D_from_detections(detections, c_img):
    successes = []
    poses = []
    for detection in detections:
        poses.append(
            transform.Pose(
                camera_T_object=detection.camera_T_object,
                scale_matrix=detection.scale_matrix,
            )
        )
        successes.append(detection.success)
    return draw_9dof_cv2_boxes(c_img, poses, successes=successes)


def solve_for_rotation_from_cov_matrix(cov_matrix):
    assert cov_matrix.shape[0] == 3
    assert cov_matrix.shape[1] == 3
    U, D, Vh = np.linalg.svd(cov_matrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # Rotation from world to points.
    rotation = np.eye(4)
    rotation[0:3, 0:3] = U
    return rotation


def compute_9D_poses(heatmap_output, vertex_output, z_centroid_output):
    peaks = extract_peaks_from_centroid(np.copy(heatmap_output))
    bboxes_ext = extract_vertices_from_peaks(
        np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output)
    )
    z_centroids = extract_z_centroid_from_peaks(
        np.copy(peaks), np.copy(z_centroid_output)
    )
    poses = []
    scores = []
    for bbox_ext, z_centroid, peak in zip(bboxes_ext, z_centroids, peaks):
        bbox_ext_flipped = bbox_ext[:, ::-1]
        # Solve for pose up to a scale factor
        error, camera_T_object, scale_matrix = optimize_for_9D(
            bbox_ext_flipped.T, solve_for_transforms=True
        )
        # Assign correct depth factor
        abs_camera_T_object, abs_scale_matrix = epnp.find_absolute_scale(
            z_centroid, camera_T_object, scale_matrix
        )
        poses.append(
            transform.Pose(
                camera_T_object=abs_camera_T_object, scale_matrix=abs_scale_matrix
            )
        )
        scores.append(heatmap_output[peak[0], peak[1]])
    return poses, scores


def draw_9dof_cv2_boxes(
    c_img, poses, camera_model=None, successes=None, class_labels=None
):
    boxes = []
    pixel_centers = []
    for pose in poses:
        # Compute the bounds of the boxes current size and location
        unit_box_homopoints = camera.convert_points_to_homopoints(
            epnp._WORLD_T_POINTS.T
        )
        morphed_homopoints = pose.camera_T_object @ (
            pose.scale_matrix @ unit_box_homopoints
        )
        morphed_pixels = camera.convert_homopixels_to_pixels(
            camera_model.K_matrix @ morphed_homopoints
        ).T
        boxes.append(morphed_pixels[:, ::-1])
        # Compute the centroid in pixel coordinates
        center_homopoints = camera.convert_points_to_homopoints(
            np.array([pose.camera_T_object[0:3, 3]]).T
        )
        center_pixels = camera.convert_homopixels_to_pixels(
            camera_model.K_matrix @ center_homopoints
        ).T
        pixel_centers.append(center_pixels[:, ::-1])

    c_img = draw_9dof_box(c_img, boxes, poses, successes=successes)
    if class_labels is None:
        return c_img
    if len(class_labels) != len(poses):
        return c_img
    return draw_class_labels(c_img, pixel_centers, class_labels)


def compute_pixel_centers(poses, camera_model):
    pixel_centers = []
    for pose in poses:
        # Compute the centroid in pixel coordinates
        center_homopoints = camera.convert_points_to_homopoints(
            np.array([pose.camera_T_object[0:3, 3]]).T
        )
        center_pixels = camera.convert_homopixels_to_pixels(
            camera_model.K_matrix @ center_homopoints
        ).T
        pixel_centers.append(center_pixels[:, ::-1])
    return pixel_centers


def draw_9dof_pyrender_poses(
    c_img, poses, camera_model, class_labels=None, scale_value=0.3
):
    # Convert to pyrender frame.
    poses_c = copy.deepcopy(poses)
    pyrender_T_opengl = transform.Transform.from_aa(
        axis=transform.X_AXIS, angle_deg=180.0
    ).matrix

    scale_matrix = np.eye(4)
    scale_matrix[0:3, 0:3] = scale_matrix[0:3, 0:3] * scale_value

    for ii in range(len(poses_c)):
        poses_c[ii].camera_T_object = (
            pyrender_T_opengl @ scale_matrix @ poses_c[ii].camera_T_object
        )
        poses_c[ii].scale_matrix = scale_matrix @ poses_c[ii].scale_matrix
    c_img = label.draw_absolute_pose(c_img, poses_c, camera_model=camera_model)

    if class_labels is None:
        return c_img
    pixel_centers = compute_pixel_centers(poses, camera_model)
    color = (250, 128, 114)  # Soft blue
    return draw_class_labels(c_img, pixel_centers, class_labels, color=color)


def draw_class_labels(c_img, object_pixel_centers, class_labels, color=None):
    assert_str = f"{len(object_pixel_centers)} == {len(class_labels)}"
    assert len(object_pixel_centers) == len(class_labels), f"{assert_str}"
    if color is None:
        color = (0, 255, 0)  # green

    for pixel_center, class_label in zip(object_pixel_centers, class_labels):
        if class_label == "null":
            continue
        class_label = " ".join(class_label.split("_"))
        # TODO: add more metadata to class labels so we can have human friendly names and styles
        is_door_state = False
        if class_label == "close":
            is_door_state = True
            class_label = "door closed"
        if class_label == "open":
            class_label = "door open"
            is_door_state = True
        pixel_x = int(pixel_center[0, 0])
        pixel_y = int(pixel_center[0, 1])
        if "null" in class_label:
            continue
        size = 0.5 if is_door_state else 0.75
        thickness = 1 if is_door_state else 2
        color = (150, 255, 0) if is_door_state else (0, 255, 0)
        c_img = cv2.putText(
            c_img.copy(),
            class_label,
            (pixel_y, pixel_x),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return c_img


def draw_9dof_box(c_img, boxes, poses, successes=None):
    if len(boxes) == 0:
        return cv2.cvtColor(np.array(c_img), cv2.COLOR_BGR2RGB)
    if successes is None:
        colors = color_stuff.get_colors(len(boxes))
    else:
        colors = []
        for success in successes:
            # TODO(michael.laskey): Move to Enum Structure
            if success == 1:
                colors.append(np.array([0, 255, 0]).astype(np.uint8))
            elif success == -1:
                colors.append(np.array([255, 255, 0]).astype(np.uint8))
            elif success == -2:
                colors.append(np.array([0, 0, 255]).astype(np.uint8))
            else:
                colors.append(np.array([255, 0, 0]).astype(np.uint8))
    c_img = cv2.cvtColor(np.array(c_img), cv2.COLOR_BGR2RGB)
    for pose, vertices, color in zip(poses, boxes, colors):
        visible_vertices = find_visable_vertices(pose)
        vertices = vertices.astype(np.int)
        points = []
        vertex_colors = (255, 0, 0)
        line_color = (int(color[0]), int(color[1]), int(color[2]))
        circle_colors = color_stuff.get_colors(8)
        for i, circle_color in zip(range(vertices.shape[0]), circle_colors):
            color = vertex_colors
            point = (int(vertices[i, 1]), int(vertices[i, 0]))
            points.append(point)
            if i in visible_vertices:
                c_img = cv2.circle(c_img, point, 1, (0, 255, 0), -1)
        # Draw the lines
        thickness = 2
        for idx0, idx1 in epnp.CUBE_EDGE_VERTEX_PAIRS:
            if idx0 in visible_vertices and idx1 in visible_vertices:
                c_img = cv2.line(
                    c_img, points[idx0], points[idx1], line_color, thickness
                )
    return c_img


def find_visable_vertices(pose):
    camera_t_focalpoint = np.array([0.0, 0.0, 0.0, 1.0]).reshape([4, 1])
    object_t_focalpoint = np.linalg.inv(pose.camera_T_object) @ camera_t_focalpoint
    object_t_focalpoint = object_t_focalpoint[:3, 0]
    half_extents = np.diag(pose.scale_matrix)[:3] * 0.5
    points = set()
    if object_t_focalpoint[0] > half_extents[0]:
        points.update([4, 5, 6, 7])  # positive x face
    if object_t_focalpoint[0] < -half_extents[0]:
        points.update([0, 1, 2, 3])  # negative x face
    if object_t_focalpoint[1] > half_extents[1]:
        points.update([2, 3, 6, 7])  # positive y face
    if object_t_focalpoint[1] < -half_extents[1]:
        points.update([0, 1, 4, 5])  # negative y face
    if object_t_focalpoint[2] > half_extents[2]:
        points.update([1, 3, 5, 7])  # positive z face
    if object_t_focalpoint[2] < -half_extents[2]:
        points.update([0, 2, 4, 6])  # negative z face
    return points
