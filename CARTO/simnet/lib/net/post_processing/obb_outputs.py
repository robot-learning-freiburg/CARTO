import numpy as np
import cv2
import copy
import IPython
import torch
import torch.nn as nn
from torch.nn import functional as F

from skimage.feature import peak_local_max

from CARTO.simnet.lib import color_stuff
from CARTO.simnet.lib import transform
from CARTO.simnet.lib.net.post_processing.epnp import optimize_for_9D
from CARTO.simnet.lib.net.post_processing import epnp
from CARTO.simnet.lib.net.post_processing import nms
from CARTO.simnet.lib.net.post_processing import pose_outputs
from CARTO.simnet.lib.net.post_processing import eval3d
from CARTO.simnet.lib.net.post_processing import utils
from CARTO.simnet.lib.net import losses

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()

# TODO(mike.laskey) Save correct robot_T_camera in short posture
_ROBOT_T_SHORT_CAMERA = np.array(
    [
        [1.78979770e-02, -3.70753997e-01, 9.28558634e-01, 5.54787371e-02],
        [-9.99816895e-01, -3.47922479e-04, 1.91325626e-02, 5.87290881e-02],
        [-6.77040762e-03, -9.28731044e-01, -3.70692337e-01, 1.18662336e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)


class OBBOutput:
    def __init__(
        self,
        heatmap,
        vertex_field,
        z_centroid_field,
        cov_field,
        hparams,
        class_field=None,
        shape_emb=None,
        arti_emb=None,
        abs_pose_field=None,
    ):
        self.heatmap = heatmap
        self.vertex_field = vertex_field
        self.z_centroid_field = z_centroid_field
        self.cov_field = cov_field
        self.is_numpy = False
        self.hparams = hparams
        self.class_field = class_field
        self.shape_emb = shape_emb
        self.arti_emb = arti_emb
        self.abs_pose_field = abs_pose_field

    # Converters for torch to numpy
    def convert_to_numpy_from_torch(self):
        self.heatmap = np.ascontiguousarray(self.heatmap.float().cpu().numpy())
        if self.heatmap.ndim == 2:
            self.heatmap = np.expand_dims(self.heatmap, 0)
        self.vertex_field = utils.to_numpy_from_torch(self.vertex_field)
        self.cov_field = utils.to_numpy_from_torch(self.cov_field, multiplier=1000.0)
        self.z_centroid_field = np.ascontiguousarray(
            self.z_centroid_field.float().cpu().numpy()
        )
        self.z_centroid_field = self.z_centroid_field / 100.0 + 1.0
        if self.z_centroid_field.ndim == 2:
            self.z_centroid_field = np.expand_dims(self.z_centroid_field, 0)

        if self.class_field is not None:
            self.class_field = np.ascontiguousarray(
                self.class_field.float().cpu().numpy()
            )
        # shape emb
        if self.shape_emb is not None:
            # print("self.shape_emb", self.shape_emb.shape)
            self.shape_emb = utils.to_numpy_from_torch(self.shape_emb)
        # arti emb
        if self.arti_emb is not None:
            self.arti_emb = utils.to_numpy_from_torch(self.arti_emb)
        # abs pose emb
        if self.abs_pose_field is not None:
            self.abs_pose_field = utils.to_numpy_from_torch(self.abs_pose_field)

        self.is_numpy = True

    def convert_to_torch_from_numpy(self):
        self.vertex_field = utils.to_torch_from_numpy(self.vertex_field)
        self.cov_field = utils.to_torch_from_numpy(self.cov_field, multiplier=1000.0)
        self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
        # Normalize z_centroid by 1.
        self.z_centroid_field = 100.0 * (self.z_centroid_field - 1.0)
        self.z_centroid_field = torch.from_numpy(
            np.ascontiguousarray(self.z_centroid_field)
        ).float()

        if self.class_field is not None:
            self.class_field = torch.from_numpy(
                np.ascontiguousarray(self.class_field)
            ).long()
        # shape embedding
        if self.shape_emb is not None:
            self.shape_emb = utils.to_torch_from_numpy(self.shape_emb)
        if self.arti_emb is not None:
            self.arti_emb = utils.to_torch_from_numpy(self.arti_emb)
        # abs pose
        if self.abs_pose_field is not None:
            self.abs_pose_field = utils.to_torch_from_numpy(self.abs_pose_field)

        self.is_numpy = False

    def get_detections(self, index, camera_model, class_list=None):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        if self.class_field is None:
            class_pred = None
        else:
            class_pred = np.argmax(self.class_field[index], axis=0)

        poses, scores, peaks, classes = compute_oriented_bounding_boxes(
            np.copy(self.heatmap[index]),
            np.copy(self.vertex_field[index]),
            np.copy(self.z_centroid_field[index]),
            np.copy(self.cov_field[index]),
            camera_model=camera_model,
            classes_outputs=class_pred,
        )
        detections = []
        for ii, pose in enumerate(poses):
            bbox = epnp.get_2d_bbox_of_9D_box(
                pose.camera_T_object, pose.scale_matrix, camera_model
            )
            if self.class_field is None or class_list is None:
                class_label = "null"
            else:
                class_label = class_list[classes[ii]]
            detections.append(
                eval3d.Detection(
                    camera_T_object=pose.camera_T_object,
                    bbox=bbox,
                    score=scores[ii],
                    class_label=class_label,
                    scale_matrix=pose.scale_matrix,
                )
            )
        detections = nms.run(detections)
        return detections

    def get_visualization_img(
        self,
        index,
        left_img,
        camera_model=None,
        class_list=None,
        poses=False,
        prune_distance=False,
        command=None,
    ):
        # Draw the utterance if it exists.
        if command:
            size = 1.0
            thickness = 2
            color = (0, 255, 0)
            left_img = cv2.putText(
                left_img.copy(),
                command,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                color,
                thickness,
                cv2.LINE_AA,
            )

        if not self.is_numpy:
            self.convert_to_numpy_from_torch()

        # if self.class_field is None or class_list is None:
        #   class_pred = None
        # else:
        #   class_pred = np.argmax(self.class_field[index], axis=0)

        if poses:
            # TODO: Is the order ensured?!
            # peaks = pose_outputs.extract_peaks_from_centroid_sorted(np.copy(self.heatmap[index]))
            # class_labels = extract_classes_from_peaks(np.copy(peaks), np.copy(class_pred))
            class_labels = None
            abs_pose_outputs, _, _, _, _, _, _, _ = self.compute_embs_and_poses(
                index=index
            )
            return pose_outputs.draw_9dof_pyrender_poses(
                left_img,
                abs_pose_outputs,
                camera_model=camera_model,
                class_labels=class_labels,
            )
        else:
            detections = self.get_detections(index, camera_model, class_list)
            # Does not affect poses!
            if prune_distance:
                detections = prune_detections_by_distance(detections)
            class_labels = []
            for detection in detections:
                class_labels.append(detection.class_label)
            return pose_outputs.draw_9dof_cv2_boxes(
                left_img,
                detections,
                camera_model=camera_model,
                class_labels=class_labels,
            )

    ## ------------------ TODO Potentially refactor ------------------
    def nms_heatmap(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    ## TODO Correct for two embeddings
    def compute_pointclouds_and_poses(self, is_target=False):
        if is_target:
            heatmap = np.ascontiguousarray(self.heatmap.clone().cpu().numpy())

            latent_emb = np.ascontiguousarray(self.latent_emb.clone().cpu().numpy())
            latent_emb = latent_emb.transpose((1, 2, 0))
            latent_emb = latent_emb / 100.0

            abs_pose_field = np.ascontiguousarray(
                self.abs_pose_field.clone().cpu().numpy()
            )
            abs_pose_field = abs_pose_field.transpose((1, 2, 0))
            abs_pose_field = abs_pose_field / 100.0
            (
                latent_embeddings,
                abs_pose_outputs,
                img,
                img_nms,
                indices,
            ) = compute_pointclouds_and_poses(heatmap, latent_emb, abs_pose_field)
        else:
            if not self.is_numpy:
                self.convert_to_numpy_from_torch()
            (
                latent_embeddings,
                abs_pose_outputs,
                img,
                img_nms,
                indices,
            ) = compute_pointclouds_and_poses(
                np.copy(self.heatmap[0]),
                np.copy(self.latent_emb[0]),
                np.copy(self.abs_pose_field[0]),
            )

        return latent_embeddings, abs_pose_outputs, img, img_nms, indices

    def compute_embs_and_poses(self, index=0, is_target=False):
        # TODO Figure out what is_target is for
        # shape emb
        if is_target:
            if self.is_numpy:
                heatmap = np.copy(self.heatmap)
                shape_emb = np.copy(self.shape_emb)
                arti_emb = np.copy(self.arti_emb)
                abs_pose_field = np.copy(self.abs_pose_field)
            else:
                heatmap = np.ascontiguousarray(self.heatmap.float().cpu().numpy())
                shape_emb = utils.to_numpy_from_torch(self.shape_emb)
                arti_emb = utils.to_numpy_from_torch(self.arti_emb)
                abs_pose_field = utils.to_numpy_from_torch(self.abs_pose_field)
            if heatmap.ndim == 2:
                heatmap = np.expand_dims(self.heatmap, 0)
            (
                abs_pose_outputs,
                img,
                img_nms,
                latent_embeddings_shape,
                scores,
                indices_shape,
                latent_embeddings_arti,
                indices_arti,
            ) = compute_embs_and_poses(
                heatmap[index], shape_emb[index], arti_emb[index], abs_pose_field[index]
            )
        else:
            if not self.is_numpy:
                self.convert_to_numpy_from_torch()
            (
                abs_pose_outputs,
                img,
                img_nms,
                latent_embeddings_shape,
                scores,
                indices_shape,
                latent_embeddings_arti,
                indices_arti,
            ) = compute_embs_and_poses(
                np.copy(self.heatmap[index]),
                np.copy(self.shape_emb[index]),
                np.copy(self.arti_emb[index]),
                np.copy(self.abs_pose_field[index]),
            )
        return (
            abs_pose_outputs,
            img,
            img_nms,
            latent_embeddings_shape,
            scores,
            indices_shape,
            latent_embeddings_arti,
            indices_arti,
        )

    ## TODO Correct for two embeddings
    def get_latent_embeddings(self, is_target=False):
        if is_target:
            heatmap = np.ascontiguousarray(self.heatmap.clone().cpu().numpy())
            latent_emb = np.ascontiguousarray(self.latent_emb.clone().cpu().numpy())
            latent_emb = latent_emb.transpose((1, 2, 0))
            latent_emb = latent_emb / 100.0
            latent_embeddings, img, indices = compute_point_cloud_embeddings(
                heatmap, latent_emb
            )
        else:
            if not self.is_numpy:
                self.convert_to_numpy_from_torch()
            latent_embeddings, img, img_nms, indices = compute_point_cloud_embeddings(
                np.copy(self.heatmap[0]), np.copy(self.latent_emb[0])
            )
        return latent_embeddings, img, img_nms, indices

    ## ^^^^^^^^^^^^^^^^^^ TODO Potentially refactor ^^^^^^^^^^^^^^^^^^

    def compute_loss(self, obb_targets, log, prefix):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        vertex_target = torch.stack(
            [obb_target.vertex_field for obb_target in obb_targets]
        )
        z_centroid_field_target = torch.stack(
            [obb_target.z_centroid_field for obb_target in obb_targets]
        )

        heatmap_target = torch.stack([obb_target.heatmap for obb_target in obb_targets])
        cov_target = torch.stack([obb_target.cov_field for obb_target in obb_targets])

        # Move to GPU
        heatmap_target = heatmap_target.to(torch.device("cuda:0"))
        vertex_target = vertex_target.to(torch.device("cuda:0"))
        z_centroid_field_target = z_centroid_field_target.to(torch.device("cuda:0"))
        cov_target = cov_target.to(torch.device("cuda:0"))

        cov_loss = _mask_l1_loss(cov_target, self.cov_field, heatmap_target)
        log[f"{prefix}/cov_loss"] = cov_loss.item()
        vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
        log[f"{prefix}/vertex_loss"] = vertex_loss.item()
        z_centroid_loss = _mask_l1_loss(
            z_centroid_field_target, self.z_centroid_field, heatmap_target
        )
        log[f"{prefix}/z_centroid"] = z_centroid_loss.item()

        heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
        log[f"{prefix}/heatmap"] = heatmap_loss.item()

        loss = 0.0
        loss += self.hparams.loss_vertex_mult * vertex_loss
        loss += self.hparams.loss_heatmap_mult * heatmap_loss
        loss += self.hparams.loss_z_centroid_mult * z_centroid_loss
        loss += self.hparams.loss_rotation_mult * cov_loss

        if self.class_field is not None and obb_targets[0].class_field is not None:
            class_target = torch.stack(
                [obb_target.class_field for obb_target in obb_targets]
            )
            class_target = class_target.to(torch.device("cuda:0"))
            # Class Loss
            class_target[heatmap_target[:, ::8, ::8] < 0.3] = -100
            class_loss = F.cross_entropy(
                self.class_field, class_target, reduction="mean", ignore_index=-100
            )
            loss += self.hparams.loss_seg_mult * class_loss
            log[f"{prefix}/class"] = class_loss.item()

        # Latent Shape Loss
        if obb_targets[0].shape_emb is not None:
            latent_emb_target = torch.stack(
                [obb_target.shape_emb for obb_target in obb_targets]
            )
            latent_emb_target = latent_emb_target.to(torch.device("cuda:0"))

            latent_emb_loss = _mask_l1_loss(
                latent_emb_target, self.shape_emb, heatmap_target
            )
            log[f"{prefix}/latent_shape_emb_loss"] = latent_emb_loss
            loss += self.hparams.loss_latent_emb_mult * latent_emb_loss

        # Latent Articulation Loss
        if obb_targets[0].arti_emb is not None:
            latent_emb_target = torch.stack(
                [obb_target.arti_emb for obb_target in obb_targets]
            )
            latent_emb_target = latent_emb_target.to(torch.device("cuda:0"))

            latent_emb_loss = _mask_l1_loss(
                latent_emb_target, self.arti_emb, heatmap_target
            )
            log[f"{prefix}/latent_arti_emb_loss"] = latent_emb_loss
            loss += self.hparams.loss_latent_emb_mult * latent_emb_loss

        # Naively regressing the absolute pose
        if obb_targets[0].abs_pose_field is not None:
            abs_pose_target = torch.stack(
                [obb_target.abs_pose_field for obb_target in obb_targets]
            )
            abs_pose_target = abs_pose_target.to(torch.device("cuda:0"))

            # symmetric orthogonalization
            # rotations = []
            # for i in range(self.abs_pose_field.shape[0]):
            #   x = self.abs_pose_field[i][:9,:,:].clone()
            #   x = x.permute(1,2,0).view(64*120,-1)
            #   x = _symmetric_orthogonalization(x)
            #   x = x.view(64,120,3,3)
            #   x = x.view(64,120,-1).permute(2,0,1).unsqueeze(0)
            #   rotations.append(x)
            # rotations = torch.cat(rotations, dim=0)

            # #faster SVD orthogonalization
            # start_time = time.time()
            # B,C,H,W = self.abs_pose_field.shape
            # rotations = self.abs_pose_field[:,:9,:,:].permute(0,2,3,1).contiguous().view(B*H*W,-1)
            # rotations = _symmetric_orthogonalization(rotations)
            # rotations= rotations.contiguous().view(B,H,W,3,3)
            # rotations=rotations.contiguous().view(B,H,W,9).permute(0,3,1,2)

            # svd rotation loss
            abs_rotation_loss = _mask_l1_loss(
                abs_pose_target[:, :9, :, :],
                self.abs_pose_field[:, :9, :, :],
                heatmap_target,
            )
            log[f"{prefix}/abs_svd_rotation_loss"] = abs_rotation_loss

            # svd translation_ scale loss
            abs_trans_scale_loss = _mask_l1_loss(
                abs_pose_target[:, 9:, :, :],
                self.abs_pose_field[:, 9:, :, :],
                heatmap_target,
            )
            log[f"{prefix}/abs_svd_translation+scale_loss"] = abs_trans_scale_loss

            abs_pose_loss = abs_rotation_loss + abs_trans_scale_loss
            log[f"{prefix}/abs_svd_pose_loss"] = abs_pose_loss

            # complete rotation loss
            # abs_pose_loss = _mask_l1_loss(abs_pose_target, self.abs_pose_field, heatmap_target)
            # log['abs_pose_loss'] = abs_pose_loss
            loss += self.hparams.loss_abs_pose_mult * abs_pose_loss

        return loss


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


def extract_classes_from_peaks(peaks, class_output, scale_factor=8):
    assert peaks.shape[1] == 2
    class_values = []
    for ii in range(peaks.shape[0]):
        index = np.zeros([2])
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)
        index = index.astype(np.int)
        class_value = class_output[index[0], index[1]]
        class_values.append(class_value)
    return class_values


def prune_detections_by_distance(detections, x_thresh=1.1, y_thresh=1.0):
    pruned_detections = []
    for detection in detections:
        robot_T_object = _ROBOT_T_SHORT_CAMERA @ detection.camera_T_object
        if np.abs(robot_T_object[1, 3]) > y_thresh:
            continue
        if robot_T_object[0, 3] > x_thresh:
            continue
        pruned_detections.append(detection)
    return pruned_detections


def draw_oriented_bounding_box_from_outputs(
    heatmap_output,
    vertex_output,
    rotation_output,
    z_centroid_output,
    c_img,
    camera_model=None,
    classes_output=None,
    class_list=None,
    command=None,
):
    poses, _, _, classes = compute_oriented_bounding_boxes(
        np.copy(heatmap_output),
        np.copy(vertex_output),
        np.copy(z_centroid_output),
        np.copy(rotation_output),
        camera_model=camera_model,
        classes_outputs=classes_output,
    )

    image = None
    if class_list is None:
        # Don't draw the class labels
        image = pose_outputs.draw_9dof_cv2_boxes(
            c_img, poses, camera_model=camera_model
        )
    else:
        # Draw the class labels
        class_labels = [class_list[int(ii)] for ii in classes]
        image = pose_outputs.draw_9dof_cv2_boxes(
            c_img, poses, camera_model=camera_model, class_labels=class_labels
        )

    # Draw the utterance if it exists.
    if command:
        size = 1.0
        thickness = 2
        color = (0, 255, 0)
        image = cv2.putText(
            image.copy(),
            command,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return image


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


def compute_oriented_bounding_boxes(
    heatmap_output,
    vertex_output,
    z_centroid_output,
    cov_matrices,
    camera_model,
    classes_outputs=None,
    ground_truth_peaks=None,
):
    peaks = pose_outputs.extract_peaks_from_centroid(np.copy(heatmap_output))
    bboxes_ext = pose_outputs.extract_vertices_from_peaks(
        np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output)
    )
    z_centroids = pose_outputs.extract_z_centroid_from_peaks(
        np.copy(peaks), np.copy(z_centroid_output)
    )
    cov_matrices = pose_outputs.extract_cov_matrices_from_peaks(
        np.copy(peaks), np.copy(cov_matrices)
    )
    poses = []
    scores = []
    classes = []
    assert len(bboxes_ext) == len(peaks)
    assert len(z_centroids) == len(peaks)
    assert len(cov_matrices) == len(peaks)
    for ii, peak in enumerate(peaks):
        bbox_ext_flipped = bboxes_ext[ii][:, ::-1]
        # Solve for pose up to a scale factor
        error, camera_T_object, scale_matrix = optimize_for_9D(
            bbox_ext_flipped.T, camera_model, solve_for_transforms=True
        )
        # Revert back to original pose.
        camera_T_object = (
            camera_T_object
            @ transform.Transform.from_aa(axis=transform.X_AXIS, angle_deg=-45.0).matrix
        )
        # Add rotation solution to pose.
        camera_T_object = camera_T_object @ solve_for_rotation_from_cov_matrix(
            cov_matrices[ii]
        )
        # Assign correct depth factor
        abs_camera_T_object, abs_object_scale = epnp.find_absolute_scale(
            -1.0 * z_centroids[ii], camera_T_object, scale_matrix
        )
        poses.append(
            transform.Pose(
                camera_T_object=abs_camera_T_object, scale_matrix=abs_object_scale
            )
        )
        scores.append(heatmap_output[peak[0], peak[1]])

    if classes_outputs is not None:
        classes = extract_classes_from_peaks(np.copy(peaks), np.copy(classes_outputs))

    return poses, scores, peaks, classes


def compute_peaks(heatmap_output, kernel=7):
    peaks = pose_outputs.extract_peaks_from_centroid_sorted(
        np.copy(heatmap_output), min_confidence=0.5  # TODO Why was this 0.3?
    )
    image = pose_outputs.draw_peaks(np.copy(heatmap_output), np.copy(peaks))

    kernel = 71  # TODO Ok?
    min_confidence = 0.5
    # kernel = 7  # TODO Ok?
    # min_confidence = 0.3
    peaks_nms, _, _, _, _ = pose_outputs.extract_peaks_from_centroid_nms(
        np.copy(heatmap_output), kernel=kernel, min_confidence=min_confidence
    )
    image_nms = pose_outputs.draw_peaks(np.copy(heatmap_output), np.copy(peaks_nms))

    peaks = peaks_nms
    return peaks, image, image_nms


def compute_embeddings(peaks, latent_emb_output):
    latent_embs, indices = pose_outputs.extract_latent_emb_from_peaks(
        np.copy(peaks), np.copy(latent_emb_output)
    )
    return latent_embs, indices


def compute_embs_and_poses(
    heatmap_output,
    shape_latent_emb_output,
    arti_latent_emb_output,
    abs_pose_output,
):
    peaks, img, img_nms = compute_peaks(np.copy(heatmap_output))
    scores = []

    hm = np.copy(heatmap_output)
    for ii in range(peaks.shape[0]):
        scores.append(hm[int(peaks[ii, 0]), int(peaks[ii, 1])])

    latent_embeddings_shape, indices_shape = compute_embeddings(
        np.copy(peaks), np.copy(shape_latent_emb_output)
    )
    latent_embeddings_arti, indices_arti = compute_embeddings(
        np.copy(peaks), np.copy(arti_latent_emb_output)
    )
    abs_pose_outputs = pose_outputs.extract_abs_pose_from_peaks(
        np.copy(peaks), abs_pose_output
    )

    return (
        abs_pose_outputs,
        img,
        img_nms,
        latent_embeddings_shape,
        scores,
        indices_shape,
        latent_embeddings_arti,
        indices_arti,
    )
