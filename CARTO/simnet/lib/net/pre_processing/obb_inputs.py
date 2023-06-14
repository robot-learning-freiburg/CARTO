import numpy as np
from scipy.stats import multivariate_normal

from CARTO.simnet.lib.net.post_processing import epnp
from CARTO.simnet.lib.net.pre_processing import pose_inputs
from CARTO.simnet.lib import datapoint

_HEATMAP_THRESHOLD = 0.3
_DOWNSCALE_VALUE = 8
_PEAK_CONCENTRATION = 0.8

# def compute_network_targets(obbs, masks, height, width, camera_model, class_index_list=None):
#   assert len(obbs) == len(masks)
#   if len(obbs) == 0:
#     height_d = int(height / _DOWNSCALE_VALUE)
#     width_d = int(width / _DOWNSCALE_VALUE)
#     return datapoint.OBB(
#         heat_map=np.zeros([height, width]),
#         vertex_target=np.zeros([height_d, width_d, 16]),
#         cov_matrices=np.zeros([height_d, width_d, 6]),
#         z_centroid=np.zeros([height_d, width_d]),
#         classes=np.zeros([height_d, width_d])
#     )
#   heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
#   vertex_target = pose_inputs.compute_vertex_field(obbs, heatmaps, camera_model)
#   z_centroid = pose_inputs.compute_z_centroid_field(obbs, heatmaps)
#   cov_matrix = compute_rotation_field(obbs, heatmaps)
#   class_target = None
#   if class_index_list is not None:
#     class_target = compute_class_field(obbs, class_index_list, heatmaps)
#   return datapoint.OBB(
#       heat_map=np.max(heatmaps, axis=0),
#       vertex_target=vertex_target,
#       cov_matrices=cov_matrix,
#       z_centroid=z_centroid,
#       classes=class_target
#   )


## Extended Targers to include the pose + latent emb
def compute_network_targets(
    obbs,
    masks,
    shape_code,
    arti_code,
    poses,
    height,
    width,
    camera_model,
    class_index_list=None,
    shape_emb_size=32,
    arti_emb_size=16,
):
    assert len(obbs) == len(masks)
    if len(obbs) == 0:
        height_d = int(height / _DOWNSCALE_VALUE)
        width_d = int(width / _DOWNSCALE_VALUE)
        return datapoint.OBB(
            heat_map=np.zeros([height, width]),
            vertex_target=np.zeros([height_d, width_d, 16]),
            cov_matrices=np.zeros([height_d, width_d, 6]),
            z_centroid=np.zeros([height_d, width_d]),
            shape_emb=np.zeros([height_d, width_d, shape_emb_size]),
            arti_emb=np.zeros([height_d, width_d, arti_emb_size]),
            abs_pose=np.zeros([height_d, width_d, 13]),
        )
    heatmaps = pose_inputs.compute_heatmaps_from_masks(masks)
    vertex_target = pose_inputs.compute_vertex_field(obbs, heatmaps, camera_model)
    z_centroid = pose_inputs.compute_z_centroid_field(obbs, heatmaps)
    cov_matrix = compute_rotation_field(obbs, heatmaps)
    shape_emb_target = compute_latent_emb(
        obbs, shape_code, heatmaps, embedding_size=shape_emb_size
    )
    arti_emb_target = compute_latent_emb(
        obbs, arti_code, heatmaps, embedding_size=arti_emb_size
    )
    abs_pose_target = compute_abspose_field(poses, heatmaps, camera_model)
    return datapoint.OBB(
        heat_map=np.max(heatmaps, axis=0),
        vertex_target=vertex_target,
        cov_matrices=cov_matrix,
        z_centroid=z_centroid,
        shape_emb=shape_emb_target,
        arti_emb=arti_emb_target,
        abs_pose=abs_pose_target,
    )


####
# How does it work?
# The first dimension represents a layer for each obbs.
# Data will be set in the according channels (last dimension)
# As the remainders entries for this layer stay zero, we can sum over the
# first dimension to get rid of it.
# TODO Nick maybe refactor and use direct indexing to save time
#   class_target[mask] = class_values
#   etc..
####
def compute_class_field(obbs, class_index_list, heat_maps, threshold=0.3):
    class_target = np.zeros([len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1]])
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for obb, heat_map, ii in zip(obbs, heat_maps, range(len(heat_maps))):
        mask = heatmap_indices == ii
        class_values = class_index_list.index(obb.category_name)
        class_target[ii, mask] = class_values
    return np.sum(class_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]


def compute_rotation_field(obbs, heat_maps, threshold=0.3):
    cov_target = np.zeros([len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1], 6])
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for obb, heat_map, ii in zip(obbs, heat_maps, range(len(heat_maps))):
        mask = heatmap_indices == ii
        cov_matrix = obb.cov_matrix
        cov_mat_values = np.array(
            [
                cov_matrix[0, 0],
                cov_matrix[1, 1],
                cov_matrix[2, 2],
                cov_matrix[0, 1],
                cov_matrix[0, 2],
                cov_matrix[1, 2],
            ]
        )
        cov_target[ii, mask] = cov_mat_values
    return np.sum(cov_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]


def compute_latent_emb(obbs, embeddings, heat_maps, embedding_size=1):
    """
    Fills each pixel with the closest embedding code according to the heatmap
    """
    latent_emb_target = np.zeros(
        [len(obbs), heat_maps[0].shape[0], heat_maps[0].shape[1], embedding_size]
    )
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for emb, ii in zip(embeddings, range(len(heat_maps))):
        mask = heatmap_indices == ii
        latent_emb_target[ii, mask] = emb
    return np.sum(latent_emb_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]


def compute_abspose_field(poses, heat_maps, camera_model, threshold=0.3):
    abs_pose_target = np.zeros(
        [len(poses), heat_maps[0].shape[0], heat_maps[0].shape[1], 13]
    )
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)
    for pose, ii in zip(poses, range(len(heat_maps))):
        mask = heatmap_indices == ii
        actual_abs_pose = camera_model.RT_matrix @ pose.camera_T_object
        rotation_matrix = actual_abs_pose[:3, :3]
        translation_vector = actual_abs_pose[:3, 3]
        scale = pose.scale_matrix[0, 0]
        abs_pose_values = np.array(
            [
                rotation_matrix[0, 0],
                rotation_matrix[0, 1],
                rotation_matrix[0, 2],
                rotation_matrix[1, 0],
                rotation_matrix[1, 1],
                rotation_matrix[1, 2],
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
                translation_vector[0],
                translation_vector[1],
                translation_vector[2],
                scale,
            ]
        )
        abs_pose_target[ii, mask] = abs_pose_values
    return np.sum(abs_pose_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE]
