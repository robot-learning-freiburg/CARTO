import copy

import numpy as np
import cv2
import IPython
from scipy.stats import multivariate_normal

_TURN_ON_PATCHES = False
_PERCENT_OFF_PATCH = 0.2
_CROP_SIZE = [768, 768]
_PATCH_VARIETY = 2

_OCC_SCORE = 0.5
_USE_IGNORE = False

_PEAK_CONCENTRATION = 0.8


def generate_crops(img_one, img_two, corr_pairs, appliances_on=True):
    cropped_corr_pairs = []
    for corr_pair in corr_pairs:
        # Create the positive cropped corr_pairs
        if corr_pair.is_appliance:
            if not appliances_on:
                continue
            is_valid, corr_cropped_pair = create_positive_cropped_appliance_pair(
                corr_pair, img_one, img_two
            )
            if is_valid:
                cropped_corr_pairs.append(corr_cropped_pair)
        else:
            if corr_pair.pixels_one.shape[0] == 0:
                continue
            if corr_pair.pixels_two.shape[0] == 0:
                continue
            if fits_in_patch(corr_pair, img_one, img_two):
                cropped_corr_pairs.append(
                    create_positive_cropped_object_pair(corr_pair, img_one, img_two)
                )

    return cropped_corr_pairs


def fits_in_patch(corr_pair, img_one, img_two):
    bbox_one = get_bbox_from_pixels(corr_pair.pixels_one)
    bbox_two = get_bbox_from_pixels(corr_pair.pixels_two)

    if not bounding_box_fits_in_image(bbox_one, img_one):
        return False
    if not bounding_box_fits_in_image(bbox_two, img_two):
        return False

    if (bbox_one["y_max"] - bbox_one["y_min"]) >= _CROP_SIZE[1]:
        return False
    if (bbox_one["x_max"] - bbox_one["x_min"]) >= _CROP_SIZE[0]:
        return False
    if (bbox_two["y_max"] - bbox_two["y_min"]) >= _CROP_SIZE[1]:
        return False
    if (bbox_two["x_max"] - bbox_two["x_min"]) >= _CROP_SIZE[0]:
        return False

    return True


def bounding_box_fits_in_image(bbox, img):
    if bbox["x_max"] >= img.shape[0]:
        return False
    if bbox["x_min"] < 0:
        return False
    if bbox["y_max"] >= img.shape[1]:
        return False
    if bbox["y_min"] < 0:
        return False
    return True


def convert_pixels_to_crop(pixels, bbox):
    assert pixels.shape[1] == 2
    pixels[:, 0] = pixels[:, 0] - bbox["x_min"]
    pixels[:, 1] = pixels[:, 1] - bbox["y_min"]
    return pixels


def crop_image_with_bbox(img, bbox):
    cropped_img = img[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]]
    return cropped_img


def get_bbox_from_pixels(pixels):
    assert pixels.shape[1] == 2
    bbox = {}
    bbox["x_min"] = np.min(pixels[:, 0])
    bbox["y_min"] = np.min(pixels[:, 1])
    bbox["x_max"] = np.max(pixels[:, 0])
    bbox["y_max"] = np.max(pixels[:, 1])
    return bbox


def get_bbox_from_object_mask(mask):
    coords = np.indices(mask.shape)
    coords = coords.reshape([2, -1]).T
    mask_f = mask.flatten()
    indices = coords[np.where(mask_f > 0)]

    bbox = {}
    bbox["x_min"] = np.min(indices[:, 0])
    bbox["x_max"] = np.max(indices[:, 0])
    bbox["y_min"] = np.min(indices[:, 1])
    bbox["y_max"] = np.max(indices[:, 1])
    return bbox


def get_box_mask_from_masks(masks):
    assert len(masks) > 0
    box_mask = np.zeros(masks[0].shape)
    for mask in masks:
        bbox = get_bbox_from_object_mask(mask)
        box_mask[bbox["x_min"] : bbox["x_max"], bbox["y_min"] : bbox["y_max"]] = 1.0
    return box_mask


def get_gaussian_from_mask(mask, intensify=True):
    coords = np.indices(mask.shape)
    coords = coords.reshape([2, -1]).T
    mask_f = mask.flatten()
    indices = coords[np.where(mask_f > 0)]
    mean_value = np.floor(np.average(indices, axis=0))
    cov = np.cov((indices - mean_value).T)
    cov = cov * _PEAK_CONCENTRATION
    multi_var = multivariate_normal(mean=mean_value, cov=cov)
    density = multi_var.pdf(coords)
    heat_map = np.zeros(mask.shape)
    heat_map[coords[:, 0], coords[:, 1]] = density
    return heat_map / np.max(heat_map)


def merge_heat_maps(heatmaps, height=480, width=640):
    if len(heatmaps) == 0:
        return np.zeros([height, width])
    return np.max(np.array(heatmaps), axis=0)


def draw_bbox_on_image(detections, color_img, colors, draw_seg_mask=False):
    color_img = copy.deepcopy(color_img)
    for detection, color in zip(detections, colors):
        pt1 = (detection.bbox["y_min"], detection.bbox["x_min"])
        pt2 = (detection.bbox["y_max"], detection.bbox["x_max"])
        color = (int(color[0]), int(color[1]), int(color[2]))
        color_img = cv2.rectangle(color_img, pt1, pt2, color)
        if draw_seg_mask:
            mask = detection.mask
            color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
            color_mask[mask > 0] = np.array(color)
            color_mask = cv2.resize(
                color_mask, (color_img.shape[1], color_img.shape[0])
            )
            color_img = cv2.addWeighted(
                color_img.astype(np.uint8), 1.0, color_mask.astype(np.uint8), 0.2, 0
            )

    return color_img


def does_bbox_fit_in_crops_size(bbox):
    if bbox["y_max"] - bbox["y_min"] < _APPLIANCE_CROP_SIZE[1]:
        if bbox["x_max"] - bbox["x_min"] < _APPLIANCE_CROP_SIZE[0]:
            return True
    return False


def is_point_in_detection_box(point, bbox):
    if bbox[0][0] > point[0]:
        return False
    if bbox[0][1] > point[1]:
        return False
    if bbox[1][0] < point[0]:
        return False
    if bbox[1][1] < point[1]:
        return False
    return True


def is_point_in_box(point, bbox):
    if bbox["x_min"] > point[0]:
        return False
    if bbox["y_min"] > point[1]:
        return False
    if bbox["x_max"] < point[0]:
        return False
    if bbox["y_max"] < point[1]:
        return False
    return True


def compute_area(bbox):
    return (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])


def compute_percent_off_image(patch_bbox, detection, heat_map):
    x_min = int(max(0, detection.bbox[0][0] - patch_bbox["x_min"]))
    y_min = int(max(0, detection.bbox[0][1] - patch_bbox["y_min"]))
    x_max = int(min(heat_map.shape[1], detection.bbox[1][0] - patch_bbox["x_min"]))
    y_max = int(min(heat_map.shape[0], detection.bbox[1][1] - patch_bbox["y_min"]))

    on_img_area = (x_max - x_min) * (y_max - y_min)
    off_img_area = (detection.bbox[1][0] - detection.bbox[0][0]) * (
        detection.bbox[1][1] - detection.bbox[0][1]
    )
    return 1.0 - on_img_area / off_img_area


def create_ignore_mask(patch_bbox, detections, heat_map):
    def is_detection_outside_img(patch_bbox, detection):
        if detection.bbox[0][0] < patch_bbox["x_min"]:
            return True
        if detection.bbox[0][1] < patch_bbox["y_min"]:
            return True
        if detection.bbox[1][0] > patch_bbox["x_max"]:
            return True
        if detection.bbox[1][1] > patch_bbox["y_max"]:
            return True
        return False

    def is_detection_partially_in_img(patch_bbox, detection):
        if (
            detection.bbox[0][0] > patch_bbox["x_min"]
            and detection.bbox[0][0] < patch_bbox["x_max"]
        ):
            if (
                detection.bbox[0][1] > patch_bbox["y_min"]
                and detection.bbox[0][1] < patch_bbox["y_max"]
            ):
                return True
        if (
            detection.bbox[1][0] < patch_bbox["x_max"]
            and detection.bbox[1][0] > patch_bbox["x_min"]
        ):
            if (
                detection.bbox[1][1] < patch_bbox["y_max"]
                and detection.bbox[1][1] > patch_bbox["y_min"]
            ):
                return True
        return False

    ignore_mask = np.zeros(heat_map.shape)
    num_objs = 0
    for detection in detections:
        if is_detection_outside_img(
            patch_bbox, detection
        ) and is_detection_partially_in_img(patch_bbox, detection):
            if (
                compute_percent_off_image(patch_bbox, detection, heat_map)
                < _PERCENT_OFF_PATCH
            ):
                continue
            x_min = int(max(0, detection.bbox[0][0] - patch_bbox["x_min"]))
            y_min = int(max(0, detection.bbox[0][1] - patch_bbox["y_min"]))
            x_max = int(
                min(heat_map.shape[1], detection.bbox[1][0] - patch_bbox["x_min"])
            )
            y_max = int(
                min(heat_map.shape[0], detection.bbox[1][1] - patch_bbox["y_min"])
            )
            ignore_mask[y_min:y_max, x_min:x_max] = 1.0
            heat_map[y_min:y_max, x_min:x_max] = 0.0
            num_objs += 1
    return ignore_mask, heat_map, num_objs


def sample_patch_containing_img_center(img, crop_size=None):
    if crop_size is None:
        crop_size = _CROP_SIZE
    x, y = img.shape[1] / 2.0, img.shape[0] / 2.0
    # sample the upper left corner of the box
    sample_range = np.array(_CROP_SIZE) / _PATCH_VARIETY
    min_x_value = np.max([0, (x - crop_size[0] / 2.0) - sample_range[0]])
    min_y_value = np.max([0, (y - crop_size[1] / 2.0) - sample_range[1]])
    max_x_value = np.min(
        [(x - crop_size[0] / 2.0) + sample_range[0], img.shape[1] - crop_size[0]]
    )
    max_y_value = np.min(
        [(y - crop_size[1] / 2.0) + sample_range[1], img.shape[0] - crop_size[1]]
    )
    patch_bbox = {}
    patch_bbox["x_min"] = int(np.random.uniform(min_x_value, max_x_value))
    patch_bbox["y_min"] = int(np.random.uniform(min_y_value, max_y_value))
    patch_bbox["x_max"] = int(patch_bbox["x_min"] + crop_size[0])
    patch_bbox["y_max"] = int(patch_bbox["y_min"] + crop_size[1])
    return patch_bbox


def draw_bboxes_on_image(detections, c_img):
    for detection in detections:
        pt1 = (int(detection.bbox[0][0]), int(detection.bbox[0][1]))
        pt2 = (int(detection.bbox[1][0]), int(detection.bbox[1][1]))
        c_img = cv2.rectangle(c_img, pt1, pt2, (0, 255, 0))
    return c_img


def crop_grasp_data(dp, scale_factor=8):
    if not _TURN_ON_PATCHES:
        dp.ignore_mask = np.zeros(dp.ignore_mask.shape)
        return dp, 0
    patch_bbox = sample_patch_containing_img_center(dp.left_color)
    dp.left_color = crop_image_with_bbox(dp.left_color, patch_bbox)
    dp.right_color = crop_image_with_bbox(dp.right_color, patch_bbox)
    dp.seg_mask = crop_image_with_bbox(dp.seg_mask, patch_bbox)
    dp.left_depth = crop_image_with_bbox(dp.left_depth, patch_bbox)
    heat_map = crop_image_with_bbox(dp.heat_map, patch_bbox)
    ignore_mask, heat_map, num_objs = create_ignore_mask(
        patch_bbox, dp.detections, heat_map
    )
    dp.ignore_mask = ignore_mask
    dp.heat_map = heat_map
    # Downsample boxes.
    patch_bbox["x_min"] = int(patch_bbox["x_min"] / scale_factor)
    patch_bbox["y_min"] = int(patch_bbox["y_min"] / scale_factor)
    patch_bbox["x_max"] = int(patch_bbox["x_max"] / scale_factor)
    patch_bbox["y_max"] = int(patch_bbox["y_max"] / scale_factor)
    dp.vertex_target = crop_image_with_bbox(dp.vertex_target, patch_bbox)
    dp.z_centroid = crop_image_with_bbox(dp.z_centroid, patch_bbox)
    dp.cov_matrices = crop_image_with_bbox(dp.cov_matrices, patch_bbox)
    return dp, num_objs


def create_ignore_mask(dp, min_occ_score, use_ignore_mask):
    occ_scores = dp.occ_scores
    instance_mask = dp.instance_mask
    heat_maps = []
    ignore_masks = []
    for ii, occ_score in enumerate(occ_scores):
        seg_mask = instance_mask == ii + 1
        if occ_score > min_occ_score:
            heat_maps.append(get_gaussian_from_mask(seg_mask))
            continue
        if use_ignore_mask:
            ignore_masks.append(seg_mask)

    if len(heat_maps):
        dp.heat_map = np.max(np.array(heat_maps), axis=0)
    else:
        dp.heat_map = np.zeros(dp.heat_map.shape)
    if len(ignore_masks):
        dp.ignore_mask = np.max(np.array(ignore_masks), axis=0)
    else:
        dp.ignore_mask = np.zeros(dp.heat_map.shape)
    return dp


def sample_patch_containing_object(object_bbox, img, crop_size=None):
    if crop_size is None:
        crop_size = _CROP_SIZE
    # sample the centroid of the box
    min_x_value = np.max([0, object_bbox["x_max"] - crop_size[0]])
    min_y_value = np.max([0, object_bbox["y_max"] - crop_size[1]])
    max_x_value = np.min([object_bbox["x_min"], img.shape[1] - crop_size[0]])
    max_y_value = np.min([object_bbox["y_min"], img.shape[0] - crop_size[1]])
    patch_bbox = {}
    patch_bbox["x_min"] = int(np.random.uniform(min_x_value, max_x_value))
    patch_bbox["y_min"] = int(np.random.uniform(min_y_value, max_y_value))
    patch_bbox["x_max"] = int(patch_bbox["x_min"] + crop_size[0])
    patch_bbox["y_max"] = int(patch_bbox["y_min"] + crop_size[1])
    return patch_bbox


def sample_patch_on_image(img):
    assert _CROP_SIZE[0] < img.shape[0]
    assert _CROP_SIZE[1] < img.shape[1]

    patch_bbox = {}
    patch_bbox["x_min"] = int(np.random.uniform(0, img.shape[0] - _CROP_SIZE[0]))
    patch_bbox["y_min"] = int(np.random.uniform(0, img.shape[1] - _CROP_SIZE[1]))
    patch_bbox["x_max"] = int(patch_bbox["x_min"] + _CROP_SIZE[0])
    patch_bbox["y_max"] = int(patch_bbox["y_min"] + _CROP_SIZE[1])
    return patch_bbox
