import cv2
import numpy as np
from scipy.stats import multivariate_normal

from CARTO.simnet.lib import datapoint

_KEYPOINT_VAR = 20


def compute_network_targets(keypoints, height, width):
    coords = np.indices((height, width))
    coords = coords.reshape([2, -1]).T
    all_targets = []
    # for each type of keypoint
    for keypoint_group in keypoints:
        # for each keypoint in each keypoint group
        individual_heat_maps = []
        for keypoint in keypoint_group:
            # for each instance of the keypoint in the image
            for px in keypoint.pixels:
                # place a Gaussian target distribution at the pixel location
                cur_heat_map = np.zeros([height, width])
                cov = np.eye(2) * _KEYPOINT_VAR
                multi_var = multivariate_normal(mean=px[::-1], cov=cov)
                density = multi_var.pdf(coords)
                cur_heat_map[coords[:, 0], coords[:, 1]] = density
                individual_heat_maps.append(cur_heat_map)
        # take a max over all pixels for this keypoint group
        if len(individual_heat_maps):
            target = np.stack(individual_heat_maps).max(0)
            target /= target.max()
        else:
            target = np.zeros([height, width])
        all_targets.append(datapoint.Keypoint(heat_map=target))
    return all_targets


def vis_network_targets(keypoints, height, width, left_img):
    target_images = []
    all_targets = compute_network_targets(keypoints, height, width)
    for target in all_targets:
        heat_map = target.heat_map
        img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        img = cv2.addWeighted(heat_map, 0.999, img.astype(float), 0.00005, 0)
        img /= img.max() / 255
        target_images.append(img.astype(np.uint8))
    return target_images
