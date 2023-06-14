import numpy as np
import cv2
import IPython
import torch
import torch.nn as nn

from CARTO.simnet.lib.net.post_processing import pose_outputs
from CARTO.simnet.lib.net.post_processing import nms
from CARTO.simnet.lib.net.post_processing.eval2d import Detection
from CARTO.simnet.lib.net import losses

_mask_l1_loss = losses.MaskedL1Loss(downscale_factor=1)
_mse_loss = losses.MaskedMSELoss()


class BoxOutput:
    def __init__(self, heatmap, vertex_field, hparams, ignore_mask=None):
        self.heatmap = heatmap
        self.vertex_field = vertex_field
        self.ignore_mask = ignore_mask
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
        self.is_numpy = True

    def convert_to_torch_from_numpy(self):
        self.vertex_field = self.vertex_field.transpose((2, 0, 1))
        self.vertex_field = 100.0 * self.vertex_field
        self.vertex_field = torch.from_numpy(
            np.ascontiguousarray(self.vertex_field)
        ).float()
        self.ignore_mask = torch.from_numpy(
            np.ascontiguousarray(self.ignore_mask)
        ).bool()
        self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
        self.is_numpy = False

    def get_detections(self, min_confidence=0.02, overlap_thresh=0.75):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        detections = create_detections_from_outputs(
            np.copy(self.heatmap[0]),
            np.copy(self.vertex_field[0]),
            min_confidence=min_confidence,
        )
        detections = nms.run(detections, overlap_thresh=overlap_thresh)
        return detections

    def get_visualization_img(self, left_img, is_pretty=False):
        if not self.is_numpy:
            self.convert_to_numpy_from_torch()
        if is_pretty:
            return draw_pretty_detection_from_outputs(
                self.heatmap[0], self.vertex_field[0], left_img
            )
        return draw_detection_from_outputs(
            self.heatmap[0], self.vertex_field[0], left_img
        )

    def compute_loss(self, pose_targets, log, prefix):
        if self.is_numpy:
            raise ValueError("Output is not in torch mode")
        vertex_target = torch.stack(
            [pose_target.vertex_field for pose_target in pose_targets]
        )
        heatmap_target = torch.stack(
            [pose_target.heatmap for pose_target in pose_targets]
        )
        ignore_target = torch.stack(
            [pose_target.ignore_mask for pose_target in pose_targets]
        )

        # Move to GPU
        heatmap_target = heatmap_target.to(torch.device("cuda:0"))
        vertex_target = vertex_target.to(torch.device("cuda:0"))
        ignore_target = ignore_target.to(torch.device("cuda:0"))

        vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
        log[f"{prefix}/vertex_loss"] = vertex_loss.item()
        heatmap_loss = _mse_loss(self.heatmap, heatmap_target, ignore_target)
        log[f"{prefix}/heatmap"] = heatmap_loss.item()
        return (
            self.hparams.loss_vertex_mult * vertex_loss
            + self.hparams.loss_heatmap_mult * heatmap_loss
        )


def draw_detection_from_outputs(
    heatmap_output, vertex_output, c_img, min_confidence=0.4
):
    c_img_gray = np.zeros(c_img.shape)
    for i in range(3):
        c_img_gray[:, :, i] = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    peaks = pose_outputs.extract_peaks_from_centroid(
        heatmap_output, min_confidence=min_confidence
    )
    peak_img = pose_outputs.draw_peaks(heatmap_output, peaks)
    bboxes_ext = extract_vertices_from_peaks(np.copy(peaks), vertex_output, c_img_gray)
    img = draw_2d_boxes(c_img_gray, bboxes_ext)
    img = cv2.addWeighted(img.astype(np.uint8), 0.9, peak_img.astype(np.uint8), 0.4, 0)
    return img


def draw_pretty_detection_from_outputs(
    heatmap_output, vertex_output, c_img, min_confidence=0.4
):
    # c_img_gray = np.zeros(c_img.shape)
    # for i in range(3):
    #  c_img_gray[:, :, i] = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)

    c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)

    peaks = pose_outputs.extract_peaks_from_centroid(
        heatmap_output, min_confidence=min_confidence
    )
    bboxes_ext = extract_vertices_from_peaks(np.copy(peaks), vertex_output, c_img)
    img = draw_2d_boxes(c_img, bboxes_ext)
    return img


def create_detections_from_outputs(heatmap_output, vertex_output, min_confidence=0.1):
    peaks = pose_outputs.extract_peaks_from_centroid(
        heatmap_output, min_confidence=min_confidence
    )
    bboxes_ext = extract_vertices_from_peaks(
        np.copy(peaks), vertex_output, heatmap_output
    )
    detections = []
    for peak, bbox_ext in zip(peaks, bboxes_ext):
        score = heatmap_output[peak[0], peak[1]]
        bbox = [
            np.array([bbox_ext[0][0], bbox_ext[0][1]]),
            np.array([bbox_ext[1][0], bbox_ext[1][1]]),
        ]
        detection = Detection(class_label="Car", bbox=bbox, score=score)
        detections.append(detection)
    return detections


def extract_vertices_from_peaks(peaks, vertex_fields, c_img, scale_factor=1):
    assert peaks.shape[1] == 2
    assert vertex_fields.shape[2] == 4
    height = vertex_fields.shape[0] * scale_factor
    width = vertex_fields.shape[1] * scale_factor
    vertex_fields[:, :, ::2] = (1.0 - vertex_fields[:, :, ::2]) * (2 * height) - height
    vertex_fields[:, :, 1::2] = (1.0 - vertex_fields[:, :, 1::2]) * (2 * width) - width
    bboxes = []
    for ii in range(peaks.shape[0]):
        bbox = get_bbox_from_vertex(
            vertex_fields, peaks[ii, :], scale_factor=scale_factor
        )
        bboxes.append(bbox)
    return bboxes


def get_bbox_from_vertex(vertex_fields, index, scale_factor=64):
    assert index.shape[0] == 2
    index[0] = int(index[0] / scale_factor)
    index[1] = int(index[1] / scale_factor)
    bbox = vertex_fields[index[0], index[1], :]
    bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
    bbox = scale_factor * (index) - bbox
    return bbox


def draw_2d_boxes_with_colors(img, bboxes, colors):
    for bounding_box, color in zip(bboxes, colors):
        bbox = bounding_box.bounding_box
        pt1 = (int(bbox[0][1]), int(bbox[0][0]))
        pt2 = (int(bbox[1][1]), int(bbox[1][0]))
        img = cv2.rectangle(img, pt1, pt2, color, 2)
    return img


def draw_2d_boxes(c_img, bboxes):
    c_img = cv2.cvtColor(np.array(c_img), cv2.COLOR_BGR2RGB)
    for bounding_box in bboxes:
        bbox = bounding_box.bbox
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[1][0]), int(bbox[1][1]))
        c_img = cv2.rectangle(c_img, pt1, pt2, (255, 0, 0), 2)
    return c_img


def draw_2d_boxes_with_labels(c_img, bboxes):
    c_img = cv2.cvtColor(np.array(c_img), cv2.COLOR_BGR2RGB)
    for bounding_box in bboxes:
        bbox = bounding_box.bbox
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[1][0]), int(bbox[1][1]))
        c_img = cv2.rectangle(c_img, pt1, pt2, (255, 0, 0), 2)
        c_img = draw_class_label(
            c_img, pt1 + ((bbox[1] - bbox[0]) / 2.0), bounding_box.class_label
        )

    return c_img


def draw_class_label(c_img, pixel_center, class_label):
    color = (0, 255, 0)  # green
    if class_label == "null":
        return c_img
    class_label = " ".join(class_label.split("_"))
    # TODO: add more metadata to class labels so we can have human friendly names and styles
    pixel_x = int(pixel_center[0])
    pixel_y = int(pixel_center[1])
    size = 0.75
    thickness = 2
    color = (0, 255, 0)
    c_img = cv2.putText(
        c_img.copy(),
        class_label,
        (pixel_x, pixel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return c_img
