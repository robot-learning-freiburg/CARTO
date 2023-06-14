import copy
import dataclasses

import numpy as np
import cv2
import IPython
import pyrender

from CARTO.simnet.lib import plotter

# from CARTO.simnet.lib.net.dataset import _HEIGHT,_WIDTH

_HEIGHT = 320
_WIDTH = 1120
EVAL_IOUS = [0.25]

################################
# Precision/recall stuff


@dataclasses.dataclass
class Detection:
    class_label: str
    bbox: np.ndarray  # abs (xy,xy)
    score: float = 1.0
    truncation: float = 0.0
    ignore: bool = False
    occlusion_level: int = 0


def is_detection_moderate(detection):
    if detection.class_label != "Car":
        return False
    if detection.occlusion_level > 1:
        return False
    if detection.truncation > 0.3:
        return False
    if detection.bbox[1][0] - detection.bbox[0][0] < 25:
        return False
    return True


def truncate_bbox(bbox):
    is_truncated = False
    if bbox[0][0] < 0:
        bbox[0][0] = 0
    if bbox[0][1] < 0:
        bbox[0][1] = 0
    if bbox[1][0] > _HEIGHT:
        bbox[1][0] = _HEIGHT
        is_truncated = True
    if bbox[1][1] > _WIDTH:
        bbox[1][1] = _WIDTH
        is_truncated = True
    return bbox, is_truncated


class Eval2d:
    def __init__(self, iou):
        self.n = 0
        self.iou = iou
        self.eval_metrics = EvalMetrics()

    def process_sample(self, detections, gt_detections):
        pred_bboxes = []
        true_bboxes = []
        pred_scores = []
        is_moderate = []

        for detection in detections:
            pred_bboxes.append(
                np.array(
                    [
                        detection.bbox[0][0],
                        detection.bbox[0][1],
                        detection.bbox[1][0],
                        detection.bbox[1][1],
                    ]
                )
            )
            pred_scores.append(detection.score)

        for detection in gt_detections:
            detection.bbox, is_outside = truncate_bbox(detection.bbox)
            if is_outside:
                detection.truncation = 1.0
            true_bboxes.append(
                np.array(
                    [
                        detection.bbox[0][0],
                        detection.bbox[0][1],
                        detection.bbox[1][0],
                        detection.bbox[1][1],
                    ]
                )
            )
            is_moderate.append(is_detection_moderate(detection))

        true_matches, pred_matches, _, _ = compute_2d_matches(
            "single_class",
            np.array(pred_bboxes),
            np.array(true_bboxes),
            np.array(pred_scores),
            [self.iou],
            debug=False,
        )
        pruned_true_matches = []
        pruned_pred_matches = []
        pruned_pred_scores = []
        # Prune any predictions that intersect with an ignore class.
        # Prune ground truth to only contain moderate samples.
        for ii in range(true_matches.shape[1]):
            if not is_moderate[ii]:
                continue
            pruned_true_matches.append(true_matches[:, ii])
        successes = []
        for ii in range(pred_matches.shape[1]):
            gt_match = pred_matches[0, ii]
            # Check if there was a match at all
            if gt_match != -1:
                successes.append(True)
                # Check if it fired on a hard example.
                if not is_moderate[int(gt_match)]:
                    continue
            else:
                successes.append(False)
            pruned_pred_matches.append(pred_matches[:, ii])
            pruned_pred_scores.append(pred_scores[ii])
        if len(pruned_true_matches) == 0:
            true_matches = np.zeros([1, 0])
        else:
            true_matches = np.array(pruned_true_matches).T
        if len(pruned_pred_matches) == 0:
            pred_matches = np.zeros([1, 0])
        else:
            pred_matches = np.array(pruned_pred_matches).T
        pred_scores = np.array(pruned_pred_scores).T

        self.eval_metrics.process_sample(
            true_matches=true_matches,
            pred_matches=pred_matches,
            pred_scores=pred_scores,
        )
        return successes

    def process_dataset(self, final=True, results_path=None):
        self.n += 1
        if not final:
            if self.n % 100 != 0:
                return

        if final:
            print("\n" + "=" * 100)

        def _print(name, x, iou):
            if final:
                prefix = "[step {self.n: 6d}] (FINAL)"
            else:
                prefix = "(incomplete)"
            if not final and name != "MEAN":
                return
            out = f"{prefix} AP@{iou:.02f}[{name}] = {x}"

        for idx, iou in enumerate(EVAL_IOUS):
            ap = {}
            ap["single_class"] = self.eval_metrics.process_dataset()[idx]

            for k in sorted(ap.keys()):
                _print(k, ap[k], iou)

            if final:
                print("+" * 50)
            _print("MEAN", np.mean(np.array(list(ap.values()))), iou)
            if final:
                print("+" * 50)
            # Plot mAP per scene
            # Draw the bar plot.
            # Plot AP curve for known depth per scene
            precisions, recalls = self.eval_metrics.get_pr_curve()
            if results_path is not None:
                plotter.draw_precision_recall_curve(
                    precisions, recalls, results_path, name="pr_curve"
                )

        if final:
            print("=" * 100 + "\n")
        return ap["single_class"]


class EvalMetrics:
    def __init__(self):
        self.true_matches = []
        self.pred_matches = []
        self.pred_scores = []

    def process_sample(self, true_matches, pred_matches, pred_scores):
        self.true_matches.append(true_matches)
        self.pred_matches.append(pred_matches)
        self.pred_scores.append(pred_scores)

    def process_dataset(self):
        true_matches = np.concatenate(self.true_matches, axis=1)
        pred_matches = np.concatenate(self.pred_matches, axis=1)
        pred_scores = np.concatenate(self.pred_scores, axis=0)
        assert true_matches.shape[0] == pred_matches.shape[0]
        num_ious = true_matches.shape[0]
        ap_per_iou = []
        for i in range(num_ious):
            ap_per_iou.append(
                compute_ap_from_matches_scores(
                    pred_matches[i, :], pred_scores, true_matches[i, :]
                )
            )
        return ap_per_iou

    def get_pr_curve(self):
        true_matches = np.concatenate(self.true_matches, axis=1)
        pred_matches = np.concatenate(self.pred_matches, axis=1)
        pred_scores = np.concatenate(self.pred_scores, axis=0)
        assert true_matches.shape[0] == pred_matches.shape[0]
        num_ious = true_matches.shape[0]
        ap_per_iou = []
        return compute_pr_curve_from_matches_scores(
            pred_matches[0, :], pred_scores, true_matches[0, :]
        )


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x


def compute_2d_matches(
    class_name,
    pred_boxes,
    true_boxes,
    pred_scores,
    iou_2d_thresholds,
    score_threshold=0,
    debug=False,
):
    """Finds matches between prediction and ground truth instances.
    Returns:
      gt_matches: 2-D array. For each GT box it has the index of the matched
            predicted box.
      pred_matches: 2-D array. For each predicted box, it has the index of
            the matched ground truth box.
      overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # assert pred_boxes.shape[1] == 4
    # assert true_boxes.shape[1] == 4

    num_pred = pred_boxes.shape[0]
    num_gt = true_boxes.shape[0]
    indices = np.zeros(0)

    if num_pred:
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_scores = pred_scores[indices].copy()
        pred_boxes = pred_boxes[indices, :].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_2d_iou(pred_boxes[i], true_boxes[j])

    # Loop through predictions and find matching ground truth boxes
    num_iou_2d_thres = len(iou_2d_thresholds)
    pred_matches = -1 * np.ones([num_iou_2d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_2d_thres, num_gt])

    for s, iou_thres in enumerate(iou_2d_thresholds):
        for i in range(num_pred):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[: low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                # print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                # print('iou: ', iou)
                if iou < iou_thres:
                    break

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    if debug:
        IPython.embed()

    return gt_matches, pred_matches, overlaps, indices


def compute_2d_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match, debug=False):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert len(pred_match.shape) == 1
    assert len(gt_match.shape) == 1
    assert pred_match.shape[0] == pred_scores.shape[0]

    if gt_match.shape[0] == 0:
        return 0.0

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    # precisions2 = precisions
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    assert 0 <= ap <= 1.0

    if debug:
        IPython.embed()

    return ap


def compute_pr_curve_from_matches_scores(
    pred_match, pred_scores, gt_match, debug=False
):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert len(pred_match.shape) == 1
    assert len(gt_match.shape) == 1
    assert pred_match.shape[0] == pred_scores.shape[0]

    if gt_match.shape[0] == 0:
        return 0.0

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    return precisions, recalls
