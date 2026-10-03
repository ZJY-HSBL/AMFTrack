from __future__ import annotations

import math
import numpy as np


def _area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def ciou_loss(box_a: np.ndarray, box_b: np.ndarray, eps: float = 1e-9) -> float:
    """Complete-IoU loss: 1 - IoU + center_distance/c^2 + alpha*v."""
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)

    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    iou = inter / max(union, eps)

    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    rho2 = (acx - bcx) ** 2 + (acy - bcy) ** 2

    cx1, cy1 = min(a[0], b[0]), min(a[1], b[1])
    cx2, cy2 = max(a[2], b[2]), max(a[3], b[3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

    aw, ah = max(a[2] - a[0], eps), max(a[3] - a[1], eps)
    bw, bh = max(b[2] - b[0], eps), max(b[3] - b[1], eps)
    v = (4.0 / (math.pi ** 2)) * (math.atan(bw / bh) - math.atan(aw / ah)) ** 2
    alpha = v / max(1.0 - iou + v, eps)

    return float(1.0 - iou + rho2 / c2 + alpha * v)


def ciou_similarity(box_a: np.ndarray, box_b: np.ndarray) -> float:
    return 1.0 - ciou_loss(box_a, box_b)


def ciou_cost_matrix(track_boxes: list[np.ndarray], det_boxes: list[np.ndarray]) -> np.ndarray:
    if not track_boxes or not det_boxes:
        return np.empty((len(track_boxes), len(det_boxes)), dtype=np.float64)
    out = np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float64)
    for i, ta in enumerate(track_boxes):
        for j, db in enumerate(det_boxes):
            out[i, j] = ciou_loss(ta, db)
    return out
