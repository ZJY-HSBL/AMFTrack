from __future__ import annotations
import numpy as np


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    x, y, a, h = np.asarray(xyah, dtype=np.float64)[:4]
    h = max(float(h), 1e-6)
    w = max(float(a) * h, 1e-6)
    return np.asarray([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float64)


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    x1 = np.clip(x1, 0, max(width - 1, 0))
    y1 = np.clip(y1, 0, max(height - 1, 0))
    x2 = np.clip(x2, x1 + 1, width)
    y2 = np.clip(y2, y1 + 1, height)
    return np.asarray([x1, y1, x2, y2], dtype=np.float64)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)
