from __future__ import annotations

from pathlib import Path
import numpy as np


def _load_mot(path: str | Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            rows.append([
                int(float(parts[0])),
                int(float(parts[1])),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            ])
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 6), dtype=np.float64)


def evaluate_motchallenge(gt_path: str | Path, result_path: str | Path, iou_threshold: float = 0.5):
    import motmetrics as mm

    gt = _load_mot(gt_path)
    pred = _load_mot(result_path)
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt[:, 0].astype(int).tolist()) | set(pred[:, 0].astype(int).tolist()))

    for frame in frames:
        g = gt[gt[:, 0] == frame]
        p = pred[pred[:, 0] == frame]
        gt_ids = g[:, 1].astype(int).tolist()
        pr_ids = p[:, 1].astype(int).tolist()
        gt_boxes = g[:, 2:6]
        pr_boxes = p[:, 2:6]
        distances = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=1.0 - iou_threshold)
        acc.update(gt_ids, pr_ids, distances)

    mh = mm.metrics.create()
    metrics = [
        "num_frames", "mota", "idf1", "mostly_tracked",
        "num_switches", "num_false_positives", "num_misses"
    ]
    return mh.compute(acc, metrics=metrics, name="AMFTrack")
