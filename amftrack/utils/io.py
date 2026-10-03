from __future__ import annotations

from pathlib import Path


def write_mot_line(fp, frame_id: int, track_id: int, xyxy, score: float = 1.0):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = x2 - x1, y2 - y1
    fp.write(
        f"{frame_id},{track_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},"
        f"{score:.5f},-1,-1,-1\n"
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
