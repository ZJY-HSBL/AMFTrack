from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import time
import cv2

from amftrack.config import load_config
from amftrack.pipeline import AMFTrackPipeline
from amftrack.metrics.mot import evaluate_motchallenge
from amftrack.utils.io import ensure_dir, write_mot_line


def sequence_fps(seq_dir: Path) -> float:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    return float(cfg["Sequence"].get("frameRate", 25))


def run_sequence(pipeline: AMFTrackPipeline, seq_dir: Path, result_path: Path):
    images = sorted((seq_dir / "img1").glob("*.jpg"))
    if not images:
        images = sorted((seq_dir / "img1").glob("*.png"))
    elapsed = 0.0
    with result_path.open("w", encoding="utf-8") as fp:
        for frame_id, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            t0 = time.perf_counter()
            tracks = pipeline.process_frame(frame, frame_id - 1)
            elapsed += time.perf_counter() - t0
            for track in tracks:
                write_mot_line(fp, frame_id, track.track_id, track.to_xyxy(), track.last_score)
    return len(images) / max(elapsed, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mot17.yaml")
    ap.add_argument("--mot-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", default="outputs/mot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    split_dir = Path(args.mot_root) / args.split
    out_dir = ensure_dir(args.output_dir)

    rows = []
    for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        print(f"Processing {seq_dir.name}")
        pipeline = AMFTrackPipeline(cfg)
        result_path = out_dir / f"{seq_dir.name}.txt"
        fps = run_sequence(pipeline, seq_dir, result_path)
        record = {"sequence": seq_dir.name, "fps": fps}
        gt = seq_dir / "gt" / "gt.txt"
        if gt.exists():
            summary = evaluate_motchallenge(
                gt,
                result_path,
                iou_threshold=float(cfg["evaluation"].get("mot_iou_threshold", 0.5)),
            )
            for col in summary.columns:
                record[col] = float(summary.iloc[0][col])
        rows.append(record)
        print(record)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
