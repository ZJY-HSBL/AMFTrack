from __future__ import annotations
import argparse

from amftrack.config import load_config
from amftrack.pipeline import AMFTrackPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mot17.yaml")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="outputs/tracked.mp4")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipeline = AMFTrackPipeline(cfg)
    stats = pipeline.process_video(args.source, args.output, args.display)
    print(stats)


if __name__ == "__main__":
    main()
