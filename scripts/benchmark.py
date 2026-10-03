from __future__ import annotations
import argparse
import json
from pathlib import Path

TARGETS = {
    "MOT16": {"MOTA": 69.7, "IDF1": 71.3, "MT": 41.5, "FPS": 55.8},
    "MOT17": {"MOTA": 77.9, "IDF1": 70.6, "MT": 45.7, "FPS": 48.7},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(TARGETS), default="MOT17")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    payload = {"dataset": args.dataset, "reference_target": TARGETS[args.dataset]}
    print(json.dumps(payload, indent=2))
    if args.save:
        Path(args.save).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
