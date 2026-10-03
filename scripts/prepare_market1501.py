from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PATTERN = re.compile(r"([-\d]+)_c(\d)")


def parse_market(root: Path):
    subsets = [
        ("bounding_box_train", "train"),
        ("bounding_box_test", "gallery"),
        ("query", "query"),
    ]
    rows = []
    for folder, split in subsets:
        d = root / folder
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg")):
            m = PATTERN.search(p.name)
            if not m:
                continue
            pid, camid = int(m.group(1)), int(m.group(2))
            if pid < 0:
                continue
            rows.append((str(p.resolve()), pid, camid, split))
    return rows


def parse_generic(root: Path):
    rows = []
    for pid_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        for img in sorted(pid_dir.rglob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            m = re.search(r"c(\d+)", img.stem, re.I)
            camid = int(m.group(1)) if m else 0
            rows.append((str(img.resolve()), pid, camid, "train"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generic-folder", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    rows = parse_generic(root) if args.generic_folder else parse_market(root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pid", "camid", "split"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
