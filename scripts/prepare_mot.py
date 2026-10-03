from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    split = Path(args.root) / args.split
    if not split.exists():
        raise FileNotFoundError(split)

    sequences = [p for p in sorted(split.iterdir()) if p.is_dir()]
    failures = []
    for seq in sequences:
        required = [seq / "img1", seq / "seqinfo.ini"]
        if args.split == "train":
            required.append(seq / "gt" / "gt.txt")
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            failures.append((seq.name, missing))
        else:
            print(f"[OK] {seq.name}")

    if failures:
        for name, missing in failures:
            print(f"[MISSING] {name}: {missing}")
        raise SystemExit(2)
    print(f"Validated {len(sequences)} sequence(s).")


if __name__ == "__main__":
    main()
