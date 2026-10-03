#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_mot.py --config configs/mot17.yaml --mot-root data/MOT17 --split train --output-dir outputs/mot17
