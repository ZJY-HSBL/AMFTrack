from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return cfg


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
