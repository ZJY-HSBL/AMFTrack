from __future__ import annotations
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(np.asarray(a).reshape(1, -1))[0]
    b = l2_normalize(np.asarray(b).reshape(1, -1))[0]
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def gallery_distance(gallery: list[np.ndarray], feature: np.ndarray | None, mode: str = "min") -> float:
    if feature is None or not gallery:
        return 1.0
    distances = [cosine_distance(g, feature) for g in gallery]
    if mode == "mean":
        return float(np.mean(distances))
    return float(np.min(distances))
