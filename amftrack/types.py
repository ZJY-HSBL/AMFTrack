from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray
    score: float
    class_id: int = 0
    osnet_feature: np.ndarray | None = None
    gf_feature: np.ndarray | None = None
    frame_index: int = -1

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float64).reshape(4)

    @property
    def tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)

    @property
    def xyah(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w / h, h], dtype=np.float64)
