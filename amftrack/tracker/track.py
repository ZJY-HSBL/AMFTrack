from __future__ import annotations

from collections import deque
from enum import IntEnum
import numpy as np

from amftrack.types import Detection
from amftrack.utils.geometry import xyah_to_xyxy


class TrackState(IntEnum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        detection: Detection,
        n_init: int,
        max_age: int,
        max_feature_history: int = 100,
        max_gf_history: int = 30,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = int(track_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.n_init = int(n_init)
        self.max_age = int(max_age)
        self.last_score = float(detection.score)

        self.features = deque(maxlen=max_feature_history)
        self.gf_features = deque(maxlen=max_gf_history)
        if detection.osnet_feature is not None:
            self.features.append(np.asarray(detection.osnet_feature, dtype=np.float64))
        if detection.gf_feature is not None:
            self.gf_features.append(np.asarray(detection.gf_feature, dtype=np.float64))

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection: Detection) -> None:
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.xyah, detection.score
        )
        self.hits += 1
        self.time_since_update = 0
        self.last_score = float(detection.score)

        if detection.osnet_feature is not None:
            self.features.append(np.asarray(detection.osnet_feature, dtype=np.float64))
        if detection.gf_feature is not None:
            self.gf_features.append(np.asarray(detection.gf_feature, dtype=np.float64))

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def to_xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    def mean_gf_feature(self) -> np.ndarray | None:
        if not self.gf_features:
            return None
        x = np.mean(np.stack(self.gf_features, axis=0), axis=0)
        n = np.linalg.norm(x)
        return x / max(n, 1e-12)
