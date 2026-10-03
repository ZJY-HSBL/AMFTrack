from __future__ import annotations

import numpy as np

from amftrack.types import Detection
from amftrack.tracker.amf_tracker import AMFTracker


TRACKER_CFG = {
    "min_detection_confidence": 0.1,
    "n_init": 2,
    "max_age": 5,
    "max_feature_history": 20,
    "max_gf_history": 10,
    "matching_threshold": 0.9,
    "second_stage_ciou_threshold": 1.2,
    "appearance_weight": 0.35,
    "geometry_weight": 0.30,
    "gf_weight": 0.20,
    "motion_weight": 0.15,
    "mahalanobis_gate": 16.0,
}
FSA_CFG = {
    "mu": 0.95,
    "confidence_window": 10,
    "noise_floor": 0.05,
    "std_weight_position": 0.05,
    "std_weight_velocity": 0.00625,
}


def unit_feature(index: int, dim: int = 16):
    x = np.zeros(dim, dtype=np.float64)
    x[index % dim] = 1.0
    return x


def main():
    rng = np.random.default_rng(7)
    tracker = AMFTracker(TRACKER_CFG, FSA_CFG)

    trajectories = {
        0: np.array([20.0, 20.0, 50.0, 90.0]),
        1: np.array([170.0, 40.0, 205.0, 120.0]),
    }
    velocities = {
        0: np.array([3.2, 1.0, 3.2, 1.0]),
        1: np.array([-2.1, 0.8, -2.1, 0.8]),
    }

    for frame in range(20):
        detections = []
        for identity in (0, 1):
            trajectories[identity] += velocities[identity]
            noise = rng.normal(0.0, 0.7, size=4)
            det = Detection(
                trajectories[identity] + noise,
                score=0.92 - 0.03 * (frame % 3),
                osnet_feature=unit_feature(identity),
                gf_feature=unit_feature(identity + 4),
                frame_index=frame,
            )
            detections.append(det)

        tracks = tracker.update(detections)
        ids = [(t.track_id, np.round(t.to_xyxy(), 1).tolist()) for t in tracks]
        print(f"frame={frame:02d} tracks={ids}")

    confirmed = [t for t in tracker.tracks if t.is_confirmed()]
    if len(confirmed) != 2:
        raise SystemExit(f"Synthetic check failed: expected 2 tracks, got {len(confirmed)}.")
    print("Synthetic core tracking check passed.")


if __name__ == "__main__":
    main()
