import numpy as np
from amftrack.types import Detection
from amftrack.tracker.amf_tracker import AMFTracker


def feature(k):
    x = np.zeros(8)
    x[k] = 1.0
    return x


def test_two_identity_tracking():
    tracker = AMFTracker(
        {
            "min_detection_confidence": 0.1,
            "n_init": 2,
            "max_age": 3,
            "max_feature_history": 20,
            "max_gf_history": 10,
            "matching_threshold": 0.95,
            "second_stage_ciou_threshold": 1.2,
            "appearance_weight": 0.4,
            "geometry_weight": 0.3,
            "gf_weight": 0.2,
            "motion_weight": 0.1,
            "mahalanobis_gate": 16.0,
        },
        {
            "mu": 0.95,
            "confidence_window": 10,
            "noise_floor": 0.05,
            "std_weight_position": 0.05,
            "std_weight_velocity": 0.00625,
        },
    )
    for frame in range(6):
        d0 = Detection(np.array([10 + frame, 10, 30 + frame, 60]), .9, osnet_feature=feature(0), gf_feature=feature(2))
        d1 = Detection(np.array([100 - frame, 20, 125 - frame, 80]), .9, osnet_feature=feature(1), gf_feature=feature(3))
        tracker.update([d0, d1])

    confirmed = [t for t in tracker.tracks if t.is_confirmed()]
    assert len(confirmed) == 2
    assert len({t.track_id for t in confirmed}) == 2
