import numpy as np
from amftrack.motion.fsa_kalman import FSAKalmanFilter


def test_fsa_initiate_predict_update():
    kf = FSAKalmanFilter(mu=0.95)
    measurement = np.array([100.0, 50.0, 0.5, 120.0])
    mean, cov = kf.initiate(measurement, 0.9)
    mean2, cov2 = kf.predict(mean, cov)
    updated, cov3 = kf.update(mean2, cov2, measurement + np.array([2.0, 1.0, 0.0, 0.5]), 0.8)
    assert updated.shape == (8,)
    assert cov3.shape == (8, 8)
    assert np.isfinite(updated).all()
    assert np.isfinite(cov3).all()
