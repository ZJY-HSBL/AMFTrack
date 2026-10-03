from __future__ import annotations

from collections import deque
import numpy as np


class FSAKalmanFilter:
    """
    DeepSORT-style constant-velocity Kalman model with confidence-aware
    measurement noise and covariance forgetting.
    State: [x, y, a, h, vx, vy, va, vh]
    Measurement: [x, y, a, h]
    """

    ndim = 4
    dt = 1.0

    def __init__(
        self,
        mu: float = 0.95,
        confidence_window: int = 10,
        noise_floor: float = 0.05,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
    ) -> None:
        if not (0.0 < mu <= 1.0):
            raise ValueError("mu must be in (0, 1].")
        self.mu = float(mu)
        self.noise_floor = float(noise_floor)
        self.std_weight_position = float(std_weight_position)
        self.std_weight_velocity = float(std_weight_velocity)
        self.confidences: deque[float] = deque(maxlen=max(1, int(confidence_window)))

        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float64)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float64)

    def initiate(self, measurement: np.ndarray, confidence: float = 1.0):
        measurement = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[measurement, np.zeros_like(measurement)]
        h = max(float(measurement[3]), 1.0)
        std = np.asarray([
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ])
        covariance = np.diag(std * std)
        self.confidences.append(float(np.clip(confidence, 0.0, 1.0)))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1.0)
        std_pos = np.asarray([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ])
        std_vel = np.asarray([
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ])
        motion_cov = np.diag(np.r_[std_pos, std_vel] ** 2)

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def _project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float):
        h = max(float(mean[3]), 1.0)
        base_std = np.asarray([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ])
        base_r = np.diag(base_std ** 2)

        confidence = float(np.clip(confidence, 0.0, 1.0))
        history_mean = float(np.mean(self.confidences)) if self.confidences else max(confidence, 1e-3)
        # Confidence-aware scaling: low current confidence increases uncertainty
        # relative to recent confidence; the floor prevents numerical collapse.
        scale = max(self.noise_floor, (1.0 - confidence) * max(history_mean, 1e-3))
        adaptive_r = base_r * scale

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + adaptive_r
        return projected_mean, projected_cov, adaptive_r

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float,
    ):
        measurement = np.asarray(measurement, dtype=np.float64)
        projected_mean, projected_cov, _ = self._project(mean, covariance, confidence)

        cross_cov = covariance @ self._update_mat.T
        kalman_gain = np.linalg.solve(projected_cov.T, cross_cov.T).T
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        # Adaptive-forgetting covariance update.
        new_cov = (1.0 / self.mu) * (identity - kalman_gain @ self._update_mat) @ covariance
        new_cov = 0.5 * (new_cov + new_cov.T)
        self.confidences.append(float(np.clip(confidence, 0.0, 1.0)))
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        confidence: float = 1.0,
    ) -> np.ndarray:
        measurements = np.asarray(measurements, dtype=np.float64)
        if measurements.size == 0:
            return np.empty((0,), dtype=np.float64)
        projected_mean, projected_cov, _ = self._project(mean, covariance, confidence)
        d = measurements - projected_mean
        try:
            chol = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(chol, d.T)
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(projected_cov)
            return np.einsum("ni,ij,nj->n", d, inv, d)
