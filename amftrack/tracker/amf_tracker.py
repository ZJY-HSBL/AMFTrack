from __future__ import annotations

import numpy as np

from amftrack.matching.appearance import gallery_distance, cosine_distance
from amftrack.matching.assignment import linear_assignment
from amftrack.matching.ciou import ciou_cost_matrix
from amftrack.motion.fsa_kalman import FSAKalmanFilter
from amftrack.tracker.track import Track
from amftrack.types import Detection


class AMFTracker:
    def __init__(self, tracker_cfg: dict, fsa_cfg: dict):
        self.cfg = tracker_cfg
        self.kf = FSAKalmanFilter(**fsa_cfg)
        self.tracks: list[Track] = []
        self._next_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.predict(self.kf)

    def _gf_distance(self, track: Track, feature: np.ndarray | None) -> float:
        base = track.mean_gf_feature()
        if base is None or feature is None:
            return 1.0
        return cosine_distance(base, feature)

    def _build_cost(self, tracks: list[Track], detections: list[Detection]) -> np.ndarray:
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float64)

        track_boxes = [t.to_xyxy() for t in tracks]
        det_boxes = [d.xyxy for d in detections]
        geo = ciou_cost_matrix(track_boxes, det_boxes)

        app = np.zeros_like(geo)
        gf = np.zeros_like(geo)
        motion = np.zeros_like(geo)
        measurements = np.stack([d.xyah for d in detections], axis=0)

        gate = float(self.cfg.get("mahalanobis_gate", 9.4877))
        for i, track in enumerate(tracks):
            gdist = self.kf.gating_distance(
                track.mean, track.covariance, measurements, confidence=track.last_score
            )
            motion[i] = np.minimum(gdist / max(gate, 1e-6), 1.0)
            for j, det in enumerate(detections):
                app[i, j] = gallery_distance(list(track.features), det.osnet_feature)
                gf[i, j] = self._gf_distance(track, det.gf_feature)
                if gdist[j] > gate:
                    geo[i, j] = np.inf
                    app[i, j] = np.inf
                    gf[i, j] = np.inf
                    motion[i, j] = np.inf

        wa = float(self.cfg.get("appearance_weight", 0.35))
        wg = float(self.cfg.get("geometry_weight", 0.30))
        wgf = float(self.cfg.get("gf_weight", 0.20))
        wm = float(self.cfg.get("motion_weight", 0.15))
        total = wa + wg + wgf + wm
        if total <= 0:
            raise ValueError("At least one association weight must be positive.")
        return (wa * app + wg * geo + wgf * gf + wm * motion) / total

    def _initiate_track(self, detection: Detection) -> None:
        mean, covariance = self.kf.initiate(detection.xyah, detection.score)
        self.tracks.append(
            Track(
                mean=mean,
                covariance=covariance,
                track_id=self._next_id,
                detection=detection,
                n_init=int(self.cfg.get("n_init", 3)),
                max_age=int(self.cfg.get("max_age", 30)),
                max_feature_history=int(self.cfg.get("max_feature_history", 100)),
                max_gf_history=int(self.cfg.get("max_gf_history", 30)),
            )
        )
        self._next_id += 1

    def update(self, detections: list[Detection]) -> list[Track]:
        self.predict()

        active_tracks = [t for t in self.tracks if not t.is_deleted()]
        cost = self._build_cost(active_tracks, detections)
        matches, unmatched_t, unmatched_d = linear_assignment(
            cost, float(self.cfg.get("matching_threshold", 0.86))
        )

        for ti, di in matches:
            active_tracks[ti].update(self.kf, detections[di])

        # CIoU-only recovery stage.
        if unmatched_t and unmatched_d:
            t2 = [active_tracks[i] for i in unmatched_t]
            d2 = [detections[j] for j in unmatched_d]
            geo = ciou_cost_matrix([t.to_xyxy() for t in t2], [d.xyxy for d in d2])
            m2, ut2, ud2 = linear_assignment(
                geo, float(self.cfg.get("second_stage_ciou_threshold", 0.90))
            )
            for a, b in m2:
                t2[a].update(self.kf, d2[b])
            unmatched_t = [unmatched_t[i] for i in ut2]
            unmatched_d = [unmatched_d[j] for j in ud2]

        for ti in unmatched_t:
            active_tracks[ti].mark_missed()

        min_conf = float(self.cfg.get("min_detection_confidence", 0.25))
        for di in unmatched_d:
            if detections[di].score >= min_conf:
                self._initiate_track(detections[di])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update == 0]
