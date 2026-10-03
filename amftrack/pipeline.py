from __future__ import annotations

from pathlib import Path
import time
import cv2

from amftrack.config import resolve_device
from amftrack.detector.yolov8 import YOLOv8Detector
from amftrack.reid.extractor import AppearanceExtractor
from amftrack.tracker.amf_tracker import AMFTracker
from amftrack.utils.visualize import draw_tracks


class AMFTrackPipeline:
    def __init__(self, cfg: dict):
        det_cfg = dict(cfg["detector"])
        reid_cfg = dict(cfg["reid"])
        tracker_cfg = dict(cfg["tracker"])
        fsa_cfg = dict(cfg["fsa"])

        det_cfg["device"] = resolve_device(det_cfg.get("device", "auto"))
        reid_cfg["device"] = resolve_device(reid_cfg.get("device", "auto"))

        self.detector = YOLOv8Detector(**{
            k: det_cfg[k] for k in (
                "model", "person_class_id", "confidence", "iou_nms", "image_size", "device"
            )
        })
        self.extractor = AppearanceExtractor(
            osnet_weights=reid_cfg.get("osnet_weights"),
            gfmodel_weights=reid_cfg.get("gfmodel_weights"),
            device=reid_cfg["device"],
        )
        self.tracker = AMFTracker(tracker_cfg, fsa_cfg)

    def process_frame(self, frame, frame_index: int):
        detections = self.detector(frame, frame_index)
        boxes = [d.xyxy for d in detections]
        os_features, gf_features = self.extractor.encode(frame, boxes)
        for d, of, gf in zip(detections, os_features, gf_features):
            d.osnet_feature = of
            d.gf_feature = gf
        tracks = self.tracker.update(detections)
        return tracks

    def process_video(self, source: str, output: str | None = None, display: bool = False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        frame_index = 0
        elapsed = 0.0
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.perf_counter()
            tracks = self.process_frame(frame, frame_index)
            elapsed += time.perf_counter() - start
            processed += 1

            vis = draw_tracks(frame, tracks)
            if writer is not None:
                writer.write(vis)
            if display:
                cv2.imshow("AMFTrack", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            frame_index += 1

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        return {
            "frames": processed,
            "seconds": elapsed,
            "fps": processed / max(elapsed, 1e-9),
        }
