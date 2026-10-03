from __future__ import annotations

import numpy as np
from amftrack.types import Detection


class YOLOv8Detector:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        person_class_id: int = 0,
        confidence: float = 0.25,
        iou_nms: float = 0.7,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.person_class_id = int(person_class_id)
        self.confidence = float(confidence)
        self.iou_nms = float(iou_nms)
        self.image_size = int(image_size)
        self.device = device

    def __call__(self, frame: np.ndarray, frame_index: int = -1) -> list[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_nms,
            imgsz=self.image_size,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, confs, classes):
            detections.append(
                Detection(box, float(score), int(cls), frame_index=frame_index)
            )
        return detections
