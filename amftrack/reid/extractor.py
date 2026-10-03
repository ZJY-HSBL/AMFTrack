from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

from amftrack.reid.osnet import OSNet
from amftrack.reid.gfmodel import GFModel
from amftrack.utils.geometry import clip_box


class AppearanceExtractor:
    def __init__(
        self,
        osnet_weights: str | None = None,
        gfmodel_weights: str | None = None,
        device: str = "cpu",
        image_size: tuple[int, int] = (256, 128),
    ) -> None:
        self.device = torch.device(device)
        self.osnet = OSNet(feature_dim=512).to(self.device).eval()
        self.gfmodel = GFModel(feature_dim=512).to(self.device).eval()

        self._load(self.osnet, osnet_weights)
        self._load(self.gfmodel, gfmodel_weights)

        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(image_size, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load(self, model, path):
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model", "osnet", "gfmodel"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    def _crop_batch(self, frame: np.ndarray, boxes: list[np.ndarray]) -> torch.Tensor:
        h, w = frame.shape[:2]
        tensors = []
        for box in boxes:
            x1, y1, x2, y2 = clip_box(box, w, h).astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(crop))
        if not tensors:
            return torch.empty((0, 3, 256, 128), device=self.device)
        return torch.stack(tensors).to(self.device)

    @torch.inference_mode()
    def encode(self, frame: np.ndarray, boxes: list[np.ndarray]):
        batch = self._crop_batch(frame, boxes)
        if batch.shape[0] == 0:
            return [], []
        os_feat = self.osnet(batch).cpu().numpy()
        gf_feat = self.gfmodel(batch).cpu().numpy()
        return [x for x in os_feat], [x for x in gf_feat]
