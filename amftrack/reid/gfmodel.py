from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class GFModel(nn.Module):
    """
    ResNet50 trajectory-context encoder.

    Three representations are produced from the same feature map:
    - global average pooled embedding;
    - spatially pooled local embedding;
    - triplet-supervised embedding.
    Their normalized mean is used as the fused trajectory feature.
    """

    def __init__(self, num_classes: int = 0, feature_dim: int = 512):
        super().__init__()
        backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        in_dim = 2048

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d((4, 1))

        self.global_fc = nn.Linear(in_dim, feature_dim)
        self.local_fc = nn.Linear(in_dim * 4, feature_dim)
        self.triplet_fc = nn.Linear(in_dim, feature_dim)

        self.classifier = nn.Linear(feature_dim, num_classes) if num_classes > 0 else None

    def forward(self, x, return_parts: bool = False, return_logits: bool = False):
        fmap = self.backbone(x)
        global_raw = self.global_pool(fmap).flatten(1)
        local_raw = self.local_pool(fmap).flatten(1)

        g = F.normalize(self.global_fc(global_raw), p=2, dim=1)
        l = F.normalize(self.local_fc(local_raw), p=2, dim=1)
        t = F.normalize(self.triplet_fc(global_raw), p=2, dim=1)
        fused = F.normalize((g + l + t) / 3.0, p=2, dim=1)

        out = fused
        if return_parts:
            out = (fused, g, l, t)
        if return_logits:
            if self.classifier is None:
                raise RuntimeError("num_classes must be > 0 for classification logits.")
            logits = self.classifier(fused)
            return out, logits
        return out
