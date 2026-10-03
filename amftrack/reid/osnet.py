from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ConvLayer(in_channels, out_channels, 1)

    def forward(self, x):
        return self.block(x)


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.mlp(self.pool(x))


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid)

        self.stream1 = LightConv3x3(mid, mid)
        self.stream2 = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.stream3 = nn.Sequential(
            LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid)
        )
        self.stream4 = nn.Sequential(
            LightConv3x3(mid, mid), LightConv3x3(mid, mid),
            LightConv3x3(mid, mid), LightConv3x3(mid, mid)
        )
        self.gates = nn.ModuleList([ChannelGate(mid) for _ in range(4)])
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        x1 = self.conv1(x)
        streams = [self.stream1(x1), self.stream2(x1), self.stream3(x1), self.stream4(x1)]
        y = sum(g(s) for g, s in zip(self.gates, streams))
        y = self.conv3(y)
        return F.relu(y + identity, inplace=True)


class OSNet(nn.Module):
    """Compact omni-scale network for person appearance representation."""

    def __init__(self, num_classes: int = 0, feature_dim: int = 512):
        super().__init__()
        channels = [64, 256, 384, 512]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            OSBlock(channels[0], channels[1]),
            OSBlock(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            OSBlock(channels[1], channels[2]),
            OSBlock(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            OSBlock(channels[2], channels[3]),
            OSBlock(channels[3], channels[3]),
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_classes) if num_classes > 0 else None

    def forward(self, x, return_logits: bool = False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x).flatten(1)
        feat = self.embedding(x)
        feat = F.normalize(feat, p=2, dim=1)
        if return_logits:
            if self.classifier is None:
                raise RuntimeError("num_classes must be > 0 for classification logits.")
            return feat, self.classifier(feat)
        return feat
