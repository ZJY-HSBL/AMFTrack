# AMFTrack — Adaptive Motion-Feature Tracking

> 中文 / English 双语仓库。面向复杂交通场景的自适应运动—特征融合多目标跟踪系统。

AMFTrack（Adaptive Motion-Feature Tracking）将 **YOLOv8、OSNet、FSA 自适应遗忘卡尔曼滤波、CIoU 关联度量、GFModel 轨迹特征融合与匈牙利匹配** 组织成统一的多目标跟踪流水线。仓库提供训练、推理、MOTChallenge 格式评估、视频可视化、ONNX 导出、单元测试和合成数据自检。

## 中文简介

复杂交通场景中的行人跟踪容易受到遮挡、尺度变化、非线性运动和相似外观干扰。AMFTrack 采用分阶段建模：

1. **YOLOv8 检测器**：逐帧定位行人；
2. **OSNet 外观编码器**：提取多尺度 ReID 特征；
3. **FSA 自适应遗忘卡尔曼滤波**：根据检测置信度动态调整测量噪声，并通过遗忘因子强化对近期运动的响应；
4. **CIoU 关联代价**：同时考虑重叠、中心距离与宽高比；
5. **GFModel 轨迹特征提取器**：以 ResNet50 为骨干融合局部与全局上下文，并支持三元组监督；
6. **Hungarian Assignment**：联合运动、外观、轨迹上下文与几何代价完成在线数据关联。

默认研究配置对应 300 轮 YOLOv8 训练、OSNet 初始学习率 `5e-5`、180000 次 ReID 训练迭代以及 FSA 遗忘因子 `μ=0.95`。所有关键参数均在 YAML 配置中可修改。

## English Overview

AMFTrack (Adaptive Motion-Feature Tracking) is a modular pedestrian multi-object tracking system for complex traffic scenes. The pipeline combines:

1. **YOLOv8** for pedestrian detection;
2. **OSNet** for omni-scale appearance representation;
3. **FSA adaptive-forgetting Kalman filtering** for confidence-aware motion estimation;
4. **CIoU association cost** for geometry-aware matching;
5. **GFModel** with a ResNet50 backbone for local/global trajectory-context representation and triplet supervision;
6. **Hungarian assignment** for online multi-cue data association.

The default research configuration uses 300 detector epochs, an OSNet initial learning rate of `5e-5`, 180,000 ReID iterations, and an FSA forgetting factor of `μ=0.95`. All core parameters are configurable through YAML files.

## Repository Layout

```text
AMFTrack/
├── configs/
│   ├── mot16.yaml
│   ├── mot17.yaml
│   └── reid.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARK_TARGETS.md
│   ├── DATASETS.md
│   ├── EVALUATION.md
│   ├── IMPLEMENTATION_NOTES.md
│   └── TRAINING.md
├── scripts/
│   ├── benchmark.py
│   ├── demo_synthetic.py
│   ├── evaluate_mot.py
│   ├── export_onnx.py
│   ├── prepare_market1501.py
│   ├── prepare_mot.py
│   ├── track_video.py
│   ├── train_detector.py
│   └── train_reid.py
├── tests/
├── weights/
├── amftrack/
│   ├── detector/
│   ├── matching/
│   ├── metrics/
│   ├── motion/
│   ├── reid/
│   ├── tracker/
│   └── utils/
├── GITHUB_DESCRIPTION.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Installation / 安装

Recommended / 推荐：

```bash
conda create -n amftrack python=3.10 -y
conda activate amftrack
pip install -e .
```

CUDA-enabled PyTorch should be installed according to the local CUDA version before installing the remaining dependencies.

如果本机已安装与 CUDA 匹配的 PyTorch，可直接执行：

```bash
pip install -e .
```

## Quick Start / 快速开始

### 1. Synthetic self-check / 合成数据自检

```bash
python scripts/demo_synthetic.py
```

This validates the FSA motion model, CIoU cost and online association logic without downloading external datasets or model weights.

该命令不需要任何外部数据集或权重，可用于检查核心跟踪链路。

### 2. Track a video / 视频跟踪

```bash
python scripts/track_video.py \
  --config configs/mot17.yaml \
  --source path/to/input.mp4 \
  --output outputs/tracked.mp4
```

### 3. Detector training / 检测器训练

```bash
python scripts/train_detector.py \
  --data path/to/pedestrian.yaml \
  --model yolov8n.pt \
  --epochs 300
```

### 4. ReID training / ReID 训练

Prepare a CSV manifest with columns `path,pid,camid,split`, then run:

准备包含 `path,pid,camid,split` 的 CSV 后执行：

```bash
python scripts/train_reid.py \
  --config configs/reid.yaml \
  --manifest data/reid_manifest.csv
```

### 5. MOT evaluation / MOT 评估

```bash
python scripts/evaluate_mot.py \
  --config configs/mot17.yaml \
  --mot-root data/MOT17 \
  --split train \
  --output-dir outputs/mot17
```

## Core Metrics / 核心指标

The evaluation module supports:

- **MOTA**: multi-object tracking accuracy;
- **IDF1**: identity F-score;
- **MT**: mostly tracked trajectories;
- **FPS**: processing speed;
- **ID switches**, false positives and false negatives through `motmetrics`.

评估脚本同时输出 MOTChallenge 标准结果文件，方便与其他跟踪器统一比较。

## Dataset Roles / 数据集用途

- **Market-1501**: appearance/ReID training;
- **CUHK03**: appearance/ReID training;
- **MOT16**: multi-object tracking evaluation;
- **MOT17**: multi-object tracking evaluation.

Dataset files and pretrained weights are intentionally not bundled in the repository. See `docs/DATASETS.md` and `weights/README.md`.

数据集与大体积预训练权重不直接打包，请按照对应文档放置。

## Reproducible Configuration / 可复核配置

Important research parameters are centralized in:

```text
configs/mot16.yaml
configs/mot17.yaml
configs/reid.yaml
```

Run configuration checks with:

```bash
python -m pytest -q
```

## Design Goals / 设计目标

中文：本项目重点解决复杂交通场景中遮挡、ID 切换、行人非线性运动以及局部噪声造成的关联不稳定问题，同时保留可部署性、模块可替换性与可解释的匹配代价。

English: The system targets occlusion, identity switches, nonlinear pedestrian motion and association instability while preserving deployability, modularity and interpretable multi-cue matching.

## License

MIT License. External datasets and model weights remain subject to their own licenses and terms.
