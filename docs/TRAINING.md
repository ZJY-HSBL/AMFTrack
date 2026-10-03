# Training / 训练说明

## 中文

### YOLOv8

默认训练轮数设为 300：

```bash
python scripts/train_detector.py \
  --data data/pedestrian.yaml \
  --model yolov8n.pt \
  --epochs 300 \
  --imgsz 640
```

实际行人检测训练集由使用者在 Ultralytics 数据 YAML 中指定。

### OSNet + GFModel

ReID 训练采用分类交叉熵与 Batch-Hard Triplet Loss 联合优化。默认初始学习率为 `5e-5`，最大迭代次数为 `180000`。

```bash
python scripts/train_reid.py \
  --config configs/reid.yaml \
  --manifest data/reid_manifest.csv \
  --output weights
```

训练输出：

```text
weights/
├── osnet_last.pth
├── gfmodel_last.pth
└── reid_training_state.pt
```

### 训练恢复

```bash
python scripts/train_reid.py \
  --config configs/reid.yaml \
  --manifest data/reid_manifest.csv \
  --output weights \
  --resume weights/reid_training_state.pt
```

## English

The detector configuration uses 300 epochs by default. ReID training jointly optimizes identity classification and batch-hard triplet objectives. The initial ReID learning rate is `5e-5`, and the default maximum training length is 180,000 iterations. Checkpoints include both encoder states, optimizer state and the current iteration for deterministic continuation.
