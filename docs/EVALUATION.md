# Evaluation / 评估

## 中文

使用 MOTChallenge 标准文本结果进行评估：

```bash
python scripts/evaluate_mot.py \
  --config configs/mot17.yaml \
  --mot-root data/MOT17 \
  --split train \
  --output-dir outputs/mot17
```

每个序列生成：

```text
outputs/mot17/MOT17-02-FRCNN.txt
```

格式：

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

主要指标：

- `MOTA = 1 - (FN + FP + IDSW) / GT`
- `IDF1 = 2*IDTP / (2*IDTP + IDFN + IDFP)`
- `MT`: 超过 80% 生命周期被成功跟踪的真实轨迹比例/数量
- `FPS`: 处理帧率

脚本使用 `motmetrics` 对存在 GT 的训练序列进行汇总。

## English

The evaluator writes standard MOTChallenge result files and uses `motmetrics` when ground-truth annotations are available. It reports MOTA, IDF1, mostly-tracked trajectories, identity switches, false positives, false negatives and measured pipeline FPS.
