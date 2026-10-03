# Benchmark Targets / 基准目标

These values are stored as **reference targets**, not as automatically achieved results. Actual values depend on detector weights, ReID training, exact dataset versions, hardware and runtime configuration.

以下数值作为**基准目标**保存，不代表下载仓库后未经训练即可直接获得。实际结果取决于检测器权重、ReID 权重、数据集版本、硬件与运行参数。

| Dataset | MOTA | IDF1 | MT | FPS |
|---|---:|---:|---:|---:|
| MOT16 | 69.7 | 71.3 | 41.5 | 55.8 |
| MOT17 | 77.9 | 70.6 | 45.7 | 48.7 |

The available technical specification also contains a narrative MOT17 IDF1 value of `76.0`, while the benchmark table lists `70.6`. The repository preserves the table value as the machine-readable benchmark target and records the discrepancy here instead of silently merging the two numbers.

现有技术说明中，MOT17 的 IDF1 在正文叙述中出现 `76.0`，而基准表中记录为 `70.6`。仓库将表格值 `70.6` 作为机器可读目标，并在此保留差异说明，不对两个数值进行擅自合并。
