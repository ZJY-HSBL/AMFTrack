# Implementation Notes / 实现说明

## 中文

公开技术说明给出了 AMFTrack 的主要模块、训练轮数/迭代数、FSA 遗忘因子和基准指标，但没有给出所有工程级超参数，例如多线索代价融合权重、Hungarian 阈值、轨迹生命周期、GFModel 各分支精确通道配置以及检测器所用完整训练数据 YAML。

因此，本仓库采用以下原则：

- 明确给出的参数保持固定默认值：`YOLOv8 epochs=300`、`OSNet lr=5e-5`、`ReID iterations=180000`、`FSA μ=0.95`；
- DeepSORT 常用的轨迹状态 `xyah + velocity` 作为运动状态空间；
- FSA 的测量噪声缩放按当前置信度与最近置信度均值构造，并对最小噪声设置数值下界；
- 未明确给出的代价融合权重与生命周期参数全部集中放入 YAML，避免隐藏常量；
- GFModel 使用 ResNet50，并实现全局、局部与三元组监督分支后进行均值融合；
- 所有无法由技术说明唯一确定的工程细节均保持可配置，而不伪装成唯一正确取值。

## English

The available technical specification defines the major AMFTrack modules, core training lengths, the FSA forgetting factor and benchmark values, but it does not uniquely specify every engineering hyperparameter. Missing details include multi-cue fusion weights, assignment thresholds, track lifecycle settings, exact GFModel branch widths and the complete detector training-data definition.

This repository therefore keeps explicitly defined values as defaults and exposes underspecified engineering choices through YAML. The implementation uses an `xyah + velocity` DeepSORT-style state space, confidence-aware FSA measurement noise, configurable cue fusion, and a ResNet50 GFModel with global/local/triplet-supervised embeddings.
