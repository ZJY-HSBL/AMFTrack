# Datasets / 数据集

## 中文

项目按用途区分 ReID 数据与 MOT 数据。

### Market-1501

用于 OSNet/GFModel 外观特征训练。推荐保留原始目录并通过：

```bash
python scripts/prepare_market1501.py \
  --root data/Market-1501-v15.09.15 \
  --output data/market1501.csv
```

生成统一 CSV。脚本从标准文件名中解析 `pid` 和 `camid`。

### CUHK03

用于补充 ReID 训练。由于不同发布版本的目录格式差异较大，本仓库不假设固定目录。将图像导出为按身份分组的目录：

```text
data/CUHK03_exported/
├── 0001/
│   ├── c1_0001.jpg
│   └── c2_0002.jpg
└── 0002/
```

然后使用 `scripts/prepare_market1501.py --generic-folder` 生成同一格式的 manifest。

### MOT16 / MOT17

用于多目标跟踪评估。保持 MOTChallenge 原始结构：

```text
data/MOT17/
├── train/
│   ├── MOT17-02-FRCNN/
│   │   ├── img1/
│   │   ├── gt/gt.txt
│   │   └── seqinfo.ini
│   └── ...
└── test/
```

运行：

```bash
python scripts/prepare_mot.py --root data/MOT17 --split train
```

进行目录完整性检查。

## English

Market-1501 and CUHK03 are used for appearance/ReID representation learning, while MOT16 and MOT17 are used for tracking evaluation. Dataset files are not redistributed by this repository. The preparation scripts only parse locally available copies into a unified manifest or validate MOTChallenge directory structure.
