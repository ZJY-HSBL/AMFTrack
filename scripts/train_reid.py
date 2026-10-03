from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image

from amftrack.config import load_config, resolve_device
from amftrack.reid.osnet import OSNet
from amftrack.reid.gfmodel import GFModel


class ReIDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, height=256, width=128):
        self.df = df.reset_index(drop=True)
        self.transform = T.Compose([
            T.ToImage(),
            T.Resize((height, width), antialias=True),
            T.RandomHorizontalFlip(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pids = sorted(self.df["pid"].unique().tolist())
        self.pid_to_label = {pid: i for i, pid in enumerate(pids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), self.pid_to_label[int(row["pid"])]


def pairwise_distance(x):
    xx = (x * x).sum(dim=1, keepdim=True)
    dist = xx + xx.t() - 2.0 * x @ x.t()
    return dist.clamp_min(1e-12).sqrt()


def batch_hard_triplet_loss(features, labels, margin=0.3):
    dist = pairwise_distance(features)
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos = same & ~eye
    neg = ~same

    hardest_pos = torch.where(pos, dist, torch.full_like(dist, -1e9)).max(dim=1).values
    hardest_neg = torch.where(neg, dist, torch.full_like(dist, 1e9)).min(dim=1).values
    valid = pos.any(dim=1) & neg.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    return torch.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reid.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default="weights")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["project"].get("seed", 42))
    torch.manual_seed(cfg["project"].get("seed", 42))

    df = pd.read_csv(args.manifest)
    if "split" in df.columns:
        train_df = df[df["split"].isin(["train"])].copy()
        if train_df.empty:
            train_df = df.copy()
    else:
        train_df = df.copy()

    data_cfg = cfg["data"]
    dataset = ReIDDataset(
        train_df,
        height=int(data_cfg.get("image_height", 256)),
        width=int(data_cfg.get("image_width", 128)),
    )
    num_classes = len(dataset.pid_to_label)
    train_cfg = cfg["training"]
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(data_cfg.get("workers", 4)),
        drop_last=True,
        pin_memory=True,
    )

    device = torch.device(resolve_device("auto"))
    feature_dim = int(cfg["model"].get("osnet_feature_dim", 512))
    osnet = OSNet(num_classes=num_classes, feature_dim=feature_dim).to(device)
    gfmodel = GFModel(num_classes=num_classes, feature_dim=int(cfg["model"].get("gf_feature_dim", 512))).to(device)

    params = list(osnet.parameters()) + list(gfmodel.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=float(train_cfg.get("initial_lr", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
    )
    ce = nn.CrossEntropyLoss()
    start_iter = 0

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        osnet.load_state_dict(state["osnet"])
        gfmodel.load_state_dict(state["gfmodel"])
        optimizer.load_state_dict(state["optimizer"])
        start_iter = int(state["iteration"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    max_iterations = int(train_cfg.get("max_iterations", 180000))
    margin = float(train_cfg.get("triplet_margin", 0.3))
    cls_w = float(train_cfg.get("classification_weight", 1.0))
    tri_w = float(train_cfg.get("triplet_weight", 1.0))
    log_interval = int(train_cfg.get("log_interval", 100))
    save_interval = int(train_cfg.get("save_interval", 10000))

    iteration = start_iter
    osnet.train()
    gfmodel.train()
    while iteration < max_iterations:
        for images, labels in loader:
            iteration += 1
            images, labels = images.to(device), labels.to(device)

            ofeat, ologits = osnet(images, return_logits=True)
            gfeat, glogits = gfmodel(images, return_logits=True)

            loss_cls = ce(ologits, labels) + ce(glogits, labels)
            loss_tri = batch_hard_triplet_loss(ofeat, labels, margin) + batch_hard_triplet_loss(gfeat, labels, margin)
            loss = cls_w * loss_cls + tri_w * loss_tri

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print(
                    f"iter={iteration}/{max_iterations} "
                    f"loss={loss.item():.4f} cls={loss_cls.item():.4f} tri={loss_tri.item():.4f}"
                )

            if iteration % save_interval == 0 or iteration >= max_iterations:
                torch.save(osnet.state_dict(), output / "osnet_last.pth")
                torch.save(gfmodel.state_dict(), output / "gfmodel_last.pth")
                torch.save(
                    {
                        "iteration": iteration,
                        "osnet": osnet.state_dict(),
                        "gfmodel": gfmodel.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    output / "reid_training_state.pt",
                )

            if iteration >= max_iterations:
                break


if __name__ == "__main__":
    main()
