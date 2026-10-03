from __future__ import annotations

import argparse
from pathlib import Path
import torch

from amftrack.reid.osnet import OSNet
from amftrack.reid.gfmodel import GFModel


def load_state(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["osnet", "gfmodel"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = OSNet(feature_dim=512) if args.model == "osnet" else GFModel(feature_dim=512)
    load_state(model, args.weights)
    model.eval()
    dummy = torch.randn(1, 3, 256, 128)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
    )
    print(args.output)


if __name__ == "__main__":
    main()
