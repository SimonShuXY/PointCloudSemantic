#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Pointcept native SemanticKITTI dataloader and model.")
    parser.add_argument("--root", default="/root/autodl-tmp/ipfp_repro")
    parser.add_argument("--config", default="configs/semantic_kitti/semseg-pt-v2m2-0-base.py")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def move_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def install_optional_dummy(name: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    pointcept_root = root / "src/Pointcept"
    sys.path.insert(0, str(pointcept_root))
    for optional_module in ("open3d", "h5py", "cv2", "SharedArray"):
        install_optional_dummy(optional_module)

    from pointcept.datasets import build_dataset
    from pointcept.datasets.utils import collate_fn
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = Config.fromfile(str(pointcept_root / args.config))
    cfg.data.train.data_root = str(root / "data/semantic_kitti")
    cfg.model.backbone.enable_checkpoint = False

    dataset = build_dataset(cfg.data.train)
    sample = dataset[args.sample_index]
    batch = collate_fn([sample])
    if "offset" not in batch:
        batch["offset"] = torch.tensor([batch["coord"].shape[0]], dtype=torch.int)
    batch = move_to_device(batch, args.device)

    model = build_model(cfg.model).to(args.device).eval()
    with torch.no_grad():
        output = model(batch)

    seg_logits = output["seg_logits"]
    loss = output.get("loss")
    result = {
        "status": "OK",
        "config": args.config,
        "dataset_len": len(dataset),
        "sample_name": sample.get("name", "<unknown>"),
        "points_after_transform": int(batch["coord"].shape[0]),
        "feat_shape": list(batch["feat"].shape),
        "grid_coord_shape": list(batch["grid_coord"].shape),
        "segment_shape": list(batch["segment"].shape),
        "offset": batch["offset"].detach().cpu().tolist(),
        "logits_shape": list(seg_logits.shape),
        "loss": float(loss.item()) if loss is not None else None,
        "finite_logits": bool(torch.isfinite(seg_logits).all().item()),
        "cuda_max_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if args.device == "cuda"
        else 0.0,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
