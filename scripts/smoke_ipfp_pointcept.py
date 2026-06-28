#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test IPFP feature back-projection with Pointcept PTv3.")
    parser.add_argument("--pointcept-root", default=os.environ.get("POINTCEPT_ROOT", "."), help="Path to Pointcept root")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-centers", type=int, default=128)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_synthetic_camera_scene(num_points: int, h: int, w: int, device: str) -> dict:
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    z = torch.empty(num_points, device=device).uniform_(5.0, 35.0, generator=gen)
    x = (torch.rand(num_points, device=device, generator=gen) - 0.5) * z * 0.9
    y = (torch.rand(num_points, device=device, generator=gen) - 0.5) * z * 0.5
    coord = torch.stack([x, y, z], dim=-1)
    intensity = torch.rand(num_points, 1, device=device, generator=gen)
    feat = torch.cat([coord / 35.0, intensity], dim=-1)

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij",
    )
    image = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0)
    metric_depth = 5.0 + 30.0 * (0.65 * yy + 0.35 * xx)

    intrinsics = torch.tensor(
        [[300.0, 0.0, w / 2.0], [0.0, 300.0, h / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    lidar_to_camera = torch.eye(4, device=device)
    return {
        "coord": coord,
        "feat": feat,
        "image": image,
        "metric_depth": metric_depth,
        "intrinsics": intrinsics,
        "lidar_to_camera": lidar_to_camera,
    }


def grid_coord_from_coord(coord: torch.Tensor, origin: torch.Tensor, grid_size: float) -> torch.Tensor:
    grid = torch.div(coord - origin, grid_size, rounding_mode="trunc").int()
    return grid.clamp_min(0)


def main() -> None:
    args = parse_args()
    pointcept_root = Path(args.pointcept_root).resolve()
    sys.path.insert(0, str(pointcept_root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from ipfp import IPFPFeatureBackProjector
    from pointcept.models import build_model
    from pointcept.models.utils.structure import Point
    from pointcept.utils.config import Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(7)
    cfg = Config.fromfile(str(pointcept_root / "configs/nuscenes/semseg-pt-v3m1-0-base.py"))
    cfg.model.backbone.enable_flash = False
    cfg.model.backbone.enc_patch_size = (64, 64, 64, 64, 64)
    cfg.model.backbone.dec_patch_size = (64, 64, 64, 64)

    model = build_model(cfg.model).to(args.device).eval()
    scene = make_synthetic_camera_scene(args.num_points, args.height, args.width, args.device)
    origin = scene["coord"].min(dim=0).values
    lidar_grid = grid_coord_from_coord(scene["coord"], origin, args.grid_size)
    offset = torch.tensor([args.num_points], device=args.device, dtype=torch.long)

    lidar_input = {
        "coord": scene["coord"],
        "grid_coord": lidar_grid,
        "feat": scene["feat"],
        "offset": offset,
    }

    with torch.no_grad():
        lidar_only = model(lidar_input)["seg_logits"]

    ipfp = IPFPFeatureBackProjector(
        image_channels=3,
        hidden_channels=64,
        out_channels=cfg.model.backbone.enc_channels[0],
        patch_size=9,
        lower_percentile=5,
        upper_percentile=95,
    ).to(args.device).eval()

    with torch.no_grad():
        extra = ipfp(
            image_chw=scene["image"],
            metric_depth_hw=scene["metric_depth"],
            points_lidar=scene["coord"],
            intrinsics=scene["intrinsics"],
            lidar_to_camera=scene["lidar_to_camera"],
            num_centers=args.num_centers,
        )

        point = Point(lidar_input)
        point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
        point.sparsify()
        point = model.backbone.embedding(point)

        extra_grid = grid_coord_from_coord(extra["coord"], origin, args.grid_size)
        merged = Point(
            {
                "coord": torch.cat([point.coord, extra["coord"]], dim=0),
                "grid_coord": torch.cat([point.grid_coord, extra_grid], dim=0),
                "feat": torch.cat([point.feat, extra["feat"]], dim=0),
                "offset": torch.tensor([point.feat.shape[0] + extra["feat"].shape[0]], device=args.device),
            }
        )
        merged.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
        merged.sparsify()
        merged = model.backbone.enc(merged)
        merged = model.backbone.dec(merged)
        ipfp_logits_for_lidar = model.seg_head(merged.feat[: args.num_points])

    print("pointcept_root", pointcept_root)
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", args.device)
    print("lidar_only_logits", tuple(lidar_only.shape))
    print("ipfp_extra_coord", tuple(extra["coord"].shape))
    print("ipfp_extra_feat", tuple(extra["feat"].shape))
    print("ipfp_training_like_logits", tuple(ipfp_logits_for_lidar.shape))
    print("smoke_test", "OK")


if __name__ == "__main__":
    main()
