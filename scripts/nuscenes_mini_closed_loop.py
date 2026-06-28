#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix


LEARNING_MAP = {
    0: -1,
    1: -1,
    2: 6,
    3: 6,
    4: 6,
    5: -1,
    6: 6,
    7: -1,
    8: -1,
    9: 0,
    10: -1,
    11: -1,
    12: 7,
    13: -1,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: -1,
    20: -1,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: -1,
    30: 15,
    31: -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data nuScenes-mini IPFP/Pointcept closed-loop smoke test.")
    parser.add_argument("--root", default="/root/autodl-tmp/ipfp_repro")
    parser.add_argument("--pointcept-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-centers", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def as_matrix(record: dict, inverse: bool = False) -> np.ndarray:
    return transform_matrix(record["translation"], Quaternion(record["rotation"]), inverse=inverse)


def lidar_to_camera_matrix(nusc: NuScenes, lidar_token: str, camera_token: str) -> np.ndarray:
    lidar_sd = nusc.get("sample_data", lidar_token)
    cam_sd = nusc.get("sample_data", camera_token)

    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    return (
        as_matrix(cam_cs, inverse=True)
        @ as_matrix(cam_pose, inverse=True)
        @ as_matrix(lidar_pose, inverse=False)
        @ as_matrix(lidar_cs, inverse=False)
    )


def project_np(points_lidar: np.ndarray, intrinsics: np.ndarray, lidar_to_cam: np.ndarray, image_hw: tuple[int, int]):
    h, w = image_hw
    ones = np.ones((points_lidar.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points_lidar, ones], axis=1)
    pts_cam = (lidar_to_cam @ pts_h.T).T[:, :3]
    depth = pts_cam[:, 2]
    uvh = (intrinsics @ pts_cam.T).T
    uv = uvh[:, :2] / np.clip(uvh[:, 2:3], 1e-6, None)
    valid = (
        (depth > 1e-6)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= w - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= h - 1)
    )
    return uv, depth, valid


def grid_coord_from_coord(coord: torch.Tensor, origin: torch.Tensor, grid_size: float) -> torch.Tensor:
    grid = torch.div(coord - origin, grid_size, rounding_mode="trunc").int()
    return grid.clamp_min(0)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    pointcept_root = Path(args.pointcept_root or root / "src/Pointcept")
    data_root = Path(args.data_root or root / "data/nuscenes_mini/raw")

    sys.path.insert(0, str(pointcept_root))
    sys.path.insert(0, str(root / "ipfp_minimal"))

    from ipfp import IPFPFeatureBackProjector
    from pointcept.models import build_model
    from pointcept.models.utils.structure import Point
    from pointcept.utils.config import Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    nusc = NuScenes(version="v1.0-mini", dataroot=str(data_root), verbose=False)
    sample = nusc.sample[args.sample_index]
    lidar_token = sample["data"]["LIDAR_TOP"]
    camera_token = sample["data"]["CAM_FRONT"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    cam_sd = nusc.get("sample_data", camera_token)

    lidar_path = data_root / lidar_sd["filename"]
    image_path = data_root / cam_sd["filename"]
    label_path = data_root / nusc.get("lidarseg", lidar_token)["filename"]

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    labels_raw = np.fromfile(label_path, dtype=np.uint8).reshape(-1)
    if labels_raw.shape[0] != points.shape[0]:
        raise RuntimeError(f"label/point count mismatch: {labels_raw.shape[0]} vs {points.shape[0]}")

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    new_w = args.image_width
    new_h = int(round(orig_h * new_w / orig_w))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    intrinsics = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)
    intrinsics[0, :] *= new_w / orig_w
    intrinsics[1, :] *= new_h / orig_h
    lidar_to_cam = lidar_to_camera_matrix(nusc, lidar_token, camera_token).astype(np.float32)

    uv, depth, valid = project_np(points[:, :3], intrinsics, lidar_to_cam, (new_h, new_w))
    valid_idx = np.flatnonzero(valid)
    invalid_idx = np.flatnonzero(~valid)
    if valid_idx.shape[0] < 32:
        raise RuntimeError(f"not enough CAM_FRONT projected LiDAR points: {valid_idx.shape[0]}")

    rng = np.random.default_rng(7)
    n = min(args.num_points, points.shape[0])
    take_valid = min(valid_idx.shape[0], max(n // 2, min(n, 512)))
    chosen_valid = rng.choice(valid_idx, size=take_valid, replace=False)
    remaining = n - chosen_valid.shape[0]
    if remaining > 0:
        pool = invalid_idx if invalid_idx.shape[0] >= remaining else np.setdiff1d(np.arange(points.shape[0]), chosen_valid)
        chosen_other = rng.choice(pool, size=remaining, replace=pool.shape[0] < remaining)
        chosen = np.concatenate([chosen_valid, chosen_other])
    else:
        chosen = chosen_valid
    rng.shuffle(chosen)

    coord_np = points[chosen, :3].astype(np.float32)
    strength_np = (points[chosen, 3:4] / 255.0).astype(np.float32)
    mapped_labels = np.vectorize(LEARNING_MAP.__getitem__)(labels_raw[chosen]).astype(np.int64)
    selected_uv, selected_depth, selected_valid = project_np(coord_np, intrinsics, lidar_to_cam, (new_h, new_w))
    projected_depth = selected_depth[selected_valid]

    d_low, d_high = np.percentile(projected_depth, [5, 95]).astype(np.float32)
    yy = np.linspace(0.0, 1.0, new_h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, new_w, dtype=np.float32)[None, :]
    metric_depth = d_low + (d_high - d_low) * (0.25 + 0.55 * yy + 0.20 * xx)
    metric_depth = np.clip(metric_depth, max(float(d_low), 1e-3), max(float(d_high), 1e-3))

    image_np = np.asarray(image).astype(np.float32) / 255.0
    coord = torch.from_numpy(coord_np).to(args.device)
    strength = torch.from_numpy(strength_np).to(args.device)
    segment = torch.from_numpy(mapped_labels).to(args.device)
    image_chw = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(args.device)
    metric_depth_hw = torch.from_numpy(metric_depth).to(args.device)
    intrinsics_t = torch.from_numpy(intrinsics).to(args.device)
    lidar_to_cam_t = torch.from_numpy(lidar_to_cam).to(args.device)

    cfg = Config.fromfile(str(pointcept_root / "configs/nuscenes/semseg-pt-v3m1-0-base.py"))
    cfg.model.backbone.enable_flash = False
    cfg.model.backbone.enc_patch_size = (64, 64, 64, 64, 64)
    cfg.model.backbone.dec_patch_size = (64, 64, 64, 64)
    model = build_model(cfg.model).to(args.device).eval()

    feat = torch.cat([coord, strength], dim=-1)
    origin = coord.min(dim=0).values
    grid_coord = grid_coord_from_coord(coord, origin, args.grid_size)
    lidar_input = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=args.device),
    }

    ipfp = IPFPFeatureBackProjector(
        image_channels=3,
        hidden_channels=64,
        out_channels=cfg.model.backbone.enc_channels[0],
        patch_size=9,
        lower_percentile=5,
        upper_percentile=95,
    ).to(args.device).eval()

    with torch.no_grad():
        lidar_logits = model(lidar_input)["seg_logits"]
        ce_loss = F.cross_entropy(lidar_logits, segment, ignore_index=-1)

        extra = ipfp(
            image_chw=image_chw,
            metric_depth_hw=metric_depth_hw,
            points_lidar=coord,
            intrinsics=intrinsics_t,
            lidar_to_camera=lidar_to_cam_t,
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
                "offset": torch.tensor([point.feat.shape[0] + extra["feat"].shape[0]], dtype=torch.long, device=args.device),
            }
        )
        merged.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
        merged.sparsify()
        merged = model.backbone.enc(merged)
        merged = model.backbone.dec(merged)
        fused_logits = model.seg_head(merged.feat[: coord.shape[0]])
        fused_ce_loss = F.cross_entropy(fused_logits, segment, ignore_index=-1)

    valid_label_count = int((segment >= 0).sum().item())
    print("closed_loop", "OK")
    print("sample_token", sample["token"])
    print("lidar_file", lidar_sd["filename"])
    print("image_file", cam_sd["filename"])
    print("raw_points", points.shape[0])
    print("selected_points", coord.shape[0])
    print("selected_cam_front_projected", int(selected_valid.sum()))
    print("valid_labels", valid_label_count)
    print("image_chw", tuple(image_chw.shape))
    print("depth_range_p5_p95", (float(d_low), float(d_high)))
    print("lidar_logits", tuple(lidar_logits.shape), "loss", float(ce_loss.item()))
    print("ipfp_extra_coord", tuple(extra["coord"].shape))
    print("ipfp_extra_feat", tuple(extra["feat"].shape))
    print("fused_logits", tuple(fused_logits.shape), "loss", float(fused_ce_loss.item()))
    print("cuda_max_memory_gb", round(torch.cuda.max_memory_allocated() / (1024**3), 3) if args.device == "cuda" else 0.0)


if __name__ == "__main__":
    main()
