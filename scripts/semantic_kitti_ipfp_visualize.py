#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


LEARNING_MAP = {
    0: -1,
    1: -1,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: -1,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: -1,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4,
}

CLASS_COLORS = np.array(
    [
        [245, 150, 100],
        [245, 230, 100],
        [150, 60, 30],
        [180, 30, 80],
        [255, 0, 0],
        [30, 30, 255],
        [200, 40, 255],
        [90, 30, 150],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [75, 0, 175],
        [0, 200, 255],
        [50, 120, 255],
        [0, 175, 0],
        [0, 60, 135],
        [80, 240, 150],
        [150, 240, 255],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SemanticKITTI + image + IPFP visualization closed-loop.")
    parser.add_argument("--root", default="/root/autodl-tmp/ipfp_repro")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--frame", default="000000")
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--num-centers", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def parse_calib(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = np.array([float(v) for v in value.strip().split()], dtype=np.float32)
        if key.startswith("P"):
            out[key] = values.reshape(3, 4)
        elif key == "Tr":
            out[key] = values.reshape(3, 4)
        else:
            out[key] = values
    return out


def as_4x4(mat34: np.ndarray) -> np.ndarray:
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return np.vstack([mat34, bottom]).astype(np.float32)


def project_kitti(points_xyz: np.ndarray, p2: np.ndarray, tr: np.ndarray, image_hw: tuple[int, int]):
    h, w = image_hw
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    cam = (tr @ points_h.T).T
    cam_h = np.concatenate([cam, ones], axis=1)
    uvw = (p2 @ cam_h.T).T
    depth = cam[:, 2]
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    valid = (
        (depth > 1e-6)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return uv, depth, valid


def depth_color(depth: np.ndarray) -> np.ndarray:
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth, [5, 95])
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    r = (255 * t).astype(np.uint8)
    g = (255 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8)
    b = (255 * (1.0 - t)).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


def draw_projection(image: Image.Image, uv: np.ndarray, colors: np.ndarray, output: Path, radius: int = 1) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for (u, v), color in zip(uv, colors):
        x, y = int(round(float(u))), int(round(float(v)))
        c = tuple(int(vv) for vv in color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=c)
    canvas.save(output)


def draw_bev(points: np.ndarray, labels: np.ndarray, output: Path, title: str, size: int = 900, extent: float = 55.0) -> None:
    canvas = Image.new("RGB", (size, size), (12, 12, 12))
    pix = canvas.load()
    valid = (points[:, 0] >= -extent) & (points[:, 0] <= extent) & (points[:, 1] >= -extent) & (points[:, 1] <= extent)
    pts = points[valid]
    labs = labels[valid]
    for p, lab in zip(pts, labs):
        x = int((p[1] + extent) / (2 * extent) * (size - 1))
        y = int((extent - p[0]) / (2 * extent) * (size - 1))
        if 0 <= lab < len(CLASS_COLORS):
            color = tuple(int(v) for v in CLASS_COLORS[int(lab)])
        else:
            color = (80, 80, 80)
        pix[x, y] = color
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    canvas.save(output)


def grid_coord_from_coord(coord: torch.Tensor, origin: torch.Tensor, grid_size: float) -> torch.Tensor:
    grid = torch.div(coord - origin, grid_size, rounding_mode="trunc").int()
    return grid.clamp_min(0)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    pointcept_root = root / "src/Pointcept"
    data_root = root / "data/semantic_kitti/dataset/sequences" / args.sequence
    output_dir = Path(args.output_dir or root / "results/semantic_kitti_ipfp" / f"seq{args.sequence}_{args.frame}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(pointcept_root))
    sys.path.insert(0, str(root / "ipfp_minimal"))

    from ipfp import IPFPFeatureBackProjector
    from pointcept.models import build_model
    from pointcept.models.utils.structure import Point
    from pointcept.utils.config import Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    velodyne_path = data_root / "velodyne" / f"{args.frame}.bin"
    label_path = data_root / "labels" / f"{args.frame}.label"
    image_path = data_root / "image_2" / f"{args.frame}.png"
    calib_path = data_root / "calib.txt"
    for path in [velodyne_path, label_path, image_path, calib_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    labels_raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1) & 0xFFFF
    labels = np.vectorize(lambda x: LEARNING_MAP.get(int(x), -1))(labels_raw).astype(np.int64)

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    new_w = args.image_width
    new_h = int(round(orig_h * new_w / orig_w))
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)

    calib = parse_calib(calib_path)
    p2 = calib["P2"].copy()
    p2[0, :] *= new_w / orig_w
    p2[1, :] *= new_h / orig_h
    tr = calib["Tr"].copy()
    lidar_to_cam_4 = as_4x4(tr)
    intrinsics = p2[:, :3].copy()

    uv_all, depth_all, valid_all = project_kitti(scan[:, :3], p2, tr, (new_h, new_w))
    valid_idx = np.flatnonzero(valid_all)
    invalid_idx = np.flatnonzero(~valid_all)
    if valid_idx.shape[0] < 64:
        raise RuntimeError(f"Too few projected points: {valid_idx.shape[0]}")

    rng = np.random.default_rng(13)
    n = min(args.num_points, scan.shape[0])
    take_valid = min(valid_idx.shape[0], max(n // 2, min(n, 1024)))
    chosen_valid = rng.choice(valid_idx, size=take_valid, replace=False)
    remaining = n - take_valid
    if remaining > 0:
        pool = invalid_idx if invalid_idx.shape[0] >= remaining else np.setdiff1d(np.arange(scan.shape[0]), chosen_valid)
        chosen_other = rng.choice(pool, size=remaining, replace=pool.shape[0] < remaining)
        chosen = np.concatenate([chosen_valid, chosen_other])
    else:
        chosen = chosen_valid
    rng.shuffle(chosen)

    coord_np = scan[chosen, :3].astype(np.float32)
    intensity_np = scan[chosen, 3:4].astype(np.float32)
    segment_np = labels[chosen].astype(np.int64)
    uv_sel, depth_sel, valid_sel = project_kitti(coord_np, p2, tr, (new_h, new_w))
    projected_depth = depth_sel[valid_sel]
    d_low, d_high = np.percentile(projected_depth, [5, 95]).astype(np.float32)
    yy = np.linspace(0.0, 1.0, new_h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, new_w, dtype=np.float32)[None, :]
    metric_depth = d_low + (d_high - d_low) * (0.20 + 0.55 * yy + 0.25 * xx)
    metric_depth = np.clip(metric_depth, max(float(d_low), 1e-3), max(float(d_high), 1e-3))

    image_np = np.asarray(image_resized).astype(np.float32) / 255.0
    coord = torch.from_numpy(coord_np).to(args.device)
    intensity = torch.from_numpy(intensity_np).to(args.device)
    segment = torch.from_numpy(segment_np).to(args.device)
    image_chw = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(args.device)
    metric_depth_hw = torch.from_numpy(metric_depth).to(args.device)
    intrinsics_t = torch.from_numpy(intrinsics).to(args.device)
    lidar_to_cam_t = torch.from_numpy(lidar_to_cam_4).to(args.device)

    cfg = Config.fromfile(str(pointcept_root / "configs/nuscenes/semseg-pt-v3m1-0-base.py"))
    cfg.model.num_classes = 19
    cfg.model.backbone.enable_flash = False
    cfg.model.backbone.enc_patch_size = (64, 64, 64, 64, 64)
    cfg.model.backbone.dec_patch_size = (64, 64, 64, 64)
    model = build_model(cfg.model).to(args.device).eval()

    feat = torch.cat([coord, intensity], dim=1)
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
        lidar_loss = F.cross_entropy(lidar_logits, segment, ignore_index=-1)
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
        fused_loss = F.cross_entropy(fused_logits, segment, ignore_index=-1)

    pred = torch.argmax(fused_logits, dim=1).detach().cpu().numpy().astype(np.int64)
    draw_projection(
        image_resized,
        uv_sel[valid_sel],
        depth_color(depth_sel[valid_sel]),
        output_dir / "image_lidar_depth_projection.png",
        radius=1,
    )
    extra_np = extra["coord"].detach().cpu().numpy()
    uv_extra, depth_extra, valid_extra = project_kitti(extra_np, p2, tr, (new_h, new_w))
    draw_projection(
        image_resized,
        uv_extra[valid_extra],
        np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (int(valid_extra.sum()), 1)),
        output_dir / "image_ipfp_extra_projection.png",
        radius=2,
    )
    draw_bev(coord_np, segment_np, output_dir / "bev_ground_truth_labels.png", "SemanticKITTI GT labels")
    draw_bev(coord_np, pred, output_dir / "bev_random_init_prediction.png", "Random-init fused prediction")
    draw_bev(extra_np, np.full(extra_np.shape[0], 4, dtype=np.int64), output_dir / "bev_ipfp_extra_points.png", "IPFP extra points")

    summary = {
        "status": "OK",
        "sequence": args.sequence,
        "frame": args.frame,
        "velodyne": str(velodyne_path),
        "image": str(image_path),
        "calib": str(calib_path),
        "raw_points": int(scan.shape[0]),
        "selected_points": int(coord_np.shape[0]),
        "projected_selected_points": int(valid_sel.sum()),
        "valid_labels": int((segment_np >= 0).sum()),
        "image_shape": [3, new_h, new_w],
        "depth_p5_p95": [float(d_low), float(d_high)],
        "lidar_logits_shape": list(lidar_logits.shape),
        "lidar_ce_loss": float(lidar_loss.item()),
        "ipfp_extra_coord_shape": list(extra["coord"].shape),
        "ipfp_extra_feat_shape": list(extra["feat"].shape),
        "fused_logits_shape": list(fused_logits.shape),
        "fused_ce_loss": float(fused_loss.item()),
        "extra_points_projected": int(valid_extra.sum()),
        "cuda_max_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if args.device == "cuda"
        else 0.0,
        "outputs": [
            "image_lidar_depth_projection.png",
            "image_ipfp_extra_projection.png",
            "bev_ground_truth_labels.png",
            "bev_random_init_prediction.png",
            "bev_ipfp_extra_points.png",
            "summary.json",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
