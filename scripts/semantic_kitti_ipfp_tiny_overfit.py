#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from semantic_kitti_ipfp_visualize import (  # noqa: E402
    LEARNING_MAP,
    as_4x4,
    depth_color,
    draw_bev,
    draw_projection,
    grid_coord_from_coord,
    parse_calib,
    project_kitti,
)


CLASS_NAMES = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]
NUM_CLASSES = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny-overfit PTv3/IPFP on one or a few SemanticKITTI frames.")
    parser.add_argument("--root", default="/root/autodl-tmp/ipfp_repro")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--frames", nargs="+", default=["000000"])
    parser.add_argument("--eval-frames", nargs="*", default=None)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-centers", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--viz-frame-count", type=int, default=8)
    parser.add_argument("--mode", choices=["fused", "lidar-only"], default="fused")
    parser.add_argument("--eval-route", choices=["same", "lidar-only", "both"], default="same")
    parser.add_argument("--depth-mode", choices=["pseudo", "lidar-inpaint"], default="pseudo")
    parser.add_argument("--loss-mode", choices=["ce", "ce-lovasz"], default="ce")
    parser.add_argument("--lovasz-weight", type=float, default=1.0)
    parser.add_argument("--extra-feature-mode", choices=["learned", "zeros"], default="learned")
    parser.add_argument("--extra-feature-scale", type=float, default=1.0)
    parser.add_argument("--ipfp-detach", action="store_true")
    parser.add_argument("--ipfp-lower-percentile", type=float, default=5.0)
    parser.add_argument("--ipfp-upper-percentile", type=float, default=95.0)
    parser.add_argument("--ipfp-discard-probability", type=float, default=0.0)
    return parser.parse_args()


def compact_frame_label(frames: list[str]) -> str:
    if not frames:
        return "none"
    try:
        values = [int(frame) for frame in frames]
    except ValueError:
        joined = "-".join(frames)
        return joined if len(joined) <= 120 else f"{frames[0]}-{frames[-1]}_n{len(frames)}"
    width = max(len(frame) for frame in frames)
    contiguous = values == list(range(values[0], values[0] + len(values)))
    if len(values) == 1:
        return f"{values[0]:0{width}d}"
    if contiguous:
        return f"{values[0]:0{width}d}-{values[-1]:0{width}d}"
    joined = "-".join(frames)
    return joined if len(joined) <= 120 else f"{values[0]:0{width}d}-{values[-1]:0{width}d}_n{len(values)}"


def remap_labels(labels_raw: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: LEARNING_MAP.get(int(x), -1))(labels_raw).astype(np.int64)


def update_confusion_matrix(confusion: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
    valid = (target >= 0) & (target < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
    if not np.any(valid):
        return
    indices = target[valid].astype(np.int64) * NUM_CLASSES + pred[valid].astype(np.int64)
    confusion += np.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def _maybe_float(value: np.floating | float, valid: bool) -> float | None:
    return float(value) if valid else None


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    true_positive = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    class_acc = np.divide(true_positive, gt_count, out=np.zeros_like(true_positive), where=gt_count > 0)
    present_iou = union > 0
    present_acc = gt_count > 0
    total_valid = int(gt_count.sum())
    overall_acc = float(true_positive.sum() / gt_count.sum()) if total_valid else None
    fw_iou = float((gt_count[present_iou] * iou[present_iou]).sum() / gt_count.sum()) if total_valid else None
    class_metrics = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_metrics.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": _maybe_float(iou[class_id], bool(union[class_id] > 0)),
                "accuracy": _maybe_float(class_acc[class_id], bool(gt_count[class_id] > 0)),
                "tp": int(true_positive[class_id]),
                "gt_count": int(gt_count[class_id]),
                "pred_count": int(pred_count[class_id]),
                "union": int(union[class_id]),
            }
        )
    return {
        "overall_accuracy": overall_acc,
        "mean_iou": float(iou[present_iou].mean()) if np.any(present_iou) else None,
        "mean_class_accuracy": float(class_acc[present_acc].mean()) if np.any(present_acc) else None,
        "frequency_weighted_iou": fw_iou,
        "valid_points": total_valid,
        "present_iou_classes": int(present_iou.sum()),
        "present_gt_classes": int(present_acc.sum()),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


def fmt_metric(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "n/a"


def append_class_iou_notes(notes: list[str], title: str, metrics: dict | None) -> None:
    if not metrics:
        return
    notes.extend(["", title, ""])
    for item in metrics["class_metrics"]:
        if item["union"] == 0:
            continue
        notes.append(
            f"- {item['class_id']:02d} {item['class_name']}: "
            f"IoU {fmt_metric(item['iou'])}, acc {fmt_metric(item['accuracy'])}, "
            f"gt {item['gt_count']}, pred {item['pred_count']}, tp {item['tp']}"
        )


def make_pseudo_metric_depth(projected_depth: np.ndarray, height: int, width: int) -> tuple[np.ndarray, list[float]]:
    d_low, d_high = np.percentile(projected_depth, [5, 95]).astype(np.float32)
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    metric_depth = d_low + (d_high - d_low) * (0.20 + 0.55 * yy + 0.25 * xx)
    metric_depth = np.clip(metric_depth, max(float(d_low), 1e-3), max(float(d_high), 1e-3))
    return metric_depth.astype(np.float32), [float(d_low), float(d_high)]


def make_lidar_inpaint_metric_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[float]]:
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    uv_valid = uv[valid]
    depth_valid = depth[valid].astype(np.float32)
    if depth_valid.shape[0] < 16:
        return make_pseudo_metric_depth(depth_valid, height, width)

    d_low, d_high = np.percentile(depth_valid, [5, 95]).astype(np.float32)
    sparse = np.zeros((height, width), dtype=np.float32)
    xs = np.rint(uv_valid[:, 0]).astype(np.int32).clip(0, width - 1)
    ys = np.rint(uv_valid[:, 1]).astype(np.int32).clip(0, height - 1)

    order = np.argsort(depth_valid)[::-1]
    for idx in order:
        sparse[ys[idx], xs[idx]] = depth_valid[idx]

    known = sparse > 0
    if int(known.sum()) < 16:
        return make_pseudo_metric_depth(depth_valid, height, width)

    holes = (~known).astype(np.uint8) * 255
    clipped = np.clip(sparse, max(float(d_low), 1e-3), max(float(d_high), 1e-3))
    try:
        inpainted = cv2.inpaint(clipped.astype(np.float32), holes, 5.0, cv2.INPAINT_NS)
    except cv2.error:
        denom = max(float(d_high - d_low), 1e-3)
        normalized = ((clipped - float(d_low)) / denom * 255.0).clip(0, 255).astype(np.uint8)
        normalized[~known] = 0
        inpainted_u8 = cv2.inpaint(normalized, holes, 5.0, cv2.INPAINT_NS)
        inpainted = inpainted_u8.astype(np.float32) / 255.0 * denom + float(d_low)

    smoothed = cv2.GaussianBlur(inpainted.astype(np.float32), (5, 5), 0)
    smoothed[known] = sparse[known]
    metric_depth = np.clip(smoothed, max(float(d_low), 1e-3), max(float(d_high), 1e-3))
    return metric_depth.astype(np.float32), [float(d_low), float(d_high)]


def build_metric_depth(
    args: argparse.Namespace,
    uv_all: np.ndarray,
    depth_all: np.ndarray,
    valid_all: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[float]]:
    projected_depth = depth_all[valid_all]
    if projected_depth.shape[0] < 16:
        raise RuntimeError(f"Too few projected points to build metric depth: {projected_depth.shape[0]}")
    if args.depth_mode == "lidar-inpaint":
        return make_lidar_inpaint_metric_depth(uv_all[valid_all], projected_depth, height, width)
    return make_pseudo_metric_depth(projected_depth, height, width)


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if not bool(valid.any()):
        return probas.sum() * 0.0
    probas = probas[valid]
    labels = labels[valid]
    losses = []
    for class_idx in range(probas.shape[1]):
        fg = (labels == class_idx).float()
        if not bool(fg.any()):
            continue
        errors = (fg - probas[:, class_idx]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def compute_loss(logits: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    class_weights = getattr(args, "class_weights_tensor", None)
    if class_weights is not None and class_weights.device != logits.device:
        class_weights = class_weights.to(logits.device)
    ce_loss = F.cross_entropy(logits, target, ignore_index=-1, weight=class_weights)
    lovasz_loss = logits.new_tensor(0.0)
    if args.loss_mode == "ce-lovasz":
        lovasz_loss = lovasz_softmax_flat(F.softmax(logits, dim=1), target)
    total = ce_loss + args.lovasz_weight * lovasz_loss
    return total, {
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "lovasz_loss": float(lovasz_loss.detach().cpu().item()),
    }


def primary_eval_route(args: argparse.Namespace) -> str:
    if args.eval_route == "same":
        return args.mode
    return "lidar-only"


def diagnostic_eval_route(args: argparse.Namespace) -> str | None:
    if args.eval_route == "both" and args.mode == "fused":
        return "fused"
    return None


def load_sample(args: argparse.Namespace, frame: str, sample_seed: int) -> dict:
    data_root = Path(args.root) / "data/semantic_kitti/dataset/sequences" / args.sequence
    velodyne_path = data_root / "velodyne" / f"{frame}.bin"
    label_path = data_root / "labels" / f"{frame}.label"
    image_path = data_root / "image_2" / f"{frame}.png"
    calib_path = data_root / "calib.txt"
    for path in [velodyne_path, label_path, image_path, calib_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    labels_raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1) & 0xFFFF
    labels = remap_labels(labels_raw)

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
        raise RuntimeError(f"Too few projected points in frame {frame}: {valid_idx.shape[0]}")
    metric_depth, depth_p5_p95 = build_metric_depth(args, uv_all, depth_all, valid_all, new_h, new_w)

    rng = np.random.default_rng(sample_seed)
    n = min(args.num_points, scan.shape[0])
    take_valid = min(valid_idx.shape[0], max(n // 2, min(n, 512)))
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
    image_np = np.asarray(image_resized).astype(np.float32) / 255.0
    return {
        "frame": frame,
        "velodyne": str(velodyne_path),
        "image": str(image_path),
        "calib": str(calib_path),
        "raw_points": int(scan.shape[0]),
        "coord_np": coord_np,
        "intensity_np": intensity_np,
        "segment_np": segment_np,
        "image_pil": image_resized,
        "image_chw_np": np.transpose(image_np, (2, 0, 1)).copy(),
        "metric_depth_np": metric_depth.astype(np.float32),
        "intrinsics_np": intrinsics.astype(np.float32),
        "lidar_to_cam_np": lidar_to_cam_4.astype(np.float32),
        "p2": p2,
        "tr": tr,
        "uv_sel": uv_sel,
        "depth_sel": depth_sel,
        "valid_sel": valid_sel,
        "depth_mode": args.depth_mode,
        "depth_p5_p95": depth_p5_p95,
    }


def sample_to_tensors(sample: dict, device: str) -> dict:
    coord = torch.from_numpy(sample["coord_np"]).to(device)
    intensity = torch.from_numpy(sample["intensity_np"]).to(device)
    segment = torch.from_numpy(sample["segment_np"]).to(device)
    image_chw = torch.from_numpy(sample["image_chw_np"]).to(device)
    metric_depth = torch.from_numpy(sample["metric_depth_np"]).to(device)
    intrinsics = torch.from_numpy(sample["intrinsics_np"]).to(device)
    lidar_to_cam = torch.from_numpy(sample["lidar_to_cam_np"]).to(device)
    return {
        "coord": coord,
        "intensity": intensity,
        "segment": segment,
        "image_chw": image_chw,
        "metric_depth": metric_depth,
        "intrinsics": intrinsics,
        "lidar_to_cam": lidar_to_cam,
    }


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj.type)
    generator.manual_seed(seed)
    return generator


def fused_forward(
    model,
    ipfp,
    point_cls,
    tensors: dict,
    grid_size: float,
    num_centers: int,
    generator_seed: int,
    extra_feature_mode: str,
    extra_feature_scale: float,
    ipfp_detach: bool,
) -> tuple[torch.Tensor, dict]:
    coord = tensors["coord"]
    intensity = tensors["intensity"]
    feat = torch.cat([coord, intensity], dim=1)
    origin = coord.min(dim=0).values
    grid_coord = grid_coord_from_coord(coord, origin, grid_size)
    lidar_input = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=coord.device),
    }
    generator = make_generator(coord.device, generator_seed)
    extra = ipfp(
        image_chw=tensors["image_chw"],
        metric_depth_hw=tensors["metric_depth"],
        points_lidar=coord,
        intrinsics=tensors["intrinsics"],
        lidar_to_camera=tensors["lidar_to_cam"],
        num_centers=num_centers,
        generator=generator,
    )
    if ipfp_detach:
        extra = {
            **extra,
            "coord": extra["coord"].detach(),
            "feat": extra["feat"].detach(),
        }
    if extra_feature_mode == "zeros":
        extra = {
            **extra,
            "feat": torch.zeros_like(extra["feat"]),
        }
    elif extra_feature_scale != 1.0:
        extra = {
            **extra,
            "feat": extra["feat"] * extra_feature_scale,
        }
    point = point_cls(lidar_input)
    point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
    point.sparsify()
    point = model.backbone.embedding(point)
    extra_grid = grid_coord_from_coord(extra["coord"], origin, grid_size)
    merged = point_cls(
        {
            "coord": torch.cat([point.coord, extra["coord"]], dim=0),
            "grid_coord": torch.cat([point.grid_coord, extra_grid], dim=0),
            "feat": torch.cat([point.feat, extra["feat"]], dim=0),
            "offset": torch.tensor([point.feat.shape[0] + extra["feat"].shape[0]], dtype=torch.long, device=coord.device),
        }
    )
    merged.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
    merged.sparsify()
    merged = model.backbone.enc(merged)
    merged = model.backbone.dec(merged)
    logits = model.seg_head(merged.feat[: coord.shape[0]])
    return logits, extra


def lidar_only_forward(
    model,
    point_cls,
    tensors: dict,
    grid_size: float,
) -> tuple[torch.Tensor, None]:
    coord = tensors["coord"]
    intensity = tensors["intensity"]
    feat = torch.cat([coord, intensity], dim=1)
    origin = coord.min(dim=0).values
    grid_coord = grid_coord_from_coord(coord, origin, grid_size)
    lidar_input = {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([coord.shape[0]], dtype=torch.long, device=coord.device),
    }
    point = point_cls(lidar_input)
    point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
    point.sparsify()
    point = model.backbone.embedding(point)
    point = model.backbone.enc(point)
    point = model.backbone.dec(point)
    logits = model.seg_head(point.feat[: coord.shape[0]])
    return logits, None


def run_forward(
    model,
    ipfp,
    point_cls,
    tensors: dict,
    args: argparse.Namespace,
    generator_seed: int,
    route: str | None = None,
) -> tuple[torch.Tensor, dict | None]:
    active_route = route or args.mode
    if active_route == "fused":
        if ipfp is None:
            raise RuntimeError("IPFP module is required in fused mode")
        return fused_forward(
            model,
            ipfp,
            point_cls,
            tensors,
            args.grid_size,
            args.num_centers,
            generator_seed,
            args.extra_feature_mode,
            args.extra_feature_scale,
            args.ipfp_detach,
        )
    return lidar_only_forward(model, point_cls, tensors, args.grid_size)


def evaluate_samples(
    model,
    ipfp,
    point_cls,
    samples: list[dict],
    args: argparse.Namespace,
    seed_base: int,
    route: str | None = None,
) -> tuple[list[dict], np.ndarray, dict | None, dict]:
    model.eval()
    if ipfp is not None:
        ipfp.eval()
    rows = []
    first_pred = None
    first_extra = None
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for index, sample in enumerate(samples):
            tensors = sample_to_tensors(sample, args.device)
            logits, extra = run_forward(
                model,
                ipfp,
                point_cls,
                tensors,
                args,
                seed_base + index,
                route=route,
            )
            loss, loss_parts = compute_loss(logits, tensors["segment"], args)
            pred = logits.argmax(dim=1)
            valid = tensors["segment"] >= 0
            acc = (
                (pred[valid] == tensors["segment"][valid]).float().mean()
                if valid.any()
                else torch.tensor(0.0, device=pred.device)
            )
            rows.append(
                {
                    "frame": sample["frame"],
                    "loss": float(loss.detach().cpu().item()),
                    **loss_parts,
                    "valid_acc": float(acc.detach().cpu().item()),
                    "valid_labels": int(valid.detach().cpu().sum().item()),
                }
            )
            if index == 0:
                first_pred = pred.detach().cpu().numpy()
                first_extra = extra
            update_confusion_matrix(
                confusion,
                tensors["segment"].detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
            )
    if first_pred is None:
        raise RuntimeError("No samples evaluated")
    return rows, first_pred, first_extra, metrics_from_confusion(confusion)


def draw_loss_curve(records: list[dict], output: Path) -> None:
    width, height = 900, 520
    margin_l, margin_t, margin_r, margin_b = 70, 40, 30, 70
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((margin_l, margin_t, width - margin_r, height - margin_b), outline=(30, 30, 30), width=2)
    if not records:
        canvas.save(output)
        return
    steps = np.array([r["step"] for r in records], dtype=np.float32)
    losses = np.array([r["loss"] for r in records], dtype=np.float32)
    x_min, x_max = float(steps.min()), float(steps.max())
    y_min, y_max = float(losses.min()), float(losses.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    pad = max((y_max - y_min) * 0.12, 0.05)
    y_min -= pad
    y_max += pad

    def xy(step: float, loss: float) -> tuple[int, int]:
        x = margin_l + int(round((step - x_min) / (x_max - x_min) * (width - margin_l - margin_r)))
        y = height - margin_b - int(round((loss - y_min) / (y_max - y_min) * (height - margin_t - margin_b)))
        return x, y

    points = [xy(float(s), float(l)) for s, l in zip(steps, losses)]
    if len(points) > 1:
        draw.line(points, fill=(30, 95, 190), width=3)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(30, 95, 190))
    draw.text((margin_l, 12), "Tiny overfit loss", fill=(10, 10, 10))
    draw.text((margin_l, height - 42), f"step {int(x_min)} to {int(x_max)}", fill=(10, 10, 10))
    draw.text((8, margin_t), f"{y_max:.3f}", fill=(10, 10, 10))
    draw.text((8, height - margin_b - 8), f"{y_min:.3f}", fill=(10, 10, 10))
    canvas.save(output)


def save_prediction_artifacts(
    output_dir: Path,
    prefix: str,
    sample: dict,
    pred: np.ndarray,
    extra: dict | None = None,
) -> None:
    draw_bev(sample["coord_np"], sample["segment_np"], output_dir / f"{prefix}_bev_ground_truth.png", "GT labels")
    draw_bev(sample["coord_np"], pred.astype(np.int64), output_dir / f"{prefix}_bev_prediction.png", f"{prefix} prediction")
    draw_projection(
        sample["image_pil"],
        sample["uv_sel"][sample["valid_sel"]],
        depth_color(sample["depth_sel"][sample["valid_sel"]]),
        output_dir / f"{prefix}_image_lidar_depth_projection.png",
        radius=1,
    )
    if extra is not None:
        extra_np = extra["coord"].detach().cpu().numpy()
        uv_extra, _, valid_extra = project_kitti(extra_np, sample["p2"], sample["tr"], sample["image_pil"].size[::-1])
        colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (int(valid_extra.sum()), 1))
        draw_projection(
            sample["image_pil"],
            uv_extra[valid_extra],
            colors,
            output_dir / f"{prefix}_image_ipfp_extra_projection.png",
            radius=2,
        )


def save_selected_frame_visualizations(
    output_dir: Path,
    model,
    ipfp,
    point_cls,
    samples: list[dict],
    args: argparse.Namespace,
    seed_base: int,
    subdir_name: str = "selected_frame_visualizations",
    montage_name: str = "selected_frames_final_montage.png",
    prediction_label: str = "final",
    route: str | None = None,
) -> list[str]:
    count = min(max(args.viz_frame_count, 0), len(samples))
    if count == 0:
        return []
    if count == len(samples):
        indices = list(range(len(samples)))
    else:
        indices = sorted(set(np.linspace(0, len(samples) - 1, count, dtype=int).tolist()))
    vis_dir = output_dir / subdir_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    cards = []
    model.eval()
    if ipfp is not None:
        ipfp.eval()
    with torch.no_grad():
        for index in indices:
            sample = samples[index]
            tensors = sample_to_tensors(sample, args.device)
            logits, _ = run_forward(
                model,
                ipfp,
                point_cls,
                tensors,
                args,
                seed_base + index,
                route=route,
            )
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            gt_path = vis_dir / f"frame_{sample['frame']}_gt.png"
            pred_path = vis_dir / f"frame_{sample['frame']}_{prediction_label}_prediction.png"
            draw_bev(sample["coord_np"], sample["segment_np"], gt_path, f"frame {sample['frame']} GT")
            draw_bev(sample["coord_np"], pred.astype(np.int64), pred_path, f"frame {sample['frame']} {prediction_label}")
            outputs.extend([str(gt_path.relative_to(output_dir)), str(pred_path.relative_to(output_dir))])
            cards.append((sample["frame"], gt_path, pred_path))

    tile = 190
    pad = 16
    title_h = 24
    card_w = tile * 2 + pad * 3
    card_h = tile + title_h + pad * 2
    cols = min(4, len(cards))
    rows = int(np.ceil(len(cards) / cols))
    montage = Image.new("RGB", (cols * card_w, rows * card_h), (248, 248, 248))
    draw = ImageDraw.Draw(montage)
    for card_index, (frame, gt_path, pred_path) in enumerate(cards):
        col = card_index % cols
        row = card_index // cols
        x0 = col * card_w
        y0 = row * card_h
        draw.rectangle((x0 + 4, y0 + 4, x0 + card_w - 4, y0 + card_h - 4), outline=(210, 210, 210))
        draw.text((x0 + pad, y0 + 6), f"frame {frame}: GT | {prediction_label}", fill=(20, 20, 20))
        gt_img = Image.open(gt_path).convert("RGB").resize((tile, tile), Image.BILINEAR)
        pred_img = Image.open(pred_path).convert("RGB").resize((tile, tile), Image.BILINEAR)
        montage.paste(gt_img, (x0 + pad, y0 + title_h + pad))
        montage.paste(pred_img, (x0 + pad * 2 + tile, y0 + title_h + pad))
    montage_path = output_dir / montage_name
    montage.save(montage_path)
    outputs.append(str(montage_path.relative_to(output_dir)))
    return outputs


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    pointcept_root = root / "src/Pointcept"
    frame_label = compact_frame_label(args.frames)
    eval_frame_label = compact_frame_label(args.eval_frames or [])
    run_label = (
        f"train{frame_label}_eval{eval_frame_label}"
        if args.eval_frames
        else frame_label
    )
    output_dir = Path(
        args.output_dir
        or root
        / "results/semantic_kitti_repro"
        / time.strftime("%Y%m%d_%H%M%S")
        / f"tiny_overfit_{args.mode}_seq{args.sequence}_{run_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_route = primary_eval_route(args)
    diagnostic_route = diagnostic_eval_route(args)

    if str(pointcept_root) not in sys.path:
        sys.path.insert(0, str(pointcept_root))
    ipfp_root = root / "ipfp_minimal"
    if str(ipfp_root) not in sys.path:
        sys.path.insert(0, str(ipfp_root))

    from ipfp import IPFPFeatureBackProjector
    from pointcept.models import build_model
    from pointcept.models.utils.structure import Point
    from pointcept.utils.config import Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    samples = [load_sample(args, frame, args.seed + idx * 1009) for idx, frame in enumerate(args.frames)]
    if not samples:
        raise RuntimeError("No samples loaded")
    eval_samples = [
        load_sample(args, frame, args.seed + 50000 + idx * 1009)
        for idx, frame in enumerate(args.eval_frames or [])
    ]

    cfg = Config.fromfile(str(pointcept_root / "configs/nuscenes/semseg-pt-v3m1-0-base.py"))
    cfg.model.num_classes = 19
    cfg.model.backbone.enable_flash = False
    cfg.model.backbone.enc_patch_size = (64, 64, 64, 64, 64)
    cfg.model.backbone.dec_patch_size = (64, 64, 64, 64)
    model = build_model(cfg.model).to(args.device)
    ipfp = None
    if args.mode == "fused":
        ipfp = IPFPFeatureBackProjector(
            image_channels=3,
            hidden_channels=64,
            out_channels=cfg.model.backbone.enc_channels[0],
            patch_size=9,
            lower_percentile=args.ipfp_lower_percentile,
            upper_percentile=args.ipfp_upper_percentile,
            discard_probability=args.ipfp_discard_probability,
        ).to(args.device)
    trainable_parameters = list(model.parameters())
    if ipfp is not None:
        trainable_parameters += list(ipfp.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_eval_by_frame, initial_pred, initial_extra, initial_eval_metrics = evaluate_samples(
        model,
        ipfp,
        Point,
        samples,
        args,
        args.seed + 999,
        route=eval_route,
    )
    save_prediction_artifacts(output_dir, "initial", samples[0], initial_pred, initial_extra)
    holdout_initial_eval_by_frame = []
    holdout_initial_eval_metrics = None
    if eval_samples:
        (
            holdout_initial_eval_by_frame,
            holdout_initial_pred,
            holdout_initial_extra,
            holdout_initial_eval_metrics,
        ) = evaluate_samples(
            model,
            ipfp,
            Point,
            eval_samples,
            args,
            args.seed + 1999,
            route=eval_route,
        )
        save_prediction_artifacts(
            output_dir,
            "holdout_initial",
            eval_samples[0],
            holdout_initial_pred,
            holdout_initial_extra,
        )

    records = []
    log_path = output_dir / "overfit_log.jsonl"
    for step in range(1, args.steps + 1):
        sample = samples[(step - 1) % len(samples)]
        tensors = sample_to_tensors(sample, args.device)
        model.train()
        if ipfp is not None:
            ipfp.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = run_forward(
            model,
            ipfp,
            Point,
            tensors,
            args,
            args.seed + step,
        )
        loss, loss_parts = compute_loss(logits, tensors["segment"], args)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()
        pred = logits.argmax(dim=1)
        valid = tensors["segment"] >= 0
        acc = (pred[valid] == tensors["segment"][valid]).float().mean() if valid.any() else torch.tensor(0.0, device=pred.device)
        record = {
            "step": step,
            "frame": sample["frame"],
            "loss": float(loss.detach().cpu().item()),
            **loss_parts,
            "valid_acc": float(acc.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "lr": args.lr,
        }
        records.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    final_eval_by_frame, final_pred, final_extra, final_eval_metrics = evaluate_samples(
        model,
        ipfp,
        Point,
        samples,
        args,
        args.seed + 999,
        route=eval_route,
    )
    save_prediction_artifacts(output_dir, "final", samples[0], final_pred, final_extra)
    holdout_final_eval_by_frame = []
    holdout_viz_outputs = []
    holdout_final_eval_metrics = None
    diagnostic_final_eval_by_frame = []
    diagnostic_final_eval_metrics = None
    diagnostic_holdout_final_eval_by_frame = []
    diagnostic_holdout_final_eval_metrics = None
    if eval_samples:
        (
            holdout_final_eval_by_frame,
            holdout_final_pred,
            holdout_final_extra,
            holdout_final_eval_metrics,
        ) = evaluate_samples(
            model,
            ipfp,
            Point,
            eval_samples,
            args,
            args.seed + 1999,
            route=eval_route,
        )
        save_prediction_artifacts(
            output_dir,
            "holdout_final",
            eval_samples[0],
            holdout_final_pred,
            holdout_final_extra,
        )
    if diagnostic_route is not None:
        (
            diagnostic_final_eval_by_frame,
            diagnostic_final_pred,
            diagnostic_final_extra,
            diagnostic_final_eval_metrics,
        ) = evaluate_samples(
            model,
            ipfp,
            Point,
            samples,
            args,
            args.seed + 2999,
            route=diagnostic_route,
        )
        save_prediction_artifacts(
            output_dir,
            f"diagnostic_{diagnostic_route}_final",
            samples[0],
            diagnostic_final_pred,
            diagnostic_final_extra,
        )
        if eval_samples:
            (
                diagnostic_holdout_final_eval_by_frame,
                diagnostic_holdout_final_pred,
                diagnostic_holdout_final_extra,
                diagnostic_holdout_final_eval_metrics,
            ) = evaluate_samples(
                model,
                ipfp,
                Point,
                eval_samples,
                args,
                args.seed + 3999,
                route=diagnostic_route,
            )
            save_prediction_artifacts(
                output_dir,
                f"diagnostic_{diagnostic_route}_holdout_final",
                eval_samples[0],
                diagnostic_holdout_final_pred,
                diagnostic_holdout_final_extra,
            )
    selected_viz_outputs = save_selected_frame_visualizations(
        output_dir,
        model,
        ipfp,
        Point,
        samples,
        args,
        args.seed + 999,
        route=eval_route,
    )
    if eval_samples:
        holdout_viz_outputs = save_selected_frame_visualizations(
            output_dir,
            model,
            ipfp,
            Point,
            eval_samples,
            args,
            args.seed + 1999,
            subdir_name="holdout_selected_frame_visualizations",
            montage_name="holdout_selected_frames_final_montage.png",
            prediction_label="holdout_final",
            route=eval_route,
        )
    draw_loss_curve(records, output_dir / "loss_curve.png")
    initial_eval_mean_loss = float(np.mean([row["loss"] for row in initial_eval_by_frame]))
    final_eval_mean_loss = float(np.mean([row["loss"] for row in final_eval_by_frame]))
    initial_eval_mean_acc = float(np.mean([row["valid_acc"] for row in initial_eval_by_frame]))
    final_eval_mean_acc = float(np.mean([row["valid_acc"] for row in final_eval_by_frame]))
    holdout_initial_eval_mean_loss = (
        float(np.mean([row["loss"] for row in holdout_initial_eval_by_frame]))
        if holdout_initial_eval_by_frame
        else None
    )
    holdout_final_eval_mean_loss = (
        float(np.mean([row["loss"] for row in holdout_final_eval_by_frame]))
        if holdout_final_eval_by_frame
        else None
    )
    holdout_initial_eval_mean_acc = (
        float(np.mean([row["valid_acc"] for row in holdout_initial_eval_by_frame]))
        if holdout_initial_eval_by_frame
        else None
    )
    holdout_final_eval_mean_acc = (
        float(np.mean([row["valid_acc"] for row in holdout_final_eval_by_frame]))
        if holdout_final_eval_by_frame
        else None
    )
    diagnostic_final_eval_mean_loss = (
        float(np.mean([row["loss"] for row in diagnostic_final_eval_by_frame]))
        if diagnostic_final_eval_by_frame
        else None
    )
    diagnostic_final_eval_mean_acc = (
        float(np.mean([row["valid_acc"] for row in diagnostic_final_eval_by_frame]))
        if diagnostic_final_eval_by_frame
        else None
    )
    diagnostic_holdout_final_eval_mean_loss = (
        float(np.mean([row["loss"] for row in diagnostic_holdout_final_eval_by_frame]))
        if diagnostic_holdout_final_eval_by_frame
        else None
    )
    diagnostic_holdout_final_eval_mean_acc = (
        float(np.mean([row["valid_acc"] for row in diagnostic_holdout_final_eval_by_frame]))
        if diagnostic_holdout_final_eval_by_frame
        else None
    )
    outputs = [
        "overfit_log.jsonl",
        "loss_curve.png",
        "initial_bev_ground_truth.png",
        "initial_bev_prediction.png",
        "initial_image_lidar_depth_projection.png",
        "final_bev_ground_truth.png",
        "final_bev_prediction.png",
        "final_image_lidar_depth_projection.png",
        "summary.json",
        "OVERFIT_NOTES.md",
        *selected_viz_outputs,
    ]
    if initial_extra is not None:
        outputs.insert(5, "initial_image_ipfp_extra_projection.png")
    if final_extra is not None:
        outputs.insert(9, "final_image_ipfp_extra_projection.png")
    if eval_samples:
        holdout_outputs = [
            "holdout_initial_bev_ground_truth.png",
            "holdout_initial_bev_prediction.png",
            "holdout_initial_image_lidar_depth_projection.png",
            "holdout_final_bev_ground_truth.png",
            "holdout_final_bev_prediction.png",
            "holdout_final_image_lidar_depth_projection.png",
            *holdout_viz_outputs,
        ]
        if holdout_initial_eval_by_frame and holdout_initial_extra is not None:
            holdout_outputs.insert(3, "holdout_initial_image_ipfp_extra_projection.png")
        if holdout_final_eval_by_frame and holdout_final_extra is not None:
            holdout_outputs.insert(7, "holdout_final_image_ipfp_extra_projection.png")
        outputs.extend(holdout_outputs)
    if diagnostic_route is not None:
        diagnostic_prefix = f"diagnostic_{diagnostic_route}_final"
        outputs.extend(
            [
                f"{diagnostic_prefix}_bev_ground_truth.png",
                f"{diagnostic_prefix}_bev_prediction.png",
                f"{diagnostic_prefix}_image_lidar_depth_projection.png",
            ]
        )
        if diagnostic_final_eval_by_frame and diagnostic_final_extra is not None:
            outputs.append(f"{diagnostic_prefix}_image_ipfp_extra_projection.png")
        if eval_samples:
            diagnostic_holdout_prefix = f"diagnostic_{diagnostic_route}_holdout_final"
            outputs.extend(
                [
                    f"{diagnostic_holdout_prefix}_bev_ground_truth.png",
                    f"{diagnostic_holdout_prefix}_bev_prediction.png",
                    f"{diagnostic_holdout_prefix}_image_lidar_depth_projection.png",
                ]
            )
            if diagnostic_holdout_final_eval_by_frame and diagnostic_holdout_final_extra is not None:
                outputs.append(f"{diagnostic_holdout_prefix}_image_ipfp_extra_projection.png")

    summary = {
        "status": "OK",
        "sequence": args.sequence,
        "mode": args.mode,
        "eval_route": args.eval_route,
        "primary_eval_route": eval_route,
        "diagnostic_eval_route": diagnostic_route,
        "depth_mode": args.depth_mode,
        "loss_mode": args.loss_mode,
        "lovasz_weight": args.lovasz_weight,
        "frames": args.frames,
        "eval_frames": args.eval_frames or [],
        "frame_label": frame_label,
        "eval_frame_label": eval_frame_label if args.eval_frames else None,
        "num_points": args.num_points,
        "num_centers": args.num_centers,
        "extra_feature_mode": args.extra_feature_mode,
        "extra_feature_scale": args.extra_feature_scale,
        "ipfp_detach": args.ipfp_detach,
        "ipfp_lower_percentile": args.ipfp_lower_percentile,
        "ipfp_upper_percentile": args.ipfp_upper_percentile,
        "ipfp_discard_probability": args.ipfp_discard_probability,
        "image_width": args.image_width,
        "viz_frame_count": args.viz_frame_count,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "initial_eval_loss": initial_eval_by_frame[0]["loss"],
        "final_eval_loss": final_eval_by_frame[0]["loss"],
        "initial_eval_acc": initial_eval_by_frame[0]["valid_acc"],
        "final_eval_acc": final_eval_by_frame[0]["valid_acc"],
        "initial_eval_mean_loss": initial_eval_mean_loss,
        "final_eval_mean_loss": final_eval_mean_loss,
        "initial_eval_mean_acc": initial_eval_mean_acc,
        "final_eval_mean_acc": final_eval_mean_acc,
        "initial_eval_mean_iou": initial_eval_metrics["mean_iou"],
        "final_eval_mean_iou": final_eval_metrics["mean_iou"],
        "initial_eval_overall_acc": initial_eval_metrics["overall_accuracy"],
        "final_eval_overall_acc": final_eval_metrics["overall_accuracy"],
        "initial_eval_by_frame": initial_eval_by_frame,
        "final_eval_by_frame": final_eval_by_frame,
        "holdout_initial_eval_mean_loss": holdout_initial_eval_mean_loss,
        "holdout_final_eval_mean_loss": holdout_final_eval_mean_loss,
        "holdout_initial_eval_mean_acc": holdout_initial_eval_mean_acc,
        "holdout_final_eval_mean_acc": holdout_final_eval_mean_acc,
        "holdout_initial_eval_mean_iou": holdout_initial_eval_metrics["mean_iou"]
        if holdout_initial_eval_metrics
        else None,
        "holdout_final_eval_mean_iou": holdout_final_eval_metrics["mean_iou"]
        if holdout_final_eval_metrics
        else None,
        "holdout_initial_eval_overall_acc": holdout_initial_eval_metrics["overall_accuracy"]
        if holdout_initial_eval_metrics
        else None,
        "holdout_final_eval_overall_acc": holdout_final_eval_metrics["overall_accuracy"]
        if holdout_final_eval_metrics
        else None,
        "holdout_initial_eval_by_frame": holdout_initial_eval_by_frame,
        "holdout_final_eval_by_frame": holdout_final_eval_by_frame,
        "diagnostic_final_eval_mean_loss": diagnostic_final_eval_mean_loss,
        "diagnostic_final_eval_mean_acc": diagnostic_final_eval_mean_acc,
        "diagnostic_final_eval_mean_iou": diagnostic_final_eval_metrics["mean_iou"]
        if diagnostic_final_eval_metrics
        else None,
        "diagnostic_final_eval_overall_acc": diagnostic_final_eval_metrics["overall_accuracy"]
        if diagnostic_final_eval_metrics
        else None,
        "diagnostic_holdout_final_eval_mean_loss": diagnostic_holdout_final_eval_mean_loss,
        "diagnostic_holdout_final_eval_mean_acc": diagnostic_holdout_final_eval_mean_acc,
        "diagnostic_holdout_final_eval_mean_iou": diagnostic_holdout_final_eval_metrics["mean_iou"]
        if diagnostic_holdout_final_eval_metrics
        else None,
        "diagnostic_holdout_final_eval_overall_acc": diagnostic_holdout_final_eval_metrics["overall_accuracy"]
        if diagnostic_holdout_final_eval_metrics
        else None,
        "diagnostic_final_eval_by_frame": diagnostic_final_eval_by_frame,
        "diagnostic_holdout_final_eval_by_frame": diagnostic_holdout_final_eval_by_frame,
        "metrics": {
            "train_initial": initial_eval_metrics,
            "train_final": final_eval_metrics,
            "holdout_initial": holdout_initial_eval_metrics,
            "holdout_final": holdout_final_eval_metrics,
            "diagnostic_train_final": diagnostic_final_eval_metrics,
            "diagnostic_holdout_final": diagnostic_holdout_final_eval_metrics,
        },
        "train_first_loss": records[0]["loss"] if records else None,
        "train_final_loss": records[-1]["loss"] if records else None,
        "train_best_loss": min((r["loss"] for r in records), default=None),
        "train_final_acc": records[-1]["valid_acc"] if records else None,
        "raw_points_first_frame": samples[0]["raw_points"],
        "valid_labels_first_sample": int((samples[0]["segment_np"] >= 0).sum()),
        "depth_p5_p95_first_frame": samples[0]["depth_p5_p95"],
        "cuda_max_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if args.device == "cuda"
        else 0.0,
        "outputs": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    notes = [
        f"# SemanticKITTI {args.mode} Tiny Overfit",
        "",
        "## What this validates",
        "",
        "- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.",
        "- Projects LiDAR into the image frame for visualization sanity checks.",
        "- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.",
        "- Checks whether a tiny training set can be memorized at least partially.",
        "",
        "## Important caveat",
        "",
        "This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.",
        "",
        "## Summary",
        "",
        f"- Train route: {args.mode}",
        f"- Primary eval route: {eval_route}",
        f"- Diagnostic eval route: {diagnostic_route or 'none'}",
        f"- Depth mode: {args.depth_mode}, first-frame p5/p95: {samples[0]['depth_p5_p95']}",
        f"- Loss mode: {args.loss_mode}, Lovasz weight: {args.lovasz_weight}",
        f"- Extra feature mode/scale: {args.extra_feature_mode}, {args.extra_feature_scale:.3f}",
        f"- IPFP depth percentile range: {args.ipfp_lower_percentile:.1f}-{args.ipfp_upper_percentile:.1f}",
        f"- IPFP discard probability: {args.ipfp_discard_probability:.3f}",
        f"- Initial eval loss, first frame: {summary['initial_eval_loss']:.6f}",
        f"- Final eval loss, first frame: {summary['final_eval_loss']:.6f}",
        f"- Initial eval mean loss: {summary['initial_eval_mean_loss']:.6f}",
        f"- Final eval mean loss: {summary['final_eval_mean_loss']:.6f}",
        f"- Initial eval mean accuracy: {summary['initial_eval_mean_acc']:.6f}",
        f"- Final eval mean accuracy: {summary['final_eval_mean_acc']:.6f}",
        f"- Initial eval mIoU: {fmt_metric(summary['initial_eval_mean_iou'])}",
        f"- Final eval mIoU: {fmt_metric(summary['final_eval_mean_iou'])}",
        f"- Initial eval overall accuracy: {fmt_metric(summary['initial_eval_overall_acc'])}",
        f"- Final eval overall accuracy: {fmt_metric(summary['final_eval_overall_acc'])}",
        f"- First train loss: {summary['train_first_loss']:.6f}",
        f"- Final train loss: {summary['train_final_loss']:.6f}",
        f"- Best train loss: {summary['train_best_loss']:.6f}",
        f"- Final train valid accuracy: {summary['train_final_acc']:.6f}",
        f"- CUDA peak memory GB: {summary['cuda_max_memory_gb']:.3f}",
        f"- Selected visualization frames: {len(selected_viz_outputs) // 2 if selected_viz_outputs else 0}",
        "",
    ]
    if holdout_final_eval_by_frame:
        notes.extend(
            [
                "## Holdout Eval",
                "",
                f"- Holdout initial mean loss: {holdout_initial_eval_mean_loss:.6f}",
                f"- Holdout final mean loss: {holdout_final_eval_mean_loss:.6f}",
                f"- Holdout initial mean accuracy: {holdout_initial_eval_mean_acc:.6f}",
                f"- Holdout final mean accuracy: {holdout_final_eval_mean_acc:.6f}",
                f"- Holdout initial mIoU: {fmt_metric(summary['holdout_initial_eval_mean_iou'])}",
                f"- Holdout final mIoU: {fmt_metric(summary['holdout_final_eval_mean_iou'])}",
                f"- Holdout initial overall accuracy: {fmt_metric(summary['holdout_initial_eval_overall_acc'])}",
                f"- Holdout final overall accuracy: {fmt_metric(summary['holdout_final_eval_overall_acc'])}",
                "",
            ]
        )
    if diagnostic_route is not None and diagnostic_final_eval_by_frame:
        notes.extend(
            [
                f"## Diagnostic {diagnostic_route} Eval",
                "",
                f"- Train final mean loss: {diagnostic_final_eval_mean_loss:.6f}",
                f"- Train final mean accuracy: {diagnostic_final_eval_mean_acc:.6f}",
                f"- Train final mIoU: {fmt_metric(summary['diagnostic_final_eval_mean_iou'])}",
                f"- Train final overall accuracy: {fmt_metric(summary['diagnostic_final_eval_overall_acc'])}",
                "",
            ]
        )
        if diagnostic_holdout_final_eval_by_frame:
            notes.extend(
                [
                    f"- Holdout final mean loss: {diagnostic_holdout_final_eval_mean_loss:.6f}",
                    f"- Holdout final mean accuracy: {diagnostic_holdout_final_eval_mean_acc:.6f}",
                    f"- Holdout final mIoU: {fmt_metric(summary['diagnostic_holdout_final_eval_mean_iou'])}",
                    f"- Holdout final overall accuracy: {fmt_metric(summary['diagnostic_holdout_final_eval_overall_acc'])}",
                    "",
                ]
            )
    notes.extend(
        [
            "## Train-Frame Eval",
            "",
        ]
    )
    for before, after in zip(initial_eval_by_frame, final_eval_by_frame):
            notes.append(
                f"- Frame {before['frame']}: loss {before['loss']:.6f} -> {after['loss']:.6f}, "
                f"accuracy {before['valid_acc']:.6f} -> {after['valid_acc']:.6f}"
            )
    append_class_iou_notes(notes, "## Train Final Class IoU", final_eval_metrics)
    if holdout_final_eval_by_frame:
        notes.extend(
            [
                "",
                "## Holdout Per-Frame Eval",
                "",
            ]
        )
        for before, after in zip(holdout_initial_eval_by_frame, holdout_final_eval_by_frame):
            notes.append(
                f"- Frame {before['frame']}: loss {before['loss']:.6f} -> {after['loss']:.6f}, "
                f"accuracy {before['valid_acc']:.6f} -> {after['valid_acc']:.6f}"
            )
        append_class_iou_notes(notes, "## Holdout Final Class IoU", holdout_final_eval_metrics)
    notes.append("")
    (output_dir / "OVERFIT_NOTES.md").write_text("\n".join(notes))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
