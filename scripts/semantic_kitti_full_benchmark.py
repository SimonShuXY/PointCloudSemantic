#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from semantic_kitti_ipfp_tiny_overfit import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    append_class_iou_notes,
    compute_loss,
    diagnostic_eval_route,
    fmt_metric,
    metrics_from_confusion,
    primary_eval_route,
    remap_labels,
    run_forward,
    update_confusion_matrix,
)
from semantic_kitti_ipfp_visualize import (  # noqa: E402
    as_4x4,
    draw_bev,
    parse_calib,
    project_kitti,
)


TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SemanticKITTI full-frame validation benchmark scaffold.")
    parser.add_argument("--root", default="/root/autodl-tmp/ipfp_repro")
    parser.add_argument("--train-sequences", nargs="+", default=TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", nargs="+", default=["08"])
    parser.add_argument("--train-frame-stride", type=int, default=1)
    parser.add_argument("--val-frame-stride", type=int, default=1)
    parser.add_argument("--train-frame-limit", type=int, default=None)
    parser.add_argument("--val-frame-limit", type=int, default=None)
    parser.add_argument("--train-sample-points", type=int, default=8192)
    parser.add_argument("--eval-chunk-points", type=int, default=16384)
    parser.add_argument("--num-centers", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--viz-frame-count", type=int, default=8)
    parser.add_argument("--mode", choices=["fused", "lidar-only"], default="lidar-only")
    parser.add_argument("--eval-route", choices=["same", "lidar-only", "both"], default="same")
    parser.add_argument("--depth-mode", choices=["pseudo", "lidar-inpaint"], default="lidar-inpaint")
    parser.add_argument("--loss-mode", choices=["ce", "ce-lovasz"], default="ce-lovasz")
    parser.add_argument("--lovasz-weight", type=float, default=1.0)
    parser.add_argument("--extra-feature-mode", choices=["learned", "zeros"], default="learned")
    parser.add_argument("--extra-feature-scale", type=float, default=0.1)
    parser.add_argument("--ipfp-detach", action="store_true")
    parser.add_argument("--ipfp-lower-percentile", type=float, default=20.0)
    parser.add_argument("--ipfp-upper-percentile", type=float, default=99.0)
    parser.add_argument("--ipfp-discard-probability", type=float, default=0.2)
    return parser.parse_args()


def apply_stride_limit(items: list[tuple[str, str]], stride: int, limit: int | None) -> list[tuple[str, str]]:
    stride = max(stride, 1)
    items = items[::stride]
    if limit is not None:
        items = items[: max(limit, 0)]
    return items


def list_frames(root: Path, sequences: list[str], stride: int, limit: int | None) -> list[tuple[str, str]]:
    all_frames: list[tuple[str, str]] = []
    seq_root = root / "data/semantic_kitti/dataset/sequences"
    for sequence in sequences:
        velodyne_dir = seq_root / sequence / "velodyne"
        label_dir = seq_root / sequence / "labels"
        if not velodyne_dir.exists():
            raise FileNotFoundError(velodyne_dir)
        frames = []
        for path in sorted(velodyne_dir.glob("*.bin")):
            if (label_dir / f"{path.stem}.label").exists():
                frames.append((sequence, path.stem))
        all_frames.extend(frames)
    return apply_stride_limit(all_frames, stride, limit)


def load_frame(args: argparse.Namespace, sequence: str, frame: str, need_image: bool) -> dict:
    data_root = Path(args.root) / "data/semantic_kitti/dataset/sequences" / sequence
    velodyne_path = data_root / "velodyne" / f"{frame}.bin"
    label_path = data_root / "labels" / f"{frame}.label"
    calib_path = data_root / "calib.txt"
    for path in [velodyne_path, label_path, calib_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    scan = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    labels_raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1) & 0xFFFF
    labels = remap_labels(labels_raw)
    if scan.shape[0] != labels.shape[0]:
        raise RuntimeError(f"label/point count mismatch for {sequence}/{frame}: {labels.shape[0]} vs {scan.shape[0]}")

    loaded = {
        "sequence": sequence,
        "frame": frame,
        "name": f"{sequence}_{frame}",
        "velodyne": str(velodyne_path),
        "label": str(label_path),
        "calib": str(calib_path),
        "scan": scan,
        "labels": labels,
        "raw_points": int(scan.shape[0]),
    }
    if not need_image:
        return loaded

    image_path = data_root / "image_2" / f"{frame}.png"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
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

    from semantic_kitti_ipfp_tiny_overfit import build_metric_depth

    metric_depth, depth_p5_p95 = build_metric_depth(args, uv_all, depth_all, valid_all, new_h, new_w)
    image_np = np.asarray(image_resized).astype(np.float32) / 255.0
    loaded.update(
        {
            "image": str(image_path),
            "image_pil": image_resized,
            "image_chw_np": np.transpose(image_np, (2, 0, 1)).copy(),
            "metric_depth_np": metric_depth.astype(np.float32),
            "intrinsics_np": intrinsics.astype(np.float32),
            "lidar_to_cam_np": lidar_to_cam_4.astype(np.float32),
            "p2": p2,
            "tr": tr,
            "uv_all": uv_all,
            "depth_all": depth_all,
            "valid_all": valid_all,
            "depth_p5_p95": depth_p5_p95,
        }
    )
    return loaded


def choose_train_indices(args: argparse.Namespace, frame_data: dict, rng: np.random.Generator) -> np.ndarray:
    labels = frame_data["labels"]
    valid_label_idx = np.flatnonzero(labels >= 0)
    if valid_label_idx.size == 0:
        raise RuntimeError(f"no valid labels in {frame_data['name']}")
    n = min(args.train_sample_points, valid_label_idx.size)

    if args.mode == "fused" and "valid_all" in frame_data:
        visible_valid = np.flatnonzero((labels >= 0) & frame_data["valid_all"])
        take_visible = min(visible_valid.size, max(n // 2, min(n, 512)))
        chosen_parts = []
        if take_visible > 0:
            chosen_visible = rng.choice(visible_valid, size=take_visible, replace=False)
            chosen_parts.append(chosen_visible)
        remaining = n - take_visible
        if remaining > 0:
            pool = np.setdiff1d(valid_label_idx, chosen_parts[0], assume_unique=False) if chosen_parts else valid_label_idx
            if pool.size == 0:
                pool = valid_label_idx
            chosen_other = rng.choice(pool, size=remaining, replace=pool.size < remaining)
            chosen_parts.append(chosen_other)
        chosen = np.concatenate(chosen_parts)
    else:
        chosen = rng.choice(valid_label_idx, size=n, replace=valid_label_idx.size < n)
    rng.shuffle(chosen)
    return chosen.astype(np.int64)


def make_sample(frame_data: dict, indices: np.ndarray, need_image: bool) -> dict:
    scan = frame_data["scan"]
    coord_np = scan[indices, :3].astype(np.float32)
    intensity_np = scan[indices, 3:4].astype(np.float32)
    segment_np = frame_data["labels"][indices].astype(np.int64)
    sample = {
        "sequence": frame_data["sequence"],
        "frame": frame_data["frame"],
        "name": frame_data["name"],
        "raw_points": frame_data["raw_points"],
        "coord_np": coord_np,
        "intensity_np": intensity_np,
        "segment_np": segment_np,
    }
    if need_image:
        uv_sel, depth_sel, valid_sel = project_kitti(coord_np, frame_data["p2"], frame_data["tr"], frame_data["image_pil"].size[::-1])
        sample.update(
            {
                "image_pil": frame_data["image_pil"],
                "image_chw_np": frame_data["image_chw_np"],
                "metric_depth_np": frame_data["metric_depth_np"],
                "intrinsics_np": frame_data["intrinsics_np"],
                "lidar_to_cam_np": frame_data["lidar_to_cam_np"],
                "p2": frame_data["p2"],
                "tr": frame_data["tr"],
                "uv_sel": uv_sel,
                "depth_sel": depth_sel,
                "valid_sel": valid_sel,
                "depth_p5_p95": frame_data["depth_p5_p95"],
            }
        )
    return sample


def sample_to_tensors(sample: dict, device: str) -> dict:
    tensors = {
        "coord": torch.from_numpy(sample["coord_np"]).to(device),
        "intensity": torch.from_numpy(sample["intensity_np"]).to(device),
        "segment": torch.from_numpy(sample["segment_np"]).to(device),
    }
    if "image_chw_np" in sample:
        tensors.update(
            {
                "image_chw": torch.from_numpy(sample["image_chw_np"]).to(device),
                "metric_depth": torch.from_numpy(sample["metric_depth_np"]).to(device),
                "intrinsics": torch.from_numpy(sample["intrinsics_np"]).to(device),
                "lidar_to_cam": torch.from_numpy(sample["lidar_to_cam_np"]).to(device),
            }
        )
    return tensors


def make_chunks(num_points: int, chunk_points: int) -> list[np.ndarray]:
    chunk_points = max(chunk_points, 1)
    return [np.arange(start, min(start + chunk_points, num_points), dtype=np.int64) for start in range(0, num_points, chunk_points)]


def selected_indices(count: int, total: int) -> set[int]:
    count = min(max(count, 0), total)
    if count == 0:
        return set()
    if count == total:
        return set(range(total))
    return set(np.linspace(0, total - 1, count, dtype=int).tolist())


def save_val_montage(output_dir: Path, cards: list[tuple[str, Path, Path]], name: str) -> str | None:
    if not cards:
        return None
    tile = 190
    pad = 16
    title_h = 24
    card_w = tile * 2 + pad * 3
    card_h = tile + title_h + pad * 2
    cols = min(4, len(cards))
    rows = int(np.ceil(len(cards) / cols))
    montage = Image.new("RGB", (cols * card_w, rows * card_h), (248, 248, 248))
    draw = ImageDraw.Draw(montage)
    for card_index, (frame_name, gt_path, pred_path) in enumerate(cards):
        col = card_index % cols
        row = card_index // cols
        x0 = col * card_w
        y0 = row * card_h
        draw.rectangle((x0 + 4, y0 + 4, x0 + card_w - 4, y0 + card_h - 4), outline=(210, 210, 210))
        draw.text((x0 + pad, y0 + 6), f"{frame_name}: GT | pred", fill=(20, 20, 20))
        gt_img = Image.open(gt_path).convert("RGB").resize((tile, tile), Image.BILINEAR)
        pred_img = Image.open(pred_path).convert("RGB").resize((tile, tile), Image.BILINEAR)
        montage.paste(gt_img, (x0 + pad, y0 + title_h + pad))
        montage.paste(pred_img, (x0 + pad * 2 + tile, y0 + title_h + pad))
    montage_path = output_dir / name
    montage.save(montage_path)
    return str(montage_path.relative_to(output_dir))


def evaluate_full_frames(
    model,
    ipfp,
    point_cls,
    frame_keys: list[tuple[str, str]],
    args: argparse.Namespace,
    route: str,
    output_dir: Path,
    step: int,
) -> dict:
    model.eval()
    if ipfp is not None:
        ipfp.eval()
    need_image = route == "fused"
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    viz_dir = output_dir / f"val_viz_step_{step:07d}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_pick = selected_indices(args.viz_frame_count, len(frame_keys))
    cards: list[tuple[str, Path, Path]] = []
    start_time = time.time()

    with torch.no_grad():
        for frame_index, (sequence, frame) in enumerate(frame_keys):
            frame_data = load_frame(args, sequence, frame, need_image=need_image)
            frame_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            pred_full = np.full(frame_data["scan"].shape[0], -1, dtype=np.int64)
            loss_weighted = 0.0
            loss_weight = 0
            for chunk_index, indices in enumerate(make_chunks(frame_data["scan"].shape[0], args.eval_chunk_points)):
                sample = make_sample(frame_data, indices, need_image=need_image)
                tensors = sample_to_tensors(sample, args.device)
                logits, _ = run_forward(
                    model,
                    ipfp,
                    point_cls,
                    tensors,
                    args,
                    args.seed + step * 100000 + frame_index * 1000 + chunk_index,
                    route=route,
                )
                loss, _ = compute_loss(logits, tensors["segment"], args)
                pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                target = tensors["segment"].detach().cpu().numpy()
                valid_count = int((target >= 0).sum())
                loss_weighted += float(loss.detach().cpu().item()) * valid_count
                loss_weight += valid_count
                pred_full[indices] = pred
                update_confusion_matrix(frame_confusion, target, pred)
                update_confusion_matrix(confusion, target, pred)

            frame_metrics = metrics_from_confusion(frame_confusion)
            rows.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "name": frame_data["name"],
                    "raw_points": frame_data["raw_points"],
                    "chunks": int(np.ceil(frame_data["scan"].shape[0] / args.eval_chunk_points)),
                    "loss": loss_weighted / max(loss_weight, 1),
                    "valid_points": frame_metrics["valid_points"],
                    "overall_accuracy": frame_metrics["overall_accuracy"],
                    "mean_iou": frame_metrics["mean_iou"],
                }
            )
            if frame_index in viz_pick:
                gt_path = viz_dir / f"{frame_data['name']}_gt.png"
                pred_path = viz_dir / f"{frame_data['name']}_pred.png"
                draw_bev(frame_data["scan"][:, :3].astype(np.float32), frame_data["labels"].astype(np.int64), gt_path, f"{frame_data['name']} GT")
                draw_bev(frame_data["scan"][:, :3].astype(np.float32), pred_full, pred_path, f"{frame_data['name']} pred")
                cards.append((frame_data["name"], gt_path, pred_path))
            if frame_index == 0 or (frame_index + 1) % max(1, len(frame_keys) // 10) == 0 or frame_index + 1 == len(frame_keys):
                print(
                    json.dumps(
                        {
                            "event": "val_progress",
                            "step": step,
                            "route": route,
                            "frames_done": frame_index + 1,
                            "frames_total": len(frame_keys),
                            "last": rows[-1],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    metrics = metrics_from_confusion(confusion)
    mean_loss = float(np.mean([row["loss"] for row in rows])) if rows else None
    montage = save_val_montage(viz_dir, cards, "val_selected_frames_montage.png")
    result = {
        "step": step,
        "route": route,
        "frames": len(frame_keys),
        "mean_loss": mean_loss,
        "mean_iou": metrics["mean_iou"],
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_class_accuracy": metrics["mean_class_accuracy"],
        "frequency_weighted_iou": metrics["frequency_weighted_iou"],
        "valid_points": metrics["valid_points"],
        "elapsed_sec": round(time.time() - start_time, 3),
        "by_frame": rows,
        "metrics": metrics,
        "outputs": [str((viz_dir / "val_selected_frames_montage.png").relative_to(output_dir))] if montage else [],
    }
    result_path = output_dir / f"val_metrics_step_{step:07d}_{route}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def save_checkpoint(path: Path, model, ipfp, optimizer, step: int, best_val_miou: float | None, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ipfp": ipfp.state_dict() if ipfp is not None else None,
            "optimizer": optimizer.state_dict(),
            "best_val_miou": best_val_miou,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model, ipfp, optimizer) -> tuple[int, float | None]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if ipfp is not None and state.get("ipfp") is not None:
        ipfp.load_state_dict(state["ipfp"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0)), state.get("best_val_miou")


def write_notes(output_dir: Path, summary: dict, val_result: dict | None) -> None:
    notes = [
        "# SemanticKITTI Full-Frame Benchmark Notes",
        "",
        "## Scope",
        "",
        "This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.",
        "",
        "## Summary",
        "",
        f"- Mode: {summary['mode']}",
        f"- Primary eval route: {summary['primary_eval_route']}",
        f"- Diagnostic eval route: {summary['diagnostic_eval_route'] or 'none'}",
        f"- Train frames: {summary['train_frame_count']}",
        f"- Val frames: {summary['val_frame_count']}",
        f"- Train sample points per step: {summary['train_sample_points']}",
        f"- Eval chunk points: {summary['eval_chunk_points']}",
        f"- Steps completed: {summary['steps_completed']}",
        f"- Best val mIoU: {fmt_metric(summary['best_val_miou'])}",
        "",
    ]
    if val_result is not None:
        notes.extend(
            [
                "## Final Validation",
                "",
                f"- Route: {val_result['route']}",
                f"- Frames: {val_result['frames']}",
                f"- Mean loss: {fmt_metric(val_result['mean_loss'])}",
                f"- mIoU: {fmt_metric(val_result['mean_iou'])}",
                f"- Overall accuracy: {fmt_metric(val_result['overall_accuracy'])}",
                f"- Valid points: {val_result['valid_points']}",
                "",
            ]
        )
        append_class_iou_notes(notes, "## Final Validation Class IoU", val_result["metrics"])
    notes.append("")
    (output_dir / "BENCHMARK_NOTES.md").write_text("\n".join(notes))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    pointcept_root = root / "src/Pointcept"
    ipfp_root = root / "ipfp_minimal"
    output_dir = Path(
        args.output_dir
        or root
        / "results/semantic_kitti_full_benchmark"
        / time.strftime("%Y%m%d_%H%M%S")
        / f"{args.mode}_seq{''.join(args.val_sequences)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(pointcept_root) not in sys.path:
        sys.path.insert(0, str(pointcept_root))
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

    train_frames = list_frames(root, args.train_sequences, args.train_frame_stride, args.train_frame_limit)
    val_frames = list_frames(root, args.val_sequences, args.val_frame_stride, args.val_frame_limit)
    if args.steps > 0 and not train_frames:
        raise RuntimeError("No training frames found")
    if not val_frames:
        raise RuntimeError("No validation frames found")

    eval_route = primary_eval_route(args)
    diag_route = diagnostic_eval_route(args)

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
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    best_val_miou = None
    if args.resume:
        start_step, best_val_miou = load_checkpoint(Path(args.resume), model, ipfp, optimizer)

    train_log_path = output_dir / "train_log.jsonl"
    rng = np.random.default_rng(args.seed + start_step)
    last_val_result = None

    run_manifest = {
        "status": "RUNNING",
        "output_dir": str(output_dir),
        "mode": args.mode,
        "eval_route": args.eval_route,
        "primary_eval_route": eval_route,
        "diagnostic_eval_route": diag_route,
        "train_sequences": args.train_sequences,
        "val_sequences": args.val_sequences,
        "train_frame_count": len(train_frames),
        "val_frame_count": len(val_frames),
        "train_sample_points": args.train_sample_points,
        "eval_chunk_points": args.eval_chunk_points,
        "steps": args.steps,
        "start_step": start_step,
        "args": vars(args),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(run_manifest, ensure_ascii=False), flush=True)

    for step in range(start_step + 1, args.steps + 1):
        sequence, frame = train_frames[int(rng.integers(0, len(train_frames)))]
        need_image = args.mode == "fused"
        frame_data = load_frame(args, sequence, frame, need_image=need_image)
        indices = choose_train_indices(args, frame_data, rng)
        sample = make_sample(frame_data, indices, need_image=need_image)
        tensors = sample_to_tensors(sample, args.device)

        model.train()
        if ipfp is not None:
            ipfp.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = run_forward(model, ipfp, Point, tensors, args, args.seed + step)
        loss, loss_parts = compute_loss(logits, tensors["segment"], args)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
        optimizer.step()

        pred = logits.argmax(dim=1)
        valid = tensors["segment"] >= 0
        acc = (pred[valid] == tensors["segment"][valid]).float().mean() if valid.any() else torch.tensor(0.0, device=pred.device)
        record = {
            "step": step,
            "sequence": sequence,
            "frame": frame,
            "loss": float(loss.detach().cpu().item()),
            **loss_parts,
            "valid_acc": float(acc.detach().cpu().item()),
            "valid_points": int(valid.detach().cpu().sum().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "lr": args.lr,
        }
        with train_log_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
            print(json.dumps({"event": "train_progress", **record}, ensure_ascii=False), flush=True)

        should_eval = args.eval_every > 0 and step % args.eval_every == 0
        if should_eval:
            last_val_result = evaluate_full_frames(model, ipfp, Point, val_frames, args, eval_route, output_dir, step)
            current = last_val_result["mean_iou"]
            if current is not None and (best_val_miou is None or current > best_val_miou):
                best_val_miou = current
                save_checkpoint(output_dir / "checkpoints/best.pth", model, ipfp, optimizer, step, best_val_miou, args)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(output_dir / "checkpoints/latest.pth", model, ipfp, optimizer, step, best_val_miou, args)

    if args.steps == 0 or args.eval_every == 0 or last_val_result is None:
        last_val_result = evaluate_full_frames(model, ipfp, Point, val_frames, args, eval_route, output_dir, args.steps)
        current = last_val_result["mean_iou"]
        if current is not None and (best_val_miou is None or current > best_val_miou):
            best_val_miou = current
            save_checkpoint(output_dir / "checkpoints/best.pth", model, ipfp, optimizer, args.steps, best_val_miou, args)
    if diag_route is not None:
        evaluate_full_frames(model, ipfp, Point, val_frames, args, diag_route, output_dir, args.steps)

    save_checkpoint(output_dir / "checkpoints/latest.pth", model, ipfp, optimizer, args.steps, best_val_miou, args)
    summary = {
        "status": "OK",
        "mode": args.mode,
        "eval_route": args.eval_route,
        "primary_eval_route": eval_route,
        "diagnostic_eval_route": diag_route,
        "train_sequences": args.train_sequences,
        "val_sequences": args.val_sequences,
        "train_frame_count": len(train_frames),
        "val_frame_count": len(val_frames),
        "train_sample_points": args.train_sample_points,
        "eval_chunk_points": args.eval_chunk_points,
        "steps_completed": args.steps,
        "best_val_miou": best_val_miou,
        "final_val": last_val_result,
        "cuda_max_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if args.device == "cuda"
        else 0.0,
        "outputs": [
            "run_manifest.json",
            "train_log.jsonl",
            "summary.json",
            "BENCHMARK_NOTES.md",
            "checkpoints/latest.pth",
            "checkpoints/best.pth",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_notes(output_dir, summary, last_val_result)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
