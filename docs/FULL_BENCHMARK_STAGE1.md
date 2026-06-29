# Full Benchmark Stage 1

This stage moves the project from sampled-point holdout diagnostics toward a
SemanticKITTI sequence `08` validation benchmark.

## Scope

The new runner is `scripts/semantic_kitti_full_benchmark.py`.

It is intentionally a scaffold rather than a final official submission trainer:

- training samples points from real SemanticKITTI training frames,
- validation covers every point in every selected validation frame,
- full-frame validation is performed with chunked inference to avoid GPU OOM,
- metrics are accumulated with a frame-level confusion matrix over 19 classes,
- checkpoints, logs, per-class IoU, and selected BEV visualizations are saved.

The first target is a LiDAR-only sequence `08` validation baseline. Once this is
stable, the gated IPFP route can be run with `extra-feature-scale=0.1`.

## Default Splits

| Split | Sequences |
| --- | --- |
| Train | `00 01 02 03 04 05 06 07 09 10` |
| Validation | `08` |

## Main Commands

Sanity run:

```bash
ROOT=/root/autodl-tmp/ipfp_repro \
STEPS=2 \
TRAIN_FRAME_LIMIT=4 \
VAL_FRAME_LIMIT=2 \
EVAL_CHUNK_POINTS=8192 \
bash scripts/run_semantic_kitti_full_benchmark_stage1.sh
```

LiDAR-only stage-1 run:

```bash
ROOT=/root/autodl-tmp/ipfp_repro \
MODE=lidar-only \
STEPS=20000 \
EVAL_EVERY=0 \
CHECKPOINT_EVERY=1000 \
TRAIN_SAMPLE_POINTS=8192 \
EVAL_CHUNK_POINTS=16384 \
VAL_FRAME_STRIDE=1 \
bash scripts/run_semantic_kitti_full_benchmark_stage1.sh
```

Gated IPFP follow-up:

```bash
ROOT=/root/autodl-tmp/ipfp_repro \
MODE=fused \
STEPS=20000 \
EVAL_EVERY=0 \
CHECKPOINT_EVERY=1000 \
TRAIN_SAMPLE_POINTS=8192 \
EVAL_CHUNK_POINTS=16384 \
VAL_FRAME_STRIDE=1 \
bash scripts/run_semantic_kitti_full_benchmark_stage1.sh
```

## Outputs

Each run writes:

- `run_manifest.json`
- `train_log.jsonl`
- `val_metrics_step_*.json`
- `summary.json`
- `BENCHMARK_NOTES.md`
- `checkpoints/latest.pth`
- `checkpoints/best.pth`
- `val_viz_step_*/val_selected_frames_montage.png`

## Sanity Result

The first remote sanity run completed successfully at:

`results/semantic_kitti_full_benchmark/sanity_lidar_20260629_223612`

| Setting | Value |
| --- | ---: |
| Mode | `lidar-only` |
| Train frame limit | `4` |
| Validation frame limit | `2` |
| Steps | `2` |
| Eval chunk points | `8192` |
| Full-frame valid validation points | `232805` |
| Validation mIoU | `4.09%` |
| Validation overall accuracy | `28.90%` |
| CUDA peak memory | `1.152 GB` |

This result is not meaningful as a trained benchmark score because it only uses
two update steps. Its purpose is to prove that full-frame sequence `08`
validation can read all points, split frames into chunks, accumulate metrics,
write checkpoints, and save BEV visualizations without OOM.

Sanity artifacts included in the repository:

- `results/semantic_kitti_full_benchmark/sanity_lidar_20260629_223612/summary.json`
- `results/semantic_kitti_full_benchmark/sanity_lidar_20260629_223612/val_metrics_step_0000002_lidar-only.json`
- `results/semantic_kitti_full_benchmark/sanity_lidar_20260629_223612/val_viz_step_0000002/val_selected_frames_montage.png`

![Sanity full-frame validation montage](../results/semantic_kitti_full_benchmark/sanity_lidar_20260629_223612/val_viz_step_0000002/val_selected_frames_montage.png)

## Active Remote Run

The first full LiDAR-only stage-1 run was launched in a detached `screen` session:

| Field | Value |
| --- | --- |
| Screen session | `semkitti_full_lidar` |
| Output directory | `/root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage1_lidar_full_20260629_223718` |
| Train sequences | `00 01 02 03 04 05 06 07 09 10` |
| Validation sequence | `08` |
| Train frame count | `19130` |
| Validation frame count | `4071` |
| Steps | `20000` |
| Train sample points | `8192` |
| Eval chunk points | `16384` |
| Eval schedule | full sequence `08` validation at the end |

Progress can be checked on the remote host with:

```bash
screen -ls
tail -n 40 /root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage1_lidar_full_20260629_223718/run.log
wc -l /root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage1_lidar_full_20260629_223718/train_log.jsonl
```

## Caveat

This is still not the SemanticKITTI official test submission path. It is the
first full-frame validation benchmark layer for sequence `08`. The official
submission route should be added only after this validation scaffold is stable.
