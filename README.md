# PointCloudSemantic

Minimal reproduction scaffold for image-assisted point-cloud semantic segmentation experiments based on Pointcept PTv3 and an IPFP-style image-to-point-cloud feature back-projection module.

This repository contains:

- `ipfp/`: minimal IPFP-style feature aggregation and back-projection code.
- `scripts/`: smoke tests, SemanticKITTI visualization, tiny-overfit experiments, and dataset download helpers.
- `results/`: small qualitative and JSON outputs from the reproduction runs. Raw datasets are not included.
- `docs/DATASETS.md`: dataset download links used in the experiments.
- `docs/EXPERIMENTS.md`: summary of completed sanity checks and tiny-overfit runs.

## What Is Implemented

The current code validates the following path:

```text
SemanticKITTI/KITTI LiDAR + labels + image + calibration
  -> LiDAR to image projection
  -> IPFP-style image feature aggregation and back-projected extra points
  -> Pointcept PTv3 fused forward
  -> cross-entropy loss
  -> tiny-overfit training loop and visualizations
```

The IPFP implementation is a compact reproduction scaffold, not an official implementation of the paper.

## Key Results

The included small-scale SemanticKITTI overfit runs were used as plumbing checks:

| Experiment | Frames | Steps | Mean eval loss | Mean eval acc |
| --- | ---: | ---: | ---: | ---: |
| 1-frame tiny overfit | 1 | 40 | 3.3334 -> 0.3642 | first-frame only |
| 5-frame tiny overfit | 5 | 100 | 3.3463 -> 0.1470 | 7.72% -> 95.60% |
| 20-frame tiny overfit | 20 | 200 | 3.3933 -> 0.1702 | 6.55% -> 94.16% |
| 50-frame tiny overfit | 50 | 500 | 3.4847 -> 0.3492 | 4.94% -> 89.14% |

Example 50-frame visualization:

![50-frame montage](results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/selected_frames_final_montage.png)

## Environment

The validated environment used:

- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- Pointcept checked out separately
- RTX 4090 D GPU for the reported smoke tests

Pointcept itself is not vendored in this repository. Clone it separately and set `PYTHONPATH`:

```bash
git clone https://github.com/Pointcept/Pointcept.git src/Pointcept
export PYTHONPATH="$PWD/src/Pointcept:$PWD"
```

Install this repository's lightweight package code by keeping the repository root on `PYTHONPATH`.

## Quick Commands

Run a SemanticKITTI + image + IPFP visualization:

```bash
python scripts/semantic_kitti_ipfp_visualize.py \
  --root /path/to/ipfp_repro \
  --sequence 00 \
  --frame 000000 \
  --num-points 8192 \
  --num-centers 256 \
  --device cuda
```

Run a tiny-overfit experiment:

```bash
python scripts/semantic_kitti_ipfp_tiny_overfit.py \
  --root /path/to/ipfp_repro \
  --sequence 00 \
  --frames 000000 000001 000002 000003 000004 \
  --num-points 2048 \
  --num-centers 64 \
  --steps 100 \
  --device cuda
```

See `docs/REPRODUCTION.md` for more detail.

## Data

No raw dataset files are committed. Use `docs/DATASETS.md` for the exact dataset URLs used during setup.

Expected data layout:

```text
data/
  semantic_kitti/
    dataset/sequences/00/velodyne/000000.bin
    dataset/sequences/00/image_2/000000.png
    dataset/sequences/00/labels/000000.label
    dataset/sequences/00/calib.txt
  nuscenes_mini/
    raw/
```

## Security

This repository intentionally excludes:

- raw datasets and archive files
- model checkpoints
- SSH commands, server addresses, and passwords
- local virtual environments
- token files or `.env` files

Before pushing, the repository was scanned for common credential patterns.
