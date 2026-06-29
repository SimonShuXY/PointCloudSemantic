# Reproduction Commands

These commands assume:

- Pointcept is cloned separately at `src/Pointcept`.
- Dataset files are extracted under the `ROOT` directory.
- The repository root and Pointcept root are on `PYTHONPATH`.

```bash
export ROOT=/path/to/ipfp_repro
export POINTCEPT_ROOT=$ROOT/src/Pointcept
export PYTHONPATH=$POINTCEPT_ROOT:/path/to/PointCloudSemantic:$PYTHONPATH
```

## Framework Smoke Test

```bash
python scripts/smoke_ipfp_pointcept.py \
  --pointcept-root "$POINTCEPT_ROOT" \
  --num-points 1024 \
  --num-centers 128
```

Expected successful output includes:

```text
lidar_only_logits (1024, 16)
ipfp_extra_coord (128, 3)
ipfp_extra_feat (128, 32)
ipfp_training_like_logits (1024, 16)
smoke_test OK
```

## nuScenes Mini Closed Loop

```bash
python scripts/nuscenes_mini_closed_loop.py \
  --root "$ROOT" \
  --device cuda
```

This checks:

- nuScenes-mini LiDAR loading
- lidarseg labels
- CAM_FRONT image loading
- LiDAR-camera projection
- IPFP extra points/features
- Pointcept PTv3 forward and CE loss

## SemanticKITTI Visualization Closed Loop

```bash
python scripts/semantic_kitti_ipfp_visualize.py \
  --root "$ROOT" \
  --sequence 00 \
  --frame 000000 \
  --num-points 8192 \
  --num-centers 256 \
  --image-width 640 \
  --device cuda
```

Expected outputs include:

- `image_lidar_depth_projection.png`
- `image_ipfp_extra_projection.png`
- `bev_ground_truth_labels.png`
- `bev_random_init_prediction.png`
- `bev_ipfp_extra_points.png`
- `summary.json`

## Tiny Overfit

The default mode is `fused`, which runs PTv3 with IPFP extra points/features. Use `--mode lidar-only` for the LiDAR-only control route.

Single frame:

```bash
python scripts/semantic_kitti_ipfp_tiny_overfit.py \
  --root "$ROOT" \
  --sequence 00 \
  --frames 000000 \
  --num-points 2048 \
  --num-centers 64 \
  --image-width 480 \
  --steps 40 \
  --lr 5e-4 \
  --device cuda
```

Five frames:

```bash
python scripts/semantic_kitti_ipfp_tiny_overfit.py \
  --root "$ROOT" \
  --sequence 00 \
  --frames 000000 000001 000002 000003 000004 \
  --num-points 2048 \
  --num-centers 64 \
  --image-width 480 \
  --steps 100 \
  --lr 5e-4 \
  --device cuda
```

Fifty frames with selected-frame visualization montage:

```bash
python scripts/semantic_kitti_ipfp_tiny_overfit.py \
  --root "$ROOT" \
  --sequence 00 \
  --frames 000000 000001 000002 000003 000004 000005 000006 000007 000008 000009 \
           000010 000011 000012 000013 000014 000015 000016 000017 000018 000019 \
           000020 000021 000022 000023 000024 000025 000026 000027 000028 000029 \
           000030 000031 000032 000033 000034 000035 000036 000037 000038 000039 \
           000040 000041 000042 000043 000044 000045 000046 000047 000048 000049 \
  --num-points 2048 \
  --num-centers 64 \
  --image-width 480 \
  --steps 500 \
  --lr 5e-4 \
  --viz-frame-count 8 \
  --mode fused \
  --device cuda
```

LiDAR-only control with the same 50-frame schedule:

```bash
python scripts/semantic_kitti_ipfp_tiny_overfit.py \
  --root "$ROOT" \
  --sequence 00 \
  --frames 000000 000001 000002 000003 000004 000005 000006 000007 000008 000009 \
           000010 000011 000012 000013 000014 000015 000016 000017 000018 000019 \
           000020 000021 000022 000023 000024 000025 000026 000027 000028 000029 \
           000030 000031 000032 000033 000034 000035 000036 000037 000038 000039 \
           000040 000041 000042 000043 000044 000045 000046 000047 000048 000049 \
  --num-points 2048 \
  --num-centers 64 \
  --image-width 480 \
  --steps 500 \
  --lr 5e-4 \
  --viz-frame-count 8 \
  --mode lidar-only \
  --device cuda
```

The tiny-overfit runs are sanity checks. They validate the training path, not benchmark generalization.
