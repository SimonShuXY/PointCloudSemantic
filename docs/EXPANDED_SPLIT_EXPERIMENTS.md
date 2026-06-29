# Expanded Split Experiments

## 100-Frame Train, 50-Frame Holdout

This run expands the previous `50/20` split and adds sampled-point IoU metrics.

- Dataset slice: SemanticKITTI sequence `00`
- Train frames: `000000-000099`
- Holdout eval frames: `000100-000149`
- Points per frame: `2048`
- IPFP centers: `64`
- Steps: `1000`
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Image width: `480`
- Routes compared: PTv3 LiDAR-only vs PTv3 + minimal IPFP extra points/features

The reported mIoU is computed on the sampled points used by this reproduction script,
not on full-frame official SemanticKITTI validation.

## Results

| Route | Train loss, mean eval | Train acc, mean eval | Train mIoU | Holdout loss, mean eval | Holdout acc, mean eval | Holdout overall acc | Holdout mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PTv3 LiDAR-only | 0.3591 | 88.83% | 48.55% | 1.7897 | 61.45% | 61.51% | 15.30% |
| PTv3 + IPFP | 0.5618 | 85.03% | 41.76% | 2.3918 | 56.64% | 56.67% | 13.41% |

## Class-Level Holdout IoU

| Class | LiDAR-only IoU | PTv3 + IPFP IoU | Delta |
| --- | ---: | ---: | ---: |
| road | 65.71% | 67.85% | +2.14 |
| sidewalk | 40.67% | 42.68% | +2.01 |
| car | 41.25% | 35.16% | -6.09 |
| building | 49.41% | 34.70% | -14.71 |
| vegetation | 35.31% | 24.40% | -10.91 |
| parking | 7.32% | 6.76% | -0.57 |
| pole | 8.54% | 8.27% | -0.27 |
| traffic-sign | 4.69% | 2.56% | -2.12 |

## Interpretation

The earlier `50/20` split showed a small holdout accuracy advantage for PTv3+IPFP.
After expanding to `100/50` and adding sampled mIoU, the current minimal fused route
underperforms the LiDAR-only baseline.

This does not invalidate IPFP as a paper idea. It says the current reproduction
scaffold is not yet a reliable fused model. The likely bottlenecks are still the
pseudo metric-depth path, the small image feature branch, and noisy extra
back-projected points.

The class-level metrics are useful: the fused route is competitive on road/sidewalk,
but loses clearly on car, building, and vegetation. That pattern points to image
back-projection noise or feature mismatch rather than a global training failure.

## Artifacts

LiDAR-only:

- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_lidar-only_seq00_train000000-000099_eval000100-000149/summary.json`
- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_lidar-only_seq00_train000000-000099_eval000100-000149/loss_curve.png`
- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_lidar-only_seq00_train000000-000099_eval000100-000149/holdout_selected_frames_final_montage.png`

PTv3 + IPFP:

- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_fused_seq00_train000000-000099_eval000100-000149/summary.json`
- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_fused_seq00_train000000-000099_eval000100-000149/loss_curve.png`
- `results/semantic_kitti_repro/expanded100_holdout50_20260629_130604/tiny_holdout_fused_seq00_train000000-000099_eval000100-000149/holdout_selected_frames_final_montage.png`

## Next Steps

The first IPFP ablations are complete; see `IPFP_ABLATION_EXPERIMENTS.md`.

1. Add an image-feature gate and sweep feature weights on the `100/50` split.
2. Keep `zero-feat64` as a geometry-only control.
3. Replace pseudo metric depth with a stronger depth source before any larger training run.
4. Only after the fused route recovers on `100/50`, scale to larger splits or sequence `08`.
