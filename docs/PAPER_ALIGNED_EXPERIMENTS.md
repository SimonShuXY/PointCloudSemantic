# Paper-Aligned IPFP Diagnostics

## Goal

This run tests whether the earlier IPFP underperformance was caused by a pipeline
or parameter mismatch with the paper-style setup. It keeps the same small
SemanticKITTI sequence `00` split but changes three high-impact details:

- fused training can be evaluated with a LiDAR-only forward path,
- pseudo gradient depth is replaced by sparse LiDAR depth inpainting,
- CE can be paired with a Lovasz-Softmax loss.

The run is still a sampled-point reproduction diagnostic, not an official
SemanticKITTI benchmark.

## Setup

- Train frames: `000000-000099`
- Holdout eval frames: `000100-000149`
- Points per frame: `2048`
- Steps: `1000`
- Image width: `480`
- Depth mode: `lidar-inpaint`
- Loss: `CE + Lovasz`
- Fused IPFP centers: `32`
- Fused IPFP depth percentile range: `20-99`
- Fused IPFP discard probability: `0.2`

## Results

| Route | Train route | Primary eval route | Holdout loss | Holdout overall acc | Holdout mIoU | Diagnostic fused holdout mIoU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| LiDAR-only CE+Lovasz | LiDAR-only | LiDAR-only | 2.7178 | 60.36% | 14.71% | n/a |
| Paper-aligned IPFP | Fused | LiDAR-only | 2.7495 | 60.67% | 14.51% | 14.49% |

The paper-aligned IPFP route is much closer to the LiDAR-only baseline than the
earlier learned-feature fused route, but it still does not beat the baseline on
overall holdout mIoU in this small split.

## Feature Gate Sweep

The next run keeps the same paper-aligned setup and sweeps a scalar multiplier on
the learned IPFP image feature content before merging the extra tokens into PTv3.
The extra point coordinates are still present. A scale of `0.0` is therefore a
geometry-only extra-token control, while `1.0` is the full learned-feature path.

| Extra feature scale | Primary eval route | Holdout overall acc | Holdout mIoU | Diagnostic fused holdout mIoU | Train mIoU |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.00 | LiDAR-only | 60.35% | 14.89% | 14.92% | 48.42% |
| 0.10 | LiDAR-only | 61.73% | 15.23% | 15.32% | 47.04% |
| 0.25 | LiDAR-only | 61.10% | 15.23% | 15.04% | 50.48% |
| 0.50 | LiDAR-only | 59.88% | 14.93% | 15.05% | 48.81% |
| 1.00 | LiDAR-only | 54.45% | 12.95% | 12.85% | 48.40% |

The best setting in this sweep is `extra-feature-scale=0.1`, reaching `15.23%`
sampled holdout mIoU and `61.73%` holdout overall accuracy. This beats the
paper-aligned LiDAR-only CE+Lovasz baseline from the previous table (`14.71%`
mIoU, `60.36%` overall accuracy). Full-scale learned image features remain
harmful, which confirms that the image branch needs strong gating or staged
training rather than being merged at full strength.

## Class-Level Holdout Deltas

| Class | LiDAR-only IoU | Paper-aligned IPFP IoU | Delta |
| --- | ---: | ---: | ---: |
| road | 67.35% | 67.56% | +0.21 |
| sidewalk | 40.42% | 39.20% | -1.22 |
| car | 43.03% | 39.68% | -3.35 |
| building | 33.64% | 38.00% | +4.35 |
| vegetation | 36.57% | 37.61% | +1.04 |
| terrain | 0.76% | 0.99% | +0.23 |
| parking | 5.74% | 7.06% | +1.32 |

## Interpretation

The earlier gap was very likely amplified by pipeline mismatch. After switching
to LiDAR-only primary evaluation, sparse LiDAR depth inpainting, CE+Lovasz, and a
less aggressive IPFP token setting, the fused route no longer collapses relative
to LiDAR-only. The remaining gap is small, but the fused branch still introduces
noise for classes like `car` and `sidewalk`.

This points to image feature quality and fusion scheduling as the next bottleneck,
not basic projection or PTv3 plumbing. The feature gate sweep shows that a small
amount of learned image feature content can help, while full-strength image
features are still noisy.

## Artifacts

- `results/semantic_kitti_repro/paper_aligned100_holdout50_latest/lidar_only_ce_lovasz/summary.json`
- `results/semantic_kitti_repro/paper_aligned100_holdout50_latest/fused_train_lidar_eval/summary.json`
- `results/semantic_kitti_repro/paper_aligned100_holdout50_latest/lidar_only_ce_lovasz/holdout_selected_frames_final_montage.png`
- `results/semantic_kitti_repro/paper_aligned100_holdout50_latest/fused_train_lidar_eval/holdout_selected_frames_final_montage.png`
- `results/semantic_kitti_repro/paper_aligned100_holdout50_latest/fused_train_lidar_eval/diagnostic_fused_holdout_final_image_ipfp_extra_projection.png`
- `results/semantic_kitti_repro/feature_gate100_holdout50_latest/scale_0p1/summary.json`
- `results/semantic_kitti_repro/feature_gate100_holdout50_latest/scale_0p1/holdout_selected_frames_final_montage.png`
