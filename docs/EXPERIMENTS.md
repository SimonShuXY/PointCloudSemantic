# Experiment Summary

## Completed Checks

### Pointcept PTv3 + IPFP Smoke

Validated that Pointcept PTv3 can ingest LiDAR points, IPFP extra points/features can be generated, and a training-like fused forward can produce segmentation logits.

### nuScenes Mini Closed Loop

Validated a real-data closed loop using:

- LiDAR point cloud
- lidarseg labels
- CAM_FRONT image
- camera/LiDAR calibration
- IPFP feature back-projection
- Pointcept PTv3 forward
- CE loss

### SemanticKITTI + KITTI Color Closed Loop

Validated:

- SemanticKITTI velodyne `.bin`
- SemanticKITTI labels `.label`
- KITTI color image `image_2`
- KITTI calibration `calib.txt`
- LiDAR-to-image projection
- IPFP extra 3D point generation
- Pointcept PTv3 forward
- qualitative visualizations

The included `results/semantic_kitti_repro/ipfp_semantic_kitti_seq00_000000` directory contains the first visualization closed-loop output.

## Tiny-Overfit Runs

| Run directory | Frames | Steps | Initial mean loss | Final mean loss | Initial mean acc | Final mean acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_overfit_seq00_000000_20260628_113535` | 1 | 40 | first frame only | first frame only | first frame only | first frame only |
| `tiny_overfit_seq00_000000-000004_20260628_115300` | 5 | 100 | 3.3463 | 0.1470 | 7.72% | 95.60% |
| `tiny_overfit_seq00_000000-000019_20260628_115458` | 20 | 200 | 3.3933 | 0.1702 | 6.55% | 94.16% |
| `tiny_overfit_seq00_000000-000049_20260628_120319` | 50 | 500 | 3.4847 | 0.3492 | 4.94% | 89.14% |
| `tiny_overfit_lidar-only_seq00_000000-000049_20260629_111458` | 50 | 500 | 3.4867 | 0.3241 | 4.92% | 89.36% |

## LiDAR-Only Control

The 50-frame LiDAR-only control slightly outperformed the current minimal PTv3+IPFP route on tiny-overfit final mean eval accuracy and loss. See `CONTROL_EXPERIMENTS.md` for the comparison table and interpretation.

## Train/Holdout Control

The first small generalization check trains on `seq00` frames `000000-000049` and
evaluates on held-out frames `000050-000069`.

| Route | Train-frame final mean loss | Train-frame final mean acc | Holdout final mean loss | Holdout final mean acc |
| --- | ---: | ---: | ---: | ---: |
| PTv3 LiDAR-only | 0.3519 | 89.31% | 0.7379 | 82.60% |
| PTv3 + IPFP | 0.3361 | 89.31% | 0.6166 | 84.94% |

The fused route has nearly identical train-frame accuracy and better holdout loss and
accuracy in this small control. See `HOLDOUT_EXPERIMENTS.md` for artifacts and caveats.

## Expanded Split With Sampled mIoU

The expanded split trains on `seq00` frames `000000-000099` and evaluates on held-out
frames `000100-000149`. This run adds sampled-point confusion matrices, per-class IoU,
and mIoU to `summary.json`.

| Route | Train-frame final mIoU | Holdout final overall acc | Holdout final mIoU |
| --- | ---: | ---: | ---: |
| PTv3 LiDAR-only | 48.55% | 61.51% | 15.30% |
| PTv3 + IPFP | 41.76% | 56.67% | 13.41% |

The expanded split reverses the small `50/20` signal: the current fused route is worse
than LiDAR-only on holdout mIoU. See `EXPANDED_SPLIT_EXPERIMENTS.md`.

## IPFP Ablations

The first IPFP diagnostics use the same `100/50` split and compare extra-point count,
zeroed extra features, and detached IPFP features.

| Route | Holdout final overall acc | Holdout final mIoU |
| --- | ---: | ---: |
| PTv3 LiDAR-only baseline | 61.51% | 15.30% |
| PTv3 + IPFP learned features, 64 centers | 56.67% | 13.41% |
| PTv3 + IPFP learned features, 16 centers | 59.70% | 14.70% |
| PTv3 + IPFP zero extra features, 64 centers | 61.95% | 15.33% |
| PTv3 + IPFP detached branch, 64 centers | 61.58% | 15.10% |

The diagnostic signal is that learned image features are currently harmful, while
extra geometry/tokens are not inherently harmful. See `IPFP_ABLATION_EXPERIMENTS.md`.

## Paper-Aligned Diagnostic

This diagnostic keeps the `100/50` split but changes the most likely
paper-mismatch points: LiDAR-only primary eval for fused training, sparse LiDAR
depth inpainting, CE+Lovasz, fewer IPFP centers, a `20-99` depth percentile
range, and IPFP token discard.

| Route | Primary eval route | Holdout final overall acc | Holdout final mIoU | Diagnostic fused holdout mIoU |
| --- | --- | ---: | ---: | ---: |
| LiDAR-only CE+Lovasz | LiDAR-only | 60.36% | 14.71% | n/a |
| Paper-aligned IPFP | LiDAR-only | 60.67% | 14.51% | 14.49% |

The follow-up feature gate sweep scales the learned IPFP feature content while
keeping the extra point geometry and the same paper-aligned controls.

| Extra feature scale | Holdout final overall acc | Holdout final mIoU | Diagnostic fused holdout mIoU |
| ---: | ---: | ---: | ---: |
| 0.00 | 60.35% | 14.89% | 14.92% |
| 0.10 | 61.73% | 15.23% | 15.32% |
| 0.25 | 61.10% | 15.23% | 15.04% |
| 0.50 | 59.88% | 14.93% | 15.05% |
| 1.00 | 54.45% | 12.95% | 12.85% |

The current best setting is `extra-feature-scale=0.1`, which beats the
paper-aligned LiDAR-only control on sampled holdout mIoU. See
`PAPER_ALIGNED_EXPERIMENTS.md`.

## 50-Frame Visualization

The 50-frame run includes a selected-frame montage:

![50-frame selected montage](../results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/selected_frames_final_montage.png)

## Important Caveats

- Tiny-overfit results are not benchmark results.
- The mIoU in the expanded split is sampled-point mIoU from this reproduction script, not official full-frame SemanticKITTI mIoU.
- The paper-aligned diagnostic now includes sparse LiDAR depth inpainting, but it still is not the paper's full Depth Anything V2 / UniDepth / IP-BASIC depth pipeline.
- Pointcept PTv2 SemanticKITTI config was not used as the main route because it required a compatible `pyg-lib>=0.6.0` binary for `voxel_grid` in the active environment.
- PointROPE CUDA was not available, so PTv3 used the PyTorch fallback path in the validated environment.

## Next Steps

1. Promote `extra-feature-scale=0.1` as the current IPFP control and test it on larger/full-frame evaluation.
2. Replace sparse LiDAR inpainting with the paper's full depth-estimator path.
3. Convert the script path into a proper SemanticKITTI dataloader and trainer.
4. Scale the gated route beyond `100/50`.
5. Add official-style full-frame mIoU evaluation on SemanticKITTI sequence `08`.
