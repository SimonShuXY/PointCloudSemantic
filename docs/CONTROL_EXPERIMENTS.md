# Control Experiments

## 50-Frame LiDAR-Only vs PTv3+IPFP Tiny Overfit

This control experiment uses the same SemanticKITTI frames and training schedule for both routes:

- Sequence: `00`
- Frames: `000000-000049`
- Points per frame: `2048`
- Steps: `500`
- Learning rate: `5e-4`
- Model route A: PTv3 LiDAR-only
- Model route B: PTv3 + minimal IPFP extra points/features

## Results

| Route | Initial mean eval loss | Final mean eval loss | Initial mean eval acc | Final mean eval acc | Final train loss | Best train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTv3 LiDAR-only | 3.4867 | 0.3241 | 4.92% | 89.36% | 0.1713 | 0.1298 |
| PTv3 + IPFP | 3.4847 | 0.3492 | 4.94% | 89.14% | 0.1468 | 0.1014 |

## Interpretation

Both routes can overfit the same 50-frame SemanticKITTI slice, which confirms that the core PTv3 training loop is healthy. In this minimal setup, the LiDAR-only baseline slightly outperforms the PTv3+IPFP route on final mean eval loss and final mean eval accuracy.

This does not disprove IPFP. It shows that the current minimal IPFP implementation and pseudo metric-depth setup are not yet providing measurable benefit over the strong LiDAR-only PTv3 baseline in a tiny-overfit setting.

Likely reasons:

- The current depth input is a lightweight pseudo metric-depth field, not a full metric-depth estimator.
- The IPFP image feature extractor is intentionally small and trained only in the tiny-overfit loop.
- The tiny-overfit metric measures memorization, not generalization.
- Extra back-projected image points may introduce noisy features before a better depth/image feature pipeline is added.

## Artifacts

LiDAR-only:

- `results/semantic_kitti_repro/tiny_overfit_lidar-only_seq00_000000-000049_20260629_111458/summary.json`
- `results/semantic_kitti_repro/tiny_overfit_lidar-only_seq00_000000-000049_20260629_111458/loss_curve.png`
- `results/semantic_kitti_repro/tiny_overfit_lidar-only_seq00_000000-000049_20260629_111458/selected_frames_final_montage.png`

PTv3 + IPFP:

- `results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/summary.json`
- `results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/loss_curve.png`
- `results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/selected_frames_final_montage.png`

## Next Experiment

The next useful step was a small train/holdout comparison:

- Train: `seq00` frames `000000-000049`
- Holdout eval: `seq00` frames `000050-000069`
- Compare LiDAR-only vs PTv3+IPFP on holdout loss/accuracy and qualitative BEV predictions.

This has now been completed. The PTv3+IPFP route matched train-frame accuracy and
improved holdout accuracy in the first small split; see `HOLDOUT_EXPERIMENTS.md`.
