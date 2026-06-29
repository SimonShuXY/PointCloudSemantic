# Train/Holdout Experiments

## 50-Frame Train, 20-Frame Holdout

This experiment is the next step after tiny-overfit. It keeps the small reproduction
scale but separates optimization frames from evaluation-only frames:

- Dataset slice: SemanticKITTI sequence `00`
- Train frames: `000000-000049`
- Holdout eval frames: `000050-000069`
- Points per frame: `2048`
- IPFP centers: `64`
- Steps: `500`
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Image width: `480`
- Routes compared: PTv3 LiDAR-only vs PTv3 + minimal IPFP extra points/features

## Results

| Route | Train loss, mean eval | Train acc, mean eval | Holdout loss, mean eval | Holdout acc, mean eval | Final train loss | CUDA peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTv3 LiDAR-only | 3.4868 -> 0.3519 | 4.94% -> 89.31% | 3.6132 -> 0.7379 | 3.20% -> 82.60% | 0.1564 | 0.902 GB |
| PTv3 + IPFP | 3.4846 -> 0.3361 | 4.94% -> 89.31% | 3.6101 -> 0.6166 | 3.17% -> 84.94% | 0.1467 | 0.957 GB |

## Interpretation

The earlier pure tiny-overfit control slightly favored LiDAR-only. That was a
memorization check, so it mainly confirmed that both training routes can learn the
same small frame set.

This train/holdout check is more informative. On the training frames, both routes end
at essentially the same mean accuracy. On held-out adjacent frames, the PTv3+IPFP
route improves final mean accuracy by about 2.34 percentage points and lowers final
mean loss by about 0.1213.

This is an encouraging first signal that image back-projection may help beyond
memorizing the train frames. It is not yet a benchmark result.

## Caveats

- The split is small and uses adjacent frames from one sequence, so temporal
  correlation is high.
- The metric is point accuracy on the sampled points, not official SemanticKITTI
  mIoU.
- The IPFP branch still uses the lightweight pseudo metric-depth path from the
  reproduction scaffold.
- Only one seed has been reported so far.

## Artifacts

LiDAR-only:

- `results/semantic_kitti_repro/tiny_holdout_lidar-only_seq00_train000000-000049_eval000050-000069_20260629_120129/summary.json`
- `results/semantic_kitti_repro/tiny_holdout_lidar-only_seq00_train000000-000049_eval000050-000069_20260629_120129/loss_curve.png`
- `results/semantic_kitti_repro/tiny_holdout_lidar-only_seq00_train000000-000049_eval000050-000069_20260629_120129/holdout_selected_frames_final_montage.png`

PTv3 + IPFP:

- `results/semantic_kitti_repro/tiny_holdout_fused_seq00_train000000-000049_eval000050-000069_20260629_120345/summary.json`
- `results/semantic_kitti_repro/tiny_holdout_fused_seq00_train000000-000049_eval000050-000069_20260629_120345/loss_curve.png`
- `results/semantic_kitti_repro/tiny_holdout_fused_seq00_train000000-000049_eval000050-000069_20260629_120345/holdout_selected_frames_final_montage.png`

## Next Steps

The direct expansion has been run as `100 train / 50 holdout` with sampled mIoU.
That larger split favors LiDAR-only rather than the current fused route; see
`EXPANDED_SPLIT_EXPERIMENTS.md`.

1. Diagnose the fused route with IPFP ablations on the `100/50` split.
2. Replace the pseudo metric-depth path with a stronger calibrated depth source.
3. Move from sampled-point mIoU to full-frame SemanticKITTI-style mIoU.
4. Turn the script path into a proper dataloader/trainer before full sequence `08` evaluation.
