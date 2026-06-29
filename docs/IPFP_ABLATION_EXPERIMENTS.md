# IPFP Ablation Experiments

## Goal

The expanded `100 train / 50 holdout` split showed that the minimal PTv3+IPFP route
underperformed LiDAR-only. This ablation set tests whether the issue comes from:

- too many extra back-projected points,
- noisy learned image features,
- or unstable end-to-end coupling between IPFP and PTv3.

All runs use the same SemanticKITTI sequence `00` split:

- Train frames: `000000-000099`
- Holdout eval frames: `000100-000149`
- Points per frame: `2048`
- Steps: `1000`
- Learning rate: `5e-4`
- Image width: `480`

## Compared Routes

| Route | Centers | Extra feature mode | IPFP detach | Train mIoU | Holdout loss | Holdout overall acc | Holdout mIoU |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| PTv3 LiDAR-only baseline | n/a | n/a | n/a | 48.55% | 1.7897 | 61.51% | 15.30% |
| PTv3 + IPFP learned features | 64 | learned | no | 41.76% | 2.3918 | 56.67% | 13.41% |
| PTv3 + IPFP learned features | 32 | learned | no | 47.12% | 1.9500 | 58.58% | 14.01% |
| PTv3 + IPFP learned features | 16 | learned | no | 47.02% | 1.9984 | 59.70% | 14.70% |
| PTv3 + IPFP zero extra features | 64 | zeros | no | 47.16% | 1.8024 | 61.95% | 15.33% |
| PTv3 + IPFP detached branch | 64 | learned | yes | 47.89% | 1.8951 | 61.58% | 15.10% |

## Diagnosis

Reducing the number of extra IPFP centers helps. Moving from `64` learned-feature
centers to `32` and `16` raises holdout mIoU from `13.41%` to `14.01%` and `14.70%`.
That suggests the extra back-projected tokens introduce noise when used too densely.

The strongest result is `zero extra features, 64 centers`: holdout mIoU reaches
`15.33%`, slightly above the LiDAR-only baseline at `15.30%`. This means the extra
tokens/coordinates are not inherently harmful. The harmful part is the current learned
image feature content attached to those tokens.

The detached branch also recovers most of the gap, reaching `15.10%` holdout mIoU and
`61.58%` holdout overall accuracy. This suggests end-to-end gradients through the
current IPFP branch are another source of instability.

## Class-Level Notes

Against the original learned-feature fused route, `zero-feat64` improves major classes:

| Class | LiDAR-only IoU | Fused64 learned IoU | Zero-feat64 IoU | Zero-feat64 vs Fused64 |
| --- | ---: | ---: | ---: | ---: |
| road | 65.71% | 67.85% | 69.23% | +1.38 |
| sidewalk | 40.67% | 42.68% | 43.84% | +1.16 |
| car | 41.25% | 35.16% | 42.96% | +7.80 |
| building | 49.41% | 34.70% | 47.23% | +12.53 |
| vegetation | 35.31% | 24.40% | 32.90% | +8.50 |

It still trails LiDAR-only on building and vegetation, so the extra-token geometry is
not a complete fix. But it removes most of the damage caused by learned image features.

## Interpretation

The current minimal IPFP path should not be scaled as-is. The next fix should target
the image feature branch and depth/projection quality before larger training:

- use a stronger or pretrained image encoder,
- replace pseudo metric depth with a more reliable depth source,
- gate or down-weight extra image tokens,
- keep a detached or staged training option,
- and keep `num_centers` as a tunable noise-control knob.

## Artifacts

- `results/semantic_kitti_repro/ipfp_ablation100_holdout50_20260629_163231/ablation_centers32_seq00_train000000-000099_eval000100-000149/summary.json`
- `results/semantic_kitti_repro/ipfp_ablation100_holdout50_20260629_163231/ablation_centers16_seq00_train000000-000099_eval000100-000149/summary.json`
- `results/semantic_kitti_repro/ipfp_ablation100_holdout50_20260629_163231/ablation_zero-feat64_seq00_train000000-000099_eval000100-000149/summary.json`
- `results/semantic_kitti_repro/ipfp_ablation100_holdout50_20260629_163231/ablation_detach64_seq00_train000000-000099_eval000100-000149/summary.json`

## Next Steps

1. Add an image-feature gate or scalar weight and sweep `0.0, 0.1, 0.25, 0.5, 1.0`.
2. Keep the best ablation, `zero-feat64`, as a geometry-only control.
3. Replace pseudo metric depth before attempting a larger split.
4. Try staged training: train LiDAR-only first, then enable IPFP with a low feature gate.
