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

## 50-Frame Visualization

The 50-frame run includes a selected-frame montage:

![50-frame selected montage](../results/semantic_kitti_repro/tiny_overfit_seq00_000000-000049_20260628_120319/selected_frames_final_montage.png)

## Important Caveats

- Tiny-overfit results are not benchmark results.
- The implementation uses a lightweight pseudo metric-depth field for the IPFP back-projection test path. A full paper-faithful experiment should replace this with a proper metric-depth source or the intended depth recovery pipeline.
- Pointcept PTv2 SemanticKITTI config was not used as the main route because it required a compatible `pyg-lib>=0.6.0` binary for `voxel_grid` in the active environment.
- PointROPE CUDA was not available, so PTv3 used the PyTorch fallback path in the validated environment.

## Next Steps

1. Convert tiny-overfit code into a proper SemanticKITTI dataloader and trainer.
2. Add a train/holdout split comparison for LiDAR-only vs PTv3+IPFP.
3. Replace pseudo metric depth with a real depth estimator or calibrated depth recovery.
4. Scale from tiny-overfit to train/validation split experiments.
5. Add quantitative mIoU evaluation on SemanticKITTI sequence 08.
