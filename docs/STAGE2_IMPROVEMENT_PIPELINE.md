# SemanticKITTI Stage-2 Improvement Pipeline

This stage upgrades the first full-frame validation scaffold into a stronger
benchmark path. The stage-1 LiDAR-only run reached `19.01%` mIoU on the full
SemanticKITTI sequence `08` validation split, but it still used a minimal sampled
training recipe. Stage 2 keeps the same validation target and improves the
training side before judging the gated IPFP fusion route.

## What Changed

The benchmark script now supports five improvement levers:

- Larger sampled training batches through `--train-sample-points`.
- Warmup plus cosine or polynomial learning-rate schedules.
- Class-frequency statistics, weighted cross entropy, and saved
  `class_stats.json`.
- Rare-class frame sampling plus optional class-balanced point sampling.
- Periodic small validation during training, followed by final full-frame
  sequence `08` validation.

The fused route can also initialize PTv3 weights from a completed LiDAR-only
checkpoint with `--init-model-from`, while leaving the IPFP projector freshly
initialized.

## Main Runner

Use:

```bash
ROOT=/root/autodl-tmp/ipfp_repro \
RUN_FUSED_AFTER_LIDAR=1 \
LIDAR_STEPS=60000 \
FUSED_STEPS=30000 \
bash scripts/run_semantic_kitti_stage2_improvement_pipeline.sh
```

Default stage-2 settings:

| Setting | Value |
| --- | --- |
| LiDAR steps | `60000` |
| Fused steps | `30000` |
| Train sample points | `32768` |
| Eval chunk points | `32768` |
| Periodic eval | every `5000` steps |
| Periodic eval subset | `256` frames, stride `8` |
| Final eval | full sequence `08` |
| LR schedule | cosine, `2000` warmup steps, min LR `1e-5` |
| Loss | CE + Lovasz |
| Class weighting | inverse square-root frequency, clipped at `5.0` |
| Rare-class frame sampling | probability `0.45` |
| Balanced point sampling | probability `0.50` |
| Spatial crop radius | `25.0` meters |
| Fused gate | IPFP `extra-feature-scale=0.1` |

## Active Remote Run

The first stage-2 run was launched on the remote server in a detached screen:

```bash
screen -ls
tail -f /root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage2_lidar_stronger_20260630_115149/run.log
```

Expected outputs:

- `/root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage2_lidar_stronger_20260630_115149`
- `/root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage2_fused_gate01_20260630_115149`

The fused directory appears after the LiDAR-only stage completes successfully.

## Interpretation

This stage should be read as a stronger validation benchmark, not yet an official
SemanticKITTI test submission. The key comparison is:

1. Does the stronger LiDAR-only recipe improve over `19.01%` full seq08 mIoU?
2. After initializing from that stronger LiDAR checkpoint, does gated IPFP with
   `extra-feature-scale=0.1` improve or damage full-frame mIoU?
3. Do rare classes improve without collapsing dominant classes such as road,
   building, vegetation, and car?

Only after this comparison is stable should the project move to official-style
test submission or larger fusion ablations.
