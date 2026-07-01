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
| Fused visible-point guard | at least `512` camera-visible positive-depth points |
| Fused sample retries | `12`, then LiDAR-only fallback for that step |
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
- `/root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage2_fused_gate01_recovery_20260630_160828`
- `/root/autodl-tmp/ipfp_repro/results/semantic_kitti_full_benchmark/stage2_fused_gate01_recovery2_20260630_162428`

The first fused directory exited early because an IPFP batch had no
camera-visible positive-depth points. The first recovery run exposed a second
IPFP sampling edge case where positive depth existed but no depth-constrained
center candidate could be sampled. The `recovery2` directory uses both fused
sampling guards and restarts from the completed LiDAR `best.pth`.

## LiDAR-Only Stage-2 Result

The stronger LiDAR-only run completed successfully:

| Split | Frames | mIoU | Overall accuracy | Mean loss |
| --- | ---: | ---: | ---: | ---: |
| Periodic validation at 55000 steps | 256 | `27.61%` | `78.58%` | `1.4824` |
| Full sequence `08` validation at 60000 steps | 4071 | `27.79%` | `75.59%` | `1.5952` |

This improves over the stage-1 full sequence `08` LiDAR-only result of
`19.01%` mIoU.

## Final Fused-Training Result

The recovery2 fused-training run completed successfully at `30000` steps. Its
primary evaluation route is still LiDAR-only inference, matching the paper-style
question of whether multimodal training improves a LiDAR model used at
inference time.

| Route | Eval route | Frames | mIoU | Overall accuracy | Mean loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage-2 stronger LiDAR-only | LiDAR-only | 4071 | `27.79%` | `75.59%` | `1.5952` |
| Stage-2 fused training | LiDAR-only | 4071 | `29.04%` | `80.12%` | `1.5005` |
| Stage-2 zero-feature fused path | LiDAR-only | 4071 | `28.93%` | `79.70%` | `1.5103` |
| Stage-2 IPFP detached branch | LiDAR-only | 4071 | `28.48%` | `79.22%` | `1.5408` |
| Stage-2 LiDAR continued | LiDAR-only | 4071 | `26.85%` | `73.40%` | `1.7084` |
| Stage-2 fused diagnostic | Fused | 4071 | `29.21%` | `80.15%` | `1.4922` |

The fused-training route improves the primary LiDAR-only inference result by
`+1.25` mIoU points and `+4.53` overall accuracy points over the stronger
LiDAR-only baseline. The diagnostic fused-inference route recorded `4066`
fallback counts, so it is not yet a clean fused-inference result.

The first controlled ablation replaced learned IPFP feature content with zeros
and still reached `28.93%` mIoU on the same full sequence `08` validation. This
recovers nearly all of the fused-training gain. The LiDAR-only continued
control reached only `26.85%`, below the original stronger LiDAR baseline, so
the current improvement is not explained by extra optimization time alone.
Together these results point to the fused path / added back-projected points as
the active mechanism, while weakening any claim that learned image semantics are
the main cause.

Full details, class-wise IoU, and visualizations are in
[`docs/STAGE2_FULL_BENCHMARK_RESULTS.md`](STAGE2_FULL_BENCHMARK_RESULTS.md).
The ablation breakdown is tracked in
[`docs/STAGE2_ABLATION_RESULTS.md`](STAGE2_ABLATION_RESULTS.md).

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
