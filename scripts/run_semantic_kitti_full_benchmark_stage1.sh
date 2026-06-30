#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODE="${MODE:-lidar-only}"
STEPS="${STEPS:-20000}"
EVAL_EVERY="${EVAL_EVERY:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
TRAIN_SAMPLE_POINTS="${TRAIN_SAMPLE_POINTS:-8192}"
EVAL_CHUNK_POINTS="${EVAL_CHUNK_POINTS:-16384}"
PERIODIC_VAL_FRAME_STRIDE="${PERIODIC_VAL_FRAME_STRIDE:-1}"
PERIODIC_VAL_FRAME_LIMIT="${PERIODIC_VAL_FRAME_LIMIT:-}"
VAL_FRAME_STRIDE="${VAL_FRAME_STRIDE:-1}"
VAL_FRAME_LIMIT="${VAL_FRAME_LIMIT:-}"
TRAIN_FRAME_STRIDE="${TRAIN_FRAME_STRIDE:-1}"
TRAIN_FRAME_LIMIT="${TRAIN_FRAME_LIMIT:-}"
LR="${LR:-5e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MIN_LR="${MIN_LR:-0.0}"
POLY_POWER="${POLY_POWER:-0.9}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"
CLASS_WEIGHT_CLIP="${CLASS_WEIGHT_CLIP:-5.0}"
RARE_CLASS_SAMPLING_PROB="${RARE_CLASS_SAMPLING_PROB:-0.0}"
BALANCED_POINT_SAMPLING_PROB="${BALANCED_POINT_SAMPLING_PROB:-0.0}"
SPATIAL_CROP_RADIUS="${SPATIAL_CROP_RADIUS:-0.0}"
INIT_MODEL_FROM="${INIT_MODEL_FROM:-}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$ROOT/results/semantic_kitti_full_benchmark/stage1_${MODE}_${RUN_STAMP}}"

mkdir -p "$OUT_BASE"

ARGS=(
  --root "$ROOT"
  --mode "$MODE"
  --steps "$STEPS"
  --eval-every "$EVAL_EVERY"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --train-sample-points "$TRAIN_SAMPLE_POINTS"
  --eval-chunk-points "$EVAL_CHUNK_POINTS"
  --periodic-val-frame-stride "$PERIODIC_VAL_FRAME_STRIDE"
  --train-frame-stride "$TRAIN_FRAME_STRIDE"
  --val-frame-stride "$VAL_FRAME_STRIDE"
  --lr "$LR"
  --lr-schedule "$LR_SCHEDULE"
  --warmup-steps "$WARMUP_STEPS"
  --min-lr "$MIN_LR"
  --poly-power "$POLY_POWER"
  --loss-mode ce-lovasz
  --depth-mode lidar-inpaint
  --class-weight-mode "$CLASS_WEIGHT_MODE"
  --class-weight-clip "$CLASS_WEIGHT_CLIP"
  --rare-class-sampling-prob "$RARE_CLASS_SAMPLING_PROB"
  --balanced-point-sampling-prob "$BALANCED_POINT_SAMPLING_PROB"
  --spatial-crop-radius "$SPATIAL_CROP_RADIUS"
  --output-dir "$OUT_BASE"
)

if [[ -n "$TRAIN_FRAME_LIMIT" ]]; then
  ARGS+=(--train-frame-limit "$TRAIN_FRAME_LIMIT")
fi
if [[ -n "$VAL_FRAME_LIMIT" ]]; then
  ARGS+=(--val-frame-limit "$VAL_FRAME_LIMIT")
fi
if [[ -n "$PERIODIC_VAL_FRAME_LIMIT" ]]; then
  ARGS+=(--periodic-val-frame-limit "$PERIODIC_VAL_FRAME_LIMIT")
fi
if [[ -n "$INIT_MODEL_FROM" ]]; then
  ARGS+=(--init-model-from "$INIT_MODEL_FROM")
fi

if [[ "$MODE" == "fused" ]]; then
  ARGS+=(
    --eval-route both
    --num-centers 32
    --ipfp-lower-percentile 20
    --ipfp-upper-percentile 99
    --ipfp-discard-probability 0.2
    --extra-feature-scale 0.1
  )
else
  ARGS+=(--eval-route same)
fi

echo "[full-benchmark-stage1] output: $OUT_BASE"
echo "[full-benchmark-stage1] mode=$MODE steps=$STEPS val_stride=$VAL_FRAME_STRIDE val_limit=${VAL_FRAME_LIMIT:-all}"
echo "[full-benchmark-stage1] lr=$LR schedule=$LR_SCHEDULE train_points=$TRAIN_SAMPLE_POINTS class_weight=$CLASS_WEIGHT_MODE rare_prob=$RARE_CLASS_SAMPLING_PROB"
"$PYTHON" "$ROOT/scripts/semantic_kitti_full_benchmark.py" "${ARGS[@]}" > "$OUT_BASE/run.log" 2>&1
echo "[full-benchmark-stage1] complete: $OUT_BASE"
