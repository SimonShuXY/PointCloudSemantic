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
VAL_FRAME_STRIDE="${VAL_FRAME_STRIDE:-1}"
VAL_FRAME_LIMIT="${VAL_FRAME_LIMIT:-}"
TRAIN_FRAME_STRIDE="${TRAIN_FRAME_STRIDE:-1}"
TRAIN_FRAME_LIMIT="${TRAIN_FRAME_LIMIT:-}"
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
  --train-frame-stride "$TRAIN_FRAME_STRIDE"
  --val-frame-stride "$VAL_FRAME_STRIDE"
  --loss-mode ce-lovasz
  --depth-mode lidar-inpaint
  --output-dir "$OUT_BASE"
)

if [[ -n "$TRAIN_FRAME_LIMIT" ]]; then
  ARGS+=(--train-frame-limit "$TRAIN_FRAME_LIMIT")
fi
if [[ -n "$VAL_FRAME_LIMIT" ]]; then
  ARGS+=(--val-frame-limit "$VAL_FRAME_LIMIT")
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
"$PYTHON" "$ROOT/scripts/semantic_kitti_full_benchmark.py" "${ARGS[@]}" > "$OUT_BASE/run.log" 2>&1
echo "[full-benchmark-stage1] complete: $OUT_BASE"
