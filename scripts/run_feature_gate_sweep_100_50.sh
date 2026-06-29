#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEQUENCE="${SEQUENCE:-00}"
STEPS="${STEPS:-1000}"
NUM_POINTS="${NUM_POINTS:-2048}"
IMAGE_WIDTH="${IMAGE_WIDTH:-480}"
VIZ_FRAME_COUNT="${VIZ_FRAME_COUNT:-4}"
SCALES="${SCALES:-0.0 0.1 0.25 0.5 1.0}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$ROOT/results/semantic_kitti_repro/feature_gate100_holdout50_$RUN_STAMP}"
mkdir -p "$OUT_BASE"

TRAIN_FRAMES=()
for i in $(seq -f "%06g" 0 99); do
  TRAIN_FRAMES+=("$i")
done

EVAL_FRAMES=()
for i in $(seq -f "%06g" 100 149); do
  EVAL_FRAMES+=("$i")
done

COMMON_ARGS=(
  --root "$ROOT"
  --sequence "$SEQUENCE"
  --frames "${TRAIN_FRAMES[@]}"
  --eval-frames "${EVAL_FRAMES[@]}"
  --num-points "$NUM_POINTS"
  --num-centers 32
  --image-width "$IMAGE_WIDTH"
  --steps "$STEPS"
  --mode fused
  --eval-route both
  --depth-mode lidar-inpaint
  --loss-mode ce-lovasz
  --ipfp-lower-percentile 20
  --ipfp-upper-percentile 99
  --ipfp-discard-probability 0.2
  --viz-frame-count "$VIZ_FRAME_COUNT"
)

echo "[feature-gate] output base: $OUT_BASE"
for scale in $SCALES; do
  label="${scale//./p}"
  out_dir="$OUT_BASE/scale_${label}"
  log_path="$OUT_BASE/scale_${label}.log"
  echo "[feature-gate] running scale=$scale"
  "$PYTHON" "$ROOT/scripts/semantic_kitti_ipfp_tiny_overfit.py" \
    "${COMMON_ARGS[@]}" \
    --extra-feature-scale "$scale" \
    --output-dir "$out_dir" \
    > "$log_path" 2>&1
done

echo "[feature-gate] complete: $OUT_BASE"
