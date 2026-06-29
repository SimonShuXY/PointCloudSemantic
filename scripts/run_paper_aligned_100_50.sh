#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEQUENCE="${SEQUENCE:-00}"
STEPS="${STEPS:-1000}"
NUM_POINTS="${NUM_POINTS:-2048}"
IMAGE_WIDTH="${IMAGE_WIDTH:-480}"
VIZ_FRAME_COUNT="${VIZ_FRAME_COUNT:-8}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${OUT_BASE:-$ROOT/results/semantic_kitti_repro/paper_aligned100_holdout50_$RUN_STAMP}"
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
  --image-width "$IMAGE_WIDTH"
  --steps "$STEPS"
  --loss-mode ce-lovasz
  --depth-mode lidar-inpaint
  --viz-frame-count "$VIZ_FRAME_COUNT"
)

echo "[paper-aligned] output base: $OUT_BASE"
echo "[paper-aligned] running LiDAR-only CE+Lovasz baseline"
"$PYTHON" "$ROOT/scripts/semantic_kitti_ipfp_tiny_overfit.py" \
  "${COMMON_ARGS[@]}" \
  --mode lidar-only \
  --eval-route same \
  --output-dir "$OUT_BASE/lidar_only_ce_lovasz" \
  > "$OUT_BASE/lidar_only_ce_lovasz.log" 2>&1

echo "[paper-aligned] running fused training with LiDAR-only primary eval and fused diagnostic eval"
"$PYTHON" "$ROOT/scripts/semantic_kitti_ipfp_tiny_overfit.py" \
  "${COMMON_ARGS[@]}" \
  --mode fused \
  --eval-route both \
  --num-centers 32 \
  --ipfp-lower-percentile 20 \
  --ipfp-upper-percentile 99 \
  --ipfp-discard-probability 0.2 \
  --output-dir "$OUT_BASE/fused_train_lidar_eval" \
  > "$OUT_BASE/fused_train_lidar_eval.log" 2>&1

echo "[paper-aligned] complete: $OUT_BASE"
