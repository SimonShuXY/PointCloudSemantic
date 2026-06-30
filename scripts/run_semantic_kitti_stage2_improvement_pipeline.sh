#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

LIDAR_STEPS="${LIDAR_STEPS:-60000}"
FUSED_STEPS="${FUSED_STEPS:-30000}"
RUN_FUSED_AFTER_LIDAR="${RUN_FUSED_AFTER_LIDAR:-1}"

TRAIN_SAMPLE_POINTS="${TRAIN_SAMPLE_POINTS:-32768}"
EVAL_CHUNK_POINTS="${EVAL_CHUNK_POINTS:-32768}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
PERIODIC_VAL_FRAME_LIMIT="${PERIODIC_VAL_FRAME_LIMIT:-256}"
PERIODIC_VAL_FRAME_STRIDE="${PERIODIC_VAL_FRAME_STRIDE:-8}"

LR="${LR:-5e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
MIN_LR="${MIN_LR:-1e-5}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-inverse_sqrt_freq}"
CLASS_WEIGHT_CLIP="${CLASS_WEIGHT_CLIP:-5.0}"
RARE_CLASS_SAMPLING_PROB="${RARE_CLASS_SAMPLING_PROB:-0.45}"
BALANCED_POINT_SAMPLING_PROB="${BALANCED_POINT_SAMPLING_PROB:-0.50}"
SPATIAL_CROP_RADIUS="${SPATIAL_CROP_RADIUS:-25.0}"

BASE_DIR="${BASE_DIR:-$ROOT/results/semantic_kitti_full_benchmark}"
LIDAR_OUT="${LIDAR_OUT:-$BASE_DIR/stage2_lidar_stronger_${RUN_STAMP}}"
FUSED_OUT="${FUSED_OUT:-$BASE_DIR/stage2_fused_gate01_${RUN_STAMP}}"

COMMON_ARGS=(
  --root "$ROOT"
  --eval-every "$EVAL_EVERY"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --train-sample-points "$TRAIN_SAMPLE_POINTS"
  --eval-chunk-points "$EVAL_CHUNK_POINTS"
  --periodic-val-frame-limit "$PERIODIC_VAL_FRAME_LIMIT"
  --periodic-val-frame-stride "$PERIODIC_VAL_FRAME_STRIDE"
  --lr "$LR"
  --lr-schedule "$LR_SCHEDULE"
  --warmup-steps "$WARMUP_STEPS"
  --min-lr "$MIN_LR"
  --loss-mode ce-lovasz
  --depth-mode lidar-inpaint
  --class-weight-mode "$CLASS_WEIGHT_MODE"
  --class-weight-clip "$CLASS_WEIGHT_CLIP"
  --rare-class-sampling-prob "$RARE_CLASS_SAMPLING_PROB"
  --balanced-point-sampling-prob "$BALANCED_POINT_SAMPLING_PROB"
  --spatial-crop-radius "$SPATIAL_CROP_RADIUS"
)

mkdir -p "$LIDAR_OUT"
echo "[stage2] stronger lidar output: $LIDAR_OUT"
"$PYTHON" "$ROOT/scripts/semantic_kitti_full_benchmark.py" \
  "${COMMON_ARGS[@]}" \
  --steps "$LIDAR_STEPS" \
  --mode lidar-only \
  --eval-route same \
  --output-dir "$LIDAR_OUT" > "$LIDAR_OUT/run.log" 2>&1

if [[ "$RUN_FUSED_AFTER_LIDAR" != "1" ]]; then
  echo "[stage2] fused route disabled; stopping after lidar baseline"
  exit 0
fi

mkdir -p "$FUSED_OUT"
echo "[stage2] gated fused output: $FUSED_OUT"
"$PYTHON" "$ROOT/scripts/semantic_kitti_full_benchmark.py" \
  "${COMMON_ARGS[@]}" \
  --steps "$FUSED_STEPS" \
  --mode fused \
  --eval-route both \
  --num-centers 32 \
  --ipfp-lower-percentile 20 \
  --ipfp-upper-percentile 99 \
  --ipfp-discard-probability 0.2 \
  --extra-feature-scale 0.1 \
  --init-model-from "$LIDAR_OUT/checkpoints/best.pth" \
  --output-dir "$FUSED_OUT" > "$FUSED_OUT/run.log" 2>&1

echo "[stage2] complete"
echo "[stage2] lidar: $LIDAR_OUT"
echo "[stage2] fused: $FUSED_OUT"
