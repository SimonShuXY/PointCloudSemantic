#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ipfp_repro}
PC="$ROOT/src/Pointcept"

printf "== host ==\n"
hostname

printf "== gpu ==\n"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

printf "== disk ==\n"
df -h /root/autodl-tmp

printf "== project paths ==\n"
for path in \
  "$ROOT" \
  "$PC" \
  "$ROOT/.venv" \
  "$ROOT/ipfp_minimal" \
  "$ROOT/scripts" \
  "$ROOT/data/semantic_kitti" \
  "$ROOT/data/nuscenes_mini"
do
  if [ -e "$path" ]; then
    printf "OK %s\n" "$path"
  else
    printf "MISSING %s\n" "$path"
  fi
done

printf "== pointcept git ==\n"
cd "$PC"
git rev-parse --short HEAD || true
git status --short || true

printf "== python env ==\n"
source "$ROOT/.venv/bin/activate"
python - <<'PY'
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

printf "== scripts ==\n"
for file in \
  "$ROOT/scripts/semantic_kitti_ipfp_visualize.py" \
  "$ROOT/scripts/semantic_kitti_native_smoke.py" \
  "$ROOT/scripts/nuscenes_mini_closed_loop.py"
do
  if [ -s "$file" ]; then
    printf "OK %s\n" "$file"
  else
    printf "MISSING_OR_EMPTY %s\n" "$file"
  fi
done

printf "== semantic kitti counts ==\n"
python - <<'PY'
from pathlib import Path

root = Path("/root/autodl-tmp/ipfp_repro/data/semantic_kitti/dataset/sequences")
print("sequences_dir", root, root.exists())
if root.exists():
    seqs = [f"{i:02d}" for i in range(22)]
    total = {"velodyne": 0, "image_2": 0, "image_3": 0, "labels": 0, "calib": 0}
    bad = []
    for seq in seqs:
        base = root / seq
        counts = {
            "velodyne": len(list((base / "velodyne").glob("*.bin"))) if (base / "velodyne").exists() else 0,
            "image_2": len(list((base / "image_2").glob("*.png"))) if (base / "image_2").exists() else 0,
            "image_3": len(list((base / "image_3").glob("*.png"))) if (base / "image_3").exists() else 0,
            "labels": len(list((base / "labels").glob("*.label"))) if (base / "labels").exists() else 0,
            "calib": 1 if (base / "calib.txt").exists() else 0,
        }
        for key, value in counts.items():
            total[key] += value
        if (
            counts["velodyne"] == 0
            or counts["image_2"] != counts["velodyne"]
            or counts["image_3"] != counts["velodyne"]
            or counts["calib"] != 1
        ):
            bad.append((seq, counts))
        if int(seq) <= 10 and counts["labels"] != counts["velodyne"]:
            bad.append((seq, counts))
    sample = root / "00" / "velodyne" / "000000.bin"
    print("total", total)
    print("bad_count", len(bad))
    print("bad_first10", bad[:10])
    print("sample_00_000000", sample.exists(), sample.stat().st_size if sample.exists() else None)
PY

printf "== existing latest results ==\n"
readlink -f "$ROOT/results/semantic_kitti_repro/LATEST" 2>/dev/null || true
find "$ROOT/results/semantic_kitti_repro" -maxdepth 3 -type f \
  \( -name "summary.json" -o -name "REPRO_NOTES.md" -o -name "*.png" \) 2>/dev/null \
  | sort \
  | tail -n 20 || true
