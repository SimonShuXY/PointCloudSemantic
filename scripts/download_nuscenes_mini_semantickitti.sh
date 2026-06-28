#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
ARCHIVES="$ROOT/data/archives"
NUSC_MINI="$ROOT/data/nuscenes_mini"
SK="$ROOT/data/semantic_kitti"
LOG_DIR="$ROOT/logs/downloads"
POINTCEPT_ROOT="$ROOT/src/Pointcept"

mkdir -p "$ARCHIVES" "$NUSC_MINI/raw" "$SK" "$LOG_DIR" "$POINTCEPT_ROOT/data"

download() {
  local url="$1"
  local output="$2"
  local connections="${3:-16}"
  echo
  echo "== download: $output =="
  date
  aria2c \
    --continue=true \
    --max-connection-per-server="$connections" \
    --split="$connections" \
    --min-split-size=4M \
    --file-allocation=none \
    --summary-interval=60 \
    --download-result=full \
    -d "$ARCHIVES" \
    -o "$output" \
    "$url"
}

extract_zip() {
  local archive="$1"
  local dest="$2"
  echo
  echo "== unzip: $(basename "$archive") -> $dest =="
  date
  unzip -q -n "$archive" -d "$dest"
}

extract_tar() {
  local archive="$1"
  local dest="$2"
  echo
  echo "== tar extract: $(basename "$archive") -> $dest =="
  date
  tar -xf "$archive" -C "$dest"
}

echo "ROOT=$ROOT"
df -h "$ROOT" || true

download "https://www.nuscenes.org/data/v1.0-mini.tgz" "v1.0-mini.tgz" 16
download "https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2" "nuScenes-lidarseg-mini-v1.0.tar.bz2" 8
download "http://www.semantic-kitti.org/assets/data_odometry_labels.zip" "data_odometry_labels.zip" 8
download "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip" "data_odometry_calib.zip" 4
download "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip" "data_odometry_poses.zip" 4
download "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip" "data_odometry_color.zip" 16
download "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" "data_odometry_velodyne.zip" 16

extract_tar "$ARCHIVES/v1.0-mini.tgz" "$NUSC_MINI/raw"
extract_tar "$ARCHIVES/nuScenes-lidarseg-mini-v1.0.tar.bz2" "$NUSC_MINI/raw"

extract_zip "$ARCHIVES/data_odometry_calib.zip" "$SK"
extract_zip "$ARCHIVES/data_odometry_poses.zip" "$SK"
extract_zip "$ARCHIVES/data_odometry_color.zip" "$SK"
extract_zip "$ARCHIVES/data_odometry_velodyne.zip" "$SK"
extract_zip "$ARCHIVES/data_odometry_labels.zip" "$SK"

ln -sfn "$SK" "$POINTCEPT_ROOT/data/semantic_kitti"
ln -sfn "$NUSC_MINI" "$POINTCEPT_ROOT/data/nuscenes_mini"

echo
echo "== final layout =="
date
du -sh "$ARCHIVES" "$NUSC_MINI" "$SK" 2>/dev/null || true
df -h "$ROOT" || true
find "$NUSC_MINI/raw" -maxdepth 2 -type d | sort | sed -n '1,80p'
find "$SK/dataset/sequences" -maxdepth 2 -type d | sort | sed -n '1,80p'

echo
echo "download_and_extract_complete"
