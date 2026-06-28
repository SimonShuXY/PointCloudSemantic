#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
ARCHIVES="$ROOT/data/archives"
NUSC_MINI="$ROOT/data/nuscenes_mini"
SK="$ROOT/data/semantic_kitti"
POINTCEPT_ROOT="$ROOT/src/Pointcept"

mkdir -p "$ARCHIVES" "$NUSC_MINI/raw" "$SK" "$POINTCEPT_ROOT/data"

download_tuned() {
  local url="$1"
  local output="$2"
  echo
  echo "== tuned download: $output =="
  date
  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=64 \
    --min-split-size=1M \
    --file-allocation=none \
    --summary-interval=60 \
    --download-result=full \
    --max-tries=0 \
    --retry-wait=3 \
    --timeout=20 \
    --connect-timeout=10 \
    --lowest-speed-limit=128K \
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

download_tuned "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip" "data_odometry_color.zip"
download_tuned "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" "data_odometry_velodyne.zip"

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
