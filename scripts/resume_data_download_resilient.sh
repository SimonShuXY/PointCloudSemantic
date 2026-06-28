#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/ipfp_repro}"
ARCHIVES="$ROOT/data/archives"
NUSC_MINI="$ROOT/data/nuscenes_mini"
SK="$ROOT/data/semantic_kitti"
POINTCEPT_ROOT="$ROOT/src/Pointcept"
ROUND_SECONDS="${ROUND_SECONDS:-300}"
USE_NETWORK_TURBO="${USE_NETWORK_TURBO:-0}"

mkdir -p "$ARCHIVES" "$NUSC_MINI/raw" "$SK" "$POINTCEPT_ROOT/data"

if [[ "$USE_NETWORK_TURBO" == "1" && -f /etc/network_turbo ]]; then
  # Mainly useful for GitHub/HuggingFace; keep opt-in for S3/KITTI downloads.
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
else
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
fi

download_resilient() {
  local url="$1"
  local output="$2"
  local path="$ARCHIVES/$output"
  local control="$path.aria2"
  local round=1

  echo
  echo "== resilient download: $output =="
  date
  echo "round_seconds=$ROUND_SECONDS use_network_turbo=$USE_NETWORK_TURBO"

  while true; do
    if [[ -f "$path" && ! -f "$control" ]]; then
      echo "already complete: $path"
      return 0
    fi

    echo
    echo "-- aria2 round $round for $output --"
    date
    set +e
    timeout --kill-after=30s "${ROUND_SECONDS}s" \
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
        --allow-overwrite=true \
        --auto-file-renaming=false \
        -d "$ARCHIVES" \
        -o "$output" \
        "$url"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
      echo "download complete: $output"
      return 0
    fi

    if [[ -f "$path" && ! -f "$control" ]]; then
      echo "download appears complete after rc=$rc: $output"
      return 0
    fi

    echo "round $round ended with rc=$rc; preserving partial file and resuming shortly"
    ls -lh "$path" "$control" 2>/dev/null || true
    round=$((round + 1))
    sleep 5
  done
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

download_resilient "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip" "data_odometry_color.zip"
download_resilient "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" "data_odometry_velodyne.zip"

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
