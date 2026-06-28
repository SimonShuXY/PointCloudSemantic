# Dataset Download Links

This file records the public dataset URLs used for the reproduction setup. Raw datasets are not included in this repository.

## nuScenes Mini

Used for the early real-data closed-loop smoke test.

| File | URL |
| --- | --- |
| nuScenes v1.0 mini | `https://www.nuscenes.org/data/v1.0-mini.tgz` |
| nuScenes lidarseg mini | `https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2` |

## SemanticKITTI Labels

Used with KITTI odometry LiDAR, color images, calibration, and poses.

| File | URL |
| --- | --- |
| SemanticKITTI labels | `http://www.semantic-kitti.org/assets/data_odometry_labels.zip` |

## KITTI Odometry Data

The SemanticKITTI experiments require the KITTI odometry sensor data and calibration files.

| File | URL |
| --- | --- |
| Calibration | `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip` |
| Poses | `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip` |
| Color images | `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip` |
| Velodyne point clouds | `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip` |

## Download Helper

The helper script `scripts/download_nuscenes_mini_semantickitti.sh` uses the links above and extracts the archives into:

```text
data/
  archives/
  nuscenes_mini/raw/
  semantic_kitti/
```

Example:

```bash
ROOT=/path/to/ipfp_repro bash scripts/download_nuscenes_mini_semantickitti.sh
```

For slow KITTI S3 downloads, `scripts/resume_data_download_resilient.sh` and `scripts/resume_data_download_tuned.sh` contain resumable `aria2c` variants.

## Disk Notes

The full KITTI odometry velodyne archive is large, and the extracted SemanticKITTI/KITTI layout with color images needs a large data disk. In the validated run, a 400 GB class data disk was used for full KITTI odometry velodyne, color images, labels, calibration, and nuScenes-mini.

## License and Terms

Follow the official nuScenes, KITTI, and SemanticKITTI license and citation requirements. This repository only records links and reproduction scripts; it does not redistribute dataset files.
