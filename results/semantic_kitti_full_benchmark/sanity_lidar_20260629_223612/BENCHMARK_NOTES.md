# SemanticKITTI Full-Frame Benchmark Notes

## Scope

This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.

## Summary

- Mode: lidar-only
- Primary eval route: lidar-only
- Diagnostic eval route: none
- Train frames: 4
- Val frames: 2
- Train sample points per step: 8192
- Eval chunk points: 8192
- Steps completed: 2
- Best val mIoU: 0.040932

## Final Validation

- Route: lidar-only
- Frames: 2
- Mean loss: 3.628789
- mIoU: 0.040932
- Overall accuracy: 0.289002
- Valid points: 232805


## Final Validation Class IoU

- 00 car: IoU 0.007240, acc 0.007561, gt 3174, pred 165, tp 24
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 39, pred 1, tp 0
- 02 motorcycle: IoU 0.000000, acc n/a, gt 0, pred 1, tp 0
- 04 other-vehicle: IoU 0.000000, acc n/a, gt 0, pred 40, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 327, pred 858, tp 0
- 06 bicyclist: IoU 0.000000, acc n/a, gt 0, pred 14, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 43, tp 0
- 08 road: IoU 0.220743, acc 0.737499, gt 42518, pred 130891, tp 31357
- 09 parking: IoU 0.000000, acc 0.000000, gt 160, pred 581, tp 0
- 10 sidewalk: IoU 0.124926, acc 0.168589, gt 61285, pred 31752, tp 10332
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 12 building: IoU 0.096637, acc 0.209845, gt 4327, pred 5977, tp 908
- 13 fence: IoU 0.000707, acc 0.000781, gt 1281, pred 135, tp 1
- 14 vegetation: IoU 0.286516, acc 0.509705, gt 48379, pred 62345, tp 24659
- 15 trunk: IoU 0.000000, acc 0.000000, gt 8033, pred 1, tp 0
- 16 terrain: IoU 0.000000, acc 0.000000, gt 60429, pred 1, tp 0
- 17 pole: IoU 0.000000, acc 0.000000, gt 853, pred 0, tp 0
- 18 traffic-sign: IoU 0.000000, acc 0.000000, gt 1998, pred 0, tp 0
