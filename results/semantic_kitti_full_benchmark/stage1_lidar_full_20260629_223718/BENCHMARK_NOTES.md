# SemanticKITTI Full-Frame Benchmark Notes

## Scope

This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.

## Summary

- Mode: lidar-only
- Primary eval route: lidar-only
- Diagnostic eval route: none
- Train frames: 19130
- Val frames: 4071
- Train sample points per step: 8192
- Eval chunk points: 16384
- Steps completed: 20000
- Best val mIoU: 0.190113

## Final Validation

- Route: lidar-only
- Frames: 4071
- Mean loss: 1.734163
- mIoU: 0.190113
- Overall accuracy: 0.669318
- Valid points: 476757723


## Final Validation Class IoU

- 00 car: IoU 0.478739, acc 0.508421, gt 30856105, pred 17600966, tp 15687881
- 01 bicycle: IoU 0.001972, acc 0.002038, gt 247796, pred 8850, tp 505
- 02 motorcycle: IoU 0.031085, acc 0.056889, gt 345868, pred 306777, tp 19676
- 03 truck: IoU 0.015243, acc 0.040452, gt 508704, pred 861872, tp 20578
- 04 other-vehicle: IoU 0.045574, acc 0.302326, gt 2222268, pred 13191477, tp 671849
- 05 person: IoU 0.016338, acc 0.020004, gt 476946, pred 116584, tp 9541
- 06 bicyclist: IoU 0.045492, acc 0.053731, gt 306860, pred 72064, tp 16488
- 07 motorcyclist: IoU 0.000000, acc 0.000000, gt 22737, pred 1888, tp 0
- 08 road: IoU 0.628278, acc 0.906640, gt 87963626, pred 118723986, tp 79751302
- 09 parking: IoU 0.003518, acc 0.003581, gt 5947692, pred 128800, tp 21300
- 10 sidewalk: IoU 0.316752, acc 0.469404, gt 60280496, pred 57346819, tp 28295898
- 11 other-ground: IoU 0.000072, acc 0.000090, gt 457650, pred 109970, tp 41
- 12 building: IoU 0.578348, acc 0.775961, gt 56878900, pred 63570508, tp 44135825
- 13 fence: IoU 0.177583, acc 0.434324, gt 12642657, pred 23769201, tp 5491008
- 14 vegetation: IoU 0.683996, acc 0.789372, gt 145937016, pred 137681676, tp 115198627
- 15 trunk: IoU 0.078675, acc 0.082889, gt 5503279, pred 750898, tp 456161
- 16 terrain: IoU 0.391941, acc 0.453450, gt 64111254, pred 39132580, tp 29071271
- 17 pole: IoU 0.089337, acc 0.092147, gt 1666427, pred 205971, tp 153556
- 18 traffic-sign: IoU 0.029204, acc 0.264701, gt 381442, pred 3176836, tp 100968
