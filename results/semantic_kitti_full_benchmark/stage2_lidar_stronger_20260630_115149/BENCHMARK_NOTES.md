# SemanticKITTI Full-Frame Benchmark Notes

## Scope

This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.

## Summary

- Mode: lidar-only
- Primary eval route: lidar-only
- Diagnostic eval route: none
- Train frames: 19130
- Val frames: 4071
- Train sample points per step: 32768
- Eval chunk points: 32768
- LR schedule: cosine
- Class weight mode: inverse_sqrt_freq
- Rare-class frame sampling probability: 0.45
- Balanced point sampling probability: 0.5
- Spatial crop radius: 25.0
- Periodic val frames: 256
- Steps completed: 60000
- Best val mIoU: 0.278662

## Final Validation

- Route: lidar-only
- Frames: 4071
- Mean loss: 1.595232
- mIoU: 0.277888
- Overall accuracy: 0.755914
- Valid points: 476757723


## Final Validation Class IoU

- 00 car: IoU 0.629944, acc 0.645703, gt 30856105, pred 20695835, tp 19923891
- 01 bicycle: IoU 0.051361, acc 0.089880, gt 247796, pred 208116, tp 22272
- 02 motorcycle: IoU 0.102917, acc 0.186849, gt 345868, pred 346691, tp 64625
- 03 truck: IoU 0.042674, acc 0.083632, gt 508704, pred 530801, tp 42544
- 04 other-vehicle: IoU 0.052111, acc 0.197651, gt 2222268, pred 6645684, tp 439233
- 05 person: IoU 0.084790, acc 0.139500, gt 476946, pred 374282, tp 66534
- 06 bicyclist: IoU 0.121150, acc 0.150013, gt 306860, pred 119140, tp 46033
- 07 motorcyclist: IoU 0.000908, acc 0.002111, gt 22737, pred 30148, tp 48
- 08 road: IoU 0.776793, acc 0.839496, gt 87963626, pred 80945552, tp 73845105
- 09 parking: IoU 0.082764, acc 0.203821, gt 5947692, pred 9911889, tp 1212266
- 10 sidewalk: IoU 0.511240, acc 0.645001, gt 60280496, pred 54652666, tp 38880962
- 11 other-ground: IoU 0.000626, acc 0.001512, gt 457650, pred 648274, tp 692
- 12 building: IoU 0.672973, acc 0.777742, gt 56878900, pred 53092018, tp 44237105
- 13 fence: IoU 0.195935, acc 0.603291, gt 12642657, pred 33911841, tp 7627201
- 14 vegetation: IoU 0.739130, acc 0.798590, gt 145937016, pred 128283779, tp 116543820
- 15 trunk: IoU 0.314814, acc 0.358027, gt 5503279, pred 2725750, tp 1970325
- 16 terrain: IoU 0.598368, acc 0.858942, gt 64111254, pred 82986649, tp 55067821
- 17 pole: IoU 0.186899, acc 0.208116, gt 1666427, pred 535980, tp 346810
- 18 traffic-sign: IoU 0.114477, acc 0.133048, gt 381442, pred 112628, tp 50750
