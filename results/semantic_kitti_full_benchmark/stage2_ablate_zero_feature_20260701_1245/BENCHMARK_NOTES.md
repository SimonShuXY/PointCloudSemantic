# SemanticKITTI Full-Frame Benchmark Notes

## Scope

This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.

## Summary

- Mode: fused
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
- Fused min visible points: 512
- Train sample retries: 12
- Fused train fallback: lidar-only
- Periodic val frames: 256
- Steps completed: 30000
- Best val mIoU: 0.289265

## Final Validation

- Route: lidar-only
- Frames: 4071
- Mean loss: 1.510347
- mIoU: 0.289265
- Overall accuracy: 0.796967
- Valid points: 476757723


## Final Validation Class IoU

- 00 car: IoU 0.658142, acc 0.675908, gt 30856105, pred 21688800, tp 20855886
- 01 bicycle: IoU 0.027741, acc 0.033104, gt 247796, pred 56105, tp 8203
- 02 motorcycle: IoU 0.126779, acc 0.202306, gt 345868, pred 276016, tp 69971
- 03 truck: IoU 0.034210, acc 0.045046, gt 508704, pred 184038, tp 22915
- 04 other-vehicle: IoU 0.053503, acc 0.174619, gt 2222268, pred 5418708, tp 388050
- 05 person: IoU 0.104522, acc 0.166084, gt 476946, pred 360126, tp 79213
- 06 bicyclist: IoU 0.083549, acc 0.095998, gt 306860, pred 75182, tp 29458
- 07 motorcyclist: IoU 0.000228, acc 0.000572, gt 22737, pred 34247, tp 13
- 08 road: IoU 0.813305, acc 0.902993, gt 87963626, pred 89130760, tp 79430498
- 09 parking: IoU 0.046997, acc 0.052630, gt 5947692, pred 1025968, tp 313030
- 10 sidewalk: IoU 0.568898, acc 0.699206, gt 60280496, pred 55955856, tp 42148455
- 11 other-ground: IoU 0.000115, acc 0.000131, gt 457650, pred 63551, tp 60
- 12 building: IoU 0.689606, acc 0.796037, gt 56878900, pred 54056149, tp 45277692
- 13 fence: IoU 0.223246, acc 0.487807, gt 12642657, pred 21149602, tp 6167177
- 14 vegetation: IoU 0.778941, acc 0.913790, gt 145937016, pred 158619977, tp 133355746
- 15 trunk: IoU 0.215492, acc 0.233828, gt 5503279, pred 1755097, tp 1286822
- 16 terrain: IoU 0.623537, acc 0.780193, gt 64111254, pred 66126373, tp 50019157
- 17 pole: IoU 0.215745, acc 0.242918, gt 1666427, pred 614687, tp 404805
- 18 traffic-sign: IoU 0.231473, acc 0.270002, gt 381442, pred 166481, tp 102990
