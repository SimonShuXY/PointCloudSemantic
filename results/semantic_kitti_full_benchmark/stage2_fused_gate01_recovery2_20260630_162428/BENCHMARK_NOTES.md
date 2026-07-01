# SemanticKITTI Full-Frame Benchmark Notes

## Scope

This is the first full-frame validation scaffold for the project. Training still uses sampled points per step, but validation covers every point in each selected validation frame through chunked inference and accumulates a full-frame confusion matrix.

## Summary

- Mode: fused
- Primary eval route: lidar-only
- Diagnostic eval route: fused
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
- Best val mIoU: 0.292154

## Final Validation

- Route: lidar-only
- Frames: 4071
- Mean loss: 1.500495
- mIoU: 0.290422
- Overall accuracy: 0.801245
- Valid points: 476757723


## Final Validation Class IoU

- 00 car: IoU 0.677922, acc 0.700354, gt 30856105, pred 22631208, tp 21610204
- 01 bicycle: IoU 0.053402, acc 0.071623, gt 247796, pred 102302, tp 17748
- 02 motorcycle: IoU 0.108801, acc 0.175078, gt 345868, pred 271245, tp 60554
- 03 truck: IoU 0.033170, acc 0.041281, gt 508704, pred 145405, tp 21000
- 04 other-vehicle: IoU 0.058522, acc 0.185710, gt 2222268, pred 5242490, tp 412698
- 05 person: IoU 0.099877, acc 0.162769, gt 476946, pred 377961, tp 77632
- 06 bicyclist: IoU 0.113816, acc 0.129111, gt 306860, pred 80855, tp 39619
- 07 motorcyclist: IoU 0.000333, acc 0.000880, gt 22737, pred 37320, tp 20
- 08 road: IoU 0.817542, acc 0.919451, gt 87963626, pred 91843142, tp 80878244
- 09 parking: IoU 0.049795, acc 0.055067, gt 5947692, pred 957180, tp 327520
- 10 sidewalk: IoU 0.586037, acc 0.721487, gt 60280496, pred 57424112, tp 43491568
- 11 other-ground: IoU 0.000209, acc 0.000269, gt 457650, pred 131405, tp 123
- 12 building: IoU 0.685880, acc 0.794147, gt 56878900, pred 54148647, tp 45170222
- 13 fence: IoU 0.221213, acc 0.509117, gt 12642657, pred 22890813, tp 6436593
- 14 vegetation: IoU 0.783654, acc 0.907598, gt 145937016, pred 155533806, tp 132452113
- 15 trunk: IoU 0.174059, acc 0.185802, gt 5503279, pred 1393797, tp 1022521
- 16 terrain: IoU 0.639862, acc 0.772113, gt 64111254, pred 62752046, tp 49501116
- 17 pole: IoU 0.200802, acc 0.227564, gt 1666427, pred 601306, tp 379218
- 18 traffic-sign: IoU 0.213128, acc 0.264431, gt 381442, pred 192683, tp 100865
