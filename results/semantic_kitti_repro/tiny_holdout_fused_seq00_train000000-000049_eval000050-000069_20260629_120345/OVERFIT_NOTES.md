# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.333907
- Final eval loss, first frame: 0.361911
- Initial eval mean loss: 3.484573
- Final eval mean loss: 0.336120
- Initial eval mean accuracy: 0.049448
- Final eval mean accuracy: 0.893087
- First train loss: 3.547000
- Final train loss: 0.146699
- Best train loss: 0.108551
- Final train valid accuracy: 0.953593
- CUDA peak memory GB: 0.957
- Selected visualization frames: 8

## Per-frame Eval

## Holdout Eval

- Holdout initial mean loss: 3.610092
- Holdout final mean loss: 0.616630
- Holdout initial mean accuracy: 0.031670
- Holdout final mean accuracy: 0.849424

- Frame 000000: loss 3.333907 -> 0.361911, accuracy 0.076536 -> 0.870594
- Frame 000001: loss 3.384583 -> 0.306356, accuracy 0.062059 -> 0.893037
- Frame 000002: loss 3.330714 -> 0.411868, accuracy 0.084643 -> 0.859605
- Frame 000003: loss 3.297498 -> 0.350903, accuracy 0.081354 -> 0.879232
- Frame 000004: loss 3.380021 -> 0.417372, accuracy 0.076845 -> 0.854398
- Frame 000005: loss 3.427721 -> 0.403640, accuracy 0.055807 -> 0.872800
- Frame 000006: loss 3.407758 -> 0.438518, accuracy 0.057876 -> 0.856568
- Frame 000007: loss 3.348264 -> 0.496877, accuracy 0.069208 -> 0.852056
- Frame 000008: loss 3.389593 -> 0.451016, accuracy 0.067202 -> 0.854564
- Frame 000009: loss 3.396632 -> 0.468776, accuracy 0.066229 -> 0.843276
- Frame 000010: loss 3.377207 -> 0.564615, accuracy 0.071249 -> 0.833417
- Frame 000011: loss 3.367865 -> 0.484166, accuracy 0.071928 -> 0.843656
- Frame 000012: loss 3.378001 -> 0.583389, accuracy 0.065934 -> 0.834166
- Frame 000013: loss 3.394866 -> 0.580750, accuracy 0.070571 -> 0.844344
- Frame 000014: loss 3.421453 -> 0.555269, accuracy 0.069604 -> 0.843766
- Frame 000015: loss 3.437813 -> 0.451624, accuracy 0.054808 -> 0.870952
- Frame 000016: loss 3.425505 -> 0.454456, accuracy 0.050576 -> 0.849775
- Frame 000017: loss 3.450124 -> 0.381420, accuracy 0.051360 -> 0.864552
- Frame 000018: loss 3.469639 -> 0.433281, accuracy 0.045317 -> 0.857503
- Frame 000019: loss 3.443097 -> 0.375867, accuracy 0.063627 -> 0.876253
- Frame 000020: loss 3.470494 -> 0.374314, accuracy 0.043325 -> 0.871537
- Frame 000021: loss 3.458236 -> 0.391193, accuracy 0.046570 -> 0.884327
- Frame 000022: loss 3.493210 -> 0.251515, accuracy 0.050582 -> 0.910976
- Frame 000023: loss 3.494985 -> 0.431558, accuracy 0.052369 -> 0.869825
- Frame 000024: loss 3.541894 -> 0.371451, accuracy 0.036036 -> 0.891892
- Frame 000025: loss 3.510348 -> 0.388572, accuracy 0.039500 -> 0.889500
- Frame 000026: loss 3.529212 -> 0.459744, accuracy 0.039880 -> 0.867896
- Frame 000027: loss 3.545861 -> 0.316432, accuracy 0.033966 -> 0.895604
- Frame 000028: loss 3.545544 -> 0.357133, accuracy 0.032869 -> 0.891434
- Frame 000029: loss 3.559941 -> 0.326522, accuracy 0.035982 -> 0.905547
- Frame 000030: loss 3.560806 -> 0.366372, accuracy 0.039841 -> 0.890438
- Frame 000031: loss 3.589530 -> 0.302056, accuracy 0.032934 -> 0.899701
- Frame 000032: loss 3.556178 -> 0.329608, accuracy 0.038308 -> 0.896020
- Frame 000033: loss 3.539431 -> 0.280321, accuracy 0.038942 -> 0.907139
- Frame 000034: loss 3.582607 -> 0.215271, accuracy 0.033284 -> 0.933433
- Frame 000035: loss 3.546824 -> 0.163995, accuracy 0.040520 -> 0.940470
- Frame 000036: loss 3.543419 -> 0.201436, accuracy 0.035105 -> 0.939819
- Frame 000037: loss 3.554879 -> 0.202689, accuracy 0.040419 -> 0.926148
- Frame 000038: loss 3.560931 -> 0.205713, accuracy 0.030622 -> 0.930221
- Frame 000039: loss 3.568980 -> 0.177630, accuracy 0.032468 -> 0.943057
- Frame 000040: loss 3.503487 -> 0.191895, accuracy 0.042692 -> 0.933199
- Frame 000041: loss 3.557346 -> 0.178461, accuracy 0.029693 -> 0.937091
- Frame 000042: loss 3.571183 -> 0.185833, accuracy 0.035447 -> 0.942586
- Frame 000043: loss 3.586266 -> 0.175734, accuracy 0.034068 -> 0.934870
- Frame 000044: loss 3.564916 -> 0.193190, accuracy 0.043325 -> 0.933501
- Frame 000045: loss 3.545640 -> 0.160024, accuracy 0.039117 -> 0.943832
- Frame 000046: loss 3.604239 -> 0.149135, accuracy 0.039920 -> 0.951597
- Frame 000047: loss 3.543357 -> 0.168785, accuracy 0.039920 -> 0.943613
- Frame 000048: loss 3.571834 -> 0.144602, accuracy 0.038558 -> 0.953931
- Frame 000049: loss 3.564835 -> 0.172737, accuracy 0.043413 -> 0.940619
