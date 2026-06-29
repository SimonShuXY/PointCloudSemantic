# SemanticKITTI lidar-only Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.334753
- Final eval loss, first frame: 0.369420
- Initial eval mean loss: 3.486791
- Final eval mean loss: 0.351930
- Initial eval mean accuracy: 0.049437
- Final eval mean accuracy: 0.893086
- First train loss: 3.560726
- Final train loss: 0.156368
- Best train loss: 0.095561
- Final train valid accuracy: 0.944611
- CUDA peak memory GB: 0.902
- Selected visualization frames: 8

## Per-frame Eval

## Holdout Eval

- Holdout initial mean loss: 3.613234
- Holdout final mean loss: 0.737896
- Holdout initial mean accuracy: 0.031989
- Holdout final mean accuracy: 0.825986

- Frame 000000: loss 3.334753 -> 0.369420, accuracy 0.077543 -> 0.874119
- Frame 000001: loss 3.387960 -> 0.333827, accuracy 0.060545 -> 0.881433
- Frame 000002: loss 3.338723 -> 0.404577, accuracy 0.086163 -> 0.859098
- Frame 000003: loss 3.299294 -> 0.503328, accuracy 0.080344 -> 0.851945
- Frame 000004: loss 3.383134 -> 0.426453, accuracy 0.073812 -> 0.851365
- Frame 000005: loss 3.425882 -> 0.510699, accuracy 0.054801 -> 0.850176
- Frame 000006: loss 3.404705 -> 0.549604, accuracy 0.063412 -> 0.851032
- Frame 000007: loss 3.348289 -> 0.562293, accuracy 0.071214 -> 0.836008
- Frame 000008: loss 3.391541 -> 0.514389, accuracy 0.063691 -> 0.840522
- Frame 000009: loss 3.407861 -> 0.488948, accuracy 0.063701 -> 0.846309
- Frame 000010: loss 3.374959 -> 0.536552, accuracy 0.070246 -> 0.850978
- Frame 000011: loss 3.375695 -> 0.475642, accuracy 0.071429 -> 0.862637
- Frame 000012: loss 3.378796 -> 0.586580, accuracy 0.066933 -> 0.848152
- Frame 000013: loss 3.399692 -> 0.481524, accuracy 0.067067 -> 0.874374
- Frame 000014: loss 3.422076 -> 0.527460, accuracy 0.068102 -> 0.852779
- Frame 000015: loss 3.438024 -> 0.482689, accuracy 0.055306 -> 0.863478
- Frame 000016: loss 3.426074 -> 0.520626, accuracy 0.053580 -> 0.851277
- Frame 000017: loss 3.449435 -> 0.550003, accuracy 0.052870 -> 0.860020
- Frame 000018: loss 3.472075 -> 0.582561, accuracy 0.043807 -> 0.834340
- Frame 000019: loss 3.446188 -> 0.426657, accuracy 0.064128 -> 0.872746
- Frame 000020: loss 3.476479 -> 0.453570, accuracy 0.042821 -> 0.853401
- Frame 000021: loss 3.476079 -> 0.485185, accuracy 0.046069 -> 0.868302
- Frame 000022: loss 3.493980 -> 0.396268, accuracy 0.050076 -> 0.871522
- Frame 000023: loss 3.497141 -> 0.450183, accuracy 0.052868 -> 0.873815
- Frame 000024: loss 3.537686 -> 0.327024, accuracy 0.036036 -> 0.898899
- Frame 000025: loss 3.505886 -> 0.286808, accuracy 0.041500 -> 0.900000
- Frame 000026: loss 3.533555 -> 0.373304, accuracy 0.036889 -> 0.889831
- Frame 000027: loss 3.534341 -> 0.278587, accuracy 0.036963 -> 0.912587
- Frame 000028: loss 3.545893 -> 0.350424, accuracy 0.031375 -> 0.896414
- Frame 000029: loss 3.557760 -> 0.273182, accuracy 0.037481 -> 0.903548
- Frame 000030: loss 3.567092 -> 0.352667, accuracy 0.038347 -> 0.901394
- Frame 000031: loss 3.585274 -> 0.242328, accuracy 0.032435 -> 0.924152
- Frame 000032: loss 3.558127 -> 0.314737, accuracy 0.036318 -> 0.906468
- Frame 000033: loss 3.550302 -> 0.293602, accuracy 0.040939 -> 0.909636
- Frame 000034: loss 3.589331 -> 0.197010, accuracy 0.034774 -> 0.937904
- Frame 000035: loss 3.551437 -> 0.170375, accuracy 0.042521 -> 0.935468
- Frame 000036: loss 3.542475 -> 0.214960, accuracy 0.037111 -> 0.930792
- Frame 000037: loss 3.554947 -> 0.216653, accuracy 0.042914 -> 0.925150
- Frame 000038: loss 3.560161 -> 0.184436, accuracy 0.030120 -> 0.939257
- Frame 000039: loss 3.572960 -> 0.149811, accuracy 0.031968 -> 0.946553
- Frame 000040: loss 3.500860 -> 0.206016, accuracy 0.040683 -> 0.938222
- Frame 000041: loss 3.558599 -> 0.161390, accuracy 0.029693 -> 0.948666
- Frame 000042: loss 3.571600 -> 0.153449, accuracy 0.036445 -> 0.946081
- Frame 000043: loss 3.587406 -> 0.171448, accuracy 0.032565 -> 0.938377
- Frame 000044: loss 3.570229 -> 0.183284, accuracy 0.043829 -> 0.934509
- Frame 000045: loss 3.547033 -> 0.211236, accuracy 0.043129 -> 0.934303
- Frame 000046: loss 3.609004 -> 0.203101, accuracy 0.039920 -> 0.930639
- Frame 000047: loss 3.557585 -> 0.152776, accuracy 0.039421 -> 0.953094
- Frame 000048: loss 3.575522 -> 0.160880, accuracy 0.035553 -> 0.945418
- Frame 000049: loss 3.565673 -> 0.147961, accuracy 0.042415 -> 0.947106
