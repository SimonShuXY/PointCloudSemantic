# SemanticKITTI PTv3/IPFP Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame and generates IPFP auxiliary image-backprojected points.
- Runs the PTv3/IPFP fused forward path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.334230
- Final eval loss, first frame: 0.373002
- Initial eval mean loss: 3.484718
- Final eval mean loss: 0.349235
- Initial eval mean accuracy: 0.049398
- Final eval mean accuracy: 0.891445
- First train loss: 3.563501
- Final train loss: 0.146846
- Best train loss: 0.101434
- Final train valid accuracy: 0.951098
- CUDA peak memory GB: 0.957
- Selected visualization frames: 8

## Per-frame Eval

- Frame 000000: loss 3.334230 -> 0.373002, accuracy 0.076536 -> 0.874119
- Frame 000001: loss 3.384816 -> 0.375890, accuracy 0.062059 -> 0.869828
- Frame 000002: loss 3.334314 -> 0.445252, accuracy 0.084643 -> 0.842372
- Frame 000003: loss 3.300689 -> 0.376007, accuracy 0.082365 -> 0.875189
- Frame 000004: loss 3.381907 -> 0.449096, accuracy 0.077351 -> 0.856926
- Frame 000005: loss 3.427530 -> 0.504011, accuracy 0.056812 -> 0.856712
- Frame 000006: loss 3.406348 -> 0.549672, accuracy 0.058883 -> 0.830901
- Frame 000007: loss 3.346841 -> 0.590190, accuracy 0.069709 -> 0.837513
- Frame 000008: loss 3.390174 -> 0.516269, accuracy 0.065196 -> 0.849047
- Frame 000009: loss 3.396746 -> 0.532930, accuracy 0.066229 -> 0.839737
- Frame 000010: loss 3.377934 -> 0.548855, accuracy 0.069242 -> 0.842950
- Frame 000011: loss 3.367142 -> 0.497416, accuracy 0.070929 -> 0.855145
- Frame 000012: loss 3.377203 -> 0.593546, accuracy 0.065934 -> 0.842657
- Frame 000013: loss 3.395409 -> 0.475837, accuracy 0.069069 -> 0.851351
- Frame 000014: loss 3.419336 -> 0.609252, accuracy 0.069104 -> 0.829745
- Frame 000015: loss 3.435602 -> 0.453486, accuracy 0.055306 -> 0.865969
- Frame 000016: loss 3.427406 -> 0.549134, accuracy 0.049074 -> 0.842263
- Frame 000017: loss 3.450437 -> 0.348044, accuracy 0.050352 -> 0.887210
- Frame 000018: loss 3.468730 -> 0.359642, accuracy 0.045317 -> 0.877140
- Frame 000019: loss 3.443617 -> 0.416650, accuracy 0.066132 -> 0.864228
- Frame 000020: loss 3.469938 -> 0.326932, accuracy 0.042317 -> 0.893703
- Frame 000021: loss 3.457758 -> 0.331055, accuracy 0.048072 -> 0.888833
- Frame 000022: loss 3.494188 -> 0.282077, accuracy 0.050076 -> 0.897319
- Frame 000023: loss 3.494224 -> 0.480666, accuracy 0.051870 -> 0.859850
- Frame 000024: loss 3.541690 -> 0.368511, accuracy 0.034535 -> 0.891391
- Frame 000025: loss 3.510290 -> 0.384974, accuracy 0.038500 -> 0.891000
- Frame 000026: loss 3.527702 -> 0.408587, accuracy 0.040877 -> 0.880857
- Frame 000027: loss 3.545371 -> 0.309722, accuracy 0.034965 -> 0.905095
- Frame 000028: loss 3.545322 -> 0.367432, accuracy 0.030876 -> 0.889940
- Frame 000029: loss 3.560853 -> 0.402322, accuracy 0.036482 -> 0.889555
- Frame 000030: loss 3.561368 -> 0.363335, accuracy 0.039841 -> 0.894422
- Frame 000031: loss 3.589125 -> 0.366487, accuracy 0.032435 -> 0.898703
- Frame 000032: loss 3.555737 -> 0.306985, accuracy 0.038806 -> 0.910448
- Frame 000033: loss 3.538257 -> 0.230367, accuracy 0.040939 -> 0.914628
- Frame 000034: loss 3.583715 -> 0.217534, accuracy 0.033284 -> 0.927968
- Frame 000035: loss 3.547438 -> 0.165078, accuracy 0.040020 -> 0.941971
- Frame 000036: loss 3.545425 -> 0.229106, accuracy 0.035105 -> 0.931795
- Frame 000037: loss 3.553257 -> 0.224958, accuracy 0.040419 -> 0.920160
- Frame 000038: loss 3.565492 -> 0.208772, accuracy 0.029116 -> 0.924197
- Frame 000039: loss 3.563221 -> 0.145444, accuracy 0.032967 -> 0.942557
- Frame 000040: loss 3.502174 -> 0.208887, accuracy 0.043697 -> 0.927172
- Frame 000041: loss 3.561478 -> 0.194288, accuracy 0.030196 -> 0.933065
- Frame 000042: loss 3.569891 -> 0.176395, accuracy 0.035946 -> 0.939091
- Frame 000043: loss 3.585283 -> 0.168388, accuracy 0.033567 -> 0.941884
- Frame 000044: loss 3.563341 -> 0.170456, accuracy 0.045340 -> 0.938539
- Frame 000045: loss 3.547002 -> 0.176837, accuracy 0.040120 -> 0.938816
- Frame 000046: loss 3.605962 -> 0.186661, accuracy 0.038922 -> 0.937625
- Frame 000047: loss 3.545096 -> 0.165671, accuracy 0.039920 -> 0.947106
- Frame 000048: loss 3.572106 -> 0.152413, accuracy 0.037556 -> 0.949424
- Frame 000049: loss 3.566762 -> 0.177236, accuracy 0.042914 -> 0.934132
