# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Train route: fused
- Primary eval route: lidar-only
- Diagnostic eval route: fused
- Depth mode: lidar-inpaint, first-frame p5/p95: [6.56657600402832, 36.63307189941406]
- Loss mode: ce-lovasz, Lovasz weight: 1.0
- Extra feature mode/scale: learned, 0.250
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.291584
- Final eval loss, first frame: 0.952511
- Initial eval mean loss: 4.492309
- Final eval mean loss: 0.790788
- Initial eval mean accuracy: 0.044691
- Final eval mean accuracy: 0.884814
- Initial eval mIoU: 0.006192
- Final eval mIoU: 0.504750
- Initial eval overall accuracy: 0.044664
- Final eval overall accuracy: 0.884778
- First train loss: 4.510600
- Final train loss: 0.726649
- Best train loss: 0.316967
- Final train valid accuracy: 0.890194
- CUDA peak memory GB: 0.961
- Selected visualization frames: 4

## Holdout Eval

- Holdout initial mean loss: 4.186562
- Holdout final mean loss: 2.653811
- Holdout initial mean accuracy: 0.102361
- Holdout final mean accuracy: 0.610259
- Holdout initial mIoU: 0.011223
- Holdout final mIoU: 0.152256
- Holdout initial overall accuracy: 0.102435
- Holdout final overall accuracy: 0.611005

## Diagnostic fused Eval

- Train final mean loss: 0.791792
- Train final mean accuracy: 0.884258
- Train final mIoU: 0.504496
- Train final overall accuracy: 0.884217

- Holdout final mean loss: 2.652999
- Holdout final mean accuracy: 0.606428
- Holdout final mIoU: 0.150355
- Holdout final overall accuracy: 0.607182

## Train-Frame Eval

- Frame 000000: loss 4.291584 -> 0.952511, accuracy 0.076032 -> 0.841390
- Frame 000001: loss 4.347650 -> 0.696749, accuracy 0.060545 -> 0.882442
- Frame 000002: loss 4.292539 -> 0.817040, accuracy 0.082615 -> 0.880385
- Frame 000003: loss 4.256271 -> 0.675303, accuracy 0.076806 -> 0.902981
- Frame 000004: loss 4.334161 -> 0.787384, accuracy 0.077856 -> 0.873610
- Frame 000005: loss 4.375578 -> 0.749646, accuracy 0.055807 -> 0.878834
- Frame 000006: loss 4.360269 -> 0.647619, accuracy 0.056870 -> 0.909411
- Frame 000007: loss 4.308752 -> 0.750658, accuracy 0.072217 -> 0.884152
- Frame 000008: loss 4.355790 -> 0.761310, accuracy 0.064694 -> 0.887161
- Frame 000009: loss 4.359302 -> 0.865083, accuracy 0.064712 -> 0.881699
- Frame 000010: loss 4.333489 -> 0.831060, accuracy 0.071751 -> 0.882589
- Frame 000011: loss 4.329862 -> 0.868864, accuracy 0.073427 -> 0.868132
- Frame 000012: loss 4.337019 -> 0.848999, accuracy 0.065435 -> 0.872128
- Frame 000013: loss 4.359932 -> 0.960130, accuracy 0.067568 -> 0.856356
- Frame 000014: loss 4.380711 -> 1.229004, accuracy 0.067601 -> 0.831748
- Frame 000015: loss 4.405172 -> 1.224658, accuracy 0.054310 -> 0.807673
- Frame 000016: loss 4.391142 -> 1.029129, accuracy 0.052078 -> 0.850776
- Frame 000017: loss 4.408065 -> 0.838882, accuracy 0.049849 -> 0.887714
- Frame 000018: loss 4.427414 -> 0.813474, accuracy 0.045317 -> 0.890735
- Frame 000019: loss 4.401720 -> 0.823509, accuracy 0.065631 -> 0.882766
- Frame 000020: loss 4.429627 -> 0.821638, accuracy 0.046348 -> 0.872040
- Frame 000021: loss 4.432491 -> 0.874179, accuracy 0.045068 -> 0.866800
- Frame 000022: loss 4.455209 -> 0.958194, accuracy 0.052099 -> 0.858877
- Frame 000023: loss 4.450344 -> 1.014351, accuracy 0.049377 -> 0.848379
- Frame 000024: loss 4.497886 -> 1.117496, accuracy 0.033033 -> 0.848849
- Frame 000025: loss 4.466932 -> 1.047871, accuracy 0.040500 -> 0.844000
- Frame 000026: loss 4.488230 -> 0.978226, accuracy 0.038385 -> 0.850947
- Frame 000027: loss 4.504421 -> 0.727212, accuracy 0.033966 -> 0.869630
- Frame 000028: loss 4.499907 -> 1.003772, accuracy 0.032371 -> 0.843127
- Frame 000029: loss 4.517327 -> 0.973952, accuracy 0.037481 -> 0.852574
- Frame 000030: loss 4.520315 -> 1.062666, accuracy 0.039343 -> 0.844622
- Frame 000031: loss 4.540865 -> 0.741595, accuracy 0.031437 -> 0.896707
- Frame 000032: loss 4.511164 -> 0.782493, accuracy 0.032836 -> 0.884577
- Frame 000033: loss 4.504049 -> 0.792760, accuracy 0.037943 -> 0.897654
- Frame 000034: loss 4.530494 -> 0.795811, accuracy 0.031793 -> 0.897665
- Frame 000035: loss 4.504936 -> 0.857922, accuracy 0.042021 -> 0.885443
- Frame 000036: loss 4.495050 -> 0.949147, accuracy 0.037111 -> 0.859579
- Frame 000037: loss 4.503438 -> 0.835323, accuracy 0.041916 -> 0.880240
- Frame 000038: loss 4.510077 -> 0.928775, accuracy 0.032129 -> 0.881024
- Frame 000039: loss 4.515746 -> 0.753541, accuracy 0.034466 -> 0.897103
- Frame 000040: loss 4.446148 -> 0.799210, accuracy 0.039679 -> 0.877951
- Frame 000041: loss 4.513802 -> 0.943950, accuracy 0.031203 -> 0.858581
- Frame 000042: loss 4.528090 -> 0.649527, accuracy 0.035946 -> 0.901648
- Frame 000043: loss 4.539755 -> 1.017033, accuracy 0.032064 -> 0.869739
- Frame 000044: loss 4.522154 -> 0.889084, accuracy 0.044332 -> 0.853401
- Frame 000045: loss 4.498232 -> 0.722665, accuracy 0.042126 -> 0.893681
- Frame 000046: loss 4.552831 -> 0.914210, accuracy 0.038423 -> 0.872256
- Frame 000047: loss 4.499376 -> 0.767061, accuracy 0.039421 -> 0.877744
- Frame 000048: loss 4.527609 -> 0.651401, accuracy 0.034051 -> 0.905859
- Frame 000049: loss 4.518470 -> 0.768633, accuracy 0.043413 -> 0.895210
- Frame 000050: loss 4.517783 -> 0.698103, accuracy 0.051915 -> 0.900202
- Frame 000051: loss 4.524095 -> 0.854204, accuracy 0.041646 -> 0.884094
- Frame 000052: loss 4.508625 -> 0.770650, accuracy 0.053892 -> 0.870259
- Frame 000053: loss 4.511474 -> 0.954661, accuracy 0.035211 -> 0.868712
- Frame 000054: loss 4.539313 -> 0.832294, accuracy 0.029397 -> 0.867464
- Frame 000055: loss 4.571939 -> 0.984324, accuracy 0.028971 -> 0.863636
- Frame 000056: loss 4.566006 -> 0.886114, accuracy 0.019451 -> 0.863840
- Frame 000057: loss 4.560922 -> 0.764419, accuracy 0.032274 -> 0.888282
- Frame 000058: loss 4.532488 -> 0.780973, accuracy 0.025050 -> 0.877756
- Frame 000059: loss 4.564690 -> 1.083384, accuracy 0.031016 -> 0.832416
- Frame 000060: loss 4.553025 -> 1.060017, accuracy 0.036300 -> 0.835405
- Frame 000061: loss 4.517518 -> 1.018679, accuracy 0.036743 -> 0.851539
- Frame 000062: loss 4.575370 -> 1.040750, accuracy 0.033547 -> 0.841145
- Frame 000063: loss 4.578002 -> 0.909876, accuracy 0.023348 -> 0.857924
- Frame 000064: loss 4.625247 -> 0.797753, accuracy 0.024826 -> 0.887785
- Frame 000065: loss 4.583311 -> 0.714741, accuracy 0.033317 -> 0.880656
- Frame 000066: loss 4.553096 -> 0.830143, accuracy 0.034448 -> 0.876685
- Frame 000067: loss 4.597992 -> 0.758450, accuracy 0.033865 -> 0.911853
- Frame 000068: loss 4.629689 -> 0.884253, accuracy 0.030561 -> 0.879760
- Frame 000069: loss 4.593617 -> 0.609095, accuracy 0.028358 -> 0.921891
- Frame 000070: loss 4.614069 -> 0.503875, accuracy 0.037568 -> 0.938705
- Frame 000071: loss 4.557831 -> 0.662324, accuracy 0.051664 -> 0.906110
- Frame 000072: loss 4.536000 -> 0.601267, accuracy 0.040119 -> 0.924220
- Frame 000073: loss 4.546833 -> 0.547760, accuracy 0.046177 -> 0.907646
- Frame 000074: loss 4.526664 -> 0.479920, accuracy 0.040319 -> 0.928323
- Frame 000075: loss 4.566236 -> 0.685060, accuracy 0.037481 -> 0.906047
- Frame 000076: loss 4.527321 -> 0.608555, accuracy 0.048563 -> 0.910307
- Frame 000077: loss 4.520794 -> 0.583843, accuracy 0.047071 -> 0.912369
- Frame 000078: loss 4.571254 -> 0.566922, accuracy 0.040581 -> 0.908317
- Frame 000079: loss 4.560225 -> 0.545457, accuracy 0.043151 -> 0.936277
- Frame 000080: loss 4.580373 -> 0.531317, accuracy 0.043978 -> 0.932534
- Frame 000081: loss 4.566557 -> 0.604522, accuracy 0.054408 -> 0.929471
- Frame 000082: loss 4.579596 -> 0.647614, accuracy 0.050274 -> 0.910403
- Frame 000083: loss 4.517658 -> 0.601835, accuracy 0.050302 -> 0.918511
- Frame 000084: loss 4.502485 -> 0.666498, accuracy 0.054162 -> 0.912237
- Frame 000085: loss 4.493141 -> 0.705433, accuracy 0.058412 -> 0.896156
- Frame 000086: loss 4.461884 -> 0.658420, accuracy 0.062187 -> 0.898195
- Frame 000087: loss 4.479425 -> 0.609140, accuracy 0.058435 -> 0.918699
- Frame 000088: loss 4.488423 -> 0.494091, accuracy 0.054244 -> 0.932697
- Frame 000089: loss 4.454966 -> 0.571084, accuracy 0.049950 -> 0.910696
- Frame 000090: loss 4.554924 -> 0.763169, accuracy 0.032978 -> 0.899036
- Frame 000091: loss 4.530593 -> 0.670174, accuracy 0.033028 -> 0.921240
- Frame 000092: loss 4.506173 -> 0.541674, accuracy 0.043855 -> 0.934727
- Frame 000093: loss 4.503770 -> 0.580515, accuracy 0.050152 -> 0.931104
- Frame 000094: loss 4.540900 -> 0.540865, accuracy 0.033300 -> 0.917255
- Frame 000095: loss 4.528875 -> 0.652509, accuracy 0.037794 -> 0.896834
- Frame 000096: loss 4.538679 -> 0.659956, accuracy 0.044547 -> 0.888377
- Frame 000097: loss 4.535807 -> 0.685267, accuracy 0.035312 -> 0.894063
- Frame 000098: loss 4.538261 -> 0.773060, accuracy 0.037283 -> 0.881512
- Frame 000099: loss 4.515570 -> 0.793107, accuracy 0.036261 -> 0.879469

## Train Final Class IoU

- 00 car: IoU 0.902717, acc 0.942306, gt 24786, pred 24443, tp 23356
- 01 bicycle: IoU 0.405405, acc 0.661765, gt 68, pred 88, tp 45
- 04 other-vehicle: IoU 0.381455, acc 0.397311, gt 818, pred 359, tp 325
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.417219, acc 0.422819, gt 149, pred 65, tp 63
- 08 road: IoU 0.909376, acc 0.971668, gt 53650, pred 55805, tp 52130
- 09 parking: IoU 0.587030, acc 0.632494, gt 17031, pred 12091, tp 10772
- 10 sidewalk: IoU 0.614493, acc 0.884144, gt 18290, pred 24197, tp 16171
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.876367, acc 0.899904, gt 53059, pred 49173, tp 47748
- 13 fence: IoU 0.257019, acc 0.268219, gt 1331, pred 415, tp 357
- 14 vegetation: IoU 0.675097, acc 0.866136, gt 23606, pred 27126, tp 20446
- 15 trunk: IoU 0.788528, acc 0.823663, gt 5274, pred 4579, tp 4344
- 16 terrain: IoU 0.125326, acc 0.198347, gt 484, pred 378, tp 96
- 17 pole: IoU 0.709066, acc 0.769457, gt 681, pred 582, tp 524
- 18 traffic-sign: IoU 0.426901, acc 0.525180, gt 278, pred 210, tp 146

## Holdout Per-Frame Eval

- Frame 000100: loss 4.503098 -> 1.318518, accuracy 0.041538 -> 0.794872
- Frame 000101: loss 4.529276 -> 1.577835, accuracy 0.035640 -> 0.736570
- Frame 000102: loss 4.460403 -> 1.718552, accuracy 0.050463 -> 0.729145
- Frame 000103: loss 4.486571 -> 1.797490, accuracy 0.048287 -> 0.701973
- Frame 000104: loss 4.414052 -> 2.420046, accuracy 0.056099 -> 0.609882
- Frame 000105: loss 4.434631 -> 2.728720, accuracy 0.059067 -> 0.583938
- Frame 000106: loss 4.417852 -> 3.257989, accuracy 0.063655 -> 0.565195
- Frame 000107: loss 4.388153 -> 3.602759, accuracy 0.054859 -> 0.495820
- Frame 000108: loss 4.352619 -> 3.744584, accuracy 0.067946 -> 0.479253
- Frame 000109: loss 4.387416 -> 3.512140, accuracy 0.072105 -> 0.510000
- Frame 000110: loss 4.354285 -> 3.817145, accuracy 0.072993 -> 0.488008
- Frame 000111: loss 4.339590 -> 3.685794, accuracy 0.077895 -> 0.497368
- Frame 000112: loss 4.331479 -> 3.789660, accuracy 0.079830 -> 0.434806
- Frame 000113: loss 4.318831 -> 3.610309, accuracy 0.071353 -> 0.467230
- Frame 000114: loss 4.283711 -> 3.497643, accuracy 0.091942 -> 0.466739
- Frame 000115: loss 4.259994 -> 3.454178, accuracy 0.102041 -> 0.455782
- Frame 000116: loss 4.267457 -> 3.219632, accuracy 0.105864 -> 0.491956
- Frame 000117: loss 4.236145 -> 2.982541, accuracy 0.100616 -> 0.536961
- Frame 000118: loss 4.223139 -> 3.241522, accuracy 0.115325 -> 0.527422
- Frame 000119: loss 4.214370 -> 2.897365, accuracy 0.103955 -> 0.524848
- Frame 000120: loss 4.199826 -> 2.802606, accuracy 0.119182 -> 0.545780
- Frame 000121: loss 4.154123 -> 2.473122, accuracy 0.119205 -> 0.614875
- Frame 000122: loss 4.108914 -> 2.578521, accuracy 0.139653 -> 0.617227
- Frame 000123: loss 4.074277 -> 2.938658, accuracy 0.125764 -> 0.553972
- Frame 000124: loss 4.149996 -> 2.839739, accuracy 0.119121 -> 0.554192
- Frame 000125: loss 4.141627 -> 2.418468, accuracy 0.113834 -> 0.646248
- Frame 000126: loss 4.047739 -> 2.621994, accuracy 0.133025 -> 0.587057
- Frame 000127: loss 4.031899 -> 2.224603, accuracy 0.151343 -> 0.637913
- Frame 000128: loss 4.144859 -> 2.534088, accuracy 0.119897 -> 0.612920
- Frame 000129: loss 3.983657 -> 2.622660, accuracy 0.137285 -> 0.622985
- Frame 000130: loss 4.003787 -> 2.156045, accuracy 0.133092 -> 0.664940
- Frame 000131: loss 4.051928 -> 2.339376, accuracy 0.132841 -> 0.647865
- Frame 000132: loss 4.014249 -> 2.711659, accuracy 0.139167 -> 0.602530
- Frame 000133: loss 4.018854 -> 2.385527, accuracy 0.139842 -> 0.674934
- Frame 000134: loss 4.064719 -> 2.218105, accuracy 0.130822 -> 0.697541
- Frame 000135: loss 4.040910 -> 2.486183, accuracy 0.130753 -> 0.618724
- Frame 000136: loss 4.026442 -> 2.491190, accuracy 0.136620 -> 0.642527
- Frame 000137: loss 4.024220 -> 2.247067, accuracy 0.134791 -> 0.679552
- Frame 000138: loss 4.058852 -> 1.965609, accuracy 0.117557 -> 0.704835
- Frame 000139: loss 4.037484 -> 2.200400, accuracy 0.139828 -> 0.676426
- Frame 000140: loss 4.091760 -> 2.308397, accuracy 0.101815 -> 0.705141
- Frame 000141: loss 4.078958 -> 2.193401, accuracy 0.109383 -> 0.710487
- Frame 000142: loss 4.066493 -> 1.922321, accuracy 0.114243 -> 0.757423
- Frame 000143: loss 4.098281 -> 2.392975, accuracy 0.092602 -> 0.682939
- Frame 000144: loss 4.040022 -> 2.501464, accuracy 0.106177 -> 0.649821
- Frame 000145: loss 4.046609 -> 2.461232, accuracy 0.109295 -> 0.677222
- Frame 000146: loss 4.067894 -> 2.398160, accuracy 0.094082 -> 0.663126
- Frame 000147: loss 4.090628 -> 2.258295, accuracy 0.113428 -> 0.687182
- Frame 000148: loss 4.097988 -> 2.809248, accuracy 0.093387 -> 0.588087
- Frame 000149: loss 4.068055 -> 2.315009, accuracy 0.098527 -> 0.690706

## Holdout Final Class IoU

- 00 car: IoU 0.430153, acc 0.768362, gt 11682, pred 18161, tp 8976
- 01 bicycle: IoU 0.006944, acc 0.009615, gt 104, pred 41, tp 1
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 8, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 18, tp 0
- 08 road: IoU 0.675808, acc 0.812497, gt 23173, pred 23515, tp 18828
- 09 parking: IoU 0.088511, acc 0.140127, gt 1413, pred 1022, tp 198
- 10 sidewalk: IoU 0.493374, acc 0.742376, gt 17052, pred 21265, tp 12659
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.322377, acc 0.345321, gt 17184, pred 7157, tp 5934
- 13 fence: IoU 0.002066, acc 0.002169, gt 461, pred 24, tp 1
- 14 vegetation: IoU 0.363250, acc 0.529497, gt 23596, pred 23293, tp 12494
- 15 trunk: IoU 0.062415, acc 0.090551, gt 508, pred 275, tp 46
- 16 terrain: IoU 0.012525, acc 0.022124, gt 1130, pred 891, tp 25
- 17 pole: IoU 0.083704, acc 0.298413, gt 315, pred 902, tp 94
- 18 traffic-sign: IoU 0.047218, acc 0.168675, gt 166, pred 455, tp 28
