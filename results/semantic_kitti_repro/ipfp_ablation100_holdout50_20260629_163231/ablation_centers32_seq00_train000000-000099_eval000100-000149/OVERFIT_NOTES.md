# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.334310
- Final eval loss, first frame: 0.423057
- Initial eval mean loss: 3.537642
- Final eval mean loss: 0.431886
- Initial eval mean accuracy: 0.044838
- Final eval mean accuracy: 0.870638
- Initial eval mIoU: 0.006238
- Final eval mIoU: 0.471228
- Initial eval overall accuracy: 0.044810
- Final eval overall accuracy: 0.870584
- First train loss: 3.537881
- Final train loss: 0.207676
- Best train loss: 0.110493
- Final train valid accuracy: 0.927477
- CUDA peak memory GB: 0.957
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.239033
- Holdout final mean loss: 1.949960
- Holdout initial mean accuracy: 0.101712
- Holdout final mean accuracy: 0.585366
- Holdout initial mIoU: 0.011097
- Holdout final mIoU: 0.140137
- Holdout initial overall accuracy: 0.101786
- Holdout final overall accuracy: 0.585806

## Train-Frame Eval

- Frame 000000: loss 3.334310 -> 0.423057, accuracy 0.078046 -> 0.860020
- Frame 000001: loss 3.383463 -> 0.345621, accuracy 0.063068 -> 0.886983
- Frame 000002: loss 3.336652 -> 0.393622, accuracy 0.081602 -> 0.876330
- Frame 000003: loss 3.299839 -> 0.433986, accuracy 0.082365 -> 0.867105
- Frame 000004: loss 3.381110 -> 0.448098, accuracy 0.073812 -> 0.852882
- Frame 000005: loss 3.426203 -> 0.508904, accuracy 0.057315 -> 0.862242
- Frame 000006: loss 3.403756 -> 0.406678, accuracy 0.056366 -> 0.872672
- Frame 000007: loss 3.347468 -> 0.449256, accuracy 0.067703 -> 0.864594
- Frame 000008: loss 3.390560 -> 0.539804, accuracy 0.065196 -> 0.850050
- Frame 000009: loss 3.400123 -> 0.524011, accuracy 0.067745 -> 0.834176
- Frame 000010: loss 3.379065 -> 0.438450, accuracy 0.073256 -> 0.867035
- Frame 000011: loss 3.371413 -> 0.530043, accuracy 0.072927 -> 0.843157
- Frame 000012: loss 3.375795 -> 0.468959, accuracy 0.068432 -> 0.855145
- Frame 000013: loss 3.398859 -> 0.700636, accuracy 0.070571 -> 0.824825
- Frame 000014: loss 3.422082 -> 0.777556, accuracy 0.066600 -> 0.812218
- Frame 000015: loss 3.434258 -> 0.681992, accuracy 0.054808 -> 0.827105
- Frame 000016: loss 3.429576 -> 0.572042, accuracy 0.049574 -> 0.845268
- Frame 000017: loss 3.452372 -> 0.705064, accuracy 0.050856 -> 0.835851
- Frame 000018: loss 3.470292 -> 0.502789, accuracy 0.043303 -> 0.859013
- Frame 000019: loss 3.444616 -> 0.453097, accuracy 0.065631 -> 0.869238
- Frame 000020: loss 3.472198 -> 0.628783, accuracy 0.042317 -> 0.854408
- Frame 000021: loss 3.461647 -> 0.645783, accuracy 0.045568 -> 0.832749
- Frame 000022: loss 3.492001 -> 0.700515, accuracy 0.049064 -> 0.816388
- Frame 000023: loss 3.494202 -> 0.720729, accuracy 0.052369 -> 0.799002
- Frame 000024: loss 3.536634 -> 0.785749, accuracy 0.037037 -> 0.813313
- Frame 000025: loss 3.505990 -> 0.707914, accuracy 0.039500 -> 0.805000
- Frame 000026: loss 3.529465 -> 0.661560, accuracy 0.038883 -> 0.819043
- Frame 000027: loss 3.545960 -> 0.691328, accuracy 0.035465 -> 0.811688
- Frame 000028: loss 3.546351 -> 0.590036, accuracy 0.033367 -> 0.836155
- Frame 000029: loss 3.563701 -> 0.596046, accuracy 0.036482 -> 0.847076
- Frame 000030: loss 3.566110 -> 0.548553, accuracy 0.040339 -> 0.848108
- Frame 000031: loss 3.594831 -> 0.405622, accuracy 0.032934 -> 0.875250
- Frame 000032: loss 3.551590 -> 0.509029, accuracy 0.037313 -> 0.866667
- Frame 000033: loss 3.542198 -> 0.419285, accuracy 0.038442 -> 0.879181
- Frame 000034: loss 3.582054 -> 0.448122, accuracy 0.032787 -> 0.872330
- Frame 000035: loss 3.549796 -> 0.434180, accuracy 0.040520 -> 0.864932
- Frame 000036: loss 3.543595 -> 0.553012, accuracy 0.036610 -> 0.840020
- Frame 000037: loss 3.547718 -> 0.502815, accuracy 0.043912 -> 0.852295
- Frame 000038: loss 3.562934 -> 0.491760, accuracy 0.029618 -> 0.851406
- Frame 000039: loss 3.564250 -> 0.378537, accuracy 0.033467 -> 0.888611
- Frame 000040: loss 3.500198 -> 0.462253, accuracy 0.043194 -> 0.864892
- Frame 000041: loss 3.559476 -> 0.555798, accuracy 0.033216 -> 0.836437
- Frame 000042: loss 3.571687 -> 0.545127, accuracy 0.037943 -> 0.835746
- Frame 000043: loss 3.588708 -> 0.560541, accuracy 0.033066 -> 0.837675
- Frame 000044: loss 3.567824 -> 0.640989, accuracy 0.045844 -> 0.815113
- Frame 000045: loss 3.553077 -> 0.518753, accuracy 0.040622 -> 0.848546
- Frame 000046: loss 3.608367 -> 0.504112, accuracy 0.038922 -> 0.847804
- Frame 000047: loss 3.548311 -> 0.382621, accuracy 0.040419 -> 0.860279
- Frame 000048: loss 3.573238 -> 0.301867, accuracy 0.038057 -> 0.896345
- Frame 000049: loss 3.565629 -> 0.333935, accuracy 0.042914 -> 0.877744
- Frame 000050: loss 3.568719 -> 0.353327, accuracy 0.051411 -> 0.874496
- Frame 000051: loss 3.565744 -> 0.376768, accuracy 0.040642 -> 0.872052
- Frame 000052: loss 3.559253 -> 0.305066, accuracy 0.049401 -> 0.888723
- Frame 000053: loss 3.561774 -> 0.496682, accuracy 0.033199 -> 0.855634
- Frame 000054: loss 3.582534 -> 0.350144, accuracy 0.027404 -> 0.881415
- Frame 000055: loss 3.616073 -> 0.472346, accuracy 0.030470 -> 0.856643
- Frame 000056: loss 3.612906 -> 0.556430, accuracy 0.018454 -> 0.838404
- Frame 000057: loss 3.609377 -> 0.438864, accuracy 0.030288 -> 0.875869
- Frame 000058: loss 3.580301 -> 0.416387, accuracy 0.025050 -> 0.869739
- Frame 000059: loss 3.618695 -> 0.568584, accuracy 0.029015 -> 0.825913
- Frame 000060: loss 3.601934 -> 0.436247, accuracy 0.035803 -> 0.853804
- Frame 000061: loss 3.576583 -> 0.458460, accuracy 0.037736 -> 0.853029
- Frame 000062: loss 3.619542 -> 0.505890, accuracy 0.034040 -> 0.851011
- Frame 000063: loss 3.619659 -> 0.473930, accuracy 0.023845 -> 0.843517
- Frame 000064: loss 3.667140 -> 0.416679, accuracy 0.023833 -> 0.878848
- Frame 000065: loss 3.631204 -> 0.341881, accuracy 0.032819 -> 0.891099
- Frame 000066: loss 3.595366 -> 0.338392, accuracy 0.034448 -> 0.894159
- Frame 000067: loss 3.642841 -> 0.267889, accuracy 0.031375 -> 0.904881
- Frame 000068: loss 3.676495 -> 0.315411, accuracy 0.031563 -> 0.889780
- Frame 000069: loss 3.641987 -> 0.273566, accuracy 0.023881 -> 0.909950
- Frame 000070: loss 3.664750 -> 0.277128, accuracy 0.037074 -> 0.898171
- Frame 000071: loss 3.610088 -> 0.308911, accuracy 0.053154 -> 0.895678
- Frame 000072: loss 3.591433 -> 0.296230, accuracy 0.038138 -> 0.893512
- Frame 000073: loss 3.590183 -> 0.255012, accuracy 0.044191 -> 0.913605
- Frame 000074: loss 3.578566 -> 0.281204, accuracy 0.041812 -> 0.912394
- Frame 000075: loss 3.609010 -> 0.237619, accuracy 0.037481 -> 0.909545
- Frame 000076: loss 3.580758 -> 0.281310, accuracy 0.047076 -> 0.914272
- Frame 000077: loss 3.571769 -> 0.278845, accuracy 0.049074 -> 0.903355
- Frame 000078: loss 3.616131 -> 0.297565, accuracy 0.042084 -> 0.901804
- Frame 000079: loss 3.605309 -> 0.276968, accuracy 0.041144 -> 0.914701
- Frame 000080: loss 3.624295 -> 0.286392, accuracy 0.046977 -> 0.903548
- Frame 000081: loss 3.610925 -> 0.278046, accuracy 0.051385 -> 0.913854
- Frame 000082: loss 3.620038 -> 0.350686, accuracy 0.049776 -> 0.887008
- Frame 000083: loss 3.562071 -> 0.275606, accuracy 0.049296 -> 0.910966
- Frame 000084: loss 3.555632 -> 0.335504, accuracy 0.055165 -> 0.895687
- Frame 000085: loss 3.532781 -> 0.389544, accuracy 0.059910 -> 0.886670
- Frame 000086: loss 3.515449 -> 0.306820, accuracy 0.060682 -> 0.892678
- Frame 000087: loss 3.528506 -> 0.231453, accuracy 0.056911 -> 0.924289
- Frame 000088: loss 3.537227 -> 0.285168, accuracy 0.052737 -> 0.910095
- Frame 000089: loss 3.506711 -> 0.211071, accuracy 0.051463 -> 0.926842
- Frame 000090: loss 3.598588 -> 0.342207, accuracy 0.032978 -> 0.904617
- Frame 000091: loss 3.576370 -> 0.248423, accuracy 0.036077 -> 0.922256
- Frame 000092: loss 3.548054 -> 0.255960, accuracy 0.045385 -> 0.917389
- Frame 000093: loss 3.545316 -> 0.229596, accuracy 0.049645 -> 0.917427
- Frame 000094: loss 3.581441 -> 0.239001, accuracy 0.035318 -> 0.924319
- Frame 000095: loss 3.572014 -> 0.254538, accuracy 0.037794 -> 0.918795
- Frame 000096: loss 3.580742 -> 0.280241, accuracy 0.044035 -> 0.898618
- Frame 000097: loss 3.579095 -> 0.292278, accuracy 0.035312 -> 0.894575
- Frame 000098: loss 3.584320 -> 0.294390, accuracy 0.039837 -> 0.896834
- Frame 000099: loss 3.566956 -> 0.318879, accuracy 0.037794 -> 0.891216

## Train Final Class IoU

- 00 car: IoU 0.905204, acc 0.964294, gt 24786, pred 25519, tp 23901
- 01 bicycle: IoU 0.293578, acc 0.470588, gt 68, pred 73, tp 32
- 04 other-vehicle: IoU 0.412100, acc 0.441320, gt 818, pred 419, tp 361
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.452229, acc 0.476510, gt 149, pred 79, tp 71
- 08 road: IoU 0.895970, acc 0.982945, gt 53650, pred 57943, tp 52735
- 09 parking: IoU 0.544352, acc 0.574717, gt 17031, pred 10738, tp 9788
- 10 sidewalk: IoU 0.546657, acc 0.843029, gt 18290, pred 25335, tp 15419
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.869205, acc 0.878871, gt 53059, pred 47222, tp 46632
- 13 fence: IoU 0.284423, acc 0.307288, gt 1331, pred 516, tp 409
- 14 vegetation: IoU 0.651849, acc 0.819961, gt 23606, pred 25444, tp 19356
- 15 trunk: IoU 0.771930, acc 0.792567, gt 5274, pred 4321, tp 4180
- 16 terrain: IoU 0.139868, acc 0.262397, gt 484, pred 551, tp 127
- 17 pole: IoU 0.487324, acc 0.762115, gt 681, pred 903, tp 519
- 18 traffic-sign: IoU 0.284956, acc 0.579137, gt 278, pred 448, tp 161

## Holdout Per-Frame Eval

- Frame 000100: loss 3.544555 -> 0.710858, accuracy 0.039487 -> 0.790256
- Frame 000101: loss 3.567357 -> 0.776964, accuracy 0.036674 -> 0.784607
- Frame 000102: loss 3.523621 -> 0.943451, accuracy 0.050978 -> 0.767250
- Frame 000103: loss 3.527608 -> 1.071629, accuracy 0.049325 -> 0.719626
- Frame 000104: loss 3.458831 -> 1.548486, accuracy 0.055584 -> 0.644879
- Frame 000105: loss 3.480632 -> 1.683897, accuracy 0.058549 -> 0.601036
- Frame 000106: loss 3.464420 -> 2.164819, accuracy 0.058522 -> 0.552361
- Frame 000107: loss 3.435700 -> 2.363853, accuracy 0.053292 -> 0.515152
- Frame 000108: loss 3.395350 -> 2.517626, accuracy 0.073133 -> 0.511411
- Frame 000109: loss 3.431173 -> 2.587937, accuracy 0.076842 -> 0.493684
- Frame 000110: loss 3.395028 -> 2.774850, accuracy 0.073514 -> 0.495308
- Frame 000111: loss 3.393406 -> 2.806270, accuracy 0.075789 -> 0.488947
- Frame 000112: loss 3.375830 -> 3.177690, accuracy 0.081958 -> 0.440128
- Frame 000113: loss 3.363749 -> 2.892757, accuracy 0.073996 -> 0.487315
- Frame 000114: loss 3.330229 -> 2.369343, accuracy 0.089237 -> 0.502975
- Frame 000115: loss 3.311806 -> 2.561865, accuracy 0.098901 -> 0.465201
- Frame 000116: loss 3.314768 -> 2.168497, accuracy 0.103269 -> 0.543851
- Frame 000117: loss 3.285625 -> 2.368974, accuracy 0.097536 -> 0.514374
- Frame 000118: loss 3.269620 -> 2.664322, accuracy 0.111738 -> 0.481292
- Frame 000119: loss 3.264626 -> 2.200341, accuracy 0.100406 -> 0.546146
- Frame 000120: loss 3.247024 -> 2.182597, accuracy 0.116624 -> 0.540665
- Frame 000121: loss 3.203905 -> 2.165271, accuracy 0.122771 -> 0.551197
- Frame 000122: loss 3.168373 -> 1.993546, accuracy 0.136086 -> 0.576453
- Frame 000123: loss 3.134483 -> 2.255273, accuracy 0.125764 -> 0.499491
- Frame 000124: loss 3.200191 -> 2.062489, accuracy 0.121677 -> 0.561350
- Frame 000125: loss 3.201024 -> 1.676795, accuracy 0.109750 -> 0.620725
- Frame 000126: loss 3.116665 -> 1.973570, accuracy 0.130971 -> 0.563945
- Frame 000127: loss 3.080401 -> 1.646000, accuracy 0.148760 -> 0.623450
- Frame 000128: loss 3.192366 -> 2.078593, accuracy 0.120930 -> 0.540052
- Frame 000129: loss 3.048396 -> 1.827220, accuracy 0.140406 -> 0.610504
- Frame 000130: loss 3.066005 -> 1.926086, accuracy 0.129467 -> 0.581046
- Frame 000131: loss 3.102317 -> 1.775835, accuracy 0.130733 -> 0.605693
- Frame 000132: loss 3.071512 -> 1.746745, accuracy 0.136531 -> 0.612019
- Frame 000133: loss 3.068323 -> 1.663858, accuracy 0.140369 -> 0.606860
- Frame 000134: loss 3.118371 -> 1.750717, accuracy 0.124019 -> 0.618524
- Frame 000135: loss 3.091610 -> 1.736141, accuracy 0.135460 -> 0.596757
- Frame 000136: loss 3.092579 -> 2.007681, accuracy 0.129944 -> 0.552131
- Frame 000137: loss 3.086378 -> 2.028182, accuracy 0.130722 -> 0.578332
- Frame 000138: loss 3.121826 -> 1.695983, accuracy 0.114504 -> 0.645802
- Frame 000139: loss 3.098194 -> 1.366894, accuracy 0.135285 -> 0.666330
- Frame 000140: loss 3.143322 -> 1.808539, accuracy 0.102319 -> 0.635081
- Frame 000141: loss 3.126139 -> 1.565738, accuracy 0.108881 -> 0.665830
- Frame 000142: loss 3.124868 -> 1.286188, accuracy 0.114746 -> 0.680926
- Frame 000143: loss 3.157385 -> 2.018945, accuracy 0.098138 -> 0.570710
- Frame 000144: loss 3.089904 -> 1.854074, accuracy 0.109750 -> 0.577335
- Frame 000145: loss 3.107933 -> 1.756374, accuracy 0.110827 -> 0.634831
- Frame 000146: loss 3.123996 -> 1.700208, accuracy 0.099140 -> 0.599899
- Frame 000147: loss 3.152241 -> 1.582284, accuracy 0.111902 -> 0.653611
- Frame 000148: loss 3.155194 -> 1.922199, accuracy 0.091873 -> 0.571933
- Frame 000149: loss 3.126794 -> 2.089563, accuracy 0.098527 -> 0.581006

## Holdout Final Class IoU

- 00 car: IoU 0.390635, acc 0.744821, gt 11682, pred 19293, tp 8701
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 9, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 6, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 108, tp 0
- 08 road: IoU 0.657359, acc 0.920381, gt 23173, pred 30600, tp 21328
- 09 parking: IoU 0.055864, acc 0.106865, gt 1413, pred 1441, tp 151
- 10 sidewalk: IoU 0.371046, acc 0.499472, gt 17052, pred 14419, tp 8517
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.394016, acc 0.425338, gt 17184, pred 8675, tp 7309
- 13 fence: IoU 0.001908, acc 0.002169, gt 461, pred 64, tp 1
- 14 vegetation: IoU 0.331384, acc 0.448974, gt 23596, pred 18967, tp 10594
- 15 trunk: IoU 0.040179, acc 0.053150, gt 508, pred 191, tp 27
- 16 terrain: IoU 0.014377, acc 0.023894, gt 1130, pred 775, tp 27
- 17 pole: IoU 0.084211, acc 0.431746, gt 315, pred 1436, tp 136
- 18 traffic-sign: IoU 0.041344, acc 0.289157, gt 166, pred 1043, tp 48
