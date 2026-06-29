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
- Extra feature mode/scale: learned, 0.100
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.290454
- Final eval loss, first frame: 0.935373
- Initial eval mean loss: 4.492375
- Final eval mean loss: 0.884748
- Initial eval mean accuracy: 0.044746
- Final eval mean accuracy: 0.873870
- Initial eval mIoU: 0.006251
- Final eval mIoU: 0.470441
- Initial eval overall accuracy: 0.044719
- Final eval overall accuracy: 0.873821
- First train loss: 4.509683
- Final train loss: 0.632670
- Best train loss: 0.342863
- Final train valid accuracy: 0.893769
- CUDA peak memory GB: 0.961
- Selected visualization frames: 4

## Holdout Eval

- Holdout initial mean loss: 4.186628
- Holdout final mean loss: 2.614010
- Holdout initial mean accuracy: 0.102250
- Holdout final mean accuracy: 0.616847
- Holdout initial mIoU: 0.011186
- Holdout final mIoU: 0.152332
- Holdout initial overall accuracy: 0.102322
- Holdout final overall accuracy: 0.617313

## Diagnostic fused Eval

- Train final mean loss: 0.889924
- Train final mean accuracy: 0.872905
- Train final mIoU: 0.466309
- Train final overall accuracy: 0.872854

- Holdout final mean loss: 2.613548
- Holdout final mean accuracy: 0.615606
- Holdout final mIoU: 0.153160
- Holdout final overall accuracy: 0.616035

## Train-Frame Eval

- Frame 000000: loss 4.290454 -> 0.935373, accuracy 0.075529 -> 0.842397
- Frame 000001: loss 4.347718 -> 0.913456, accuracy 0.060545 -> 0.860242
- Frame 000002: loss 4.292550 -> 0.960178, accuracy 0.081095 -> 0.865687
- Frame 000003: loss 4.257922 -> 0.830544, accuracy 0.079838 -> 0.877211
- Frame 000004: loss 4.334030 -> 0.948257, accuracy 0.078362 -> 0.859454
- Frame 000005: loss 4.376268 -> 0.904859, accuracy 0.055807 -> 0.880845
- Frame 000006: loss 4.359628 -> 0.867625, accuracy 0.057373 -> 0.877705
- Frame 000007: loss 4.309052 -> 0.941633, accuracy 0.072718 -> 0.869107
- Frame 000008: loss 4.355280 -> 0.970470, accuracy 0.063691 -> 0.861585
- Frame 000009: loss 4.359986 -> 1.012257, accuracy 0.065217 -> 0.857937
- Frame 000010: loss 4.333689 -> 1.003889, accuracy 0.071249 -> 0.869042
- Frame 000011: loss 4.330659 -> 1.034758, accuracy 0.070430 -> 0.849650
- Frame 000012: loss 4.338854 -> 1.056484, accuracy 0.064436 -> 0.876623
- Frame 000013: loss 4.361062 -> 1.052352, accuracy 0.067568 -> 0.842843
- Frame 000014: loss 4.380494 -> 1.333023, accuracy 0.067101 -> 0.824737
- Frame 000015: loss 4.404830 -> 1.343372, accuracy 0.054808 -> 0.818635
- Frame 000016: loss 4.392282 -> 1.111382, accuracy 0.053080 -> 0.841262
- Frame 000017: loss 4.407873 -> 1.092667, accuracy 0.049849 -> 0.849446
- Frame 000018: loss 4.428964 -> 0.975888, accuracy 0.043807 -> 0.853474
- Frame 000019: loss 4.404348 -> 0.944310, accuracy 0.065631 -> 0.857715
- Frame 000020: loss 4.430857 -> 1.002149, accuracy 0.047355 -> 0.863476
- Frame 000021: loss 4.432273 -> 1.027617, accuracy 0.044066 -> 0.856785
- Frame 000022: loss 4.457365 -> 1.032281, accuracy 0.050582 -> 0.854831
- Frame 000023: loss 4.449932 -> 1.187132, accuracy 0.049875 -> 0.840399
- Frame 000024: loss 4.497808 -> 1.225043, accuracy 0.033033 -> 0.838839
- Frame 000025: loss 4.468760 -> 1.180615, accuracy 0.040000 -> 0.834500
- Frame 000026: loss 4.489631 -> 1.121961, accuracy 0.038883 -> 0.850947
- Frame 000027: loss 4.504224 -> 0.965624, accuracy 0.034965 -> 0.849650
- Frame 000028: loss 4.498932 -> 1.068668, accuracy 0.032371 -> 0.860060
- Frame 000029: loss 4.519519 -> 1.142306, accuracy 0.036482 -> 0.862569
- Frame 000030: loss 4.521461 -> 1.223974, accuracy 0.039343 -> 0.828187
- Frame 000031: loss 4.539450 -> 0.992663, accuracy 0.031936 -> 0.877744
- Frame 000032: loss 4.511074 -> 0.948877, accuracy 0.033831 -> 0.864677
- Frame 000033: loss 4.502346 -> 1.020624, accuracy 0.041438 -> 0.859211
- Frame 000034: loss 4.529029 -> 0.759613, accuracy 0.034277 -> 0.901639
- Frame 000035: loss 4.503929 -> 0.837806, accuracy 0.042021 -> 0.888945
- Frame 000036: loss 4.497344 -> 0.927046, accuracy 0.039117 -> 0.860080
- Frame 000037: loss 4.505861 -> 0.870813, accuracy 0.043912 -> 0.875749
- Frame 000038: loss 4.511147 -> 0.899539, accuracy 0.030120 -> 0.873494
- Frame 000039: loss 4.516031 -> 0.824604, accuracy 0.033966 -> 0.888611
- Frame 000040: loss 4.450143 -> 0.890112, accuracy 0.040683 -> 0.873430
- Frame 000041: loss 4.510322 -> 0.947664, accuracy 0.031203 -> 0.880725
- Frame 000042: loss 4.529039 -> 0.896590, accuracy 0.035447 -> 0.862207
- Frame 000043: loss 4.537968 -> 1.135822, accuracy 0.032064 -> 0.841182
- Frame 000044: loss 4.520801 -> 1.052024, accuracy 0.042317 -> 0.834257
- Frame 000045: loss 4.497778 -> 0.920562, accuracy 0.042126 -> 0.869107
- Frame 000046: loss 4.555288 -> 1.114889, accuracy 0.037924 -> 0.832834
- Frame 000047: loss 4.500817 -> 0.968531, accuracy 0.038423 -> 0.862774
- Frame 000048: loss 4.527078 -> 0.955553, accuracy 0.036054 -> 0.861793
- Frame 000049: loss 4.517334 -> 0.962228, accuracy 0.042415 -> 0.858283
- Frame 000050: loss 4.515802 -> 0.862332, accuracy 0.053427 -> 0.867944
- Frame 000051: loss 4.525723 -> 1.079129, accuracy 0.040642 -> 0.840442
- Frame 000052: loss 4.513604 -> 0.979383, accuracy 0.049401 -> 0.840319
- Frame 000053: loss 4.511569 -> 1.154131, accuracy 0.036217 -> 0.826962
- Frame 000054: loss 4.539930 -> 1.086880, accuracy 0.030394 -> 0.837569
- Frame 000055: loss 4.571929 -> 1.095867, accuracy 0.028472 -> 0.850150
- Frame 000056: loss 4.566489 -> 1.078129, accuracy 0.019451 -> 0.830424
- Frame 000057: loss 4.560937 -> 0.842483, accuracy 0.032771 -> 0.868918
- Frame 000058: loss 4.529865 -> 0.893587, accuracy 0.025050 -> 0.870241
- Frame 000059: loss 4.565470 -> 1.087595, accuracy 0.032016 -> 0.832416
- Frame 000060: loss 4.551235 -> 0.986395, accuracy 0.036798 -> 0.849826
- Frame 000061: loss 4.518074 -> 1.112338, accuracy 0.036743 -> 0.850546
- Frame 000062: loss 4.575908 -> 0.933963, accuracy 0.032560 -> 0.858905
- Frame 000063: loss 4.579570 -> 0.944378, accuracy 0.023348 -> 0.855440
- Frame 000064: loss 4.626273 -> 0.853236, accuracy 0.025819 -> 0.873386
- Frame 000065: loss 4.581199 -> 0.813278, accuracy 0.032322 -> 0.878667
- Frame 000066: loss 4.554815 -> 0.752132, accuracy 0.034948 -> 0.905642
- Frame 000067: loss 4.596808 -> 0.767896, accuracy 0.034363 -> 0.916833
- Frame 000068: loss 4.624096 -> 0.832334, accuracy 0.031062 -> 0.888277
- Frame 000069: loss 4.591411 -> 0.585131, accuracy 0.028358 -> 0.917911
- Frame 000070: loss 4.612396 -> 0.524143, accuracy 0.038062 -> 0.918932
- Frame 000071: loss 4.557644 -> 0.710802, accuracy 0.052161 -> 0.895678
- Frame 000072: loss 4.534377 -> 0.630854, accuracy 0.038138 -> 0.910352
- Frame 000073: loss 4.547366 -> 0.522784, accuracy 0.048163 -> 0.916087
- Frame 000074: loss 4.527189 -> 0.551890, accuracy 0.041812 -> 0.918865
- Frame 000075: loss 4.565489 -> 0.868212, accuracy 0.037481 -> 0.864568
- Frame 000076: loss 4.527236 -> 0.685691, accuracy 0.048067 -> 0.905847
- Frame 000077: loss 4.520401 -> 0.643254, accuracy 0.046069 -> 0.895343
- Frame 000078: loss 4.571118 -> 0.576605, accuracy 0.042084 -> 0.916834
- Frame 000079: loss 4.561244 -> 0.564798, accuracy 0.042649 -> 0.916207
- Frame 000080: loss 4.578698 -> 0.568227, accuracy 0.044478 -> 0.915043
- Frame 000081: loss 4.564831 -> 0.718385, accuracy 0.054912 -> 0.901763
- Frame 000082: loss 4.580342 -> 0.665567, accuracy 0.048283 -> 0.905923
- Frame 000083: loss 4.520039 -> 0.650533, accuracy 0.050302 -> 0.910966
- Frame 000084: loss 4.500638 -> 0.686398, accuracy 0.055667 -> 0.913741
- Frame 000085: loss 4.490545 -> 0.886479, accuracy 0.057414 -> 0.867698
- Frame 000086: loss 4.460506 -> 0.749629, accuracy 0.063190 -> 0.882648
- Frame 000087: loss 4.479492 -> 0.568396, accuracy 0.057419 -> 0.926321
- Frame 000088: loss 4.489429 -> 0.697444, accuracy 0.055249 -> 0.898543
- Frame 000089: loss 4.455327 -> 0.539753, accuracy 0.049950 -> 0.925328
- Frame 000090: loss 4.554498 -> 0.718407, accuracy 0.031963 -> 0.897007
- Frame 000091: loss 4.529914 -> 0.629117, accuracy 0.033028 -> 0.928862
- Frame 000092: loss 4.506181 -> 0.577253, accuracy 0.044875 -> 0.920959
- Frame 000093: loss 4.503380 -> 0.631319, accuracy 0.049645 -> 0.919453
- Frame 000094: loss 4.539608 -> 0.524026, accuracy 0.034813 -> 0.920283
- Frame 000095: loss 4.528509 -> 0.609774, accuracy 0.037283 -> 0.911645
- Frame 000096: loss 4.539485 -> 0.611542, accuracy 0.044547 -> 0.913979
- Frame 000097: loss 4.535792 -> 0.647822, accuracy 0.035824 -> 0.903787
- Frame 000098: loss 4.540211 -> 0.698436, accuracy 0.037283 -> 0.888151
- Frame 000099: loss 4.515449 -> 0.740963, accuracy 0.035240 -> 0.872829

## Train Final Class IoU

- 00 car: IoU 0.894266, acc 0.940087, gt 24786, pred 24571, tp 23301
- 01 bicycle: IoU 0.269231, acc 0.617647, gt 68, pred 130, tp 42
- 04 other-vehicle: IoU 0.198009, acc 0.218826, gt 818, pred 265, tp 179
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.483660, acc 0.496644, gt 149, pred 78, tp 74
- 08 road: IoU 0.873405, acc 0.984790, gt 53650, pred 59676, tp 52834
- 09 parking: IoU 0.483328, acc 0.502143, gt 17031, pred 9215, tp 8552
- 10 sidewalk: IoU 0.532325, acc 0.759923, gt 18290, pred 21719, tp 13899
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.901907, acc 0.933150, gt 53059, pred 51350, tp 49512
- 13 fence: IoU 0.137427, acc 0.141247, gt 1331, pred 225, tp 188
- 14 vegetation: IoU 0.698524, acc 0.880242, gt 23606, pred 26920, tp 20779
- 15 trunk: IoU 0.780475, acc 0.817027, gt 5274, pred 4556, tp 4309
- 16 terrain: IoU 0.170813, acc 0.212810, gt 484, pred 222, tp 103
- 17 pole: IoU 0.615721, acc 0.621145, gt 681, pred 429, tp 423
- 18 traffic-sign: IoU 0.487973, acc 0.510791, gt 278, pred 155, tp 142

## Holdout Per-Frame Eval

- Frame 000100: loss 4.502586 -> 1.264231, accuracy 0.041026 -> 0.807692
- Frame 000101: loss 4.529016 -> 1.592991, accuracy 0.035640 -> 0.759297
- Frame 000102: loss 4.462615 -> 1.550142, accuracy 0.049949 -> 0.771885
- Frame 000103: loss 4.487285 -> 1.641642, accuracy 0.048806 -> 0.731568
- Frame 000104: loss 4.414391 -> 2.142551, accuracy 0.056613 -> 0.679362
- Frame 000105: loss 4.433777 -> 2.468582, accuracy 0.059585 -> 0.612435
- Frame 000106: loss 4.419362 -> 2.754616, accuracy 0.061088 -> 0.599589
- Frame 000107: loss 4.386530 -> 3.281823, accuracy 0.055381 -> 0.524033
- Frame 000108: loss 4.352805 -> 3.186440, accuracy 0.068465 -> 0.546162
- Frame 000109: loss 4.385810 -> 3.174987, accuracy 0.073158 -> 0.531579
- Frame 000110: loss 4.354399 -> 3.724727, accuracy 0.074035 -> 0.481230
- Frame 000111: loss 4.339562 -> 3.479243, accuracy 0.078947 -> 0.500526
- Frame 000112: loss 4.327701 -> 3.524762, accuracy 0.081426 -> 0.507717
- Frame 000113: loss 4.318940 -> 3.461939, accuracy 0.074524 -> 0.495772
- Frame 000114: loss 4.286531 -> 3.143302, accuracy 0.089237 -> 0.532180
- Frame 000115: loss 4.261898 -> 3.212734, accuracy 0.103611 -> 0.512820
- Frame 000116: loss 4.268469 -> 2.866840, accuracy 0.102750 -> 0.562532
- Frame 000117: loss 4.235785 -> 2.950538, accuracy 0.101129 -> 0.540554
- Frame 000118: loss 4.221671 -> 3.127293, accuracy 0.117376 -> 0.517171
- Frame 000119: loss 4.221683 -> 2.648245, accuracy 0.102434 -> 0.559838
- Frame 000120: loss 4.197339 -> 2.661423, accuracy 0.115090 -> 0.567775
- Frame 000121: loss 4.153171 -> 2.404742, accuracy 0.119205 -> 0.624554
- Frame 000122: loss 4.110054 -> 2.439226, accuracy 0.138634 -> 0.625382
- Frame 000123: loss 4.071923 -> 3.018003, accuracy 0.124745 -> 0.541242
- Frame 000124: loss 4.149584 -> 2.785056, accuracy 0.121677 -> 0.589980
- Frame 000125: loss 4.140788 -> 2.460542, accuracy 0.112813 -> 0.628382
- Frame 000126: loss 4.045741 -> 2.572551, accuracy 0.133025 -> 0.585003
- Frame 000127: loss 4.031945 -> 2.218156, accuracy 0.154442 -> 0.660640
- Frame 000128: loss 4.143367 -> 2.596549, accuracy 0.119380 -> 0.602067
- Frame 000129: loss 3.986136 -> 2.529670, accuracy 0.138326 -> 0.619345
- Frame 000130: loss 4.006090 -> 2.594216, accuracy 0.132056 -> 0.611082
- Frame 000131: loss 4.050820 -> 2.212159, accuracy 0.131260 -> 0.681603
- Frame 000132: loss 4.016533 -> 2.684481, accuracy 0.138640 -> 0.616763
- Frame 000133: loss 4.018696 -> 2.304060, accuracy 0.138259 -> 0.677045
- Frame 000134: loss 4.066174 -> 2.368982, accuracy 0.130822 -> 0.663527
- Frame 000135: loss 4.041370 -> 2.269527, accuracy 0.131799 -> 0.625523
- Frame 000136: loss 4.026021 -> 2.695184, accuracy 0.136107 -> 0.628659
- Frame 000137: loss 4.024655 -> 2.493209, accuracy 0.134791 -> 0.641404
- Frame 000138: loss 4.063285 -> 2.115193, accuracy 0.116031 -> 0.716031
- Frame 000139: loss 4.037079 -> 2.169549, accuracy 0.141343 -> 0.689551
- Frame 000140: loss 4.089217 -> 2.378879, accuracy 0.101310 -> 0.697077
- Frame 000141: loss 4.080057 -> 2.385953, accuracy 0.107376 -> 0.700452
- Frame 000142: loss 4.067700 -> 2.047760, accuracy 0.113236 -> 0.733769
- Frame 000143: loss 4.096816 -> 2.772298, accuracy 0.094615 -> 0.618520
- Frame 000144: loss 4.038421 -> 3.020941, accuracy 0.104645 -> 0.566616
- Frame 000145: loss 4.046134 -> 2.667789, accuracy 0.110317 -> 0.645046
- Frame 000146: loss 4.066474 -> 2.609477, accuracy 0.094082 -> 0.616085
- Frame 000147: loss 4.089515 -> 2.446392, accuracy 0.112411 -> 0.667345
- Frame 000148: loss 4.098166 -> 2.759989, accuracy 0.092378 -> 0.601716
- Frame 000149: loss 4.067292 -> 2.820921, accuracy 0.098527 -> 0.626206

## Holdout Final Class IoU

- 00 car: IoU 0.412839, acc 0.752525, gt 11682, pred 18403, tp 8791
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 22, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 58, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 6, tp 0
- 08 road: IoU 0.687620, acc 0.930911, gt 23173, pred 29771, tp 21572
- 09 parking: IoU 0.065593, acc 0.121019, gt 1413, pred 1365, tp 171
- 10 sidewalk: IoU 0.387633, acc 0.498886, gt 17052, pred 13401, tp 8507
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.398752, acc 0.446345, gt 17184, pred 9721, tp 7670
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 12, tp 0
- 14 vegetation: IoU 0.386778, acc 0.553399, gt 23596, pred 23223, tp 13058
- 15 trunk: IoU 0.049813, acc 0.078740, gt 508, pred 335, tp 40
- 16 terrain: IoU 0.010982, acc 0.015044, gt 1130, pred 435, tp 17
- 17 pole: IoU 0.136816, acc 0.174603, gt 315, pred 142, tp 55
- 18 traffic-sign: IoU 0.052817, acc 0.090361, gt 166, pred 133, tp 15
