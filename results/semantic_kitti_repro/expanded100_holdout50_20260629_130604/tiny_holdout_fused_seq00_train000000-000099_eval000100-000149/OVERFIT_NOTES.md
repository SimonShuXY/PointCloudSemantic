# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.333940
- Final eval loss, first frame: 0.528103
- Initial eval mean loss: 3.536851
- Final eval mean loss: 0.561801
- Initial eval mean accuracy: 0.044873
- Final eval mean accuracy: 0.850303
- Initial eval mIoU: 0.006179
- Final eval mIoU: 0.417617
- Initial eval overall accuracy: 0.044845
- Final eval overall accuracy: 0.850219
- First train loss: 3.543744
- Final train loss: 0.224641
- Best train loss: 0.112703
- Final train valid accuracy: 0.922370
- CUDA peak memory GB: 0.957
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.238278
- Holdout final mean loss: 2.391842
- Holdout initial mean accuracy: 0.102355
- Holdout final mean accuracy: 0.566377
- Holdout initial mIoU: 0.011189
- Holdout final mIoU: 0.134098
- Holdout initial overall accuracy: 0.102435
- Holdout final overall accuracy: 0.566749

## Train-Frame Eval

- Frame 000000: loss 3.333940 -> 0.528103, accuracy 0.076536 -> 0.828298
- Frame 000001: loss 3.384060 -> 0.419993, accuracy 0.063068 -> 0.861251
- Frame 000002: loss 3.332426 -> 0.480972, accuracy 0.086670 -> 0.859605
- Frame 000003: loss 3.296952 -> 0.462917, accuracy 0.081354 -> 0.861546
- Frame 000004: loss 3.381515 -> 0.528061, accuracy 0.077856 -> 0.846815
- Frame 000005: loss 3.429251 -> 0.517573, accuracy 0.055807 -> 0.860231
- Frame 000006: loss 3.406815 -> 0.365577, accuracy 0.057373 -> 0.880725
- Frame 000007: loss 3.347826 -> 0.617364, accuracy 0.068205 -> 0.846540
- Frame 000008: loss 3.390149 -> 0.607299, accuracy 0.064694 -> 0.853059
- Frame 000009: loss 3.395935 -> 0.755923, accuracy 0.068756 -> 0.824570
- Frame 000010: loss 3.376718 -> 0.711291, accuracy 0.070748 -> 0.834922
- Frame 000011: loss 3.367874 -> 0.706320, accuracy 0.072428 -> 0.830669
- Frame 000012: loss 3.377655 -> 0.703625, accuracy 0.064935 -> 0.834665
- Frame 000013: loss 3.394881 -> 0.828512, accuracy 0.069570 -> 0.819319
- Frame 000014: loss 3.422476 -> 0.999493, accuracy 0.069104 -> 0.797196
- Frame 000015: loss 3.436115 -> 1.033044, accuracy 0.054808 -> 0.789736
- Frame 000016: loss 3.424052 -> 0.776931, accuracy 0.052579 -> 0.833250
- Frame 000017: loss 3.450001 -> 0.838158, accuracy 0.050352 -> 0.824270
- Frame 000018: loss 3.469862 -> 0.741969, accuracy 0.046828 -> 0.839879
- Frame 000019: loss 3.442600 -> 0.576537, accuracy 0.063627 -> 0.850200
- Frame 000020: loss 3.469570 -> 0.639713, accuracy 0.043325 -> 0.847355
- Frame 000021: loss 3.458032 -> 0.837743, accuracy 0.046570 -> 0.814722
- Frame 000022: loss 3.494014 -> 0.686989, accuracy 0.049570 -> 0.846232
- Frame 000023: loss 3.493162 -> 1.037469, accuracy 0.050873 -> 0.776559
- Frame 000024: loss 3.542142 -> 0.916086, accuracy 0.036036 -> 0.798799
- Frame 000025: loss 3.511716 -> 0.869225, accuracy 0.039500 -> 0.806500
- Frame 000026: loss 3.528938 -> 0.977602, accuracy 0.041376 -> 0.784646
- Frame 000027: loss 3.544495 -> 0.826279, accuracy 0.035465 -> 0.806194
- Frame 000028: loss 3.546748 -> 0.830415, accuracy 0.033865 -> 0.808267
- Frame 000029: loss 3.561099 -> 0.705866, accuracy 0.035982 -> 0.832084
- Frame 000030: loss 3.560190 -> 0.824045, accuracy 0.039343 -> 0.812251
- Frame 000031: loss 3.591082 -> 0.656773, accuracy 0.032435 -> 0.842814
- Frame 000032: loss 3.555473 -> 0.699104, accuracy 0.039303 -> 0.834826
- Frame 000033: loss 3.537450 -> 0.584370, accuracy 0.039441 -> 0.851722
- Frame 000034: loss 3.582863 -> 0.563343, accuracy 0.032787 -> 0.848982
- Frame 000035: loss 3.545421 -> 0.504342, accuracy 0.039020 -> 0.861931
- Frame 000036: loss 3.542134 -> 0.637348, accuracy 0.036610 -> 0.829990
- Frame 000037: loss 3.548087 -> 0.731604, accuracy 0.041916 -> 0.817365
- Frame 000038: loss 3.565013 -> 0.665776, accuracy 0.030622 -> 0.831827
- Frame 000039: loss 3.568380 -> 0.572775, accuracy 0.032468 -> 0.851648
- Frame 000040: loss 3.498774 -> 0.618160, accuracy 0.043194 -> 0.836263
- Frame 000041: loss 3.559318 -> 0.655620, accuracy 0.030196 -> 0.814796
- Frame 000042: loss 3.571983 -> 0.645532, accuracy 0.036945 -> 0.838742
- Frame 000043: loss 3.584695 -> 0.730104, accuracy 0.033567 -> 0.825651
- Frame 000044: loss 3.565686 -> 0.696409, accuracy 0.044332 -> 0.805038
- Frame 000045: loss 3.546184 -> 0.546427, accuracy 0.039619 -> 0.846038
- Frame 000046: loss 3.604045 -> 0.648582, accuracy 0.038922 -> 0.818363
- Frame 000047: loss 3.543795 -> 0.611003, accuracy 0.040918 -> 0.829341
- Frame 000048: loss 3.571326 -> 0.491673, accuracy 0.039559 -> 0.844767
- Frame 000049: loss 3.568008 -> 0.401342, accuracy 0.042415 -> 0.856786
- Frame 000050: loss 3.566834 -> 0.617537, accuracy 0.049395 -> 0.813004
- Frame 000051: loss 3.569072 -> 0.531417, accuracy 0.040642 -> 0.845459
- Frame 000052: loss 3.563102 -> 0.570626, accuracy 0.047405 -> 0.830339
- Frame 000053: loss 3.561170 -> 0.735646, accuracy 0.034205 -> 0.814386
- Frame 000054: loss 3.585968 -> 0.531060, accuracy 0.029895 -> 0.819133
- Frame 000055: loss 3.617249 -> 0.619523, accuracy 0.029471 -> 0.823676
- Frame 000056: loss 3.611888 -> 0.678861, accuracy 0.019950 -> 0.817456
- Frame 000057: loss 3.604891 -> 0.580038, accuracy 0.029295 -> 0.829692
- Frame 000058: loss 3.575706 -> 0.569629, accuracy 0.024549 -> 0.835671
- Frame 000059: loss 3.621376 -> 0.697843, accuracy 0.031016 -> 0.810905
- Frame 000060: loss 3.599029 -> 0.441808, accuracy 0.034809 -> 0.852809
- Frame 000061: loss 3.575832 -> 0.572081, accuracy 0.037736 -> 0.843098
- Frame 000062: loss 3.615385 -> 0.614901, accuracy 0.031080 -> 0.822398
- Frame 000063: loss 3.618101 -> 0.637202, accuracy 0.025335 -> 0.810730
- Frame 000064: loss 3.666569 -> 0.518767, accuracy 0.023337 -> 0.845581
- Frame 000065: loss 3.630090 -> 0.417833, accuracy 0.033814 -> 0.867727
- Frame 000066: loss 3.593514 -> 0.460729, accuracy 0.034948 -> 0.869695
- Frame 000067: loss 3.642549 -> 0.414439, accuracy 0.033367 -> 0.868028
- Frame 000068: loss 3.676542 -> 0.423493, accuracy 0.029559 -> 0.861222
- Frame 000069: loss 3.641117 -> 0.327149, accuracy 0.026368 -> 0.885075
- Frame 000070: loss 3.661105 -> 0.365539, accuracy 0.038062 -> 0.885319
- Frame 000071: loss 3.607404 -> 0.372580, accuracy 0.054645 -> 0.874814
- Frame 000072: loss 3.592580 -> 0.300957, accuracy 0.038633 -> 0.909361
- Frame 000073: loss 3.588392 -> 0.386871, accuracy 0.047666 -> 0.881331
- Frame 000074: loss 3.576283 -> 0.254337, accuracy 0.042807 -> 0.913888
- Frame 000075: loss 3.610590 -> 0.271384, accuracy 0.036482 -> 0.911044
- Frame 000076: loss 3.574437 -> 0.462356, accuracy 0.049058 -> 0.884539
- Frame 000077: loss 3.568812 -> 0.327676, accuracy 0.048573 -> 0.893340
- Frame 000078: loss 3.618064 -> 0.305757, accuracy 0.041082 -> 0.900301
- Frame 000079: loss 3.603535 -> 0.244504, accuracy 0.041144 -> 0.919217
- Frame 000080: loss 3.624865 -> 0.418315, accuracy 0.043478 -> 0.875562
- Frame 000081: loss 3.610883 -> 0.427481, accuracy 0.050882 -> 0.878086
- Frame 000082: loss 3.622424 -> 0.629607, accuracy 0.048283 -> 0.845694
- Frame 000083: loss 3.562377 -> 0.356586, accuracy 0.049799 -> 0.891851
- Frame 000084: loss 3.553814 -> 0.383816, accuracy 0.054162 -> 0.887161
- Frame 000085: loss 3.539079 -> 0.630070, accuracy 0.057414 -> 0.847229
- Frame 000086: loss 3.513477 -> 0.587557, accuracy 0.061184 -> 0.842026
- Frame 000087: loss 3.529171 -> 0.299898, accuracy 0.055894 -> 0.907520
- Frame 000088: loss 3.536800 -> 0.410866, accuracy 0.054244 -> 0.884982
- Frame 000089: loss 3.505135 -> 0.357039, accuracy 0.050959 -> 0.884965
- Frame 000090: loss 3.594116 -> 0.434594, accuracy 0.033486 -> 0.886352
- Frame 000091: loss 3.577402 -> 0.299612, accuracy 0.035569 -> 0.897866
- Frame 000092: loss 3.550248 -> 0.373505, accuracy 0.044875 -> 0.895971
- Frame 000093: loss 3.545964 -> 0.280994, accuracy 0.049139 -> 0.915907
- Frame 000094: loss 3.581631 -> 0.321109, accuracy 0.036831 -> 0.898587
- Frame 000095: loss 3.570730 -> 0.311365, accuracy 0.035240 -> 0.901941
- Frame 000096: loss 3.580738 -> 0.348357, accuracy 0.045571 -> 0.892473
- Frame 000097: loss 3.577782 -> 0.284006, accuracy 0.037359 -> 0.904811
- Frame 000098: loss 3.585880 -> 0.347685, accuracy 0.038815 -> 0.889683
- Frame 000099: loss 3.564440 -> 0.413694, accuracy 0.038304 -> 0.865169

## Train Final Class IoU

- 00 car: IoU 0.883690, acc 0.969257, gt 24786, pred 26424, tp 24024
- 01 bicycle: IoU 0.268293, acc 0.323529, gt 68, pred 36, tp 22
- 04 other-vehicle: IoU 0.271663, acc 0.283619, gt 818, pred 268, tp 232
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.278107, acc 0.315436, gt 149, pred 67, tp 47
- 08 road: IoU 0.889346, acc 0.979143, gt 53650, pred 57948, tp 52531
- 09 parking: IoU 0.466591, acc 0.480947, gt 17031, pred 8715, tp 8191
- 10 sidewalk: IoU 0.491327, acc 0.847075, gt 18290, pred 28736, tp 15493
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.856046, acc 0.865678, gt 53059, pred 46529, tp 45932
- 13 fence: IoU 0.156364, acc 0.161533, gt 1331, pred 259, tp 215
- 14 vegetation: IoU 0.603398, acc 0.765738, gt 23606, pred 24427, tp 18076
- 15 trunk: IoU 0.766277, acc 0.783276, gt 5274, pred 4248, tp 4131
- 16 terrain: IoU 0.136145, acc 0.233471, gt 484, pred 459, tp 113
- 17 pole: IoU 0.424437, acc 0.775330, gt 681, pred 1091, tp 528
- 18 traffic-sign: IoU 0.190184, acc 0.334532, gt 278, pred 304, tp 93

## Holdout Per-Frame Eval

- Frame 000100: loss 3.545971 -> 0.774084, accuracy 0.039487 -> 0.796923
- Frame 000101: loss 3.568427 -> 1.049087, accuracy 0.038740 -> 0.740186
- Frame 000102: loss 3.525544 -> 1.086260, accuracy 0.047889 -> 0.753862
- Frame 000103: loss 3.525811 -> 1.115105, accuracy 0.048287 -> 0.733126
- Frame 000104: loss 3.456679 -> 1.668151, accuracy 0.056613 -> 0.630468
- Frame 000105: loss 3.477961 -> 2.146145, accuracy 0.056477 -> 0.575130
- Frame 000106: loss 3.461704 -> 2.508994, accuracy 0.062628 -> 0.546715
- Frame 000107: loss 3.433818 -> 3.020180, accuracy 0.056426 -> 0.482236
- Frame 000108: loss 3.398436 -> 2.967861, accuracy 0.071058 -> 0.462137
- Frame 000109: loss 3.429154 -> 2.722910, accuracy 0.076316 -> 0.483684
- Frame 000110: loss 3.392885 -> 3.111070, accuracy 0.069343 -> 0.480188
- Frame 000111: loss 3.385629 -> 3.238715, accuracy 0.074737 -> 0.473158
- Frame 000112: loss 3.375004 -> 3.378305, accuracy 0.084619 -> 0.457690
- Frame 000113: loss 3.364390 -> 3.239284, accuracy 0.070825 -> 0.495243
- Frame 000114: loss 3.329727 -> 3.136713, accuracy 0.088156 -> 0.489454
- Frame 000115: loss 3.304393 -> 3.265255, accuracy 0.097855 -> 0.469388
- Frame 000116: loss 3.315361 -> 2.788399, accuracy 0.103269 -> 0.512714
- Frame 000117: loss 3.287613 -> 3.029498, accuracy 0.096509 -> 0.461499
- Frame 000118: loss 3.268774 -> 3.396710, accuracy 0.110200 -> 0.474116
- Frame 000119: loss 3.267013 -> 2.757298, accuracy 0.104462 -> 0.535497
- Frame 000120: loss 3.247160 -> 2.917714, accuracy 0.113043 -> 0.510997
- Frame 000121: loss 3.206630 -> 2.399950, accuracy 0.123281 -> 0.570555
- Frame 000122: loss 3.163153 -> 2.617646, accuracy 0.136595 -> 0.548420
- Frame 000123: loss 3.136351 -> 2.991021, accuracy 0.129328 -> 0.495927
- Frame 000124: loss 3.205891 -> 2.637796, accuracy 0.120143 -> 0.546524
- Frame 000125: loss 3.203026 -> 2.055234, accuracy 0.110771 -> 0.593160
- Frame 000126: loss 3.113045 -> 2.400868, accuracy 0.135080 -> 0.551618
- Frame 000127: loss 3.073357 -> 2.279850, accuracy 0.154442 -> 0.578512
- Frame 000128: loss 3.189603 -> 2.364605, accuracy 0.116279 -> 0.543669
- Frame 000129: loss 3.051187 -> 2.493785, accuracy 0.139366 -> 0.561102
- Frame 000130: loss 3.060477 -> 2.431853, accuracy 0.134127 -> 0.553081
- Frame 000131: loss 3.103316 -> 2.018233, accuracy 0.132841 -> 0.620453
- Frame 000132: loss 3.071408 -> 2.236148, accuracy 0.135477 -> 0.593042
- Frame 000133: loss 3.066986 -> 2.366609, accuracy 0.144063 -> 0.585752
- Frame 000134: loss 3.121371 -> 2.192315, accuracy 0.129252 -> 0.596546
- Frame 000135: loss 3.089835 -> 2.188469, accuracy 0.132845 -> 0.583159
- Frame 000136: loss 3.088510 -> 2.672815, accuracy 0.132512 -> 0.504366
- Frame 000137: loss 3.089638 -> 2.552825, accuracy 0.132248 -> 0.550356
- Frame 000138: loss 3.114318 -> 2.004383, accuracy 0.119084 -> 0.625954
- Frame 000139: loss 3.099557 -> 2.161891, accuracy 0.136295 -> 0.591620
- Frame 000140: loss 3.140177 -> 1.903778, accuracy 0.102823 -> 0.638609
- Frame 000141: loss 3.125912 -> 1.951510, accuracy 0.109383 -> 0.645760
- Frame 000142: loss 3.122747 -> 1.686448, accuracy 0.112733 -> 0.671364
- Frame 000143: loss 3.159700 -> 2.289507, accuracy 0.098641 -> 0.586311
- Frame 000144: loss 3.090282 -> 2.437937, accuracy 0.112813 -> 0.548239
- Frame 000145: loss 3.108227 -> 2.023708, accuracy 0.110827 -> 0.620531
- Frame 000146: loss 3.122638 -> 2.094362, accuracy 0.100152 -> 0.584219
- Frame 000147: loss 3.152658 -> 2.035674, accuracy 0.112920 -> 0.586979
- Frame 000148: loss 3.155624 -> 2.538315, accuracy 0.092378 -> 0.517920
- Frame 000149: loss 3.126810 -> 2.246809, accuracy 0.104114 -> 0.560691

## Holdout Final Class IoU

- 00 car: IoU 0.351650, acc 0.839240, gt 11682, pred 26002, tp 9804
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 7, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 2, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 76, tp 0
- 08 road: IoU 0.678474, acc 0.891857, gt 23173, pred 27955, tp 20667
- 09 parking: IoU 0.067562, acc 0.110403, gt 1413, pred 1052, tp 156
- 10 sidewalk: IoU 0.426835, acc 0.607436, gt 17052, pred 17573, tp 10358
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.346998, acc 0.366969, gt 17184, pred 7295, tp 6306
- 13 fence: IoU 0.002110, acc 0.002169, gt 461, pred 14, tp 1
- 14 vegetation: IoU 0.244005, acc 0.319122, gt 23596, pred 14794, tp 7530
- 15 trunk: IoU 0.049933, acc 0.072835, gt 508, pred 270, tp 37
- 16 terrain: IoU 0.003736, acc 0.005310, gt 1130, pred 482, tp 6
- 17 pole: IoU 0.082718, acc 0.355556, gt 315, pred 1151, tp 112
- 18 traffic-sign: IoU 0.025641, acc 0.078313, gt 166, pred 354, tp 13
