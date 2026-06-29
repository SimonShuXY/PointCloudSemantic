# SemanticKITTI lidar-only Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.333655
- Final eval loss, first frame: 0.408639
- Initial eval mean loss: 3.538900
- Final eval mean loss: 0.359060
- Initial eval mean accuracy: 0.044821
- Final eval mean accuracy: 0.888287
- Initial eval mIoU: 0.006154
- Final eval mIoU: 0.485505
- Initial eval overall accuracy: 0.044795
- Final eval overall accuracy: 0.888247
- First train loss: 3.566836
- Final train loss: 0.206171
- Best train loss: 0.117464
- Final train valid accuracy: 0.925945
- CUDA peak memory GB: 0.902
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.239762
- Holdout final mean loss: 1.789750
- Holdout initial mean accuracy: 0.102206
- Holdout final mean accuracy: 0.614489
- Holdout initial mIoU: 0.011212
- Holdout final mIoU: 0.152981
- Holdout initial overall accuracy: 0.102291
- Holdout final overall accuracy: 0.615148

## Train-Frame Eval

- Frame 000000: loss 3.333655 -> 0.408639, accuracy 0.078550 -> 0.879658
- Frame 000001: loss 3.387509 -> 0.380581, accuracy 0.060545 -> 0.882442
- Frame 000002: loss 3.337654 -> 0.413376, accuracy 0.084136 -> 0.875824
- Frame 000003: loss 3.299856 -> 0.367561, accuracy 0.079333 -> 0.889843
- Frame 000004: loss 3.383796 -> 0.404912, accuracy 0.073306 -> 0.878160
- Frame 000005: loss 3.425136 -> 0.405784, accuracy 0.055304 -> 0.878834
- Frame 000006: loss 3.403912 -> 0.326658, accuracy 0.063915 -> 0.892300
- Frame 000007: loss 3.349069 -> 0.360598, accuracy 0.071214 -> 0.882146
- Frame 000008: loss 3.390967 -> 0.479074, accuracy 0.064694 -> 0.861585
- Frame 000009: loss 3.406552 -> 0.439533, accuracy 0.064712 -> 0.854904
- Frame 000010: loss 3.373401 -> 0.389078, accuracy 0.070246 -> 0.882087
- Frame 000011: loss 3.373614 -> 0.405665, accuracy 0.070929 -> 0.881618
- Frame 000012: loss 3.381200 -> 0.458557, accuracy 0.067433 -> 0.850649
- Frame 000013: loss 3.399279 -> 0.516041, accuracy 0.068569 -> 0.862362
- Frame 000014: loss 3.423144 -> 0.697552, accuracy 0.067101 -> 0.826239
- Frame 000015: loss 3.437812 -> 0.767321, accuracy 0.055306 -> 0.811659
- Frame 000016: loss 3.427381 -> 0.455015, accuracy 0.052579 -> 0.867802
- Frame 000017: loss 3.449925 -> 0.623730, accuracy 0.051863 -> 0.855488
- Frame 000018: loss 3.471463 -> 0.370947, accuracy 0.043303 -> 0.886203
- Frame 000019: loss 3.446081 -> 0.363072, accuracy 0.062625 -> 0.889279
- Frame 000020: loss 3.476287 -> 0.415360, accuracy 0.043829 -> 0.863476
- Frame 000021: loss 3.476585 -> 0.388089, accuracy 0.046069 -> 0.872809
- Frame 000022: loss 3.497001 -> 0.425292, accuracy 0.050582 -> 0.870005
- Frame 000023: loss 3.496261 -> 0.638021, accuracy 0.052868 -> 0.840898
- Frame 000024: loss 3.536651 -> 0.609359, accuracy 0.035536 -> 0.841341
- Frame 000025: loss 3.503986 -> 0.681132, accuracy 0.041500 -> 0.832000
- Frame 000026: loss 3.533286 -> 0.727256, accuracy 0.038385 -> 0.827517
- Frame 000027: loss 3.536727 -> 0.426558, accuracy 0.036464 -> 0.853147
- Frame 000028: loss 3.544312 -> 0.601379, accuracy 0.030378 -> 0.843626
- Frame 000029: loss 3.558941 -> 0.495571, accuracy 0.036482 -> 0.861569
- Frame 000030: loss 3.567410 -> 0.550072, accuracy 0.039841 -> 0.850598
- Frame 000031: loss 3.586969 -> 0.480268, accuracy 0.033433 -> 0.864271
- Frame 000032: loss 3.558896 -> 0.436040, accuracy 0.035323 -> 0.877114
- Frame 000033: loss 3.551244 -> 0.465098, accuracy 0.037943 -> 0.866201
- Frame 000034: loss 3.588292 -> 0.376216, accuracy 0.034277 -> 0.881272
- Frame 000035: loss 3.551481 -> 0.307017, accuracy 0.040020 -> 0.901451
- Frame 000036: loss 3.548650 -> 0.494717, accuracy 0.035607 -> 0.860582
- Frame 000037: loss 3.548593 -> 0.347657, accuracy 0.043413 -> 0.881737
- Frame 000038: loss 3.561051 -> 0.306663, accuracy 0.029618 -> 0.898092
- Frame 000039: loss 3.572644 -> 0.384036, accuracy 0.034466 -> 0.889610
- Frame 000040: loss 3.498714 -> 0.384153, accuracy 0.039679 -> 0.879458
- Frame 000041: loss 3.557656 -> 0.339057, accuracy 0.030700 -> 0.890790
- Frame 000042: loss 3.569762 -> 0.282893, accuracy 0.034948 -> 0.900649
- Frame 000043: loss 3.587224 -> 0.400517, accuracy 0.032565 -> 0.869739
- Frame 000044: loss 3.571584 -> 0.396275, accuracy 0.044332 -> 0.869018
- Frame 000045: loss 3.547711 -> 0.274959, accuracy 0.043631 -> 0.894183
- Frame 000046: loss 3.608939 -> 0.314563, accuracy 0.039920 -> 0.894212
- Frame 000047: loss 3.557914 -> 0.299342, accuracy 0.040419 -> 0.899701
- Frame 000048: loss 3.572575 -> 0.285022, accuracy 0.036054 -> 0.900851
- Frame 000049: loss 3.566550 -> 0.270306, accuracy 0.042914 -> 0.907186
- Frame 000050: loss 3.562936 -> 0.248090, accuracy 0.050907 -> 0.919355
- Frame 000051: loss 3.570659 -> 0.246569, accuracy 0.042649 -> 0.912694
- Frame 000052: loss 3.562802 -> 0.297935, accuracy 0.051896 -> 0.894212
- Frame 000053: loss 3.565484 -> 0.288683, accuracy 0.035714 -> 0.899396
- Frame 000054: loss 3.582163 -> 0.253470, accuracy 0.029397 -> 0.909317
- Frame 000055: loss 3.624958 -> 0.294395, accuracy 0.027972 -> 0.896603
- Frame 000056: loss 3.611532 -> 0.401379, accuracy 0.019950 -> 0.867332
- Frame 000057: loss 3.608931 -> 0.303260, accuracy 0.032274 -> 0.901688
- Frame 000058: loss 3.578527 -> 0.355447, accuracy 0.025050 -> 0.882265
- Frame 000059: loss 3.614070 -> 0.489342, accuracy 0.033017 -> 0.851426
- Frame 000060: loss 3.598408 -> 0.366518, accuracy 0.036300 -> 0.871706
- Frame 000061: loss 3.577439 -> 0.458963, accuracy 0.034260 -> 0.855511
- Frame 000062: loss 3.624532 -> 0.442179, accuracy 0.032067 -> 0.869758
- Frame 000063: loss 3.620487 -> 0.356809, accuracy 0.023845 -> 0.874814
- Frame 000064: loss 3.668837 -> 0.374517, accuracy 0.024330 -> 0.882324
- Frame 000065: loss 3.634582 -> 0.268127, accuracy 0.032819 -> 0.903531
- Frame 000066: loss 3.591435 -> 0.330117, accuracy 0.034948 -> 0.886171
- Frame 000067: loss 3.650142 -> 0.278539, accuracy 0.032371 -> 0.902888
- Frame 000068: loss 3.680573 -> 0.326589, accuracy 0.029559 -> 0.890281
- Frame 000069: loss 3.640319 -> 0.221109, accuracy 0.028856 -> 0.922886
- Frame 000070: loss 3.664642 -> 0.254666, accuracy 0.038557 -> 0.921898
- Frame 000071: loss 3.610391 -> 0.262094, accuracy 0.052658 -> 0.913562
- Frame 000072: loss 3.587741 -> 0.228864, accuracy 0.038138 -> 0.925706
- Frame 000073: loss 3.596308 -> 0.232801, accuracy 0.046177 -> 0.912115
- Frame 000074: loss 3.588017 -> 0.236167, accuracy 0.043803 -> 0.921852
- Frame 000075: loss 3.621267 -> 0.236962, accuracy 0.036982 -> 0.912544
- Frame 000076: loss 3.571873 -> 0.213487, accuracy 0.047572 -> 0.929633
- Frame 000077: loss 3.573039 -> 0.207302, accuracy 0.047571 -> 0.930896
- Frame 000078: loss 3.617546 -> 0.297151, accuracy 0.040080 -> 0.901303
- Frame 000079: loss 3.607575 -> 0.182693, accuracy 0.041646 -> 0.937280
- Frame 000080: loss 3.623054 -> 0.203477, accuracy 0.042979 -> 0.929035
- Frame 000081: loss 3.608365 -> 0.253133, accuracy 0.051889 -> 0.914358
- Frame 000082: loss 3.622047 -> 0.302185, accuracy 0.050772 -> 0.898955
- Frame 000083: loss 3.566693 -> 0.246126, accuracy 0.050805 -> 0.920523
- Frame 000084: loss 3.554142 -> 0.252999, accuracy 0.054664 -> 0.914744
- Frame 000085: loss 3.539428 -> 0.349893, accuracy 0.059411 -> 0.887169
- Frame 000086: loss 3.513233 -> 0.308555, accuracy 0.060682 -> 0.899198
- Frame 000087: loss 3.523536 -> 0.220428, accuracy 0.056911 -> 0.931402
- Frame 000088: loss 3.540213 -> 0.234298, accuracy 0.055249 -> 0.913611
- Frame 000089: loss 3.504334 -> 0.242962, accuracy 0.052472 -> 0.917760
- Frame 000090: loss 3.594601 -> 0.277362, accuracy 0.035515 -> 0.906646
- Frame 000091: loss 3.575316 -> 0.215877, accuracy 0.034045 -> 0.928862
- Frame 000092: loss 3.558602 -> 0.232803, accuracy 0.044365 -> 0.922489
- Frame 000093: loss 3.550056 -> 0.216997, accuracy 0.047112 -> 0.916413
- Frame 000094: loss 3.587160 -> 0.197629, accuracy 0.036831 -> 0.931887
- Frame 000095: loss 3.572118 -> 0.248920, accuracy 0.035751 -> 0.914709
- Frame 000096: loss 3.582540 -> 0.266502, accuracy 0.044547 -> 0.906298
- Frame 000097: loss 3.584218 -> 0.279469, accuracy 0.035312 -> 0.900205
- Frame 000098: loss 3.585008 -> 0.242468, accuracy 0.039837 -> 0.914198
- Frame 000099: loss 3.562820 -> 0.323160, accuracy 0.034729 -> 0.887130

## Train Final Class IoU

- 00 car: IoU 0.909418, acc 0.968087, gt 24786, pred 25594, tp 23995
- 01 bicycle: IoU 0.318182, acc 0.411765, gt 68, pred 48, tp 28
- 04 other-vehicle: IoU 0.413253, acc 0.419315, gt 818, pred 355, tp 343
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.373563, acc 0.436242, gt 149, pred 90, tp 65
- 08 road: IoU 0.910548, acc 0.976365, gt 53650, pred 56260, tp 52382
- 09 parking: IoU 0.571160, acc 0.606071, gt 17031, pred 11363, tp 10322
- 10 sidewalk: IoU 0.574584, acc 0.890432, gt 18290, pred 26340, tp 16286
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.905551, acc 0.925178, gt 53059, pred 50239, tp 49089
- 13 fence: IoU 0.234443, acc 0.243426, gt 1331, pred 375, tp 324
- 14 vegetation: IoU 0.705568, acc 0.821359, gt 23606, pred 23263, tp 19389
- 15 trunk: IoU 0.798257, acc 0.816268, gt 5274, pred 4424, tp 4305
- 16 terrain: IoU 0.123684, acc 0.194215, gt 484, pred 370, tp 94
- 17 pole: IoU 0.573830, acc 0.701909, gt 681, pred 630, tp 478
- 18 traffic-sign: IoU 0.356037, acc 0.413669, gt 278, pred 160, tp 115

## Holdout Per-Frame Eval

- Frame 000100: loss 3.542444 -> 0.654597, accuracy 0.039487 -> 0.822564
- Frame 000101: loss 3.573861 -> 0.856136, accuracy 0.036157 -> 0.770145
- Frame 000102: loss 3.522551 -> 0.973165, accuracy 0.048919 -> 0.742534
- Frame 000103: loss 3.531600 -> 0.923575, accuracy 0.047248 -> 0.741952
- Frame 000104: loss 3.462417 -> 1.376498, accuracy 0.055069 -> 0.667010
- Frame 000105: loss 3.483951 -> 1.757463, accuracy 0.059585 -> 0.601554
- Frame 000106: loss 3.462843 -> 2.389959, accuracy 0.060575 -> 0.537474
- Frame 000107: loss 3.431740 -> 2.549729, accuracy 0.056949 -> 0.493730
- Frame 000108: loss 3.393529 -> 2.844673, accuracy 0.073133 -> 0.477178
- Frame 000109: loss 3.435507 -> 2.537648, accuracy 0.072105 -> 0.504211
- Frame 000110: loss 3.398078 -> 2.811418, accuracy 0.072471 -> 0.478102
- Frame 000111: loss 3.386570 -> 2.735363, accuracy 0.077895 -> 0.496316
- Frame 000112: loss 3.379181 -> 2.818837, accuracy 0.080894 -> 0.445450
- Frame 000113: loss 3.362488 -> 2.821545, accuracy 0.071882 -> 0.468816
- Frame 000114: loss 3.333135 -> 2.364561, accuracy 0.086533 -> 0.516495
- Frame 000115: loss 3.304972 -> 2.516407, accuracy 0.105704 -> 0.469388
- Frame 000116: loss 3.321365 -> 2.038315, accuracy 0.101194 -> 0.571873
- Frame 000117: loss 3.282322 -> 2.119288, accuracy 0.095996 -> 0.548255
- Frame 000118: loss 3.271286 -> 2.160709, accuracy 0.113788 -> 0.584316
- Frame 000119: loss 3.263963 -> 1.860100, accuracy 0.102434 -> 0.595842
- Frame 000120: loss 3.249763 -> 1.967383, accuracy 0.116113 -> 0.585166
- Frame 000121: loss 3.218086 -> 1.939256, accuracy 0.122771 -> 0.585838
- Frame 000122: loss 3.167550 -> 1.726994, accuracy 0.134557 -> 0.602446
- Frame 000123: loss 3.138807 -> 2.298776, accuracy 0.124745 -> 0.528004
- Frame 000124: loss 3.203809 -> 1.873267, accuracy 0.121166 -> 0.600716
- Frame 000125: loss 3.202408 -> 1.754865, accuracy 0.115365 -> 0.596223
- Frame 000126: loss 3.108544 -> 1.741362, accuracy 0.137648 -> 0.600411
- Frame 000127: loss 3.084707 -> 1.602911, accuracy 0.147727 -> 0.631715
- Frame 000128: loss 3.191061 -> 2.010117, accuracy 0.124031 -> 0.561757
- Frame 000129: loss 3.044332 -> 1.928065, accuracy 0.136765 -> 0.595424
- Frame 000130: loss 3.068155 -> 1.821419, accuracy 0.134127 -> 0.611600
- Frame 000131: loss 3.106156 -> 1.448652, accuracy 0.131787 -> 0.672114
- Frame 000132: loss 3.072982 -> 1.850394, accuracy 0.136004 -> 0.601476
- Frame 000133: loss 3.071897 -> 1.564903, accuracy 0.136148 -> 0.664380
- Frame 000134: loss 3.118170 -> 1.440146, accuracy 0.130822 -> 0.672423
- Frame 000135: loss 3.095755 -> 1.380048, accuracy 0.129184 -> 0.665272
- Frame 000136: loss 3.088790 -> 1.498106, accuracy 0.137134 -> 0.656908
- Frame 000137: loss 3.082179 -> 1.631938, accuracy 0.129196 -> 0.649034
- Frame 000138: loss 3.111847 -> 1.331213, accuracy 0.117557 -> 0.682443
- Frame 000139: loss 3.097926 -> 1.221049, accuracy 0.139828 -> 0.679455
- Frame 000140: loss 3.148153 -> 1.373963, accuracy 0.104335 -> 0.702117
- Frame 000141: loss 3.125177 -> 1.300605, accuracy 0.112895 -> 0.710487
- Frame 000142: loss 3.127875 -> 1.242870, accuracy 0.116256 -> 0.727227
- Frame 000143: loss 3.161271 -> 1.578837, accuracy 0.099144 -> 0.669854
- Frame 000144: loss 3.092087 -> 1.593978, accuracy 0.109239 -> 0.638591
- Frame 000145: loss 3.109742 -> 1.402815, accuracy 0.107763 -> 0.699183
- Frame 000146: loss 3.120406 -> 1.458514, accuracy 0.094082 -> 0.663632
- Frame 000147: loss 3.154061 -> 1.404807, accuracy 0.114446 -> 0.668362
- Frame 000148: loss 3.152932 -> 1.610601, accuracy 0.090358 -> 0.613327
- Frame 000149: loss 3.129686 -> 1.379645, accuracy 0.101067 -> 0.655663

## Holdout Final Class IoU

- 00 car: IoU 0.412518, acc 0.778035, gt 11682, pred 19440, tp 9089
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 5, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 5, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 85, tp 0
- 08 road: IoU 0.657106, acc 0.865102, gt 23173, pred 27382, tp 20047
- 09 parking: IoU 0.073233, acc 0.121019, gt 1413, pred 1093, tp 171
- 10 sidewalk: IoU 0.406750, acc 0.605735, gt 17052, pred 18671, tp 10329
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.494084, acc 0.522463, gt 17184, pred 9965, tp 8978
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 46, tp 0
- 14 vegetation: IoU 0.353103, acc 0.461985, gt 23596, pred 18177, tp 10901
- 15 trunk: IoU 0.065749, acc 0.084646, gt 508, pred 189, tp 43
- 16 terrain: IoU 0.005851, acc 0.009735, gt 1130, pred 761, tp 11
- 17 pole: IoU 0.085409, acc 0.304762, gt 315, pred 905, tp 96
- 18 traffic-sign: IoU 0.046875, acc 0.126506, gt 166, pred 303, tp 21
