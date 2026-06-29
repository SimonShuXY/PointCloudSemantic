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
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.289805
- Final eval loss, first frame: 1.006349
- Initial eval mean loss: 4.492340
- Final eval mean loss: 1.047726
- Initial eval mean accuracy: 0.044846
- Final eval mean accuracy: 0.845754
- Initial eval mIoU: 0.006180
- Final eval mIoU: 0.430700
- Initial eval overall accuracy: 0.044820
- Final eval overall accuracy: 0.845663
- First train loss: 4.505824
- Final train loss: 0.691977
- Best train loss: 0.286409
- Final train valid accuracy: 0.895812
- CUDA peak memory GB: 0.961
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 4.186509
- Holdout final mean loss: 2.749454
- Holdout initial mean accuracy: 0.101977
- Holdout final mean accuracy: 0.606198
- Holdout initial mIoU: 0.011169
- Holdout final mIoU: 0.145103
- Holdout initial overall accuracy: 0.102054
- Holdout final overall accuracy: 0.606718

## Diagnostic fused Eval

- Train final mean loss: 1.054596
- Train final mean accuracy: 0.844871
- Train final mIoU: 0.426145
- Train final overall accuracy: 0.844780

- Holdout final mean loss: 2.741911
- Holdout final mean accuracy: 0.605232
- Holdout final mIoU: 0.144861
- Holdout final overall accuracy: 0.605739

## Train-Frame Eval

- Frame 000000: loss 4.289805 -> 1.006349, accuracy 0.077039 -> 0.845921
- Frame 000001: loss 4.348124 -> 0.972749, accuracy 0.060545 -> 0.848638
- Frame 000002: loss 4.294207 -> 1.031251, accuracy 0.081602 -> 0.845920
- Frame 000003: loss 4.256695 -> 0.967552, accuracy 0.078828 -> 0.865589
- Frame 000004: loss 4.334603 -> 1.082919, accuracy 0.077856 -> 0.842265
- Frame 000005: loss 4.374938 -> 0.889040, accuracy 0.055304 -> 0.865259
- Frame 000006: loss 4.359840 -> 1.001800, accuracy 0.057876 -> 0.857574
- Frame 000007: loss 4.309050 -> 1.119857, accuracy 0.073220 -> 0.836008
- Frame 000008: loss 4.355690 -> 1.285651, accuracy 0.064193 -> 0.829488
- Frame 000009: loss 4.360531 -> 1.342737, accuracy 0.064712 -> 0.814965
- Frame 000010: loss 4.332570 -> 1.187273, accuracy 0.073758 -> 0.833417
- Frame 000011: loss 4.330702 -> 1.264567, accuracy 0.070929 -> 0.827672
- Frame 000012: loss 4.337707 -> 1.205121, accuracy 0.064935 -> 0.830669
- Frame 000013: loss 4.359585 -> 1.277448, accuracy 0.068569 -> 0.814314
- Frame 000014: loss 4.381386 -> 1.469701, accuracy 0.066099 -> 0.807211
- Frame 000015: loss 4.404507 -> 1.558805, accuracy 0.055306 -> 0.796712
- Frame 000016: loss 4.391712 -> 1.324501, accuracy 0.052579 -> 0.825739
- Frame 000017: loss 4.407859 -> 1.487276, accuracy 0.050352 -> 0.799597
- Frame 000018: loss 4.428618 -> 1.283434, accuracy 0.044814 -> 0.821249
- Frame 000019: loss 4.402597 -> 1.147874, accuracy 0.064629 -> 0.824148
- Frame 000020: loss 4.431789 -> 1.211858, accuracy 0.047355 -> 0.833249
- Frame 000021: loss 4.432362 -> 1.315067, accuracy 0.045068 -> 0.817226
- Frame 000022: loss 4.455882 -> 1.285237, accuracy 0.052099 -> 0.815377
- Frame 000023: loss 4.449142 -> 1.494370, accuracy 0.049875 -> 0.774564
- Frame 000024: loss 4.497835 -> 1.491707, accuracy 0.032032 -> 0.799800
- Frame 000025: loss 4.467153 -> 1.389337, accuracy 0.039500 -> 0.805000
- Frame 000026: loss 4.488836 -> 1.443554, accuracy 0.038883 -> 0.790130
- Frame 000027: loss 4.502529 -> 1.208995, accuracy 0.034965 -> 0.790210
- Frame 000028: loss 4.500430 -> 1.344010, accuracy 0.031375 -> 0.801793
- Frame 000029: loss 4.519269 -> 1.385312, accuracy 0.037481 -> 0.808596
- Frame 000030: loss 4.521501 -> 1.524554, accuracy 0.040339 -> 0.771414
- Frame 000031: loss 4.542971 -> 1.185900, accuracy 0.030938 -> 0.818363
- Frame 000032: loss 4.512513 -> 1.225733, accuracy 0.033333 -> 0.829353
- Frame 000033: loss 4.501283 -> 1.154303, accuracy 0.043435 -> 0.830255
- Frame 000034: loss 4.530350 -> 1.065522, accuracy 0.032787 -> 0.847491
- Frame 000035: loss 4.504712 -> 0.924225, accuracy 0.042021 -> 0.865933
- Frame 000036: loss 4.493053 -> 1.144598, accuracy 0.037613 -> 0.835005
- Frame 000037: loss 4.502813 -> 1.190048, accuracy 0.041916 -> 0.805389
- Frame 000038: loss 4.512566 -> 1.192140, accuracy 0.031124 -> 0.831827
- Frame 000039: loss 4.513015 -> 0.957155, accuracy 0.033966 -> 0.848152
- Frame 000040: loss 4.447541 -> 0.990562, accuracy 0.039679 -> 0.836263
- Frame 000041: loss 4.514977 -> 1.224587, accuracy 0.030196 -> 0.828384
- Frame 000042: loss 4.529269 -> 1.059414, accuracy 0.036945 -> 0.842736
- Frame 000043: loss 4.539284 -> 1.276853, accuracy 0.032565 -> 0.805110
- Frame 000044: loss 4.520099 -> 1.269229, accuracy 0.042317 -> 0.791940
- Frame 000045: loss 4.497058 -> 1.185562, accuracy 0.041625 -> 0.816449
- Frame 000046: loss 4.554283 -> 1.169919, accuracy 0.038922 -> 0.806387
- Frame 000047: loss 4.500263 -> 1.148992, accuracy 0.038922 -> 0.834331
- Frame 000048: loss 4.526610 -> 0.958971, accuracy 0.036054 -> 0.849274
- Frame 000049: loss 4.518613 -> 0.990025, accuracy 0.043912 -> 0.849301
- Frame 000050: loss 4.516599 -> 1.032491, accuracy 0.052419 -> 0.839214
- Frame 000051: loss 4.525437 -> 1.036214, accuracy 0.039639 -> 0.857501
- Frame 000052: loss 4.513021 -> 1.233312, accuracy 0.051896 -> 0.812874
- Frame 000053: loss 4.511411 -> 1.353709, accuracy 0.037726 -> 0.816901
- Frame 000054: loss 4.538886 -> 1.050611, accuracy 0.028401 -> 0.829098
- Frame 000055: loss 4.570799 -> 1.143874, accuracy 0.028971 -> 0.830170
- Frame 000056: loss 4.567722 -> 1.119257, accuracy 0.019451 -> 0.816459
- Frame 000057: loss 4.561201 -> 0.990725, accuracy 0.034260 -> 0.846077
- Frame 000058: loss 4.533166 -> 1.026501, accuracy 0.025551 -> 0.848697
- Frame 000059: loss 4.563667 -> 1.241765, accuracy 0.032016 -> 0.816408
- Frame 000060: loss 4.552032 -> 1.053463, accuracy 0.035803 -> 0.831924
- Frame 000061: loss 4.516449 -> 1.302943, accuracy 0.037736 -> 0.817776
- Frame 000062: loss 4.574547 -> 1.335567, accuracy 0.033054 -> 0.798224
- Frame 000063: loss 4.579351 -> 1.195961, accuracy 0.023348 -> 0.801292
- Frame 000064: loss 4.625187 -> 1.119460, accuracy 0.025819 -> 0.840616
- Frame 000065: loss 4.583478 -> 0.850213, accuracy 0.031825 -> 0.856290
- Frame 000066: loss 4.553668 -> 0.958714, accuracy 0.034948 -> 0.869695
- Frame 000067: loss 4.596909 -> 0.779511, accuracy 0.034363 -> 0.901892
- Frame 000068: loss 4.626334 -> 1.034739, accuracy 0.031062 -> 0.839679
- Frame 000069: loss 4.592255 -> 0.842027, accuracy 0.028358 -> 0.878607
- Frame 000070: loss 4.612700 -> 0.672112, accuracy 0.039051 -> 0.901631
- Frame 000071: loss 4.554973 -> 0.712353, accuracy 0.054645 -> 0.889717
- Frame 000072: loss 4.535453 -> 0.756219, accuracy 0.039128 -> 0.892026
- Frame 000073: loss 4.548565 -> 0.747590, accuracy 0.046673 -> 0.879345
- Frame 000074: loss 4.527542 -> 0.733431, accuracy 0.040319 -> 0.889995
- Frame 000075: loss 4.569540 -> 0.801184, accuracy 0.035982 -> 0.875562
- Frame 000076: loss 4.527307 -> 0.894800, accuracy 0.048067 -> 0.866204
- Frame 000077: loss 4.523336 -> 0.713590, accuracy 0.045568 -> 0.877817
- Frame 000078: loss 4.571421 -> 0.724377, accuracy 0.042084 -> 0.883267
- Frame 000079: loss 4.560109 -> 0.812096, accuracy 0.042649 -> 0.880080
- Frame 000080: loss 4.578773 -> 0.758774, accuracy 0.044478 -> 0.885058
- Frame 000081: loss 4.565015 -> 0.744061, accuracy 0.054912 -> 0.910831
- Frame 000082: loss 4.580556 -> 0.894493, accuracy 0.050274 -> 0.841215
- Frame 000083: loss 4.517680 -> 0.788575, accuracy 0.050302 -> 0.885815
- Frame 000084: loss 4.503659 -> 0.704450, accuracy 0.055165 -> 0.894183
- Frame 000085: loss 4.491639 -> 0.955631, accuracy 0.057913 -> 0.846231
- Frame 000086: loss 4.460702 -> 0.773679, accuracy 0.063691 -> 0.881645
- Frame 000087: loss 4.481573 -> 0.629956, accuracy 0.057419 -> 0.914634
- Frame 000088: loss 4.490334 -> 0.707622, accuracy 0.054244 -> 0.894023
- Frame 000089: loss 4.455044 -> 0.616655, accuracy 0.049445 -> 0.902119
- Frame 000090: loss 4.554533 -> 0.799576, accuracy 0.032471 -> 0.901573
- Frame 000091: loss 4.530490 -> 0.698568, accuracy 0.033028 -> 0.924289
- Frame 000092: loss 4.507718 -> 0.597563, accuracy 0.043855 -> 0.918409
- Frame 000093: loss 4.505341 -> 0.746787, accuracy 0.049139 -> 0.905775
- Frame 000094: loss 4.539444 -> 0.608835, accuracy 0.035318 -> 0.899596
- Frame 000095: loss 4.526837 -> 0.697660, accuracy 0.037794 -> 0.906027
- Frame 000096: loss 4.535904 -> 0.771903, accuracy 0.044547 -> 0.891449
- Frame 000097: loss 4.534838 -> 0.749738, accuracy 0.035824 -> 0.885363
- Frame 000098: loss 4.537958 -> 0.917663, accuracy 0.037794 -> 0.851379
- Frame 000099: loss 4.514145 -> 1.027977, accuracy 0.035751 -> 0.839122

## Train Final Class IoU

- 00 car: IoU 0.895966, acc 0.947188, gt 24786, pred 24894, tp 23477
- 01 bicycle: IoU 0.294574, acc 0.558824, gt 68, pred 99, tp 38
- 04 other-vehicle: IoU 0.163241, acc 0.165037, gt 818, pred 144, tp 135
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.434211, acc 0.442953, gt 149, pred 69, tp 66
- 08 road: IoU 0.875533, acc 0.976272, gt 53650, pred 58550, tp 52377
- 09 parking: IoU 0.328525, acc 0.336974, gt 17031, pred 6177, tp 5739
- 10 sidewalk: IoU 0.496707, acc 0.841170, gt 18290, pred 28069, tp 15385
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.846195, acc 0.867374, gt 53059, pred 47350, tp 46022
- 13 fence: IoU 0.056759, acc 0.057100, gt 1331, pred 84, tp 76
- 14 vegetation: IoU 0.649162, acc 0.872405, gt 23606, pred 28712, tp 20594
- 15 trunk: IoU 0.764630, acc 0.782897, gt 5274, pred 4255, tp 4129
- 16 terrain: IoU 0.127630, acc 0.188017, gt 484, pred 320, tp 91
- 17 pole: IoU 0.580902, acc 0.643172, gt 681, pred 511, tp 438
- 18 traffic-sign: IoU 0.377171, acc 0.546763, gt 278, pred 277, tp 152

## Holdout Per-Frame Eval

- Frame 000100: loss 4.501843 -> 1.428716, accuracy 0.041026 -> 0.772820
- Frame 000101: loss 4.528025 -> 1.824009, accuracy 0.037190 -> 0.723140
- Frame 000102: loss 4.463128 -> 1.948489, accuracy 0.048919 -> 0.703399
- Frame 000103: loss 4.486926 -> 2.051250, accuracy 0.046729 -> 0.676532
- Frame 000104: loss 4.415072 -> 2.601710, accuracy 0.056099 -> 0.611426
- Frame 000105: loss 4.434125 -> 2.906750, accuracy 0.059067 -> 0.554404
- Frame 000106: loss 4.418055 -> 3.405775, accuracy 0.062628 -> 0.535934
- Frame 000107: loss 4.387998 -> 3.626726, accuracy 0.055381 -> 0.514107
- Frame 000108: loss 4.352147 -> 3.694982, accuracy 0.067946 -> 0.508299
- Frame 000109: loss 4.386124 -> 3.534466, accuracy 0.073158 -> 0.498947
- Frame 000110: loss 4.354245 -> 4.077741, accuracy 0.072471 -> 0.459854
- Frame 000111: loss 4.341015 -> 3.917156, accuracy 0.076316 -> 0.499474
- Frame 000112: loss 4.329408 -> 3.872016, accuracy 0.083555 -> 0.494944
- Frame 000113: loss 4.320632 -> 3.693155, accuracy 0.071882 -> 0.497357
- Frame 000114: loss 4.284357 -> 3.509356, accuracy 0.091942 -> 0.503515
- Frame 000115: loss 4.260357 -> 3.473940, accuracy 0.102041 -> 0.514390
- Frame 000116: loss 4.272770 -> 3.131789, accuracy 0.100156 -> 0.555267
- Frame 000117: loss 4.233438 -> 3.199511, accuracy 0.099589 -> 0.543635
- Frame 000118: loss 4.224600 -> 3.442889, accuracy 0.117888 -> 0.537673
- Frame 000119: loss 4.218496 -> 3.282048, accuracy 0.105984 -> 0.549189
- Frame 000120: loss 4.202554 -> 3.041169, accuracy 0.114578 -> 0.539642
- Frame 000121: loss 4.149655 -> 2.662588, accuracy 0.121243 -> 0.611819
- Frame 000122: loss 4.109711 -> 2.506954, accuracy 0.137615 -> 0.622834
- Frame 000123: loss 4.076339 -> 3.048318, accuracy 0.124236 -> 0.523931
- Frame 000124: loss 4.145699 -> 2.512820, accuracy 0.122188 -> 0.614008
- Frame 000125: loss 4.139657 -> 2.288811, accuracy 0.112302 -> 0.648290
- Frame 000126: loss 4.049004 -> 2.722945, accuracy 0.134052 -> 0.565999
- Frame 000127: loss 4.029918 -> 2.301729, accuracy 0.152893 -> 0.657025
- Frame 000128: loss 4.141544 -> 2.499544, accuracy 0.123514 -> 0.628941
- Frame 000129: loss 3.983853 -> 2.569461, accuracy 0.135205 -> 0.628185
- Frame 000130: loss 4.004301 -> 2.281147, accuracy 0.133610 -> 0.646297
- Frame 000131: loss 4.050711 -> 2.412919, accuracy 0.129151 -> 0.653664
- Frame 000132: loss 4.012705 -> 2.583088, accuracy 0.137059 -> 0.628361
- Frame 000133: loss 4.016615 -> 2.168373, accuracy 0.137731 -> 0.677045
- Frame 000134: loss 4.066134 -> 2.439274, accuracy 0.126635 -> 0.621664
- Frame 000135: loss 4.043663 -> 2.292523, accuracy 0.130230 -> 0.659519
- Frame 000136: loss 4.027618 -> 2.589518, accuracy 0.135080 -> 0.609142
- Frame 000137: loss 4.025321 -> 2.499136, accuracy 0.133266 -> 0.634791
- Frame 000138: loss 4.062459 -> 2.209329, accuracy 0.117048 -> 0.691094
- Frame 000139: loss 4.037533 -> 2.260259, accuracy 0.139828 -> 0.688541
- Frame 000140: loss 4.088743 -> 2.453517, accuracy 0.101310 -> 0.688508
- Frame 000141: loss 4.079394 -> 2.259888, accuracy 0.106372 -> 0.705971
- Frame 000142: loss 4.067485 -> 2.102473, accuracy 0.113236 -> 0.701560
- Frame 000143: loss 4.095843 -> 2.556018, accuracy 0.094112 -> 0.645697
- Frame 000144: loss 4.038278 -> 2.640289, accuracy 0.106687 -> 0.620725
- Frame 000145: loss 4.045665 -> 2.542877, accuracy 0.109806 -> 0.673647
- Frame 000146: loss 4.067164 -> 2.508942, accuracy 0.096105 -> 0.635306
- Frame 000147: loss 4.089802 -> 2.454180, accuracy 0.111394 -> 0.660732
- Frame 000148: loss 4.096088 -> 2.724105, accuracy 0.094397 -> 0.584553
- Frame 000149: loss 4.069231 -> 2.718045, accuracy 0.098019 -> 0.588116

## Holdout Final Class IoU

- 00 car: IoU 0.396800, acc 0.770502, gt 11682, pred 20003, tp 9001
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 22, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 25, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 10, tp 0
- 08 road: IoU 0.675631, acc 0.909636, gt 23173, pred 29105, tp 21079
- 09 parking: IoU 0.070572, acc 0.108280, gt 1413, pred 908, tp 153
- 10 sidewalk: IoU 0.392036, acc 0.535773, gt 17052, pred 15388, tp 9136
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.379961, acc 0.405610, gt 17184, pred 8130, tp 6970
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 0, tp 0
- 14 vegetation: IoU 0.376139, acc 0.524835, gt 23596, pred 21712, tp 12384
- 15 trunk: IoU 0.062327, acc 0.088583, gt 508, pred 259, tp 45
- 16 terrain: IoU 0.009901, acc 0.014159, gt 1130, pred 502, tp 16
- 17 pole: IoU 0.080963, acc 0.234921, gt 315, pred 673, tp 74
- 18 traffic-sign: IoU 0.022422, acc 0.060241, gt 166, pred 290, tp 10
