# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.332027
- Final eval loss, first frame: 0.387751
- Initial eval mean loss: 3.537977
- Final eval mean loss: 0.364617
- Initial eval mean accuracy: 0.044608
- Final eval mean accuracy: 0.890133
- Initial eval mIoU: 0.006164
- Final eval mIoU: 0.470235
- Initial eval overall accuracy: 0.044579
- Final eval overall accuracy: 0.890076
- First train loss: 3.539464
- Final train loss: 0.189285
- Best train loss: 0.110316
- Final train valid accuracy: 0.940756
- CUDA peak memory GB: 0.956
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.239231
- Holdout final mean loss: 1.998411
- Holdout initial mean accuracy: 0.102004
- Holdout final mean accuracy: 0.596683
- Holdout initial mIoU: 0.011191
- Holdout final mIoU: 0.147044
- Holdout initial overall accuracy: 0.102085
- Holdout final overall accuracy: 0.596978

## Train-Frame Eval

- Frame 000000: loss 3.332027 -> 0.387751, accuracy 0.076536 -> 0.873112
- Frame 000001: loss 3.386313 -> 0.403638, accuracy 0.063572 -> 0.871847
- Frame 000002: loss 3.338791 -> 0.421860, accuracy 0.083629 -> 0.869742
- Frame 000003: loss 3.301579 -> 0.483006, accuracy 0.080344 -> 0.863062
- Frame 000004: loss 3.383584 -> 0.449571, accuracy 0.072801 -> 0.864510
- Frame 000005: loss 3.424561 -> 0.376256, accuracy 0.055304 -> 0.884867
- Frame 000006: loss 3.403206 -> 0.357866, accuracy 0.056366 -> 0.881228
- Frame 000007: loss 3.349075 -> 0.391874, accuracy 0.067202 -> 0.880642
- Frame 000008: loss 3.390809 -> 0.403530, accuracy 0.064193 -> 0.879639
- Frame 000009: loss 3.400106 -> 0.376706, accuracy 0.066229 -> 0.886754
- Frame 000010: loss 3.376604 -> 0.419009, accuracy 0.068239 -> 0.879579
- Frame 000011: loss 3.372137 -> 0.359267, accuracy 0.072428 -> 0.872627
- Frame 000012: loss 3.375144 -> 0.429891, accuracy 0.066434 -> 0.871129
- Frame 000013: loss 3.398769 -> 0.484727, accuracy 0.068569 -> 0.863363
- Frame 000014: loss 3.422103 -> 0.539658, accuracy 0.063595 -> 0.838758
- Frame 000015: loss 3.433971 -> 0.633042, accuracy 0.051819 -> 0.835077
- Frame 000016: loss 3.427627 -> 0.487925, accuracy 0.049574 -> 0.847271
- Frame 000017: loss 3.453127 -> 0.550657, accuracy 0.051360 -> 0.850957
- Frame 000018: loss 3.471414 -> 0.339315, accuracy 0.043303 -> 0.884189
- Frame 000019: loss 3.446011 -> 0.367510, accuracy 0.066132 -> 0.883267
- Frame 000020: loss 3.470838 -> 0.359207, accuracy 0.044332 -> 0.884131
- Frame 000021: loss 3.466051 -> 0.422606, accuracy 0.046570 -> 0.879820
- Frame 000022: loss 3.497029 -> 0.403460, accuracy 0.051088 -> 0.878098
- Frame 000023: loss 3.495218 -> 0.511342, accuracy 0.049875 -> 0.858354
- Frame 000024: loss 3.536407 -> 0.554116, accuracy 0.037037 -> 0.867868
- Frame 000025: loss 3.507047 -> 0.476507, accuracy 0.040000 -> 0.856500
- Frame 000026: loss 3.528722 -> 0.615317, accuracy 0.038883 -> 0.845464
- Frame 000027: loss 3.545123 -> 0.394964, accuracy 0.033467 -> 0.877123
- Frame 000028: loss 3.543662 -> 0.493282, accuracy 0.034363 -> 0.867530
- Frame 000029: loss 3.560477 -> 0.547505, accuracy 0.038481 -> 0.864068
- Frame 000030: loss 3.562625 -> 0.523316, accuracy 0.041833 -> 0.858566
- Frame 000031: loss 3.594957 -> 0.353269, accuracy 0.032435 -> 0.897206
- Frame 000032: loss 3.550403 -> 0.442157, accuracy 0.035323 -> 0.880597
- Frame 000033: loss 3.539117 -> 0.349088, accuracy 0.041937 -> 0.895157
- Frame 000034: loss 3.581365 -> 0.344506, accuracy 0.033780 -> 0.894188
- Frame 000035: loss 3.549362 -> 0.296763, accuracy 0.042521 -> 0.904452
- Frame 000036: loss 3.543000 -> 0.353446, accuracy 0.037613 -> 0.897192
- Frame 000037: loss 3.548303 -> 0.395913, accuracy 0.043912 -> 0.888224
- Frame 000038: loss 3.568302 -> 0.375953, accuracy 0.028614 -> 0.893574
- Frame 000039: loss 3.569729 -> 0.344286, accuracy 0.033467 -> 0.908591
- Frame 000040: loss 3.503120 -> 0.508345, accuracy 0.042692 -> 0.872928
- Frame 000041: loss 3.556193 -> 0.399644, accuracy 0.034222 -> 0.893306
- Frame 000042: loss 3.572409 -> 0.443221, accuracy 0.036945 -> 0.887169
- Frame 000043: loss 3.588283 -> 0.622056, accuracy 0.032064 -> 0.863727
- Frame 000044: loss 3.570592 -> 0.514120, accuracy 0.043829 -> 0.861965
- Frame 000045: loss 3.552602 -> 0.437754, accuracy 0.040120 -> 0.884152
- Frame 000046: loss 3.608188 -> 0.522175, accuracy 0.039421 -> 0.853792
- Frame 000047: loss 3.549470 -> 0.383800, accuracy 0.041417 -> 0.892715
- Frame 000048: loss 3.573367 -> 0.304790, accuracy 0.035053 -> 0.901853
- Frame 000049: loss 3.564651 -> 0.374786, accuracy 0.041916 -> 0.877245
- Frame 000050: loss 3.566043 -> 0.372703, accuracy 0.050403 -> 0.877016
- Frame 000051: loss 3.568999 -> 0.410225, accuracy 0.039639 -> 0.880080
- Frame 000052: loss 3.564296 -> 0.314592, accuracy 0.049401 -> 0.897705
- Frame 000053: loss 3.562702 -> 0.427891, accuracy 0.032696 -> 0.865694
- Frame 000054: loss 3.587220 -> 0.368662, accuracy 0.030394 -> 0.876433
- Frame 000055: loss 3.616525 -> 0.432884, accuracy 0.028971 -> 0.881119
- Frame 000056: loss 3.611973 -> 0.453361, accuracy 0.018454 -> 0.861845
- Frame 000057: loss 3.608315 -> 0.380073, accuracy 0.028798 -> 0.888779
- Frame 000058: loss 3.576823 -> 0.416812, accuracy 0.025551 -> 0.875751
- Frame 000059: loss 3.617944 -> 0.510082, accuracy 0.030015 -> 0.851926
- Frame 000060: loss 3.602817 -> 0.401894, accuracy 0.034311 -> 0.873695
- Frame 000061: loss 3.579056 -> 0.458161, accuracy 0.036743 -> 0.855015
- Frame 000062: loss 3.621930 -> 0.417170, accuracy 0.032560 -> 0.859398
- Frame 000063: loss 3.620298 -> 0.403716, accuracy 0.026329 -> 0.860904
- Frame 000064: loss 3.665282 -> 0.379050, accuracy 0.023337 -> 0.888282
- Frame 000065: loss 3.633514 -> 0.269718, accuracy 0.031328 -> 0.912979
- Frame 000066: loss 3.595900 -> 0.265511, accuracy 0.032451 -> 0.907639
- Frame 000067: loss 3.645514 -> 0.285821, accuracy 0.031873 -> 0.905877
- Frame 000068: loss 3.679405 -> 0.284685, accuracy 0.032064 -> 0.913828
- Frame 000069: loss 3.641639 -> 0.235708, accuracy 0.024876 -> 0.923881
- Frame 000070: loss 3.663918 -> 0.200559, accuracy 0.037568 -> 0.928324
- Frame 000071: loss 3.608201 -> 0.288931, accuracy 0.051664 -> 0.908097
- Frame 000072: loss 3.590439 -> 0.239206, accuracy 0.038138 -> 0.919267
- Frame 000073: loss 3.593024 -> 0.252747, accuracy 0.046177 -> 0.904667
- Frame 000074: loss 3.578224 -> 0.229129, accuracy 0.042310 -> 0.922847
- Frame 000075: loss 3.607370 -> 0.229652, accuracy 0.036482 -> 0.929035
- Frame 000076: loss 3.573526 -> 0.255879, accuracy 0.045094 -> 0.911794
- Frame 000077: loss 3.571145 -> 0.200230, accuracy 0.050075 -> 0.928893
- Frame 000078: loss 3.615521 -> 0.240361, accuracy 0.042084 -> 0.916834
- Frame 000079: loss 3.605364 -> 0.218522, accuracy 0.041144 -> 0.922730
- Frame 000080: loss 3.623892 -> 0.224809, accuracy 0.045477 -> 0.924038
- Frame 000081: loss 3.613873 -> 0.234118, accuracy 0.053401 -> 0.926448
- Frame 000082: loss 3.621875 -> 0.274920, accuracy 0.048780 -> 0.907417
- Frame 000083: loss 3.564084 -> 0.288605, accuracy 0.048793 -> 0.909960
- Frame 000084: loss 3.553228 -> 0.297085, accuracy 0.054162 -> 0.907723
- Frame 000085: loss 3.535918 -> 0.363765, accuracy 0.059411 -> 0.890664
- Frame 000086: loss 3.513553 -> 0.306447, accuracy 0.062688 -> 0.898195
- Frame 000087: loss 3.524625 -> 0.223573, accuracy 0.056402 -> 0.921748
- Frame 000088: loss 3.538915 -> 0.227129, accuracy 0.053240 -> 0.927172
- Frame 000089: loss 3.508247 -> 0.226794, accuracy 0.050454 -> 0.926842
- Frame 000090: loss 3.597897 -> 0.315063, accuracy 0.033993 -> 0.905124
- Frame 000091: loss 3.578874 -> 0.195754, accuracy 0.035061 -> 0.932419
- Frame 000092: loss 3.551787 -> 0.234194, accuracy 0.045385 -> 0.921469
- Frame 000093: loss 3.545995 -> 0.228018, accuracy 0.049645 -> 0.922492
- Frame 000094: loss 3.581286 -> 0.215507, accuracy 0.036831 -> 0.930373
- Frame 000095: loss 3.569107 -> 0.176694, accuracy 0.037794 -> 0.936159
- Frame 000096: loss 3.582600 -> 0.198172, accuracy 0.043523 -> 0.935996
- Frame 000097: loss 3.582821 -> 0.216512, accuracy 0.036336 -> 0.917605
- Frame 000098: loss 3.585888 -> 0.196689, accuracy 0.037794 -> 0.938713
- Frame 000099: loss 3.566726 -> 0.243133, accuracy 0.039837 -> 0.915220

## Train Final Class IoU

- 00 car: IoU 0.899699, acc 0.965182, gt 24786, pred 25727, tp 23923
- 01 bicycle: IoU 0.225000, acc 0.264706, gt 68, pred 30, tp 18
- 04 other-vehicle: IoU 0.552781, acc 0.595355, gt 818, pred 550, tp 487
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.236025, acc 0.255034, gt 149, pred 50, tp 38
- 08 road: IoU 0.876883, acc 0.988630, gt 53650, pred 59877, tp 53040
- 09 parking: IoU 0.607737, acc 0.650343, gt 17031, pred 12270, tp 11076
- 10 sidewalk: IoU 0.550871, acc 0.764352, gt 18290, pred 21068, tp 13980
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.923388, acc 0.947700, gt 53059, pred 51681, tp 50284
- 13 fence: IoU 0.105539, acc 0.105935, gt 1331, pred 146, tp 141
- 14 vegetation: IoU 0.732617, acc 0.820851, gt 23606, pred 22220, tp 19377
- 15 trunk: IoU 0.836645, acc 0.862344, gt 5274, pred 4710, tp 4548
- 16 terrain: IoU 0.137652, acc 0.210744, gt 484, pred 359, tp 102
- 17 pole: IoU 0.517937, acc 0.678414, gt 681, pred 673, tp 462
- 18 traffic-sign: IoU 0.320988, acc 0.374101, gt 278, pred 150, tp 104

## Holdout Per-Frame Eval

- Frame 000100: loss 3.544796 -> 0.668619, accuracy 0.038974 -> 0.818974
- Frame 000101: loss 3.570029 -> 0.694913, accuracy 0.036157 -> 0.808884
- Frame 000102: loss 3.525676 -> 0.823938, accuracy 0.049949 -> 0.789907
- Frame 000103: loss 3.528010 -> 0.898327, accuracy 0.049325 -> 0.759605
- Frame 000104: loss 3.457258 -> 1.278875, accuracy 0.058672 -> 0.715903
- Frame 000105: loss 3.478106 -> 1.404613, accuracy 0.060104 -> 0.660622
- Frame 000106: loss 3.462479 -> 1.997935, accuracy 0.060062 -> 0.601643
- Frame 000107: loss 3.436971 -> 2.178212, accuracy 0.056949 -> 0.528213
- Frame 000108: loss 3.396640 -> 2.699769, accuracy 0.072095 -> 0.515560
- Frame 000109: loss 3.429037 -> 2.359533, accuracy 0.072105 -> 0.510000
- Frame 000110: loss 3.396393 -> 2.781000, accuracy 0.071950 -> 0.503650
- Frame 000111: loss 3.389596 -> 2.940776, accuracy 0.075263 -> 0.490526
- Frame 000112: loss 3.371788 -> 2.718860, accuracy 0.083555 -> 0.517296
- Frame 000113: loss 3.365879 -> 2.576720, accuracy 0.072410 -> 0.509514
- Frame 000114: loss 3.329622 -> 2.272230, accuracy 0.087074 -> 0.521363
- Frame 000115: loss 3.311871 -> 2.552676, accuracy 0.096808 -> 0.503401
- Frame 000116: loss 3.310642 -> 2.175389, accuracy 0.104826 -> 0.564089
- Frame 000117: loss 3.286686 -> 2.285499, accuracy 0.095996 -> 0.505134
- Frame 000118: loss 3.270576 -> 2.592938, accuracy 0.111738 -> 0.541261
- Frame 000119: loss 3.263457 -> 2.407813, accuracy 0.106491 -> 0.532961
- Frame 000120: loss 3.252786 -> 2.261831, accuracy 0.112532 -> 0.522762
- Frame 000121: loss 3.205392 -> 2.029076, accuracy 0.123281 -> 0.571574
- Frame 000122: loss 3.168121 -> 2.286126, accuracy 0.140673 -> 0.564730
- Frame 000123: loss 3.136978 -> 2.413726, accuracy 0.125255 -> 0.530041
- Frame 000124: loss 3.206265 -> 2.409414, accuracy 0.117076 -> 0.571575
- Frame 000125: loss 3.198940 -> 1.924419, accuracy 0.113323 -> 0.597243
- Frame 000126: loss 3.116830 -> 2.056059, accuracy 0.133025 -> 0.581407
- Frame 000127: loss 3.079110 -> 1.782459, accuracy 0.153926 -> 0.645145
- Frame 000128: loss 3.192465 -> 2.174466, accuracy 0.120930 -> 0.552972
- Frame 000129: loss 3.050437 -> 1.742660, accuracy 0.136245 -> 0.629745
- Frame 000130: loss 3.072219 -> 2.043765, accuracy 0.127913 -> 0.595029
- Frame 000131: loss 3.107840 -> 1.795992, accuracy 0.130206 -> 0.619926
- Frame 000132: loss 3.073228 -> 1.733734, accuracy 0.136004 -> 0.617290
- Frame 000133: loss 3.066438 -> 1.864486, accuracy 0.139314 -> 0.621108
- Frame 000134: loss 3.120564 -> 2.191826, accuracy 0.129252 -> 0.550497
- Frame 000135: loss 3.091557 -> 1.551309, accuracy 0.134937 -> 0.650105
- Frame 000136: loss 3.092382 -> 2.240303, accuracy 0.131998 -> 0.557268
- Frame 000137: loss 3.085261 -> 2.325536, accuracy 0.133774 -> 0.544252
- Frame 000138: loss 3.116943 -> 1.830947, accuracy 0.119084 -> 0.617303
- Frame 000139: loss 3.095181 -> 1.853067, accuracy 0.133771 -> 0.617870
- Frame 000140: loss 3.145198 -> 2.006310, accuracy 0.102823 -> 0.646673
- Frame 000141: loss 3.128229 -> 1.875405, accuracy 0.107376 -> 0.650778
- Frame 000142: loss 3.122032 -> 1.546222, accuracy 0.109210 -> 0.678913
- Frame 000143: loss 3.159986 -> 2.089882, accuracy 0.099144 -> 0.595370
- Frame 000144: loss 3.088160 -> 2.102579, accuracy 0.112813 -> 0.566616
- Frame 000145: loss 3.109660 -> 1.852005, accuracy 0.110317 -> 0.626660
- Frame 000146: loss 3.121732 -> 1.719738, accuracy 0.098128 -> 0.624684
- Frame 000147: loss 3.150754 -> 1.744589, accuracy 0.111902 -> 0.640895
- Frame 000148: loss 3.153098 -> 2.071189, accuracy 0.090863 -> 0.593640
- Frame 000149: loss 3.128250 -> 2.092816, accuracy 0.104622 -> 0.553581

## Holdout Final Class IoU

- 00 car: IoU 0.390229, acc 0.818439, gt 11682, pred 22380, tp 9561
- 01 bicycle: IoU 0.017699, acc 0.019231, gt 104, pred 11, tp 2
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 20, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 25, tp 0
- 08 road: IoU 0.681140, acc 0.943037, gt 23173, pred 30763, tp 21853
- 09 parking: IoU 0.066271, acc 0.134466, gt 1413, pred 1644, tp 190
- 10 sidewalk: IoU 0.369757, acc 0.483404, gt 17052, pred 13484, tp 8243
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.478198, acc 0.555226, gt 17184, pred 12309, tp 9541
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 6, tp 0
- 14 vegetation: IoU 0.282301, acc 0.354001, gt 23596, pred 14346, tp 8353
- 15 trunk: IoU 0.044776, acc 0.076772, gt 508, pred 402, tp 39
- 16 terrain: IoU 0.012876, acc 0.018584, gt 1130, pred 522, tp 21
- 17 pole: IoU 0.084715, acc 0.292063, gt 315, pred 863, tp 92
- 18 traffic-sign: IoU 0.071795, acc 0.168675, gt 166, pred 252, tp 28
