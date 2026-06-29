# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.332923
- Final eval loss, first frame: 0.440165
- Initial eval mean loss: 3.536975
- Final eval mean loss: 0.422395
- Initial eval mean accuracy: 0.044942
- Final eval mean accuracy: 0.877290
- Initial eval mIoU: 0.006217
- Final eval mIoU: 0.471552
- Initial eval overall accuracy: 0.044915
- Final eval overall accuracy: 0.877225
- First train loss: 3.546340
- Final train loss: 0.220376
- Best train loss: 0.118401
- Final train valid accuracy: 0.919816
- CUDA peak memory GB: 0.904
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.238462
- Holdout final mean loss: 1.802448
- Holdout initial mean accuracy: 0.102131
- Holdout final mean accuracy: 0.618866
- Holdout initial mIoU: 0.011180
- Holdout final mIoU: 0.153317
- Holdout initial overall accuracy: 0.102209
- Holdout final overall accuracy: 0.619456

## Train-Frame Eval

- Frame 000000: loss 3.332923 -> 0.440165, accuracy 0.077039 -> 0.861027
- Frame 000001: loss 3.384600 -> 0.362582, accuracy 0.064077 -> 0.879415
- Frame 000002: loss 3.332671 -> 0.402660, accuracy 0.082108 -> 0.882413
- Frame 000003: loss 3.298828 -> 0.368137, accuracy 0.081860 -> 0.885801
- Frame 000004: loss 3.381678 -> 0.486163, accuracy 0.077351 -> 0.867543
- Frame 000005: loss 3.429058 -> 0.414107, accuracy 0.056812 -> 0.890900
- Frame 000006: loss 3.406444 -> 0.379022, accuracy 0.057373 -> 0.884751
- Frame 000007: loss 3.347037 -> 0.467322, accuracy 0.070211 -> 0.869609
- Frame 000008: loss 3.388322 -> 0.541014, accuracy 0.068205 -> 0.869609
- Frame 000009: loss 3.396600 -> 0.642981, accuracy 0.067745 -> 0.859454
- Frame 000010: loss 3.377422 -> 0.580830, accuracy 0.069744 -> 0.868540
- Frame 000011: loss 3.367042 -> 0.613164, accuracy 0.072428 -> 0.852148
- Frame 000012: loss 3.379940 -> 0.618052, accuracy 0.066434 -> 0.851149
- Frame 000013: loss 3.394364 -> 0.614257, accuracy 0.070070 -> 0.862362
- Frame 000014: loss 3.422591 -> 0.773100, accuracy 0.069604 -> 0.841763
- Frame 000015: loss 3.436578 -> 0.714511, accuracy 0.055306 -> 0.845042
- Frame 000016: loss 3.425942 -> 0.569363, accuracy 0.052078 -> 0.835754
- Frame 000017: loss 3.451646 -> 0.662989, accuracy 0.049849 -> 0.846425
- Frame 000018: loss 3.469874 -> 0.521940, accuracy 0.046324 -> 0.876133
- Frame 000019: loss 3.442630 -> 0.393907, accuracy 0.063627 -> 0.870241
- Frame 000020: loss 3.468619 -> 0.630172, accuracy 0.044332 -> 0.849874
- Frame 000021: loss 3.460087 -> 0.602147, accuracy 0.047571 -> 0.854782
- Frame 000022: loss 3.494841 -> 0.587656, accuracy 0.050076 -> 0.859889
- Frame 000023: loss 3.495005 -> 0.679875, accuracy 0.050374 -> 0.832419
- Frame 000024: loss 3.541297 -> 0.626779, accuracy 0.035536 -> 0.849349
- Frame 000025: loss 3.510426 -> 0.564454, accuracy 0.039500 -> 0.843000
- Frame 000026: loss 3.529959 -> 0.769824, accuracy 0.038883 -> 0.822532
- Frame 000027: loss 3.543887 -> 0.523862, accuracy 0.035465 -> 0.840160
- Frame 000028: loss 3.544830 -> 0.618639, accuracy 0.034363 -> 0.847112
- Frame 000029: loss 3.559660 -> 0.505841, accuracy 0.035982 -> 0.863068
- Frame 000030: loss 3.560755 -> 0.609632, accuracy 0.040837 -> 0.848606
- Frame 000031: loss 3.590198 -> 0.460234, accuracy 0.033932 -> 0.866267
- Frame 000032: loss 3.556094 -> 0.526502, accuracy 0.038806 -> 0.864677
- Frame 000033: loss 3.539248 -> 0.350576, accuracy 0.038942 -> 0.891662
- Frame 000034: loss 3.583842 -> 0.346487, accuracy 0.033780 -> 0.889717
- Frame 000035: loss 3.547911 -> 0.366511, accuracy 0.038519 -> 0.882441
- Frame 000036: loss 3.540551 -> 0.450689, accuracy 0.037613 -> 0.876630
- Frame 000037: loss 3.547280 -> 0.442707, accuracy 0.042415 -> 0.869262
- Frame 000038: loss 3.563043 -> 0.417460, accuracy 0.030622 -> 0.888554
- Frame 000039: loss 3.567087 -> 0.377860, accuracy 0.032967 -> 0.893606
- Frame 000040: loss 3.498526 -> 0.439722, accuracy 0.044199 -> 0.875439
- Frame 000041: loss 3.558676 -> 0.444548, accuracy 0.032209 -> 0.872672
- Frame 000042: loss 3.570279 -> 0.441422, accuracy 0.035447 -> 0.864703
- Frame 000043: loss 3.588832 -> 0.531371, accuracy 0.032064 -> 0.847194
- Frame 000044: loss 3.564261 -> 0.570208, accuracy 0.043829 -> 0.831234
- Frame 000045: loss 3.547030 -> 0.419678, accuracy 0.039619 -> 0.860582
- Frame 000046: loss 3.605432 -> 0.475332, accuracy 0.038922 -> 0.846806
- Frame 000047: loss 3.544074 -> 0.513961, accuracy 0.040419 -> 0.856786
- Frame 000048: loss 3.570344 -> 0.328314, accuracy 0.036555 -> 0.896845
- Frame 000049: loss 3.566332 -> 0.286853, accuracy 0.043912 -> 0.904691
- Frame 000050: loss 3.565061 -> 0.343342, accuracy 0.051915 -> 0.882056
- Frame 000051: loss 3.572104 -> 0.373041, accuracy 0.040140 -> 0.880582
- Frame 000052: loss 3.561687 -> 0.376792, accuracy 0.048403 -> 0.875250
- Frame 000053: loss 3.561402 -> 0.520125, accuracy 0.032696 -> 0.832495
- Frame 000054: loss 3.588007 -> 0.362460, accuracy 0.030394 -> 0.880917
- Frame 000055: loss 3.617286 -> 0.354853, accuracy 0.027972 -> 0.885115
- Frame 000056: loss 3.611460 -> 0.459810, accuracy 0.019950 -> 0.853865
- Frame 000057: loss 3.605789 -> 0.358314, accuracy 0.030288 -> 0.885303
- Frame 000058: loss 3.576461 -> 0.367605, accuracy 0.023046 -> 0.874749
- Frame 000059: loss 3.620217 -> 0.538265, accuracy 0.030515 -> 0.846923
- Frame 000060: loss 3.600309 -> 0.362731, accuracy 0.036798 -> 0.874192
- Frame 000061: loss 3.575757 -> 0.532916, accuracy 0.037736 -> 0.830685
- Frame 000062: loss 3.617229 -> 0.380269, accuracy 0.032560 -> 0.874198
- Frame 000063: loss 3.617568 -> 0.458590, accuracy 0.024839 -> 0.844014
- Frame 000064: loss 3.664206 -> 0.410867, accuracy 0.023833 -> 0.865938
- Frame 000065: loss 3.630670 -> 0.335699, accuracy 0.034311 -> 0.889110
- Frame 000066: loss 3.594100 -> 0.349167, accuracy 0.034448 -> 0.885671
- Frame 000067: loss 3.643457 -> 0.279299, accuracy 0.032869 -> 0.902390
- Frame 000068: loss 3.676763 -> 0.350804, accuracy 0.029559 -> 0.888778
- Frame 000069: loss 3.640964 -> 0.261073, accuracy 0.027861 -> 0.905970
- Frame 000070: loss 3.659796 -> 0.290046, accuracy 0.037074 -> 0.896194
- Frame 000071: loss 3.607602 -> 0.299131, accuracy 0.053154 -> 0.901639
- Frame 000072: loss 3.593572 -> 0.296861, accuracy 0.038633 -> 0.894502
- Frame 000073: loss 3.591558 -> 0.269644, accuracy 0.045680 -> 0.898212
- Frame 000074: loss 3.578935 -> 0.244694, accuracy 0.042807 -> 0.909408
- Frame 000075: loss 3.612384 -> 0.398870, accuracy 0.035982 -> 0.874563
- Frame 000076: loss 3.572702 -> 0.294388, accuracy 0.048563 -> 0.902379
- Frame 000077: loss 3.569779 -> 0.342400, accuracy 0.048573 -> 0.886830
- Frame 000078: loss 3.614192 -> 0.308908, accuracy 0.041082 -> 0.890782
- Frame 000079: loss 3.605206 -> 0.261881, accuracy 0.041144 -> 0.913698
- Frame 000080: loss 3.619285 -> 0.297555, accuracy 0.045477 -> 0.902049
- Frame 000081: loss 3.609797 -> 0.292400, accuracy 0.051385 -> 0.911335
- Frame 000082: loss 3.622463 -> 0.393404, accuracy 0.048283 -> 0.878547
- Frame 000083: loss 3.562967 -> 0.318582, accuracy 0.050302 -> 0.893360
- Frame 000084: loss 3.553287 -> 0.317357, accuracy 0.056169 -> 0.904714
- Frame 000085: loss 3.539074 -> 0.501157, accuracy 0.056915 -> 0.867199
- Frame 000086: loss 3.515696 -> 0.341627, accuracy 0.060682 -> 0.889669
- Frame 000087: loss 3.529483 -> 0.288057, accuracy 0.056911 -> 0.909045
- Frame 000088: loss 3.537455 -> 0.330839, accuracy 0.054244 -> 0.906580
- Frame 000089: loss 3.504214 -> 0.255118, accuracy 0.051463 -> 0.912210
- Frame 000090: loss 3.594913 -> 0.352210, accuracy 0.032978 -> 0.908168
- Frame 000091: loss 3.577314 -> 0.217976, accuracy 0.036077 -> 0.920732
- Frame 000092: loss 3.550344 -> 0.275407, accuracy 0.044365 -> 0.899031
- Frame 000093: loss 3.546733 -> 0.228927, accuracy 0.050152 -> 0.918946
- Frame 000094: loss 3.582543 -> 0.230147, accuracy 0.036831 -> 0.921796
- Frame 000095: loss 3.569472 -> 0.255967, accuracy 0.035240 -> 0.916752
- Frame 000096: loss 3.579874 -> 0.328331, accuracy 0.045571 -> 0.904250
- Frame 000097: loss 3.578172 -> 0.251179, accuracy 0.037359 -> 0.912999
- Frame 000098: loss 3.587820 -> 0.252657, accuracy 0.037283 -> 0.916752
- Frame 000099: loss 3.565766 -> 0.252166, accuracy 0.037794 -> 0.917773

## Train Final Class IoU

- 00 car: IoU 0.902447, acc 0.942024, gt 24786, pred 24436, tp 23349
- 01 bicycle: IoU 0.287500, acc 0.338235, gt 68, pred 35, tp 23
- 04 other-vehicle: IoU 0.558753, acc 0.569682, gt 818, pred 482, tp 466
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.370861, acc 0.375839, gt 149, pred 58, tp 56
- 08 road: IoU 0.901793, acc 0.978844, gt 53650, pred 57099, tp 52515
- 09 parking: IoU 0.394922, acc 0.406377, gt 17031, pred 7415, tp 6921
- 10 sidewalk: IoU 0.515746, acc 0.855987, gt 18290, pred 27722, tp 15656
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.922703, acc 0.958631, gt 53059, pred 52930, tp 50864
- 13 fence: IoU 0.179026, acc 0.179564, gt 1331, pred 243, tp 239
- 14 vegetation: IoU 0.731355, acc 0.836228, gt 23606, pred 23125, tp 19740
- 15 trunk: IoU 0.816558, acc 0.858362, gt 5274, pred 4797, tp 4527
- 16 terrain: IoU 0.158322, acc 0.241736, gt 484, pred 372, tp 117
- 17 pole: IoU 0.511059, acc 0.644640, gt 681, pred 617, tp 439
- 18 traffic-sign: IoU 0.293785, acc 0.374101, gt 278, pred 180, tp 104

## Holdout Per-Frame Eval

- Frame 000100: loss 3.545188 -> 0.667512, accuracy 0.040513 -> 0.810769
- Frame 000101: loss 3.567881 -> 0.715870, accuracy 0.038740 -> 0.802686
- Frame 000102: loss 3.524832 -> 0.963128, accuracy 0.048404 -> 0.748713
- Frame 000103: loss 3.527360 -> 1.130229, accuracy 0.048287 -> 0.725857
- Frame 000104: loss 3.457810 -> 1.486694, accuracy 0.058157 -> 0.648996
- Frame 000105: loss 3.480192 -> 1.600728, accuracy 0.055440 -> 0.637306
- Frame 000106: loss 3.464139 -> 2.233924, accuracy 0.062628 -> 0.541581
- Frame 000107: loss 3.433049 -> 2.647653, accuracy 0.055904 -> 0.494253
- Frame 000108: loss 3.397370 -> 2.596322, accuracy 0.071058 -> 0.506224
- Frame 000109: loss 3.430680 -> 2.444960, accuracy 0.073684 -> 0.497368
- Frame 000110: loss 3.393370 -> 2.825703, accuracy 0.072471 -> 0.477581
- Frame 000111: loss 3.385322 -> 2.675244, accuracy 0.074737 -> 0.480526
- Frame 000112: loss 3.371914 -> 2.373440, accuracy 0.084619 -> 0.503459
- Frame 000113: loss 3.366594 -> 2.810196, accuracy 0.073996 -> 0.482030
- Frame 000114: loss 3.329447 -> 2.322684, accuracy 0.085452 -> 0.532720
- Frame 000115: loss 3.309139 -> 2.538459, accuracy 0.098901 -> 0.499738
- Frame 000116: loss 3.313558 -> 1.976011, accuracy 0.103788 -> 0.564608
- Frame 000117: loss 3.289584 -> 2.344736, accuracy 0.095996 -> 0.500000
- Frame 000118: loss 3.268865 -> 2.304789, accuracy 0.115325 -> 0.553050
- Frame 000119: loss 3.267123 -> 2.049777, accuracy 0.101420 -> 0.560852
- Frame 000120: loss 3.245688 -> 2.000183, accuracy 0.117136 -> 0.556010
- Frame 000121: loss 3.209879 -> 1.904617, accuracy 0.121243 -> 0.599593
- Frame 000122: loss 3.163391 -> 1.787891, accuracy 0.137615 -> 0.626402
- Frame 000123: loss 3.138865 -> 2.309412, accuracy 0.124236 -> 0.543279
- Frame 000124: loss 3.204348 -> 1.822236, accuracy 0.119632 -> 0.584356
- Frame 000125: loss 3.198918 -> 1.848199, accuracy 0.111792 -> 0.609495
- Frame 000126: loss 3.112283 -> 1.671971, accuracy 0.131998 -> 0.611197
- Frame 000127: loss 3.074224 -> 1.452202, accuracy 0.153926 -> 0.680269
- Frame 000128: loss 3.188593 -> 1.752608, accuracy 0.117829 -> 0.614987
- Frame 000129: loss 3.048503 -> 1.554381, accuracy 0.135725 -> 0.663547
- Frame 000130: loss 3.062102 -> 1.895909, accuracy 0.133092 -> 0.603832
- Frame 000131: loss 3.106457 -> 1.679736, accuracy 0.131260 -> 0.636795
- Frame 000132: loss 3.074599 -> 1.693076, accuracy 0.133368 -> 0.631523
- Frame 000133: loss 3.065867 -> 1.601631, accuracy 0.144591 -> 0.652770
- Frame 000134: loss 3.122107 -> 1.405682, accuracy 0.129775 -> 0.684458
- Frame 000135: loss 3.088756 -> 1.606195, accuracy 0.135460 -> 0.634414
- Frame 000136: loss 3.086872 -> 1.646369, accuracy 0.129430 -> 0.640986
- Frame 000137: loss 3.090714 -> 1.601764, accuracy 0.131231 -> 0.668362
- Frame 000138: loss 3.115551 -> 1.548822, accuracy 0.118066 -> 0.673282
- Frame 000139: loss 3.096079 -> 1.291235, accuracy 0.136295 -> 0.698637
- Frame 000140: loss 3.141464 -> 1.440697, accuracy 0.103327 -> 0.705645
- Frame 000141: loss 3.125800 -> 1.562559, accuracy 0.106874 -> 0.687406
- Frame 000142: loss 3.122660 -> 1.295203, accuracy 0.109713 -> 0.713639
- Frame 000143: loss 3.158875 -> 1.701925, accuracy 0.097635 -> 0.642677
- Frame 000144: loss 3.090637 -> 1.657529, accuracy 0.114855 -> 0.658499
- Frame 000145: loss 3.110607 -> 1.587538, accuracy 0.111338 -> 0.672114
- Frame 000146: loss 3.120620 -> 1.444423, accuracy 0.099140 -> 0.684876
- Frame 000147: loss 3.152639 -> 1.407634, accuracy 0.113428 -> 0.678535
- Frame 000148: loss 3.154031 -> 1.610896, accuracy 0.092882 -> 0.670369
- Frame 000149: loss 3.128554 -> 1.631804, accuracy 0.104114 -> 0.647029

## Holdout Final Class IoU

- 00 car: IoU 0.429608, acc 0.759887, gt 11682, pred 17858, tp 8877
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 14, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.047619, acc 0.083333, gt 12, pred 10, tp 1
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 65, tp 0
- 08 road: IoU 0.692262, acc 0.902602, gt 23173, pred 27957, tp 20916
- 09 parking: IoU 0.036890, acc 0.046001, gt 1413, pred 414, tp 65
- 10 sidewalk: IoU 0.438442, acc 0.610662, gt 17052, pred 17111, tp 10413
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.472270, acc 0.554527, gt 17184, pred 12522, tp 9529
- 13 fence: IoU 0.008511, acc 0.008677, gt 461, pred 13, tp 4
- 14 vegetation: IoU 0.329023, acc 0.427022, gt 23596, pred 17104, tp 10076
- 15 trunk: IoU 0.034703, acc 0.096457, gt 508, pred 953, tp 49
- 16 terrain: IoU 0.021156, acc 0.036283, gt 1130, pred 849, tp 41
- 17 pole: IoU 0.062316, acc 0.336508, gt 315, pred 1492, tp 106
- 18 traffic-sign: IoU 0.033582, acc 0.162651, gt 166, pred 665, tp 27
