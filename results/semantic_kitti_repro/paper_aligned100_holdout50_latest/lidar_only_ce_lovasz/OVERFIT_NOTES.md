# SemanticKITTI lidar-only Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Train route: lidar-only
- Primary eval route: lidar-only
- Diagnostic eval route: none
- Depth mode: lidar-inpaint, first-frame p5/p95: [6.56657600402832, 36.63307189941406]
- Loss mode: ce-lovasz, Lovasz weight: 1.0
- IPFP depth percentile range: 5.0-95.0
- IPFP discard probability: 0.000
- Initial eval loss, first frame: 4.287948
- Final eval loss, first frame: 1.074649
- Initial eval mean loss: 4.492167
- Final eval mean loss: 1.036112
- Initial eval mean accuracy: 0.044817
- Final eval mean accuracy: 0.839450
- Initial eval mIoU: 0.006147
- Final eval mIoU: 0.498318
- Initial eval overall accuracy: 0.044790
- Final eval overall accuracy: 0.839352
- First train loss: 4.529469
- Final train loss: 0.603613
- Best train loss: 0.365021
- Final train valid accuracy: 0.923391
- CUDA peak memory GB: 0.902
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 4.186532
- Holdout final mean loss: 2.717767
- Holdout initial mean accuracy: 0.102278
- Holdout final mean accuracy: 0.603015
- Holdout initial mIoU: 0.011199
- Holdout final mIoU: 0.147070
- Holdout initial overall accuracy: 0.102363
- Holdout final overall accuracy: 0.603626

## Train-Frame Eval

- Frame 000000: loss 4.287948 -> 1.074649, accuracy 0.077543 -> 0.832326
- Frame 000001: loss 4.348660 -> 0.955993, accuracy 0.062059 -> 0.849647
- Frame 000002: loss 4.292764 -> 1.126013, accuracy 0.085656 -> 0.834769
- Frame 000003: loss 4.256472 -> 0.933825, accuracy 0.078828 -> 0.862052
- Frame 000004: loss 4.338703 -> 1.028153, accuracy 0.074823 -> 0.841254
- Frame 000005: loss 4.378551 -> 1.002560, accuracy 0.054299 -> 0.855204
- Frame 000006: loss 4.359931 -> 1.065120, accuracy 0.063412 -> 0.839960
- Frame 000007: loss 4.306545 -> 1.005552, accuracy 0.071715 -> 0.851555
- Frame 000008: loss 4.350931 -> 1.058596, accuracy 0.063190 -> 0.837513
- Frame 000009: loss 4.361938 -> 1.332384, accuracy 0.063701 -> 0.808392
- Frame 000010: loss 4.331000 -> 1.242509, accuracy 0.070246 -> 0.828901
- Frame 000011: loss 4.333470 -> 1.298429, accuracy 0.070430 -> 0.825175
- Frame 000012: loss 4.334184 -> 1.270842, accuracy 0.066933 -> 0.817183
- Frame 000013: loss 4.359519 -> 1.382905, accuracy 0.068569 -> 0.796296
- Frame 000014: loss 4.382591 -> 1.572252, accuracy 0.066600 -> 0.783676
- Frame 000015: loss 4.398003 -> 1.721675, accuracy 0.055306 -> 0.762332
- Frame 000016: loss 4.388776 -> 1.199350, accuracy 0.051077 -> 0.828242
- Frame 000017: loss 4.404327 -> 1.479144, accuracy 0.052870 -> 0.804632
- Frame 000018: loss 4.426990 -> 1.098714, accuracy 0.044310 -> 0.843404
- Frame 000019: loss 4.403453 -> 1.001045, accuracy 0.063627 -> 0.847695
- Frame 000020: loss 4.432528 -> 1.275506, accuracy 0.045844 -> 0.818640
- Frame 000021: loss 4.433502 -> 1.216588, accuracy 0.047071 -> 0.810716
- Frame 000022: loss 4.452173 -> 1.253803, accuracy 0.052099 -> 0.813354
- Frame 000023: loss 4.452392 -> 1.603772, accuracy 0.050374 -> 0.745636
- Frame 000024: loss 4.493004 -> 1.375227, accuracy 0.036537 -> 0.794795
- Frame 000025: loss 4.461839 -> 1.417500, accuracy 0.040000 -> 0.783000
- Frame 000026: loss 4.488909 -> 1.264005, accuracy 0.037388 -> 0.809571
- Frame 000027: loss 4.494819 -> 1.129874, accuracy 0.035964 -> 0.795205
- Frame 000028: loss 4.501324 -> 1.441339, accuracy 0.031375 -> 0.789343
- Frame 000029: loss 4.513028 -> 1.228919, accuracy 0.036982 -> 0.815592
- Frame 000030: loss 4.522477 -> 1.269505, accuracy 0.039841 -> 0.811255
- Frame 000031: loss 4.540292 -> 1.124967, accuracy 0.033433 -> 0.842814
- Frame 000032: loss 4.511308 -> 1.246964, accuracy 0.036318 -> 0.815920
- Frame 000033: loss 4.504380 -> 1.198811, accuracy 0.039441 -> 0.818772
- Frame 000034: loss 4.544264 -> 1.079523, accuracy 0.032290 -> 0.836562
- Frame 000035: loss 4.500105 -> 1.109186, accuracy 0.042021 -> 0.846923
- Frame 000036: loss 4.491864 -> 1.365924, accuracy 0.038114 -> 0.794885
- Frame 000037: loss 4.500148 -> 1.075667, accuracy 0.042914 -> 0.822854
- Frame 000038: loss 4.509117 -> 1.105771, accuracy 0.031124 -> 0.838353
- Frame 000039: loss 4.519932 -> 0.957776, accuracy 0.034466 -> 0.854146
- Frame 000040: loss 4.450198 -> 0.889207, accuracy 0.042190 -> 0.851331
- Frame 000041: loss 4.510059 -> 1.134566, accuracy 0.029693 -> 0.829391
- Frame 000042: loss 4.520315 -> 0.970294, accuracy 0.036945 -> 0.830255
- Frame 000043: loss 4.540018 -> 1.148198, accuracy 0.033066 -> 0.805110
- Frame 000044: loss 4.521173 -> 1.277497, accuracy 0.042821 -> 0.778338
- Frame 000045: loss 4.497715 -> 0.879233, accuracy 0.044634 -> 0.843029
- Frame 000046: loss 4.562046 -> 1.208800, accuracy 0.039421 -> 0.810379
- Frame 000047: loss 4.504608 -> 1.020862, accuracy 0.038922 -> 0.830339
- Frame 000048: loss 4.525826 -> 0.678911, accuracy 0.034051 -> 0.879820
- Frame 000049: loss 4.515891 -> 1.006144, accuracy 0.042914 -> 0.842814
- Frame 000050: loss 4.509365 -> 1.054426, accuracy 0.051915 -> 0.822077
- Frame 000051: loss 4.526515 -> 1.001767, accuracy 0.043653 -> 0.846964
- Frame 000052: loss 4.513366 -> 1.018489, accuracy 0.053393 -> 0.832335
- Frame 000053: loss 4.513515 -> 1.293243, accuracy 0.036217 -> 0.809859
- Frame 000054: loss 4.533084 -> 0.861146, accuracy 0.026906 -> 0.856004
- Frame 000055: loss 4.577257 -> 1.136263, accuracy 0.026973 -> 0.811688
- Frame 000056: loss 4.563448 -> 1.069042, accuracy 0.019451 -> 0.814963
- Frame 000057: loss 4.562811 -> 0.863055, accuracy 0.032274 -> 0.853525
- Frame 000058: loss 4.527745 -> 1.028518, accuracy 0.025551 -> 0.831663
- Frame 000059: loss 4.564799 -> 1.120063, accuracy 0.031516 -> 0.806403
- Frame 000060: loss 4.552075 -> 1.053430, accuracy 0.036300 -> 0.827449
- Frame 000061: loss 4.521333 -> 1.380166, accuracy 0.035253 -> 0.789970
- Frame 000062: loss 4.575757 -> 1.133844, accuracy 0.031574 -> 0.814997
- Frame 000063: loss 4.574172 -> 1.348565, accuracy 0.023845 -> 0.782414
- Frame 000064: loss 4.622681 -> 0.922879, accuracy 0.023833 -> 0.840616
- Frame 000065: loss 4.584698 -> 0.796854, accuracy 0.032819 -> 0.866733
- Frame 000066: loss 4.552192 -> 1.021766, accuracy 0.034948 -> 0.835247
- Frame 000067: loss 4.601921 -> 0.863056, accuracy 0.031375 -> 0.866534
- Frame 000068: loss 4.626919 -> 0.940019, accuracy 0.030060 -> 0.846192
- Frame 000069: loss 4.592798 -> 0.715665, accuracy 0.028358 -> 0.886567
- Frame 000070: loss 4.611268 -> 0.849253, accuracy 0.037568 -> 0.859614
- Frame 000071: loss 4.560492 -> 0.887998, accuracy 0.053154 -> 0.849975
- Frame 000072: loss 4.539721 -> 0.742225, accuracy 0.037642 -> 0.872709
- Frame 000073: loss 4.544927 -> 0.792618, accuracy 0.046673 -> 0.859484
- Frame 000074: loss 4.533774 -> 0.672756, accuracy 0.042807 -> 0.890493
- Frame 000075: loss 4.568625 -> 1.028461, accuracy 0.035982 -> 0.825087
- Frame 000076: loss 4.526240 -> 0.962469, accuracy 0.046581 -> 0.852329
- Frame 000077: loss 4.519134 -> 0.847111, accuracy 0.047071 -> 0.849775
- Frame 000078: loss 4.567451 -> 0.634115, accuracy 0.041583 -> 0.883768
- Frame 000079: loss 4.558544 -> 0.741466, accuracy 0.041144 -> 0.880582
- Frame 000080: loss 4.581075 -> 0.809182, accuracy 0.042979 -> 0.862069
- Frame 000081: loss 4.560848 -> 0.924889, accuracy 0.052393 -> 0.869521
- Frame 000082: loss 4.575143 -> 0.944425, accuracy 0.051269 -> 0.853161
- Frame 000083: loss 4.522096 -> 0.828717, accuracy 0.050805 -> 0.881791
- Frame 000084: loss 4.503757 -> 0.844372, accuracy 0.053159 -> 0.875125
- Frame 000085: loss 4.492575 -> 1.111767, accuracy 0.059411 -> 0.830754
- Frame 000086: loss 4.466233 -> 0.900956, accuracy 0.061685 -> 0.863089
- Frame 000087: loss 4.479025 -> 0.700748, accuracy 0.057419 -> 0.895833
- Frame 000088: loss 4.498022 -> 0.746186, accuracy 0.054746 -> 0.883476
- Frame 000089: loss 4.455637 -> 0.703187, accuracy 0.051968 -> 0.872351
- Frame 000090: loss 4.549095 -> 0.972494, accuracy 0.034500 -> 0.853374
- Frame 000091: loss 4.529553 -> 0.675469, accuracy 0.033537 -> 0.916158
- Frame 000092: loss 4.512609 -> 0.617562, accuracy 0.044875 -> 0.919939
- Frame 000093: loss 4.505177 -> 0.808722, accuracy 0.048126 -> 0.882979
- Frame 000094: loss 4.541544 -> 0.557188, accuracy 0.036831 -> 0.911705
- Frame 000095: loss 4.528016 -> 0.693768, accuracy 0.034729 -> 0.891726
- Frame 000096: loss 4.538797 -> 0.782313, accuracy 0.044547 -> 0.890937
- Frame 000097: loss 4.537742 -> 0.789076, accuracy 0.035312 -> 0.873081
- Frame 000098: loss 4.540551 -> 0.806437, accuracy 0.038815 -> 0.863126
- Frame 000099: loss 4.512603 -> 0.808969, accuracy 0.034729 -> 0.879469

## Train Final Class IoU

- 00 car: IoU 0.900487, acc 0.931695, gt 24786, pred 23952, tp 23093
- 01 bicycle: IoU 0.512195, acc 0.617647, gt 68, pred 56, tp 42
- 04 other-vehicle: IoU 0.490588, acc 0.509780, gt 818, pred 449, tp 417
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.627329, acc 0.677852, gt 149, pred 113, tp 101
- 08 road: IoU 0.871213, acc 0.986785, gt 53650, pred 60058, tp 52941
- 09 parking: IoU 0.364030, acc 0.380189, gt 17031, pred 7231, tp 6475
- 10 sidewalk: IoU 0.480092, acc 0.832641, gt 18290, pred 28660, tp 15229
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.842353, acc 0.857800, gt 53059, pred 46487, tp 45514
- 13 fence: IoU 0.195175, acc 0.200601, gt 1331, pred 304, tp 267
- 14 vegetation: IoU 0.577287, acc 0.776963, gt 23606, pred 26506, tp 18341
- 15 trunk: IoU 0.784361, acc 0.800721, gt 5274, pred 4333, tp 4223
- 16 terrain: IoU 0.132653, acc 0.241736, gt 484, pred 515, tp 117
- 17 pole: IoU 0.704762, acc 0.760646, gt 681, pred 572, tp 518
- 18 traffic-sign: IoU 0.490566, acc 0.654676, gt 278, pred 275, tp 182

## Holdout Per-Frame Eval

- Frame 000100: loss 4.501221 -> 1.330073, accuracy 0.040000 -> 0.790769
- Frame 000101: loss 4.529736 -> 1.896456, accuracy 0.036157 -> 0.707645
- Frame 000102: loss 4.460260 -> 1.924462, accuracy 0.048919 -> 0.712152
- Frame 000103: loss 4.487556 -> 1.810106, accuracy 0.047248 -> 0.719107
- Frame 000104: loss 4.417054 -> 2.534536, accuracy 0.056099 -> 0.617602
- Frame 000105: loss 4.441101 -> 2.686334, accuracy 0.059585 -> 0.609326
- Frame 000106: loss 4.418763 -> 3.162608, accuracy 0.060575 -> 0.569815
- Frame 000107: loss 4.382964 -> 3.582450, accuracy 0.060084 -> 0.498433
- Frame 000108: loss 4.350698 -> 3.775633, accuracy 0.073133 -> 0.481847
- Frame 000109: loss 4.389034 -> 3.357531, accuracy 0.071579 -> 0.512632
- Frame 000110: loss 4.350954 -> 3.616238, accuracy 0.074035 -> 0.483837
- Frame 000111: loss 4.342547 -> 3.638413, accuracy 0.076316 -> 0.491053
- Frame 000112: loss 4.329615 -> 3.688833, accuracy 0.083023 -> 0.469399
- Frame 000113: loss 4.318064 -> 3.474079, accuracy 0.073467 -> 0.471459
- Frame 000114: loss 4.283979 -> 3.285234, accuracy 0.085452 -> 0.500270
- Frame 000115: loss 4.261123 -> 3.353846, accuracy 0.103611 -> 0.476714
- Frame 000116: loss 4.272125 -> 2.847164, accuracy 0.101713 -> 0.570836
- Frame 000117: loss 4.239962 -> 2.964042, accuracy 0.096509 -> 0.548255
- Frame 000118: loss 4.228266 -> 3.189794, accuracy 0.111738 -> 0.551000
- Frame 000119: loss 4.216657 -> 2.936633, accuracy 0.103448 -> 0.565923
- Frame 000120: loss 4.199338 -> 2.983157, accuracy 0.116113 -> 0.538619
- Frame 000121: loss 4.150846 -> 2.763496, accuracy 0.122262 -> 0.581253
- Frame 000122: loss 4.105722 -> 2.692635, accuracy 0.137615 -> 0.592762
- Frame 000123: loss 4.072413 -> 2.699640, accuracy 0.122200 -> 0.580448
- Frame 000124: loss 4.151690 -> 2.538847, accuracy 0.120654 -> 0.605317
- Frame 000125: loss 4.139478 -> 2.385233, accuracy 0.113834 -> 0.649821
- Frame 000126: loss 4.049554 -> 2.491497, accuracy 0.137134 -> 0.630714
- Frame 000127: loss 4.025711 -> 2.402043, accuracy 0.149793 -> 0.619835
- Frame 000128: loss 4.137639 -> 2.698279, accuracy 0.121964 -> 0.583463
- Frame 000129: loss 3.982360 -> 2.606996, accuracy 0.136245 -> 0.595944
- Frame 000130: loss 4.003391 -> 2.507202, accuracy 0.132574 -> 0.617815
- Frame 000131: loss 4.054266 -> 2.413960, accuracy 0.128624 -> 0.643648
- Frame 000132: loss 4.012959 -> 2.712195, accuracy 0.134423 -> 0.591460
- Frame 000133: loss 4.016452 -> 2.528891, accuracy 0.136675 -> 0.624274
- Frame 000134: loss 4.063485 -> 2.674720, accuracy 0.130298 -> 0.621141
- Frame 000135: loss 4.039163 -> 2.401021, accuracy 0.133891 -> 0.642259
- Frame 000136: loss 4.029527 -> 2.767552, accuracy 0.137134 -> 0.609656
- Frame 000137: loss 4.032144 -> 2.668841, accuracy 0.129705 -> 0.616989
- Frame 000138: loss 4.059722 -> 2.149819, accuracy 0.117557 -> 0.686005
- Frame 000139: loss 4.040631 -> 2.284388, accuracy 0.140333 -> 0.659768
- Frame 000140: loss 4.089414 -> 2.362173, accuracy 0.104335 -> 0.684980
- Frame 000141: loss 4.073440 -> 2.373584, accuracy 0.111892 -> 0.691922
- Frame 000142: loss 4.068551 -> 2.093346, accuracy 0.114746 -> 0.711122
- Frame 000143: loss 4.095098 -> 2.712688, accuracy 0.096628 -> 0.634625
- Frame 000144: loss 4.039449 -> 2.949288, accuracy 0.112302 -> 0.589076
- Frame 000145: loss 4.051708 -> 2.419775, accuracy 0.107763 -> 0.693054
- Frame 000146: loss 4.062329 -> 2.457927, accuracy 0.097623 -> 0.641882
- Frame 000147: loss 4.087996 -> 2.531461, accuracy 0.115463 -> 0.658189
- Frame 000148: loss 4.100152 -> 2.859401, accuracy 0.092882 -> 0.593135
- Frame 000149: loss 4.070296 -> 2.703828, accuracy 0.098527 -> 0.613509

## Holdout Final Class IoU

- 00 car: IoU 0.430251, acc 0.751926, gt 11682, pred 17518, tp 8784
- 01 bicycle: IoU 0.009259, acc 0.009615, gt 104, pred 5, tp 1
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 6, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 24, tp 0
- 08 road: IoU 0.673498, acc 0.904846, gt 23173, pred 28928, tp 20968
- 09 parking: IoU 0.057377, acc 0.069356, gt 1413, pred 393, tp 98
- 10 sidewalk: IoU 0.404212, acc 0.593186, gt 17052, pred 18087, tp 10115
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.336427, acc 0.350209, gt 17184, pred 6722, tp 6018
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 1, tp 0
- 14 vegetation: IoU 0.365739, acc 0.525343, gt 23596, pred 22693, tp 12396
- 15 trunk: IoU 0.060484, acc 0.088583, gt 508, pred 281, tp 45
- 16 terrain: IoU 0.007640, acc 0.015929, gt 1130, pred 1244, tp 18
- 17 pole: IoU 0.105206, acc 0.307937, gt 315, pred 704, tp 97
- 18 traffic-sign: IoU 0.050089, acc 0.168675, gt 166, pred 421, tp 28
