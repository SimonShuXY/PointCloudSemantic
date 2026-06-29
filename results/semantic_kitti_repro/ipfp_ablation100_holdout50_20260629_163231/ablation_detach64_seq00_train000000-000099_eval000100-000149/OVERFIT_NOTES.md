# SemanticKITTI fused Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.333072
- Final eval loss, first frame: 0.427098
- Initial eval mean loss: 3.536873
- Final eval mean loss: 0.340755
- Initial eval mean accuracy: 0.045003
- Final eval mean accuracy: 0.892117
- Initial eval mIoU: 0.006268
- Final eval mIoU: 0.478909
- Initial eval overall accuracy: 0.044975
- Final eval overall accuracy: 0.892071
- First train loss: 3.544146
- Final train loss: 0.195590
- Best train loss: 0.112670
- Final train valid accuracy: 0.929520
- CUDA peak memory GB: 0.904
- Selected visualization frames: 8

## Holdout Eval

- Holdout initial mean loss: 3.238349
- Holdout final mean loss: 1.895070
- Holdout initial mean accuracy: 0.102147
- Holdout final mean accuracy: 0.615289
- Holdout initial mIoU: 0.011180
- Holdout final mIoU: 0.151041
- Holdout initial overall accuracy: 0.102219
- Holdout final overall accuracy: 0.615829

## Train-Frame Eval

- Frame 000000: loss 3.333072 -> 0.427098, accuracy 0.076032 -> 0.862538
- Frame 000001: loss 3.383420 -> 0.368050, accuracy 0.063572 -> 0.883451
- Frame 000002: loss 3.332548 -> 0.422723, accuracy 0.084136 -> 0.870248
- Frame 000003: loss 3.299715 -> 0.397477, accuracy 0.081860 -> 0.880748
- Frame 000004: loss 3.380540 -> 0.410159, accuracy 0.077351 -> 0.873104
- Frame 000005: loss 3.428648 -> 0.394399, accuracy 0.056812 -> 0.881347
- Frame 000006: loss 3.408937 -> 0.328872, accuracy 0.057876 -> 0.893810
- Frame 000007: loss 3.348249 -> 0.382040, accuracy 0.069709 -> 0.882146
- Frame 000008: loss 3.389391 -> 0.376875, accuracy 0.065697 -> 0.876630
- Frame 000009: loss 3.394706 -> 0.359425, accuracy 0.067745 -> 0.884227
- Frame 000010: loss 3.377566 -> 0.392720, accuracy 0.069744 -> 0.878073
- Frame 000011: loss 3.367157 -> 0.405067, accuracy 0.071928 -> 0.868132
- Frame 000012: loss 3.379137 -> 0.479104, accuracy 0.065934 -> 0.853147
- Frame 000013: loss 3.395451 -> 0.449929, accuracy 0.071071 -> 0.871872
- Frame 000014: loss 3.421835 -> 0.707616, accuracy 0.068603 -> 0.835754
- Frame 000015: loss 3.437027 -> 0.574058, accuracy 0.054310 -> 0.843049
- Frame 000016: loss 3.425035 -> 0.510536, accuracy 0.051577 -> 0.833751
- Frame 000017: loss 3.450137 -> 0.640671, accuracy 0.050856 -> 0.835347
- Frame 000018: loss 3.469474 -> 0.387703, accuracy 0.045821 -> 0.879154
- Frame 000019: loss 3.441621 -> 0.357082, accuracy 0.065130 -> 0.872244
- Frame 000020: loss 3.470493 -> 0.382103, accuracy 0.041814 -> 0.868514
- Frame 000021: loss 3.460442 -> 0.406449, accuracy 0.046570 -> 0.861793
- Frame 000022: loss 3.494085 -> 0.397795, accuracy 0.051088 -> 0.856348
- Frame 000023: loss 3.493963 -> 0.518696, accuracy 0.050374 -> 0.843391
- Frame 000024: loss 3.542701 -> 0.531542, accuracy 0.036036 -> 0.851351
- Frame 000025: loss 3.510380 -> 0.475555, accuracy 0.039500 -> 0.841000
- Frame 000026: loss 3.528790 -> 0.690597, accuracy 0.041376 -> 0.814556
- Frame 000027: loss 3.545205 -> 0.394488, accuracy 0.034466 -> 0.861638
- Frame 000028: loss 3.545493 -> 0.483656, accuracy 0.032371 -> 0.850598
- Frame 000029: loss 3.560477 -> 0.483201, accuracy 0.036982 -> 0.846077
- Frame 000030: loss 3.561498 -> 0.582218, accuracy 0.039841 -> 0.847610
- Frame 000031: loss 3.590363 -> 0.433568, accuracy 0.033932 -> 0.871756
- Frame 000032: loss 3.556638 -> 0.414188, accuracy 0.038806 -> 0.880100
- Frame 000033: loss 3.537115 -> 0.381294, accuracy 0.040939 -> 0.883675
- Frame 000034: loss 3.581384 -> 0.360315, accuracy 0.033780 -> 0.885743
- Frame 000035: loss 3.545405 -> 0.297845, accuracy 0.041021 -> 0.900950
- Frame 000036: loss 3.541324 -> 0.402693, accuracy 0.034604 -> 0.882648
- Frame 000037: loss 3.549603 -> 0.380865, accuracy 0.041916 -> 0.880240
- Frame 000038: loss 3.560263 -> 0.360353, accuracy 0.030622 -> 0.886546
- Frame 000039: loss 3.567145 -> 0.324089, accuracy 0.031968 -> 0.901099
- Frame 000040: loss 3.499699 -> 0.381857, accuracy 0.044199 -> 0.893019
- Frame 000041: loss 3.560239 -> 0.346414, accuracy 0.030700 -> 0.889280
- Frame 000042: loss 3.571410 -> 0.281366, accuracy 0.037943 -> 0.906640
- Frame 000043: loss 3.586227 -> 0.417088, accuracy 0.033066 -> 0.879760
- Frame 000044: loss 3.565613 -> 0.430523, accuracy 0.042317 -> 0.854408
- Frame 000045: loss 3.546814 -> 0.308108, accuracy 0.038616 -> 0.898696
- Frame 000046: loss 3.604636 -> 0.297340, accuracy 0.040419 -> 0.904691
- Frame 000047: loss 3.542354 -> 0.312814, accuracy 0.041417 -> 0.890220
- Frame 000048: loss 3.572294 -> 0.242251, accuracy 0.037556 -> 0.916375
- Frame 000049: loss 3.566222 -> 0.261387, accuracy 0.043912 -> 0.903194
- Frame 000050: loss 3.565449 -> 0.278315, accuracy 0.051411 -> 0.903226
- Frame 000051: loss 3.569757 -> 0.333341, accuracy 0.040642 -> 0.897642
- Frame 000052: loss 3.561287 -> 0.326113, accuracy 0.048902 -> 0.893713
- Frame 000053: loss 3.560915 -> 0.323065, accuracy 0.033199 -> 0.893863
- Frame 000054: loss 3.585525 -> 0.281560, accuracy 0.030394 -> 0.903338
- Frame 000055: loss 3.616402 -> 0.310333, accuracy 0.028472 -> 0.890609
- Frame 000056: loss 3.611969 -> 0.505745, accuracy 0.020449 -> 0.849875
- Frame 000057: loss 3.604546 -> 0.318675, accuracy 0.029791 -> 0.896723
- Frame 000058: loss 3.575541 -> 0.372265, accuracy 0.025551 -> 0.881263
- Frame 000059: loss 3.617086 -> 0.419104, accuracy 0.032016 -> 0.871436
- Frame 000060: loss 3.599921 -> 0.353700, accuracy 0.034311 -> 0.884137
- Frame 000061: loss 3.576430 -> 0.431899, accuracy 0.037736 -> 0.858491
- Frame 000062: loss 3.617809 -> 0.335286, accuracy 0.032560 -> 0.881105
- Frame 000063: loss 3.617774 -> 0.378143, accuracy 0.024839 -> 0.875807
- Frame 000064: loss 3.664828 -> 0.313278, accuracy 0.023833 -> 0.894240
- Frame 000065: loss 3.630344 -> 0.293863, accuracy 0.033814 -> 0.905022
- Frame 000066: loss 3.591790 -> 0.249495, accuracy 0.034948 -> 0.912132
- Frame 000067: loss 3.642188 -> 0.226549, accuracy 0.033367 -> 0.920817
- Frame 000068: loss 3.674886 -> 0.276920, accuracy 0.028557 -> 0.911824
- Frame 000069: loss 3.641091 -> 0.210405, accuracy 0.026368 -> 0.928856
- Frame 000070: loss 3.661446 -> 0.201489, accuracy 0.036579 -> 0.929807
- Frame 000071: loss 3.608161 -> 0.245192, accuracy 0.055638 -> 0.917039
- Frame 000072: loss 3.594093 -> 0.198423, accuracy 0.038633 -> 0.934621
- Frame 000073: loss 3.594085 -> 0.188416, accuracy 0.048163 -> 0.934459
- Frame 000074: loss 3.580169 -> 0.183180, accuracy 0.042807 -> 0.936784
- Frame 000075: loss 3.609013 -> 0.225585, accuracy 0.035482 -> 0.925038
- Frame 000076: loss 3.574465 -> 0.211460, accuracy 0.049554 -> 0.932111
- Frame 000077: loss 3.568818 -> 0.178219, accuracy 0.049574 -> 0.936405
- Frame 000078: loss 3.617564 -> 0.230324, accuracy 0.039579 -> 0.918838
- Frame 000079: loss 3.607534 -> 0.207274, accuracy 0.042148 -> 0.927245
- Frame 000080: loss 3.621893 -> 0.227573, accuracy 0.045477 -> 0.923538
- Frame 000081: loss 3.610971 -> 0.255958, accuracy 0.051385 -> 0.919899
- Frame 000082: loss 3.621576 -> 0.310464, accuracy 0.048283 -> 0.904928
- Frame 000083: loss 3.562326 -> 0.258793, accuracy 0.050805 -> 0.914487
- Frame 000084: loss 3.553860 -> 0.259842, accuracy 0.055165 -> 0.915246
- Frame 000085: loss 3.539105 -> 0.322277, accuracy 0.056415 -> 0.908138
- Frame 000086: loss 3.515273 -> 0.281725, accuracy 0.060181 -> 0.904213
- Frame 000087: loss 3.529214 -> 0.203578, accuracy 0.056402 -> 0.926321
- Frame 000088: loss 3.537899 -> 0.225593, accuracy 0.054746 -> 0.921145
- Frame 000089: loss 3.505961 -> 0.216720, accuracy 0.050959 -> 0.931382
- Frame 000090: loss 3.594860 -> 0.246858, accuracy 0.032978 -> 0.920852
- Frame 000091: loss 3.575806 -> 0.164691, accuracy 0.036077 -> 0.945630
- Frame 000092: loss 3.548714 -> 0.204779, accuracy 0.045385 -> 0.938297
- Frame 000093: loss 3.544910 -> 0.186241, accuracy 0.049645 -> 0.940223
- Frame 000094: loss 3.582554 -> 0.189420, accuracy 0.036831 -> 0.934410
- Frame 000095: loss 3.570258 -> 0.185313, accuracy 0.036261 -> 0.936159
- Frame 000096: loss 3.581468 -> 0.255647, accuracy 0.045571 -> 0.920635
- Frame 000097: loss 3.577888 -> 0.225391, accuracy 0.037359 -> 0.919140
- Frame 000098: loss 3.584039 -> 0.247213, accuracy 0.037794 -> 0.922370
- Frame 000099: loss 3.565852 -> 0.247589, accuracy 0.039326 -> 0.907559

## Train Final Class IoU

- 00 car: IoU 0.913293, acc 0.963810, gt 24786, pred 25260, tp 23889
- 01 bicycle: IoU 0.287356, acc 0.367647, gt 68, pred 44, tp 25
- 04 other-vehicle: IoU 0.379553, acc 0.394866, gt 818, pred 356, tp 323
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.360000, acc 0.362416, gt 149, pred 55, tp 54
- 08 road: IoU 0.901553, acc 0.983541, gt 53650, pred 57646, tp 52767
- 09 parking: IoU 0.579909, acc 0.613176, gt 17031, pred 11420, tp 10443
- 10 sidewalk: IoU 0.569853, acc 0.833406, gt 18290, pred 23702, tp 15243
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.914506, acc 0.941480, gt 53059, pred 51519, tp 49954
- 13 fence: IoU 0.254412, acc 0.259955, gt 1331, pred 375, tp 346
- 14 vegetation: IoU 0.730233, acc 0.845463, gt 23606, pred 23683, tp 19958
- 15 trunk: IoU 0.803512, acc 0.824232, gt 5274, pred 4483, tp 4347
- 16 terrain: IoU 0.132911, acc 0.173554, gt 484, pred 232, tp 84
- 17 pole: IoU 0.547995, acc 0.662261, gt 681, pred 593, tp 451
- 18 traffic-sign: IoU 0.287462, acc 0.338129, gt 278, pred 143, tp 94

## Holdout Per-Frame Eval

- Frame 000100: loss 3.546124 -> 0.631171, accuracy 0.040000 -> 0.822051
- Frame 000101: loss 3.568759 -> 0.906403, accuracy 0.036674 -> 0.772211
- Frame 000102: loss 3.524819 -> 0.997543, accuracy 0.047889 -> 0.751287
- Frame 000103: loss 3.526246 -> 1.119086, accuracy 0.048287 -> 0.725857
- Frame 000104: loss 3.456680 -> 1.692540, accuracy 0.057128 -> 0.635615
- Frame 000105: loss 3.480598 -> 1.949021, accuracy 0.054922 -> 0.609845
- Frame 000106: loss 3.463438 -> 2.373956, accuracy 0.062628 -> 0.563142
- Frame 000107: loss 3.433467 -> 2.717949, accuracy 0.057994 -> 0.513062
- Frame 000108: loss 3.398055 -> 2.876358, accuracy 0.071058 -> 0.483921
- Frame 000109: loss 3.430927 -> 2.778870, accuracy 0.074737 -> 0.491053
- Frame 000110: loss 3.393797 -> 3.080006, accuracy 0.069343 -> 0.496350
- Frame 000111: loss 3.387065 -> 3.052361, accuracy 0.075263 -> 0.473684
- Frame 000112: loss 3.374974 -> 2.968216, accuracy 0.085684 -> 0.513039
- Frame 000113: loss 3.366257 -> 3.032462, accuracy 0.070825 -> 0.492600
- Frame 000114: loss 3.326819 -> 2.727492, accuracy 0.087615 -> 0.496485
- Frame 000115: loss 3.306209 -> 2.689707, accuracy 0.100471 -> 0.492412
- Frame 000116: loss 3.314566 -> 2.233317, accuracy 0.104826 -> 0.550597
- Frame 000117: loss 3.283307 -> 2.462094, accuracy 0.098049 -> 0.505134
- Frame 000118: loss 3.268233 -> 2.626203, accuracy 0.110712 -> 0.532035
- Frame 000119: loss 3.267311 -> 2.428423, accuracy 0.105984 -> 0.554260
- Frame 000120: loss 3.246413 -> 2.237633, accuracy 0.113043 -> 0.550384
- Frame 000121: loss 3.207957 -> 1.785266, accuracy 0.121752 -> 0.610290
- Frame 000122: loss 3.164541 -> 1.899715, accuracy 0.135576 -> 0.607034
- Frame 000123: loss 3.136262 -> 2.189483, accuracy 0.128819 -> 0.573829
- Frame 000124: loss 3.207112 -> 1.695557, accuracy 0.119121 -> 0.607873
- Frame 000125: loss 3.203738 -> 1.615306, accuracy 0.110260 -> 0.639612
- Frame 000126: loss 3.109859 -> 1.579823, accuracy 0.135080 -> 0.640472
- Frame 000127: loss 3.073791 -> 1.419114, accuracy 0.154959 -> 0.675103
- Frame 000128: loss 3.188246 -> 1.817096, accuracy 0.118346 -> 0.627907
- Frame 000129: loss 3.050052 -> 1.913431, accuracy 0.139366 -> 0.628705
- Frame 000130: loss 3.062504 -> 1.635316, accuracy 0.132056 -> 0.655619
- Frame 000131: loss 3.106455 -> 1.571836, accuracy 0.131260 -> 0.662625
- Frame 000132: loss 3.071054 -> 1.802562, accuracy 0.137059 -> 0.617290
- Frame 000133: loss 3.067216 -> 1.758445, accuracy 0.143008 -> 0.638522
- Frame 000134: loss 3.120247 -> 1.571975, accuracy 0.130822 -> 0.666143
- Frame 000135: loss 3.088315 -> 1.298878, accuracy 0.135983 -> 0.688807
- Frame 000136: loss 3.087053 -> 1.723318, accuracy 0.132512 -> 0.637391
- Frame 000137: loss 3.090352 -> 1.834695, accuracy 0.130722 -> 0.605290
- Frame 000138: loss 3.115328 -> 1.710513, accuracy 0.117048 -> 0.648346
- Frame 000139: loss 3.097792 -> 1.507491, accuracy 0.135285 -> 0.647653
- Frame 000140: loss 3.140249 -> 1.698391, accuracy 0.102823 -> 0.656754
- Frame 000141: loss 3.125888 -> 1.565256, accuracy 0.106372 -> 0.663322
- Frame 000142: loss 3.124119 -> 1.078816, accuracy 0.110216 -> 0.755410
- Frame 000143: loss 3.159205 -> 1.735482, accuracy 0.099144 -> 0.620533
- Frame 000144: loss 3.090268 -> 1.670002, accuracy 0.111792 -> 0.638591
- Frame 000145: loss 3.109182 -> 1.494897, accuracy 0.110317 -> 0.679775
- Frame 000146: loss 3.121536 -> 1.388956, accuracy 0.097623 -> 0.662114
- Frame 000147: loss 3.152499 -> 1.345123, accuracy 0.112920 -> 0.684639
- Frame 000148: loss 3.154327 -> 1.484737, accuracy 0.091368 -> 0.647653
- Frame 000149: loss 3.128248 -> 1.381215, accuracy 0.102590 -> 0.652108

## Holdout Final Class IoU

- 00 car: IoU 0.383927, acc 0.812190, gt 11682, pred 22519, tp 9488
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 16, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 8, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 27, tp 0
- 08 road: IoU 0.689786, acc 0.939887, gt 23173, pred 30182, tp 21780
- 09 parking: IoU 0.063158, acc 0.118896, gt 1413, pred 1415, tp 168
- 10 sidewalk: IoU 0.401370, acc 0.522285, gt 17052, pred 14043, tp 8906
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.481499, acc 0.520251, gt 17184, pred 10323, tp 8940
- 13 fence: IoU 0.002062, acc 0.002169, gt 461, pred 25, tp 1
- 14 vegetation: IoU 0.344142, acc 0.436727, gt 23596, pred 16653, tp 10305
- 15 trunk: IoU 0.051512, acc 0.090551, gt 508, pred 431, tp 46
- 16 terrain: IoU 0.003564, acc 0.004425, gt 1130, pred 278, tp 5
- 17 pole: IoU 0.081107, acc 0.269841, gt 315, pred 818, tp 85
- 18 traffic-sign: IoU 0.065574, acc 0.168675, gt 166, pred 289, tp 28
