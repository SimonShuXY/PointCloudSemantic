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
- Extra feature mode/scale: learned, 0.000
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.290009
- Final eval loss, first frame: 0.939059
- Initial eval mean loss: 4.492275
- Final eval mean loss: 0.911456
- Initial eval mean accuracy: 0.044771
- Final eval mean accuracy: 0.868081
- Initial eval mIoU: 0.006216
- Final eval mIoU: 0.484178
- Initial eval overall accuracy: 0.044744
- Final eval overall accuracy: 0.868052
- First train loss: 4.508390
- Final train loss: 0.673004
- Best train loss: 0.330389
- Final train valid accuracy: 0.902452
- CUDA peak memory GB: 0.961
- Selected visualization frames: 4

## Holdout Eval

- Holdout initial mean loss: 4.186751
- Holdout final mean loss: 2.704452
- Holdout initial mean accuracy: 0.102247
- Holdout final mean accuracy: 0.603000
- Holdout initial mIoU: 0.011160
- Holdout final mIoU: 0.148871
- Holdout initial overall accuracy: 0.102322
- Holdout final overall accuracy: 0.603533

## Diagnostic fused Eval

- Train final mean loss: 0.924538
- Train final mean accuracy: 0.866308
- Train final mIoU: 0.482842
- Train final overall accuracy: 0.866283

- Holdout final mean loss: 2.683675
- Holdout final mean accuracy: 0.604459
- Holdout final mIoU: 0.149225
- Holdout final overall accuracy: 0.604986

## Train-Frame Eval

- Frame 000000: loss 4.290009 -> 0.939059, accuracy 0.077039 -> 0.851964
- Frame 000001: loss 4.348690 -> 0.882258, accuracy 0.060545 -> 0.858224
- Frame 000002: loss 4.292889 -> 0.971968, accuracy 0.082108 -> 0.858591
- Frame 000003: loss 4.256770 -> 1.017581, accuracy 0.079838 -> 0.862557
- Frame 000004: loss 4.332286 -> 1.084741, accuracy 0.078868 -> 0.846309
- Frame 000005: loss 4.374794 -> 0.935815, accuracy 0.055807 -> 0.865762
- Frame 000006: loss 4.360440 -> 0.869061, accuracy 0.057373 -> 0.868646
- Frame 000007: loss 4.307240 -> 0.971129, accuracy 0.073220 -> 0.865597
- Frame 000008: loss 4.355901 -> 1.036731, accuracy 0.064193 -> 0.843029
- Frame 000009: loss 4.359200 -> 1.137742, accuracy 0.064712 -> 0.834176
- Frame 000010: loss 4.332207 -> 1.089664, accuracy 0.072253 -> 0.842950
- Frame 000011: loss 4.329985 -> 0.995438, accuracy 0.072428 -> 0.841159
- Frame 000012: loss 4.337732 -> 1.048115, accuracy 0.066434 -> 0.861139
- Frame 000013: loss 4.360869 -> 1.024627, accuracy 0.068569 -> 0.848348
- Frame 000014: loss 4.380485 -> 1.333160, accuracy 0.065098 -> 0.818728
- Frame 000015: loss 4.405584 -> 1.185252, accuracy 0.055306 -> 0.817638
- Frame 000016: loss 4.389128 -> 1.035104, accuracy 0.053080 -> 0.835754
- Frame 000017: loss 4.408273 -> 1.286525, accuracy 0.049345 -> 0.825277
- Frame 000018: loss 4.427804 -> 1.006802, accuracy 0.045317 -> 0.852971
- Frame 000019: loss 4.404308 -> 0.943208, accuracy 0.065631 -> 0.863727
- Frame 000020: loss 4.431609 -> 0.940844, accuracy 0.047859 -> 0.867003
- Frame 000021: loss 4.432189 -> 1.015148, accuracy 0.045568 -> 0.843265
- Frame 000022: loss 4.455400 -> 1.018728, accuracy 0.052099 -> 0.855336
- Frame 000023: loss 4.449438 -> 1.211038, accuracy 0.049377 -> 0.821446
- Frame 000024: loss 4.497798 -> 1.040955, accuracy 0.032533 -> 0.850851
- Frame 000025: loss 4.466661 -> 1.011065, accuracy 0.040000 -> 0.845000
- Frame 000026: loss 4.487041 -> 1.181010, accuracy 0.038385 -> 0.827517
- Frame 000027: loss 4.504963 -> 1.077617, accuracy 0.034466 -> 0.825674
- Frame 000028: loss 4.500302 -> 1.091376, accuracy 0.031873 -> 0.842629
- Frame 000029: loss 4.517587 -> 1.091179, accuracy 0.036482 -> 0.849575
- Frame 000030: loss 4.521002 -> 1.172940, accuracy 0.038845 -> 0.837649
- Frame 000031: loss 4.541506 -> 0.926141, accuracy 0.032435 -> 0.882236
- Frame 000032: loss 4.508430 -> 0.923529, accuracy 0.034328 -> 0.883582
- Frame 000033: loss 4.502607 -> 1.016870, accuracy 0.040939 -> 0.875686
- Frame 000034: loss 4.531545 -> 0.831728, accuracy 0.032787 -> 0.892201
- Frame 000035: loss 4.505777 -> 0.838235, accuracy 0.043522 -> 0.894948
- Frame 000036: loss 4.496000 -> 0.992411, accuracy 0.037613 -> 0.869609
- Frame 000037: loss 4.500035 -> 0.892784, accuracy 0.041916 -> 0.881737
- Frame 000038: loss 4.513186 -> 0.985212, accuracy 0.030120 -> 0.873996
- Frame 000039: loss 4.518538 -> 0.798783, accuracy 0.033966 -> 0.895105
- Frame 000040: loss 4.449989 -> 0.978921, accuracy 0.041688 -> 0.877951
- Frame 000041: loss 4.517178 -> 1.035655, accuracy 0.029693 -> 0.872672
- Frame 000042: loss 4.528982 -> 0.969750, accuracy 0.035946 -> 0.867698
- Frame 000043: loss 4.538374 -> 0.942821, accuracy 0.032064 -> 0.864729
- Frame 000044: loss 4.519904 -> 1.019795, accuracy 0.042821 -> 0.828715
- Frame 000045: loss 4.497012 -> 1.090452, accuracy 0.042628 -> 0.835506
- Frame 000046: loss 4.554059 -> 1.009525, accuracy 0.039421 -> 0.845309
- Frame 000047: loss 4.500578 -> 1.016232, accuracy 0.038423 -> 0.855289
- Frame 000048: loss 4.526463 -> 0.759410, accuracy 0.034552 -> 0.885829
- Frame 000049: loss 4.516829 -> 0.932382, accuracy 0.040918 -> 0.851796
- Frame 000050: loss 4.513658 -> 0.911580, accuracy 0.052923 -> 0.858367
- Frame 000051: loss 4.527451 -> 0.913107, accuracy 0.041144 -> 0.875063
- Frame 000052: loss 4.511946 -> 0.956567, accuracy 0.052894 -> 0.844311
- Frame 000053: loss 4.513604 -> 1.109564, accuracy 0.036720 -> 0.836016
- Frame 000054: loss 4.539797 -> 1.089272, accuracy 0.028899 -> 0.833084
- Frame 000055: loss 4.569319 -> 1.163254, accuracy 0.028971 -> 0.832667
- Frame 000056: loss 4.566538 -> 1.192769, accuracy 0.018454 -> 0.828928
- Frame 000057: loss 4.563115 -> 0.895193, accuracy 0.032274 -> 0.859980
- Frame 000058: loss 4.527710 -> 0.963723, accuracy 0.024549 -> 0.853707
- Frame 000059: loss 4.566139 -> 1.105707, accuracy 0.030515 -> 0.847424
- Frame 000060: loss 4.551909 -> 0.966355, accuracy 0.035803 -> 0.846842
- Frame 000061: loss 4.518244 -> 1.223084, accuracy 0.036743 -> 0.837140
- Frame 000062: loss 4.575348 -> 0.943483, accuracy 0.034040 -> 0.831771
- Frame 000063: loss 4.577754 -> 0.970574, accuracy 0.022851 -> 0.839543
- Frame 000064: loss 4.625834 -> 0.917005, accuracy 0.025819 -> 0.864449
- Frame 000065: loss 4.581339 -> 0.760492, accuracy 0.033814 -> 0.891099
- Frame 000066: loss 4.553736 -> 0.761506, accuracy 0.034948 -> 0.896655
- Frame 000067: loss 4.597013 -> 0.722092, accuracy 0.032869 -> 0.907371
- Frame 000068: loss 4.625675 -> 0.812711, accuracy 0.032064 -> 0.900802
- Frame 000069: loss 4.592572 -> 0.675749, accuracy 0.028358 -> 0.907960
- Frame 000070: loss 4.613852 -> 0.554571, accuracy 0.037568 -> 0.926841
- Frame 000071: loss 4.556538 -> 0.675660, accuracy 0.052161 -> 0.893691
- Frame 000072: loss 4.536897 -> 0.566618, accuracy 0.037642 -> 0.924220
- Frame 000073: loss 4.547432 -> 0.521947, accuracy 0.046177 -> 0.899702
- Frame 000074: loss 4.527015 -> 0.612823, accuracy 0.041314 -> 0.920358
- Frame 000075: loss 4.567374 -> 0.691109, accuracy 0.036482 -> 0.895552
- Frame 000076: loss 4.525929 -> 0.691296, accuracy 0.049058 -> 0.898910
- Frame 000077: loss 4.521411 -> 0.665867, accuracy 0.047571 -> 0.889835
- Frame 000078: loss 4.571420 -> 0.620585, accuracy 0.041583 -> 0.899299
- Frame 000079: loss 4.561209 -> 0.570376, accuracy 0.043151 -> 0.922730
- Frame 000080: loss 4.579389 -> 0.662932, accuracy 0.044478 -> 0.906547
- Frame 000081: loss 4.565608 -> 0.755449, accuracy 0.054912 -> 0.889169
- Frame 000082: loss 4.580617 -> 0.923357, accuracy 0.049776 -> 0.850672
- Frame 000083: loss 4.518922 -> 0.827758, accuracy 0.049799 -> 0.879779
- Frame 000084: loss 4.502429 -> 0.850782, accuracy 0.054664 -> 0.886660
- Frame 000085: loss 4.489989 -> 0.920060, accuracy 0.057913 -> 0.856216
- Frame 000086: loss 4.460620 -> 0.845467, accuracy 0.061685 -> 0.875627
- Frame 000087: loss 4.479947 -> 0.758286, accuracy 0.057927 -> 0.886179
- Frame 000088: loss 4.489769 -> 0.749374, accuracy 0.053742 -> 0.892516
- Frame 000089: loss 4.454793 -> 0.679832, accuracy 0.050959 -> 0.899596
- Frame 000090: loss 4.554526 -> 0.793083, accuracy 0.032978 -> 0.898529
- Frame 000091: loss 4.530816 -> 0.674778, accuracy 0.033028 -> 0.921240
- Frame 000092: loss 4.506895 -> 0.608953, accuracy 0.043855 -> 0.917899
- Frame 000093: loss 4.504906 -> 0.776045, accuracy 0.049645 -> 0.899696
- Frame 000094: loss 4.539731 -> 0.592789, accuracy 0.034309 -> 0.905146
- Frame 000095: loss 4.528322 -> 0.690641, accuracy 0.037283 -> 0.894791
- Frame 000096: loss 4.537903 -> 0.723511, accuracy 0.044547 -> 0.896057
- Frame 000097: loss 4.535607 -> 0.779519, accuracy 0.035824 -> 0.877687
- Frame 000098: loss 4.538583 -> 0.827010, accuracy 0.036772 -> 0.871297
- Frame 000099: loss 4.514812 -> 0.874834, accuracy 0.035240 -> 0.875383

## Train Final Class IoU

- 00 car: IoU 0.894321, acc 0.941661, gt 24786, pred 24652, tp 23340
- 01 bicycle: IoU 0.448718, acc 0.514706, gt 68, pred 45, tp 35
- 04 other-vehicle: IoU 0.267753, acc 0.281174, gt 818, pred 271, tp 230
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.565789, acc 0.577181, gt 149, pred 89, tp 86
- 08 road: IoU 0.837738, acc 0.994949, gt 53650, pred 63447, tp 53379
- 09 parking: IoU 0.492173, acc 0.515061, gt 17031, pred 9564, tp 8772
- 10 sidewalk: IoU 0.549068, acc 0.748551, gt 18290, pred 20336, tp 13691
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.886987, acc 0.917846, gt 53059, pred 50546, tp 48700
- 13 fence: IoU 0.154478, acc 0.155522, gt 1331, pred 216, tp 207
- 14 vegetation: IoU 0.700607, acc 0.856096, gt 23606, pred 25448, tp 20209
- 15 trunk: IoU 0.717819, acc 0.726394, gt 5274, pred 3894, tp 3831
- 16 terrain: IoU 0.116788, acc 0.165289, gt 484, pred 281, tp 80
- 17 pole: IoU 0.648980, acc 0.700441, gt 681, pred 531, tp 477
- 18 traffic-sign: IoU 0.465625, acc 0.535971, gt 278, pred 191, tp 149

## Holdout Per-Frame Eval

- Frame 000100: loss 4.503073 -> 1.333585, accuracy 0.041538 -> 0.789744
- Frame 000101: loss 4.530070 -> 1.704574, accuracy 0.036674 -> 0.744835
- Frame 000102: loss 4.462923 -> 1.625073, accuracy 0.049949 -> 0.742534
- Frame 000103: loss 4.487807 -> 1.791685, accuracy 0.048287 -> 0.696262
- Frame 000104: loss 4.414119 -> 2.359391, accuracy 0.057128 -> 0.614514
- Frame 000105: loss 4.433463 -> 2.621067, accuracy 0.057513 -> 0.580311
- Frame 000106: loss 4.416684 -> 3.300843, accuracy 0.062628 -> 0.527721
- Frame 000107: loss 4.387249 -> 3.562632, accuracy 0.055904 -> 0.504180
- Frame 000108: loss 4.353221 -> 3.445515, accuracy 0.068465 -> 0.507261
- Frame 000109: loss 4.385929 -> 3.493480, accuracy 0.073684 -> 0.493684
- Frame 000110: loss 4.354631 -> 3.740314, accuracy 0.070386 -> 0.474974
- Frame 000111: loss 4.340509 -> 3.497211, accuracy 0.076316 -> 0.503158
- Frame 000112: loss 4.330801 -> 3.643562, accuracy 0.081958 -> 0.499734
- Frame 000113: loss 4.318883 -> 3.324670, accuracy 0.073996 -> 0.497886
- Frame 000114: loss 4.286860 -> 3.309153, accuracy 0.089237 -> 0.506760
- Frame 000115: loss 4.262505 -> 3.152538, accuracy 0.104657 -> 0.515960
- Frame 000116: loss 4.267370 -> 2.914353, accuracy 0.103788 -> 0.551635
- Frame 000117: loss 4.233470 -> 2.860847, accuracy 0.099076 -> 0.542094
- Frame 000118: loss 4.225513 -> 3.412522, accuracy 0.118913 -> 0.501794
- Frame 000119: loss 4.210886 -> 2.654803, accuracy 0.103448 -> 0.582657
- Frame 000120: loss 4.202587 -> 2.881202, accuracy 0.114578 -> 0.561125
- Frame 000121: loss 4.153789 -> 2.704650, accuracy 0.120224 -> 0.599593
- Frame 000122: loss 4.108832 -> 2.695517, accuracy 0.137105 -> 0.580020
- Frame 000123: loss 4.071012 -> 3.044471, accuracy 0.124745 -> 0.530550
- Frame 000124: loss 4.151955 -> 2.798864, accuracy 0.120654 -> 0.583845
- Frame 000125: loss 4.144497 -> 2.720409, accuracy 0.112813 -> 0.590607
- Frame 000126: loss 4.045893 -> 2.608971, accuracy 0.135593 -> 0.575244
- Frame 000127: loss 4.032195 -> 2.459066, accuracy 0.152376 -> 0.627583
- Frame 000128: loss 4.144076 -> 2.501571, accuracy 0.122481 -> 0.609302
- Frame 000129: loss 3.984821 -> 2.523534, accuracy 0.137806 -> 0.621945
- Frame 000130: loss 4.003598 -> 2.380454, accuracy 0.133092 -> 0.653030
- Frame 000131: loss 4.051548 -> 2.428559, accuracy 0.129678 -> 0.655245
- Frame 000132: loss 4.013374 -> 2.785486, accuracy 0.138640 -> 0.574591
- Frame 000133: loss 4.018130 -> 2.581798, accuracy 0.139314 -> 0.632190
- Frame 000134: loss 4.065660 -> 2.516087, accuracy 0.129252 -> 0.656201
- Frame 000135: loss 4.040895 -> 2.557329, accuracy 0.131276 -> 0.612971
- Frame 000136: loss 4.028896 -> 2.556243, accuracy 0.136620 -> 0.643554
- Frame 000137: loss 4.024942 -> 2.994825, accuracy 0.129705 -> 0.560529
- Frame 000138: loss 4.061117 -> 2.147452, accuracy 0.115522 -> 0.686514
- Frame 000139: loss 4.037933 -> 2.301738, accuracy 0.139324 -> 0.660777
- Frame 000140: loss 4.089342 -> 2.623430, accuracy 0.101815 -> 0.670363
- Frame 000141: loss 4.079110 -> 2.386381, accuracy 0.109885 -> 0.681887
- Frame 000142: loss 4.066930 -> 2.077282, accuracy 0.113236 -> 0.724711
- Frame 000143: loss 4.098420 -> 2.646422, accuracy 0.095622 -> 0.669854
- Frame 000144: loss 4.041223 -> 2.640634, accuracy 0.108218 -> 0.627361
- Frame 000145: loss 4.046011 -> 2.594504, accuracy 0.110827 -> 0.654239
- Frame 000146: loss 4.069359 -> 2.518496, accuracy 0.093070 -> 0.607992
- Frame 000147: loss 4.089808 -> 2.530433, accuracy 0.113428 -> 0.661750
- Frame 000148: loss 4.097392 -> 2.728235, accuracy 0.093387 -> 0.629480
- Frame 000149: loss 4.068215 -> 2.540761, accuracy 0.098527 -> 0.629253

## Holdout Final Class IoU

- 00 car: IoU 0.396830, acc 0.771443, gt 11682, pred 20040, tp 9012
- 01 bicycle: IoU 0.009259, acc 0.009615, gt 104, pred 5, tp 1
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 8, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 37, tp 0
- 08 road: IoU 0.649688, acc 0.953912, gt 23173, pred 32956, tp 22105
- 09 parking: IoU 0.061523, acc 0.086341, gt 1413, pred 692, tp 122
- 10 sidewalk: IoU 0.378458, acc 0.486922, gt 17052, pred 13190, tp 8303
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.427711, acc 0.462756, gt 17184, pred 9360, tp 7952
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 2, tp 0
- 14 vegetation: IoU 0.339396, acc 0.462621, gt 23596, pred 19483, tp 10916
- 15 trunk: IoU 0.044190, acc 0.053150, gt 508, pred 130, tp 27
- 16 terrain: IoU 0.004367, acc 0.006195, gt 1130, pred 480, tp 7
- 17 pole: IoU 0.151013, acc 0.260317, gt 315, pred 310, tp 82
- 18 traffic-sign: IoU 0.068376, acc 0.192771, gt 166, pred 334, tp 32
