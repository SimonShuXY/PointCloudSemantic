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
- Extra feature mode/scale: learned, 1.000
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.290528
- Final eval loss, first frame: 0.981758
- Initial eval mean loss: 4.492270
- Final eval mean loss: 0.883951
- Initial eval mean accuracy: 0.044796
- Final eval mean accuracy: 0.863480
- Initial eval mIoU: 0.006190
- Final eval mIoU: 0.484009
- Initial eval overall accuracy: 0.044769
- Final eval overall accuracy: 0.863426
- First train loss: 4.506416
- Final train loss: 0.722936
- Best train loss: 0.344794
- Final train valid accuracy: 0.890194
- CUDA peak memory GB: 0.961
- Selected visualization frames: 4

## Holdout Eval

- Holdout initial mean loss: 4.186430
- Holdout final mean loss: 3.302454
- Holdout initial mean accuracy: 0.102235
- Holdout final mean accuracy: 0.544093
- Holdout initial mIoU: 0.011170
- Holdout final mIoU: 0.129543
- Holdout initial overall accuracy: 0.102312
- Holdout final overall accuracy: 0.544446

## Diagnostic fused Eval

- Train final mean loss: 0.886325
- Train final mean accuracy: 0.863330
- Train final mIoU: 0.482779
- Train final overall accuracy: 0.863276

- Holdout final mean loss: 3.310526
- Holdout final mean accuracy: 0.541827
- Holdout final mIoU: 0.128500
- Holdout final overall accuracy: 0.542189

## Train-Frame Eval

- Frame 000000: loss 4.290528 -> 0.981758, accuracy 0.077039 -> 0.843907
- Frame 000001: loss 4.349031 -> 0.815527, accuracy 0.059536 -> 0.855701
- Frame 000002: loss 4.292087 -> 1.003130, accuracy 0.082108 -> 0.831222
- Frame 000003: loss 4.256783 -> 0.975342, accuracy 0.081354 -> 0.835776
- Frame 000004: loss 4.333007 -> 0.934051, accuracy 0.077856 -> 0.858948
- Frame 000005: loss 4.374909 -> 0.907948, accuracy 0.056310 -> 0.857718
- Frame 000006: loss 4.358929 -> 0.902279, accuracy 0.056366 -> 0.851032
- Frame 000007: loss 4.308273 -> 0.858411, accuracy 0.071214 -> 0.858576
- Frame 000008: loss 4.355700 -> 0.941812, accuracy 0.064694 -> 0.850552
- Frame 000009: loss 4.358656 -> 1.131393, accuracy 0.067240 -> 0.819009
- Frame 000010: loss 4.331933 -> 1.112499, accuracy 0.073256 -> 0.834922
- Frame 000011: loss 4.329230 -> 1.019994, accuracy 0.072428 -> 0.845654
- Frame 000012: loss 4.340292 -> 1.127458, accuracy 0.065435 -> 0.832667
- Frame 000013: loss 4.361177 -> 1.044634, accuracy 0.068068 -> 0.836837
- Frame 000014: loss 4.382357 -> 1.261321, accuracy 0.065098 -> 0.818728
- Frame 000015: loss 4.405955 -> 1.299788, accuracy 0.054310 -> 0.795715
- Frame 000016: loss 4.392172 -> 0.892549, accuracy 0.050576 -> 0.848272
- Frame 000017: loss 4.407812 -> 1.162994, accuracy 0.050352 -> 0.831823
- Frame 000018: loss 4.428176 -> 0.774951, accuracy 0.045317 -> 0.863041
- Frame 000019: loss 4.403918 -> 0.915370, accuracy 0.065631 -> 0.850701
- Frame 000020: loss 4.432537 -> 1.082013, accuracy 0.046348 -> 0.847859
- Frame 000021: loss 4.431120 -> 1.091693, accuracy 0.046570 -> 0.831748
- Frame 000022: loss 4.455586 -> 1.112245, accuracy 0.051088 -> 0.825999
- Frame 000023: loss 4.449797 -> 1.432477, accuracy 0.049875 -> 0.780050
- Frame 000024: loss 4.497989 -> 1.131390, accuracy 0.032533 -> 0.838338
- Frame 000025: loss 4.467013 -> 1.194587, accuracy 0.039500 -> 0.812500
- Frame 000026: loss 4.487206 -> 1.180843, accuracy 0.038385 -> 0.817547
- Frame 000027: loss 4.503575 -> 0.907334, accuracy 0.035465 -> 0.837662
- Frame 000028: loss 4.499789 -> 1.171947, accuracy 0.031873 -> 0.837151
- Frame 000029: loss 4.519717 -> 1.064322, accuracy 0.036482 -> 0.852074
- Frame 000030: loss 4.521564 -> 1.225809, accuracy 0.038347 -> 0.821713
- Frame 000031: loss 4.541203 -> 1.033155, accuracy 0.031936 -> 0.853293
- Frame 000032: loss 4.510898 -> 0.966179, accuracy 0.033333 -> 0.852736
- Frame 000033: loss 4.504143 -> 1.038416, accuracy 0.039441 -> 0.841737
- Frame 000034: loss 4.529969 -> 0.960708, accuracy 0.032787 -> 0.860904
- Frame 000035: loss 4.506396 -> 0.841482, accuracy 0.042021 -> 0.878940
- Frame 000036: loss 4.494706 -> 1.024059, accuracy 0.039117 -> 0.865597
- Frame 000037: loss 4.503107 -> 0.818927, accuracy 0.041417 -> 0.862774
- Frame 000038: loss 4.512470 -> 0.962608, accuracy 0.031124 -> 0.854920
- Frame 000039: loss 4.517360 -> 0.888121, accuracy 0.034466 -> 0.867632
- Frame 000040: loss 4.448464 -> 1.009158, accuracy 0.041185 -> 0.842793
- Frame 000041: loss 4.510316 -> 1.097311, accuracy 0.028686 -> 0.842476
- Frame 000042: loss 4.528552 -> 0.926785, accuracy 0.035946 -> 0.859211
- Frame 000043: loss 4.540429 -> 1.048056, accuracy 0.032565 -> 0.833667
- Frame 000044: loss 4.520544 -> 0.934936, accuracy 0.041814 -> 0.822166
- Frame 000045: loss 4.497547 -> 0.800888, accuracy 0.043631 -> 0.843531
- Frame 000046: loss 4.553342 -> 1.033706, accuracy 0.038423 -> 0.836826
- Frame 000047: loss 4.501908 -> 1.120830, accuracy 0.037924 -> 0.828842
- Frame 000048: loss 4.526611 -> 0.732053, accuracy 0.035053 -> 0.870806
- Frame 000049: loss 4.516377 -> 0.910224, accuracy 0.044411 -> 0.853792
- Frame 000050: loss 4.517299 -> 1.005636, accuracy 0.053427 -> 0.827621
- Frame 000051: loss 4.525660 -> 0.996194, accuracy 0.040642 -> 0.846463
- Frame 000052: loss 4.511130 -> 0.979164, accuracy 0.051397 -> 0.846307
- Frame 000053: loss 4.512437 -> 1.172954, accuracy 0.036217 -> 0.812374
- Frame 000054: loss 4.537980 -> 0.889541, accuracy 0.029895 -> 0.844544
- Frame 000055: loss 4.570273 -> 1.076025, accuracy 0.028472 -> 0.840659
- Frame 000056: loss 4.567816 -> 1.034497, accuracy 0.018953 -> 0.840399
- Frame 000057: loss 4.560173 -> 0.778813, accuracy 0.032274 -> 0.863456
- Frame 000058: loss 4.530936 -> 0.858061, accuracy 0.025551 -> 0.858717
- Frame 000059: loss 4.563755 -> 1.074599, accuracy 0.031516 -> 0.826913
- Frame 000060: loss 4.553113 -> 0.850035, accuracy 0.035306 -> 0.854799
- Frame 000061: loss 4.517201 -> 1.109997, accuracy 0.036743 -> 0.824230
- Frame 000062: loss 4.575243 -> 1.053300, accuracy 0.033547 -> 0.814997
- Frame 000063: loss 4.577114 -> 1.050490, accuracy 0.023348 -> 0.834079
- Frame 000064: loss 4.624222 -> 0.883924, accuracy 0.025323 -> 0.873386
- Frame 000065: loss 4.582477 -> 0.788170, accuracy 0.032819 -> 0.879165
- Frame 000066: loss 4.554321 -> 0.878096, accuracy 0.035447 -> 0.881678
- Frame 000067: loss 4.597955 -> 0.725837, accuracy 0.032869 -> 0.900398
- Frame 000068: loss 4.625159 -> 0.761570, accuracy 0.031062 -> 0.895291
- Frame 000069: loss 4.593780 -> 0.635119, accuracy 0.027861 -> 0.914428
- Frame 000070: loss 4.611569 -> 0.538746, accuracy 0.038062 -> 0.927830
- Frame 000071: loss 4.558980 -> 0.571532, accuracy 0.054148 -> 0.908594
- Frame 000072: loss 4.536978 -> 0.582412, accuracy 0.040119 -> 0.916295
- Frame 000073: loss 4.547190 -> 0.592517, accuracy 0.045680 -> 0.897716
- Frame 000074: loss 4.526848 -> 0.496867, accuracy 0.042310 -> 0.933798
- Frame 000075: loss 4.566823 -> 0.636343, accuracy 0.037981 -> 0.900550
- Frame 000076: loss 4.526461 -> 0.638308, accuracy 0.048563 -> 0.907830
- Frame 000077: loss 4.520562 -> 0.598173, accuracy 0.047071 -> 0.893340
- Frame 000078: loss 4.570358 -> 0.561819, accuracy 0.043086 -> 0.910321
- Frame 000079: loss 4.561895 -> 0.575007, accuracy 0.043653 -> 0.910186
- Frame 000080: loss 4.578345 -> 0.613869, accuracy 0.044478 -> 0.909046
- Frame 000081: loss 4.565705 -> 0.690475, accuracy 0.054912 -> 0.901763
- Frame 000082: loss 4.580228 -> 0.769839, accuracy 0.050772 -> 0.876058
- Frame 000083: loss 4.516726 -> 0.676007, accuracy 0.051811 -> 0.899899
- Frame 000084: loss 4.501155 -> 0.646060, accuracy 0.053661 -> 0.908225
- Frame 000085: loss 4.492502 -> 0.771086, accuracy 0.058412 -> 0.868697
- Frame 000086: loss 4.460897 -> 0.735957, accuracy 0.063691 -> 0.879639
- Frame 000087: loss 4.479928 -> 0.572831, accuracy 0.056402 -> 0.916667
- Frame 000088: loss 4.488821 -> 0.552109, accuracy 0.054244 -> 0.916625
- Frame 000089: loss 4.455208 -> 0.554265, accuracy 0.048436 -> 0.917760
- Frame 000090: loss 4.554653 -> 0.673356, accuracy 0.032471 -> 0.907154
- Frame 000091: loss 4.530367 -> 0.571742, accuracy 0.033028 -> 0.935467
- Frame 000092: loss 4.506673 -> 0.566604, accuracy 0.044365 -> 0.933707
- Frame 000093: loss 4.504733 -> 0.656381, accuracy 0.049645 -> 0.921986
- Frame 000094: loss 4.540391 -> 0.509444, accuracy 0.035318 -> 0.921292
- Frame 000095: loss 4.527252 -> 0.627256, accuracy 0.038304 -> 0.905005
- Frame 000096: loss 4.536156 -> 0.660290, accuracy 0.044035 -> 0.906810
- Frame 000097: loss 4.536089 -> 0.759415, accuracy 0.035312 -> 0.882805
- Frame 000098: loss 4.538894 -> 0.714464, accuracy 0.036261 -> 0.887130
- Frame 000099: loss 4.515416 -> 0.874432, accuracy 0.034729 -> 0.869765

## Train Final Class IoU

- 00 car: IoU 0.893187, acc 0.952070, gt 24786, pred 25232, tp 23598
- 01 bicycle: IoU 0.437500, acc 0.720588, gt 68, pred 93, tp 49
- 04 other-vehicle: IoU 0.254545, acc 0.273839, gt 818, pred 286, tp 224
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.409091, acc 0.422819, gt 149, pred 68, tp 63
- 08 road: IoU 0.878057, acc 0.986337, gt 53650, pred 59533, tp 52917
- 09 parking: IoU 0.532415, acc 0.567553, gt 17031, pred 10790, tp 9666
- 10 sidewalk: IoU 0.526975, acc 0.807490, gt 18290, pred 24505, tp 14769
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.864524, acc 0.883017, gt 53059, pred 47987, tp 46852
- 13 fence: IoU 0.280566, acc 0.327573, gt 1331, pred 659, tp 436
- 14 vegetation: IoU 0.629701, acc 0.788740, gt 23606, pred 24581, tp 18619
- 15 trunk: IoU 0.795147, acc 0.813993, gt 5274, pred 4418, tp 4293
- 16 terrain: IoU 0.142070, acc 0.266529, gt 484, pred 553, tp 129
- 17 pole: IoU 0.660563, acc 0.688693, gt 681, pred 498, tp 469
- 18 traffic-sign: IoU 0.439803, acc 0.643885, gt 278, pred 308, tp 179

## Holdout Per-Frame Eval

- Frame 000100: loss 4.502933 -> 1.431732, accuracy 0.041026 -> 0.769744
- Frame 000101: loss 4.528954 -> 1.940787, accuracy 0.037190 -> 0.676136
- Frame 000102: loss 4.461323 -> 1.877384, accuracy 0.049434 -> 0.688980
- Frame 000103: loss 4.487579 -> 2.197836, accuracy 0.049325 -> 0.667705
- Frame 000104: loss 4.414414 -> 2.593710, accuracy 0.056099 -> 0.628924
- Frame 000105: loss 4.434140 -> 3.031255, accuracy 0.057513 -> 0.560104
- Frame 000106: loss 4.418216 -> 3.388097, accuracy 0.061088 -> 0.505134
- Frame 000107: loss 4.385551 -> 3.886563, accuracy 0.054859 -> 0.476489
- Frame 000108: loss 4.353287 -> 3.902127, accuracy 0.069502 -> 0.455394
- Frame 000109: loss 4.387566 -> 3.968653, accuracy 0.073684 -> 0.453684
- Frame 000110: loss 4.353438 -> 4.147912, accuracy 0.074035 -> 0.456726
- Frame 000111: loss 4.339154 -> 4.128673, accuracy 0.077368 -> 0.485263
- Frame 000112: loss 4.328756 -> 4.398594, accuracy 0.082491 -> 0.463012
- Frame 000113: loss 4.319828 -> 4.277506, accuracy 0.071882 -> 0.464588
- Frame 000114: loss 4.286570 -> 4.042488, accuracy 0.088697 -> 0.495403
- Frame 000115: loss 4.258786 -> 3.918114, accuracy 0.104134 -> 0.492412
- Frame 000116: loss 4.267995 -> 3.811411, accuracy 0.104307 -> 0.502335
- Frame 000117: loss 4.232920 -> 3.722796, accuracy 0.101129 -> 0.495893
- Frame 000118: loss 4.222859 -> 3.926297, accuracy 0.115838 -> 0.491030
- Frame 000119: loss 4.222762 -> 3.591954, accuracy 0.103448 -> 0.527890
- Frame 000120: loss 4.198245 -> 3.522986, accuracy 0.118670 -> 0.511509
- Frame 000121: loss 4.152168 -> 3.634609, accuracy 0.121243 -> 0.520632
- Frame 000122: loss 4.109742 -> 3.508000, accuracy 0.138634 -> 0.528033
- Frame 000123: loss 4.076797 -> 3.953811, accuracy 0.122709 -> 0.474033
- Frame 000124: loss 4.147182 -> 3.358197, accuracy 0.123211 -> 0.552658
- Frame 000125: loss 4.140123 -> 3.129579, accuracy 0.112813 -> 0.552833
- Frame 000126: loss 4.046296 -> 3.151113, accuracy 0.132512 -> 0.544427
- Frame 000127: loss 4.032135 -> 3.159020, accuracy 0.153409 -> 0.521694
- Frame 000128: loss 4.143591 -> 3.476935, accuracy 0.117313 -> 0.520930
- Frame 000129: loss 3.984342 -> 3.488367, accuracy 0.135725 -> 0.521581
- Frame 000130: loss 4.003943 -> 3.405587, accuracy 0.134645 -> 0.530295
- Frame 000131: loss 4.050962 -> 3.134032, accuracy 0.128097 -> 0.573537
- Frame 000132: loss 4.012842 -> 3.576611, accuracy 0.140221 -> 0.506062
- Frame 000133: loss 4.015128 -> 3.119120, accuracy 0.136675 -> 0.573087
- Frame 000134: loss 4.067052 -> 3.048008, accuracy 0.129252 -> 0.563579
- Frame 000135: loss 4.042162 -> 3.220675, accuracy 0.132322 -> 0.561192
- Frame 000136: loss 4.026177 -> 3.453969, accuracy 0.136620 -> 0.513097
- Frame 000137: loss 4.025682 -> 3.715515, accuracy 0.132248 -> 0.492370
- Frame 000138: loss 4.061358 -> 2.806830, accuracy 0.116539 -> 0.612214
- Frame 000139: loss 4.036799 -> 2.969969, accuracy 0.139324 -> 0.592630
- Frame 000140: loss 4.091235 -> 3.060054, accuracy 0.102319 -> 0.614415
- Frame 000141: loss 4.080067 -> 2.730515, accuracy 0.105871 -> 0.629202
- Frame 000142: loss 4.065771 -> 2.558030, accuracy 0.115249 -> 0.650226
- Frame 000143: loss 4.096245 -> 3.254561, accuracy 0.095118 -> 0.526925
- Frame 000144: loss 4.039574 -> 3.594016, accuracy 0.107198 -> 0.480347
- Frame 000145: loss 4.045935 -> 3.156357, accuracy 0.110317 -> 0.548519
- Frame 000146: loss 4.067042 -> 2.791485, accuracy 0.094082 -> 0.582195
- Frame 000147: loss 4.089659 -> 2.587001, accuracy 0.112920 -> 0.612411
- Frame 000148: loss 4.098756 -> 3.322485, accuracy 0.092882 -> 0.481575
- Frame 000149: loss 4.067448 -> 3.051383, accuracy 0.100559 -> 0.555612

## Holdout Final Class IoU

- 00 car: IoU 0.328625, acc 0.804657, gt 11682, pred 26322, tp 9400
- 01 bicycle: IoU 0.000000, acc 0.000000, gt 104, pred 44, tp 0
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 3, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 12, tp 0
- 08 road: IoU 0.656999, acc 0.933543, gt 23173, pred 31387, tp 21633
- 09 parking: IoU 0.050112, acc 0.111111, gt 1413, pred 1877, tp 157
- 10 sidewalk: IoU 0.331190, acc 0.443995, gt 17052, pred 13379, tp 7571
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.378977, acc 0.410382, gt 17184, pred 8476, tp 7052
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 62, tp 0
- 14 vegetation: IoU 0.227570, acc 0.290770, gt 23596, pred 13414, tp 6861
- 15 trunk: IoU 0.040000, acc 0.059055, gt 508, pred 272, tp 30
- 16 terrain: IoU 0.006734, acc 0.012389, gt 1130, pred 963, tp 14
- 17 pole: IoU 0.115254, acc 0.215873, gt 315, pred 343, tp 68
- 18 traffic-sign: IoU 0.066778, acc 0.240964, gt 166, pred 473, tp 40
