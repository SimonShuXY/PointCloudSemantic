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
- Extra feature mode/scale: learned, 0.500
- IPFP depth percentile range: 20.0-99.0
- IPFP discard probability: 0.200
- Initial eval loss, first frame: 4.290620
- Final eval loss, first frame: 0.920739
- Initial eval mean loss: 4.492242
- Final eval mean loss: 0.846261
- Initial eval mean accuracy: 0.044736
- Final eval mean accuracy: 0.877198
- Initial eval mIoU: 0.006242
- Final eval mIoU: 0.488105
- Initial eval overall accuracy: 0.044709
- Final eval overall accuracy: 0.877155
- First train loss: 4.509361
- Final train loss: 0.714517
- Best train loss: 0.361924
- Final train valid accuracy: 0.900919
- CUDA peak memory GB: 0.961
- Selected visualization frames: 4

## Holdout Eval

- Holdout initial mean loss: 4.186363
- Holdout final mean loss: 2.789501
- Holdout initial mean accuracy: 0.102192
- Holdout final mean accuracy: 0.598263
- Holdout initial mIoU: 0.011191
- Holdout final mIoU: 0.149251
- Holdout initial overall accuracy: 0.102260
- Holdout final overall accuracy: 0.598844

## Diagnostic fused Eval

- Train final mean loss: 0.847745
- Train final mean accuracy: 0.877455
- Train final mIoU: 0.487367
- Train final overall accuracy: 0.877405

- Holdout final mean loss: 2.777769
- Holdout final mean accuracy: 0.599370
- Holdout final mIoU: 0.150490
- Holdout final overall accuracy: 0.599936

## Train-Frame Eval

- Frame 000000: loss 4.290620 -> 0.920739, accuracy 0.076536 -> 0.852971
- Frame 000001: loss 4.346815 -> 0.824314, accuracy 0.061554 -> 0.869828
- Frame 000002: loss 4.293909 -> 0.914036, accuracy 0.081602 -> 0.862139
- Frame 000003: loss 4.256694 -> 0.820112, accuracy 0.080849 -> 0.876200
- Frame 000004: loss 4.334881 -> 0.920349, accuracy 0.078362 -> 0.858443
- Frame 000005: loss 4.373769 -> 0.925577, accuracy 0.056310 -> 0.861237
- Frame 000006: loss 4.360111 -> 0.906796, accuracy 0.057876 -> 0.858077
- Frame 000007: loss 4.308707 -> 0.923037, accuracy 0.071715 -> 0.860582
- Frame 000008: loss 4.355264 -> 0.843771, accuracy 0.064193 -> 0.863089
- Frame 000009: loss 4.358256 -> 1.002169, accuracy 0.066229 -> 0.851871
- Frame 000010: loss 4.333368 -> 1.113544, accuracy 0.070246 -> 0.834922
- Frame 000011: loss 4.331148 -> 0.958123, accuracy 0.070929 -> 0.847153
- Frame 000012: loss 4.337664 -> 0.986604, accuracy 0.064935 -> 0.855644
- Frame 000013: loss 4.360801 -> 1.103757, accuracy 0.068068 -> 0.841341
- Frame 000014: loss 4.380568 -> 1.343700, accuracy 0.065098 -> 0.816725
- Frame 000015: loss 4.404449 -> 1.412975, accuracy 0.055306 -> 0.787245
- Frame 000016: loss 4.392647 -> 1.200360, accuracy 0.050075 -> 0.805709
- Frame 000017: loss 4.407366 -> 1.071344, accuracy 0.049345 -> 0.844411
- Frame 000018: loss 4.427920 -> 1.053816, accuracy 0.043807 -> 0.829809
- Frame 000019: loss 4.403957 -> 0.873656, accuracy 0.066132 -> 0.861222
- Frame 000020: loss 4.431027 -> 0.906885, accuracy 0.047859 -> 0.867003
- Frame 000021: loss 4.433137 -> 0.938400, accuracy 0.044066 -> 0.858287
- Frame 000022: loss 4.456376 -> 1.043338, accuracy 0.052099 -> 0.849772
- Frame 000023: loss 4.452061 -> 1.167828, accuracy 0.049377 -> 0.826434
- Frame 000024: loss 4.496680 -> 1.027115, accuracy 0.033033 -> 0.856857
- Frame 000025: loss 4.466917 -> 1.060512, accuracy 0.039500 -> 0.837000
- Frame 000026: loss 4.487603 -> 1.176453, accuracy 0.038883 -> 0.831505
- Frame 000027: loss 4.503293 -> 0.828445, accuracy 0.033966 -> 0.851648
- Frame 000028: loss 4.501226 -> 1.071354, accuracy 0.031873 -> 0.836653
- Frame 000029: loss 4.519518 -> 0.980819, accuracy 0.036482 -> 0.876062
- Frame 000030: loss 4.521130 -> 1.219169, accuracy 0.038845 -> 0.831673
- Frame 000031: loss 4.542085 -> 0.897116, accuracy 0.031437 -> 0.878743
- Frame 000032: loss 4.511669 -> 0.937838, accuracy 0.034328 -> 0.868657
- Frame 000033: loss 4.504533 -> 0.992052, accuracy 0.038942 -> 0.872192
- Frame 000034: loss 4.530375 -> 0.821436, accuracy 0.031793 -> 0.876801
- Frame 000035: loss 4.506146 -> 0.811588, accuracy 0.041521 -> 0.896448
- Frame 000036: loss 4.494656 -> 0.892901, accuracy 0.038616 -> 0.879639
- Frame 000037: loss 4.501177 -> 0.794200, accuracy 0.042415 -> 0.872755
- Frame 000038: loss 4.511950 -> 0.831907, accuracy 0.033133 -> 0.899598
- Frame 000039: loss 4.514106 -> 0.671712, accuracy 0.033467 -> 0.898601
- Frame 000040: loss 4.446725 -> 0.845425, accuracy 0.042190 -> 0.878955
- Frame 000041: loss 4.512251 -> 0.904098, accuracy 0.029693 -> 0.886261
- Frame 000042: loss 4.527932 -> 0.807665, accuracy 0.037943 -> 0.886670
- Frame 000043: loss 4.539068 -> 0.831221, accuracy 0.033066 -> 0.879259
- Frame 000044: loss 4.520700 -> 1.005059, accuracy 0.041814 -> 0.837783
- Frame 000045: loss 4.497377 -> 0.808564, accuracy 0.043631 -> 0.882648
- Frame 000046: loss 4.554191 -> 0.946600, accuracy 0.038423 -> 0.853792
- Frame 000047: loss 4.499152 -> 0.863954, accuracy 0.039421 -> 0.875250
- Frame 000048: loss 4.526505 -> 0.648927, accuracy 0.035053 -> 0.906860
- Frame 000049: loss 4.518776 -> 0.820527, accuracy 0.042415 -> 0.884731
- Frame 000050: loss 4.514670 -> 0.823502, accuracy 0.053427 -> 0.884577
- Frame 000051: loss 4.527042 -> 0.946600, accuracy 0.041144 -> 0.881586
- Frame 000052: loss 4.512489 -> 0.887428, accuracy 0.050898 -> 0.858782
- Frame 000053: loss 4.513575 -> 1.038685, accuracy 0.035714 -> 0.859658
- Frame 000054: loss 4.538205 -> 0.842842, accuracy 0.029895 -> 0.866966
- Frame 000055: loss 4.571918 -> 0.971287, accuracy 0.028971 -> 0.863636
- Frame 000056: loss 4.567906 -> 0.928243, accuracy 0.019451 -> 0.856359
- Frame 000057: loss 4.560519 -> 0.817033, accuracy 0.032274 -> 0.872890
- Frame 000058: loss 4.530785 -> 0.909963, accuracy 0.025050 -> 0.867234
- Frame 000059: loss 4.563774 -> 1.011582, accuracy 0.031016 -> 0.854927
- Frame 000060: loss 4.550907 -> 0.911787, accuracy 0.037295 -> 0.863252
- Frame 000061: loss 4.516413 -> 1.093610, accuracy 0.036246 -> 0.842105
- Frame 000062: loss 4.576148 -> 0.870895, accuracy 0.034534 -> 0.861865
- Frame 000063: loss 4.577150 -> 0.841291, accuracy 0.023348 -> 0.871336
- Frame 000064: loss 4.626397 -> 0.762468, accuracy 0.024826 -> 0.889275
- Frame 000065: loss 4.583757 -> 0.738304, accuracy 0.032819 -> 0.888613
- Frame 000066: loss 4.554108 -> 0.742600, accuracy 0.035447 -> 0.897154
- Frame 000067: loss 4.597458 -> 0.745584, accuracy 0.033367 -> 0.911853
- Frame 000068: loss 4.626422 -> 0.726382, accuracy 0.031062 -> 0.908317
- Frame 000069: loss 4.591921 -> 0.740070, accuracy 0.028358 -> 0.908458
- Frame 000070: loss 4.613832 -> 0.518508, accuracy 0.039051 -> 0.921898
- Frame 000071: loss 4.556072 -> 0.640326, accuracy 0.052161 -> 0.910084
- Frame 000072: loss 4.537657 -> 0.663570, accuracy 0.039128 -> 0.910847
- Frame 000073: loss 4.547033 -> 0.515594, accuracy 0.046673 -> 0.912612
- Frame 000074: loss 4.526225 -> 0.534701, accuracy 0.040319 -> 0.919861
- Frame 000075: loss 4.568514 -> 0.737220, accuracy 0.037481 -> 0.877061
- Frame 000076: loss 4.525339 -> 0.696933, accuracy 0.048563 -> 0.916254
- Frame 000077: loss 4.520331 -> 0.615917, accuracy 0.046570 -> 0.894342
- Frame 000078: loss 4.569459 -> 0.642246, accuracy 0.042084 -> 0.914830
- Frame 000079: loss 4.558682 -> 0.560340, accuracy 0.043151 -> 0.920723
- Frame 000080: loss 4.577824 -> 0.603030, accuracy 0.044478 -> 0.918541
- Frame 000081: loss 4.566505 -> 0.689693, accuracy 0.054408 -> 0.916877
- Frame 000082: loss 4.580449 -> 0.769228, accuracy 0.049776 -> 0.871080
- Frame 000083: loss 4.518702 -> 0.704255, accuracy 0.049799 -> 0.910463
- Frame 000084: loss 4.503001 -> 0.781229, accuracy 0.053159 -> 0.891174
- Frame 000085: loss 4.489349 -> 0.762118, accuracy 0.058412 -> 0.883175
- Frame 000086: loss 4.460675 -> 0.683831, accuracy 0.062688 -> 0.891675
- Frame 000087: loss 4.479259 -> 0.644497, accuracy 0.056911 -> 0.914126
- Frame 000088: loss 4.489471 -> 0.587512, accuracy 0.054244 -> 0.915118
- Frame 000089: loss 4.455047 -> 0.619634, accuracy 0.049445 -> 0.919778
- Frame 000090: loss 4.553637 -> 0.659451, accuracy 0.033486 -> 0.918823
- Frame 000091: loss 4.530827 -> 0.636152, accuracy 0.033028 -> 0.931911
- Frame 000092: loss 4.507244 -> 0.663265, accuracy 0.044365 -> 0.914839
- Frame 000093: loss 4.504894 -> 0.665279, accuracy 0.049139 -> 0.926039
- Frame 000094: loss 4.540033 -> 0.543636, accuracy 0.034813 -> 0.920283
- Frame 000095: loss 4.528164 -> 0.633642, accuracy 0.036772 -> 0.909602
- Frame 000096: loss 4.538649 -> 0.677729, accuracy 0.043523 -> 0.905274
- Frame 000097: loss 4.535354 -> 0.668735, accuracy 0.035824 -> 0.901228
- Frame 000098: loss 4.538969 -> 0.706688, accuracy 0.037794 -> 0.892748
- Frame 000099: loss 4.514200 -> 0.779117, accuracy 0.036772 -> 0.882533

## Train Final Class IoU

- 00 car: IoU 0.887582, acc 0.940652, gt 24786, pred 24797, tp 23315
- 01 bicycle: IoU 0.337349, acc 0.411765, gt 68, pred 43, tp 28
- 04 other-vehicle: IoU 0.402110, acc 0.419315, gt 818, pred 378, tp 343
- 06 bicyclist: IoU 0.000000, acc 0.000000, gt 2, pred 0, tp 0
- 07 motorcyclist: IoU 0.496732, acc 0.510067, gt 149, pred 80, tp 76
- 08 road: IoU 0.881999, acc 0.978024, gt 53650, pred 58312, tp 52471
- 09 parking: IoU 0.533341, acc 0.571076, gt 17031, pred 10931, tp 9726
- 10 sidewalk: IoU 0.548364, acc 0.819847, gt 18290, pred 24050, tp 14995
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 4, pred 0, tp 0
- 12 building: IoU 0.894194, acc 0.947078, gt 53059, pred 53389, tp 50251
- 13 fence: IoU 0.103116, acc 0.104433, gt 1331, pred 156, tp 139
- 14 vegetation: IoU 0.691273, acc 0.788232, gt 23606, pred 21918, tp 18607
- 15 trunk: IoU 0.805643, acc 0.828403, gt 5274, pred 4518, tp 4369
- 16 terrain: IoU 0.119253, acc 0.171488, gt 484, pred 295, tp 83
- 17 pole: IoU 0.656338, acc 0.684288, gt 681, pred 495, tp 466
- 18 traffic-sign: IoU 0.452381, acc 0.478417, gt 278, pred 149, tp 133

## Holdout Per-Frame Eval

- Frame 000100: loss 4.502810 -> 1.205339, accuracy 0.040513 -> 0.795897
- Frame 000101: loss 4.528673 -> 1.628049, accuracy 0.037190 -> 0.739153
- Frame 000102: loss 4.462729 -> 1.628946, accuracy 0.049434 -> 0.745623
- Frame 000103: loss 4.487474 -> 1.717954, accuracy 0.048287 -> 0.714434
- Frame 000104: loss 4.412469 -> 2.424319, accuracy 0.056099 -> 0.597015
- Frame 000105: loss 4.433637 -> 2.734509, accuracy 0.059067 -> 0.574093
- Frame 000106: loss 4.416377 -> 3.214881, accuracy 0.061602 -> 0.527207
- Frame 000107: loss 4.386223 -> 3.480329, accuracy 0.055904 -> 0.497910
- Frame 000108: loss 4.352262 -> 3.601731, accuracy 0.070539 -> 0.442946
- Frame 000109: loss 4.389334 -> 3.421072, accuracy 0.072105 -> 0.483158
- Frame 000110: loss 4.353050 -> 3.751265, accuracy 0.072471 -> 0.471324
- Frame 000111: loss 4.339737 -> 3.873965, accuracy 0.078947 -> 0.468947
- Frame 000112: loss 4.331147 -> 3.920873, accuracy 0.080362 -> 0.419372
- Frame 000113: loss 4.319055 -> 3.611378, accuracy 0.071353 -> 0.479387
- Frame 000114: loss 4.285300 -> 3.596255, accuracy 0.093564 -> 0.471065
- Frame 000115: loss 4.262356 -> 3.436298, accuracy 0.103611 -> 0.464155
- Frame 000116: loss 4.269670 -> 3.070375, accuracy 0.102750 -> 0.545407
- Frame 000117: loss 4.234519 -> 2.944843, accuracy 0.099076 -> 0.561088
- Frame 000118: loss 4.222927 -> 3.407246, accuracy 0.114300 -> 0.501794
- Frame 000119: loss 4.215374 -> 3.034239, accuracy 0.105477 -> 0.557302
- Frame 000120: loss 4.203381 -> 3.123581, accuracy 0.114066 -> 0.525320
- Frame 000121: loss 4.151345 -> 2.681816, accuracy 0.119205 -> 0.613856
- Frame 000122: loss 4.109320 -> 2.945020, accuracy 0.136595 -> 0.589195
- Frame 000123: loss 4.073084 -> 3.453051, accuracy 0.125255 -> 0.522912
- Frame 000124: loss 4.146470 -> 2.869934, accuracy 0.118098 -> 0.576176
- Frame 000125: loss 4.140177 -> 2.719144, accuracy 0.112302 -> 0.603369
- Frame 000126: loss 4.045874 -> 2.766078, accuracy 0.135080 -> 0.594761
- Frame 000127: loss 4.030192 -> 2.509882, accuracy 0.154959 -> 0.623450
- Frame 000128: loss 4.144808 -> 2.634041, accuracy 0.121447 -> 0.627907
- Frame 000129: loss 3.987703 -> 2.515256, accuracy 0.137806 -> 0.653666
- Frame 000130: loss 4.004505 -> 2.540140, accuracy 0.132056 -> 0.635940
- Frame 000131: loss 4.050807 -> 2.459765, accuracy 0.134423 -> 0.649974
- Frame 000132: loss 4.011869 -> 2.612777, accuracy 0.139167 -> 0.630469
- Frame 000133: loss 4.014896 -> 2.377978, accuracy 0.139314 -> 0.663325
- Frame 000134: loss 4.068678 -> 2.512304, accuracy 0.128205 -> 0.675563
- Frame 000135: loss 4.042440 -> 2.476047, accuracy 0.130753 -> 0.648013
- Frame 000136: loss 4.026081 -> 2.765739, accuracy 0.136107 -> 0.614278
- Frame 000137: loss 4.023821 -> 2.697924, accuracy 0.132757 -> 0.626144
- Frame 000138: loss 4.060924 -> 2.194965, accuracy 0.116031 -> 0.692112
- Frame 000139: loss 4.039866 -> 2.529336, accuracy 0.137809 -> 0.627461
- Frame 000140: loss 4.089350 -> 2.651088, accuracy 0.099798 -> 0.662298
- Frame 000141: loss 4.079247 -> 2.652855, accuracy 0.108379 -> 0.672353
- Frame 000142: loss 4.064910 -> 2.190765, accuracy 0.112733 -> 0.718671
- Frame 000143: loss 4.096942 -> 2.787136, accuracy 0.092602 -> 0.626069
- Frame 000144: loss 4.038393 -> 2.921979, accuracy 0.107708 -> 0.591118
- Frame 000145: loss 4.045950 -> 2.882479, accuracy 0.109806 -> 0.617978
- Frame 000146: loss 4.068449 -> 2.552513, accuracy 0.095094 -> 0.640364
- Frame 000147: loss 4.088479 -> 2.352570, accuracy 0.113937 -> 0.677518
- Frame 000148: loss 4.097158 -> 2.611948, accuracy 0.093387 -> 0.646138
- Frame 000149: loss 4.067928 -> 2.783051, accuracy 0.102082 -> 0.609446

## Holdout Final Class IoU

- 00 car: IoU 0.372909, acc 0.811162, gt 11682, pred 23205, tp 9476
- 01 bicycle: IoU 0.008696, acc 0.009615, gt 104, pred 12, tp 1
- 02 motorcycle: IoU 0.000000, acc 0.000000, gt 191, pred 0, tp 0
- 04 other-vehicle: IoU 0.000000, acc 0.000000, gt 12, pred 28, tp 0
- 05 person: IoU 0.000000, acc 0.000000, gt 29, pred 0, tp 0
- 07 motorcyclist: IoU 0.000000, acc n/a, gt 0, pred 23, tp 0
- 08 road: IoU 0.660390, acc 0.870194, gt 23173, pred 27527, tp 20165
- 09 parking: IoU 0.058413, acc 0.112527, gt 1413, pred 1468, tp 159
- 10 sidewalk: IoU 0.416476, acc 0.598288, gt 17052, pred 17646, tp 10202
- 11 other-ground: IoU 0.000000, acc 0.000000, gt 11, pred 0, tp 0
- 12 building: IoU 0.461535, acc 0.529271, gt 17184, pred 11617, tp 9095
- 13 fence: IoU 0.000000, acc 0.000000, gt 461, pred 5, tp 0
- 14 vegetation: IoU 0.306341, acc 0.375318, gt 23596, pred 14169, tp 8856
- 15 trunk: IoU 0.062426, acc 0.104331, gt 508, pred 394, tp 53
- 16 terrain: IoU 0.008464, acc 0.012389, gt 1130, pred 538, tp 14
- 17 pole: IoU 0.139216, acc 0.225397, gt 315, pred 266, tp 71
- 18 traffic-sign: IoU 0.042403, acc 0.072289, gt 166, pred 129, tp 12
