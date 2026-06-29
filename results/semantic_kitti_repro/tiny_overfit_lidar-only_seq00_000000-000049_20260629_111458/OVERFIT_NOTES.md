# SemanticKITTI lidar-only Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame for visualization sanity checks.
- Runs the selected PTv3 training path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.335353
- Final eval loss, first frame: 0.382305
- Initial eval mean loss: 3.486653
- Final eval mean loss: 0.324141
- Initial eval mean accuracy: 0.049238
- Final eval mean accuracy: 0.893634
- First train loss: 3.553271
- Final train loss: 0.171293
- Best train loss: 0.129773
- Final train valid accuracy: 0.938623
- CUDA peak memory GB: 0.902
- Selected visualization frames: 8

## Per-frame Eval

- Frame 000000: loss 3.335353 -> 0.382305, accuracy 0.076536 -> 0.865055
- Frame 000001: loss 3.388242 -> 0.408557, accuracy 0.060545 -> 0.853179
- Frame 000002: loss 3.336213 -> 0.461825, accuracy 0.084643 -> 0.834263
- Frame 000003: loss 3.299490 -> 0.494780, accuracy 0.077817 -> 0.824659
- Frame 000004: loss 3.381280 -> 0.572599, accuracy 0.073812 -> 0.808898
- Frame 000005: loss 3.424664 -> 0.537413, accuracy 0.053293 -> 0.839618
- Frame 000006: loss 3.404328 -> 0.552084, accuracy 0.063915 -> 0.839456
- Frame 000007: loss 3.348777 -> 0.627693, accuracy 0.071214 -> 0.814945
- Frame 000008: loss 3.389984 -> 0.517734, accuracy 0.065196 -> 0.844534
- Frame 000009: loss 3.405047 -> 0.542314, accuracy 0.065723 -> 0.843276
- Frame 000010: loss 3.374335 -> 0.532289, accuracy 0.069744 -> 0.852484
- Frame 000011: loss 3.374096 -> 0.485241, accuracy 0.071928 -> 0.842657
- Frame 000012: loss 3.378459 -> 0.540045, accuracy 0.067932 -> 0.835664
- Frame 000013: loss 3.401592 -> 0.446725, accuracy 0.069069 -> 0.849349
- Frame 000014: loss 3.423616 -> 0.458370, accuracy 0.065598 -> 0.857787
- Frame 000015: loss 3.438371 -> 0.366114, accuracy 0.055306 -> 0.874439
- Frame 000016: loss 3.426907 -> 0.406618, accuracy 0.053580 -> 0.858287
- Frame 000017: loss 3.448913 -> 0.412150, accuracy 0.051863 -> 0.875126
- Frame 000018: loss 3.472995 -> 0.444935, accuracy 0.043303 -> 0.859517
- Frame 000019: loss 3.445987 -> 0.419442, accuracy 0.065631 -> 0.871744
- Frame 000020: loss 3.476244 -> 0.360804, accuracy 0.043325 -> 0.886146
- Frame 000021: loss 3.476415 -> 0.392601, accuracy 0.045568 -> 0.870305
- Frame 000022: loss 3.494071 -> 0.307914, accuracy 0.052099 -> 0.901366
- Frame 000023: loss 3.497717 -> 0.308253, accuracy 0.051870 -> 0.892269
- Frame 000024: loss 3.536470 -> 0.305074, accuracy 0.035536 -> 0.891391
- Frame 000025: loss 3.505594 -> 0.258497, accuracy 0.040500 -> 0.911500
- Frame 000026: loss 3.532983 -> 0.297785, accuracy 0.037886 -> 0.894816
- Frame 000027: loss 3.536357 -> 0.276883, accuracy 0.034965 -> 0.896104
- Frame 000028: loss 3.546594 -> 0.282468, accuracy 0.030378 -> 0.900896
- Frame 000029: loss 3.558045 -> 0.204396, accuracy 0.036482 -> 0.920040
- Frame 000030: loss 3.567276 -> 0.302723, accuracy 0.038347 -> 0.906375
- Frame 000031: loss 3.587349 -> 0.245166, accuracy 0.032934 -> 0.912675
- Frame 000032: loss 3.559157 -> 0.250620, accuracy 0.034826 -> 0.911940
- Frame 000033: loss 3.549948 -> 0.235126, accuracy 0.039940 -> 0.918622
- Frame 000034: loss 3.589076 -> 0.214815, accuracy 0.032290 -> 0.922007
- Frame 000035: loss 3.552671 -> 0.168240, accuracy 0.042521 -> 0.936969
- Frame 000036: loss 3.543657 -> 0.188091, accuracy 0.037111 -> 0.935306
- Frame 000037: loss 3.549637 -> 0.215208, accuracy 0.041916 -> 0.927146
- Frame 000038: loss 3.561001 -> 0.203141, accuracy 0.029116 -> 0.929719
- Frame 000039: loss 3.570688 -> 0.144192, accuracy 0.032967 -> 0.947053
- Frame 000040: loss 3.502084 -> 0.169172, accuracy 0.041185 -> 0.939729
- Frame 000041: loss 3.555753 -> 0.160294, accuracy 0.031706 -> 0.949673
- Frame 000042: loss 3.572729 -> 0.150452, accuracy 0.035946 -> 0.951073
- Frame 000043: loss 3.588210 -> 0.130022, accuracy 0.033066 -> 0.954910
- Frame 000044: loss 3.574155 -> 0.142770, accuracy 0.042821 -> 0.953149
- Frame 000045: loss 3.545747 -> 0.127765, accuracy 0.044132 -> 0.957372
- Frame 000046: loss 3.610252 -> 0.137707, accuracy 0.038922 -> 0.954591
- Frame 000047: loss 3.553878 -> 0.134102, accuracy 0.040419 -> 0.954092
- Frame 000048: loss 3.575337 -> 0.123224, accuracy 0.034051 -> 0.960441
- Frame 000049: loss 3.564881 -> 0.160304, accuracy 0.042415 -> 0.949102
