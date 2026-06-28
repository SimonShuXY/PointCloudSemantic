# SemanticKITTI PTv3/IPFP Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame and generates IPFP auxiliary image-backprojected points.
- Runs the PTv3/IPFP fused forward path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.334178
- Final eval loss, first frame: 0.112793
- Initial eval mean loss: 3.346268
- Final eval mean loss: 0.146984
- Initial eval mean accuracy: 0.077196
- Final eval mean accuracy: 0.956046
- First train loss: 3.562997
- Final train loss: 0.009180
- Best train loss: 0.007195
- Final train valid accuracy: 0.998483
- CUDA peak memory GB: 0.957

## Per-frame Eval

- Frame 000000: loss 3.334178 -> 0.112793, accuracy 0.077039 -> 0.964753
- Frame 000001: loss 3.384166 -> 0.155173, accuracy 0.065086 -> 0.957619
- Frame 000002: loss 3.334597 -> 0.132507, accuracy 0.085656 -> 0.960973
- Frame 000003: loss 3.299203 -> 0.143859, accuracy 0.080849 -> 0.956544
- Frame 000004: loss 3.379198 -> 0.190588, accuracy 0.077351 -> 0.940344
