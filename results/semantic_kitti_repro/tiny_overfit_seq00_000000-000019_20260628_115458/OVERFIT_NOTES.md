# SemanticKITTI PTv3/IPFP Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame and generates IPFP auxiliary image-backprojected points.
- Runs the PTv3/IPFP fused forward path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss, first frame: 3.334400
- Final eval loss, first frame: 0.192539
- Initial eval mean loss: 3.393293
- Final eval mean loss: 0.170169
- Initial eval mean accuracy: 0.065512
- Final eval mean accuracy: 0.941562
- First train loss: 3.549637
- Final train loss: 0.115203
- Best train loss: 0.110380
- Final train valid accuracy: 0.958918
- CUDA peak memory GB: 0.957

## Per-frame Eval

- Frame 000000: loss 3.334400 -> 0.192539, accuracy 0.077543 -> 0.938066
- Frame 000001: loss 3.384203 -> 0.198366, accuracy 0.061049 -> 0.933905
- Frame 000002: loss 3.333110 -> 0.217622, accuracy 0.083122 -> 0.924987
- Frame 000003: loss 3.298739 -> 0.188645, accuracy 0.084891 -> 0.937342
- Frame 000004: loss 3.381288 -> 0.223892, accuracy 0.077351 -> 0.923660
- Frame 000005: loss 3.428445 -> 0.209520, accuracy 0.057315 -> 0.930116
- Frame 000006: loss 3.407074 -> 0.201300, accuracy 0.057373 -> 0.929039
- Frame 000007: loss 3.347612 -> 0.181432, accuracy 0.068205 -> 0.937312
- Frame 000008: loss 3.391191 -> 0.216990, accuracy 0.067202 -> 0.929789
- Frame 000009: loss 3.396827 -> 0.177931, accuracy 0.065723 -> 0.939333
- Frame 000010: loss 3.375875 -> 0.171652, accuracy 0.070246 -> 0.929754
- Frame 000011: loss 3.368306 -> 0.162427, accuracy 0.073427 -> 0.942557
- Frame 000012: loss 3.378512 -> 0.149532, accuracy 0.065934 -> 0.950549
- Frame 000013: loss 3.394899 -> 0.153073, accuracy 0.069069 -> 0.948949
- Frame 000014: loss 3.419729 -> 0.153370, accuracy 0.067601 -> 0.942414
- Frame 000015: loss 3.434877 -> 0.142040, accuracy 0.054310 -> 0.950673
- Frame 000016: loss 3.428135 -> 0.108258, accuracy 0.051077 -> 0.964947
- Frame 000017: loss 3.450282 -> 0.124602, accuracy 0.048842 -> 0.958711
- Frame 000018: loss 3.468401 -> 0.121915, accuracy 0.046324 -> 0.958207
- Frame 000019: loss 3.443961 -> 0.108284, accuracy 0.063627 -> 0.960922
