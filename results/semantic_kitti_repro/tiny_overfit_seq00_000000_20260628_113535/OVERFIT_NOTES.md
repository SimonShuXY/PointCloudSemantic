# SemanticKITTI PTv3/IPFP Tiny Overfit

## What this validates

- Loads real SemanticKITTI LiDAR, labels, KITTI color image, and calibration.
- Projects LiDAR into the image frame and generates IPFP auxiliary image-backprojected points.
- Runs the PTv3/IPFP fused forward path with gradients, backward, and optimizer updates.
- Checks whether a tiny training set can be memorized at least partially.

## Important caveat

This is a reproduction plumbing test, not a benchmark result. A useful run should show the training loss moving down, but it does not measure generalization.

## Summary

- Initial eval loss: 3.333441
- Final eval loss: 0.364151
- First train loss: 3.559571
- Final train loss: 0.005274
- Best train loss: 0.005274
- Final train valid accuracy: 0.999496
- CUDA peak memory GB: 0.956
