from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ProjectionResult:
    uv: Tensor
    depth: Tensor
    valid: Tensor
    points_cam: Tensor


def _as_4x4(transform: Tensor) -> Tensor:
    if transform.shape == (4, 4):
        return transform
    if transform.shape == (3, 4):
        bottom = transform.new_tensor([[0.0, 0.0, 0.0, 1.0]])
        return torch.cat([transform, bottom], dim=0)
    raise ValueError(f"Expected a 3x4 or 4x4 transform, got {tuple(transform.shape)}")


def _image_size_hw(depth_or_image: Tensor) -> Tuple[int, int]:
    if depth_or_image.ndim < 2:
        raise ValueError("Expected an image/depth tensor with at least 2 dimensions")
    return int(depth_or_image.shape[-2]), int(depth_or_image.shape[-1])


def project_lidar_to_image(
    points_lidar: Tensor,
    intrinsics: Tensor,
    lidar_to_camera: Tensor,
    image_hw: Tuple[int, int],
    eps: float = 1e-6,
) -> ProjectionResult:
    """Project LiDAR-frame points to image pixels.

    Args:
        points_lidar: ``[N, 3]`` point coordinates in LiDAR frame.
        intrinsics: ``[3, 3]`` camera matrix.
        lidar_to_camera: ``[4, 4]`` or ``[3, 4]`` transform from LiDAR to camera.
        image_hw: image size as ``(height, width)``.
    """

    if points_lidar.ndim != 2 or points_lidar.shape[-1] != 3:
        raise ValueError("points_lidar must have shape [N, 3]")

    h, w = image_hw
    transform = _as_4x4(lidar_to_camera).to(points_lidar)
    intrinsics = intrinsics.to(points_lidar)

    ones = torch.ones(points_lidar.shape[0], 1, device=points_lidar.device, dtype=points_lidar.dtype)
    points_h = torch.cat([points_lidar, ones], dim=-1)
    points_cam_h = points_h @ transform.T
    points_cam = points_cam_h[:, :3]
    depth = points_cam[:, 2]

    pixels_h = points_cam @ intrinsics.T
    uv = pixels_h[:, :2] / depth.clamp_min(eps).unsqueeze(-1)
    valid = (
        (depth > eps)
        & torch.isfinite(uv).all(dim=-1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= w - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= h - 1)
    )
    return ProjectionResult(uv=uv, depth=depth, valid=valid, points_cam=points_cam)


def sample_map_at_uv(values_hw: Tensor, uv: Tensor, mode: str = "bilinear") -> Tensor:
    """Sample a ``[H, W]`` map at pixel coordinates ``[u, v]``."""

    if values_hw.ndim != 2:
        raise ValueError("values_hw must have shape [H, W]")
    if uv.numel() == 0:
        return values_hw.new_empty((0,))

    h, w = _image_size_hw(values_hw)
    x = uv[:, 0] / max(w - 1, 1) * 2 - 1
    y = uv[:, 1] / max(h - 1, 1) * 2 - 1
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        values_hw.view(1, 1, h, w),
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def fit_metric_depth_from_sparse_projection(
    relative_depth: Tensor,
    projected_uv: Tensor,
    projected_metric_depth: Tensor,
    residual_sigma: float = 2.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Recover metric depth from relative depth using sparse LiDAR projections.

    The paper describes a scale-and-shift recovery from projected sparse LiDAR
    depth. This helper implements one robust least-squares pass:
    ``metric_depth = scale * relative_depth + shift``.
    """

    if projected_uv.numel() == 0:
        raise ValueError("At least one valid projected point is required")

    rel_samples = sample_map_at_uv(relative_depth, projected_uv)
    target = projected_metric_depth.to(rel_samples)
    valid = torch.isfinite(rel_samples) & torch.isfinite(target) & (target > 0)
    if valid.sum() < 2:
        raise ValueError("Need at least two valid projected depths to fit scale and shift")

    x = rel_samples[valid]
    y = target[valid]
    design = torch.stack([x, torch.ones_like(x)], dim=-1)
    solution = torch.linalg.lstsq(design, y).solution
    residual = y - design @ solution
    keep = residual.abs() <= residual.std(unbiased=False).clamp_min(1e-6) * residual_sigma

    if keep.sum() >= 2 and keep.sum() < keep.numel():
        design = design[keep]
        y = y[keep]
        solution = torch.linalg.lstsq(design, y).solution

    scale, shift = solution[0], solution[1]
    metric_depth = (relative_depth * scale + shift).clamp_min(1e-3)
    return metric_depth, scale, shift


def sample_depth_constrained_centers(
    metric_depth: Tensor,
    projected_depth: Tensor,
    num_centers: int,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Uniformly sample image cluster centers inside the LiDAR depth range."""

    if num_centers <= 0:
        return metric_depth.new_empty((0, 2))

    valid_depth = projected_depth[torch.isfinite(projected_depth) & (projected_depth > 0)]
    if valid_depth.numel() == 0:
        raise ValueError("projected_depth contains no valid positive depths")

    q = metric_depth.new_tensor([lower_percentile / 100.0, upper_percentile / 100.0])
    d_low, d_high = torch.quantile(valid_depth.to(metric_depth), q)
    mask = torch.isfinite(metric_depth) & (metric_depth >= d_low) & (metric_depth <= d_high)
    candidates_yx = mask.nonzero(as_tuple=False)
    if candidates_yx.numel() == 0:
        raise ValueError("No depth-constrained center candidates found")

    replace = candidates_yx.shape[0] < num_centers
    if replace:
        choice = torch.randint(
            candidates_yx.shape[0],
            (num_centers,),
            device=metric_depth.device,
            generator=generator,
        )
    else:
        choice = torch.randperm(
            candidates_yx.shape[0],
            device=metric_depth.device,
            generator=generator,
        )[:num_centers]

    centers_yx = candidates_yx[choice].to(metric_depth.dtype)
    return torch.stack([centers_yx[:, 1], centers_yx[:, 0]], dim=-1)


class PatchAffinityAggregator(nn.Module):
    """Depth-center image feature aggregation from the IPFP paper.

    The implementation follows the paper's local patch affinity idea: image
    features are mapped into similarity and value spaces, then each sampled
    cluster center aggregates its local patch with a learnable sigmoid weight.
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 32,
        patch_size: int = 9,
    ) -> None:
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.sim_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.beta0 = nn.Parameter(torch.tensor(1.0))
        self.beta1 = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_chw: Tensor, centers_uv: Tensor) -> Tensor:
        if image_chw.ndim == 3:
            image_bchw = image_chw.unsqueeze(0)
        elif image_chw.ndim == 4 and image_chw.shape[0] == 1:
            image_bchw = image_chw
        else:
            raise ValueError("image_chw must have shape [C, H, W] or [1, C, H, W]")

        if centers_uv.numel() == 0:
            return image_bchw.new_empty((0, self.value_proj.out_channels))

        feat = self.stem(image_bchw)
        sim = F.normalize(self.sim_proj(feat)[0], dim=0)
        value = self.value_proj(feat)[0]
        _, h, w = sim.shape

        radius = self.patch_size // 2
        sim_pad = F.pad(sim, (radius, radius, radius, radius), mode="replicate")
        value_pad = F.pad(value, (radius, radius, radius, radius), mode="replicate")

        rounded = centers_uv.round().long()
        xs = rounded[:, 0].clamp(0, w - 1)
        ys = rounded[:, 1].clamp(0, h - 1)

        aggregated = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            px, py = x + radius, y + radius
            center_sim = sim[:, y, x]
            center_value = value[:, y, x]
            patch_sim = sim_pad[:, py - radius : py + radius + 1, px - radius : px + radius + 1]
            patch_value = value_pad[:, py - radius : py + radius + 1, px - radius : px + radius + 1]

            patch_sim_flat = patch_sim.flatten(1).T
            patch_value_flat = patch_value.flatten(1).T
            similarity = (patch_sim_flat * center_sim.unsqueeze(0)).sum(dim=-1)
            weights = torch.sigmoid(self.beta0 * similarity + self.beta1)
            numerator = center_value + (weights.unsqueeze(-1) * patch_value_flat).sum(dim=0)
            denominator = 1.0 + weights.sum()
            aggregated.append(numerator / denominator.clamp_min(1e-6))

        return torch.stack(aggregated, dim=0)


def backproject_pixels_to_lidar(
    centers_uv: Tensor,
    center_depth: Tensor,
    intrinsics: Tensor,
    lidar_to_camera: Tensor,
) -> Tensor:
    """Back-project image pixels with metric depth into the LiDAR frame."""

    if centers_uv.numel() == 0:
        return centers_uv.new_empty((0, 3))

    intrinsics = intrinsics.to(centers_uv)
    transform = _as_4x4(lidar_to_camera).to(centers_uv)
    camera_to_lidar = torch.linalg.inv(transform)

    ones = torch.ones(centers_uv.shape[0], 1, device=centers_uv.device, dtype=centers_uv.dtype)
    uv1 = torch.cat([centers_uv, ones], dim=-1)
    rays = uv1 @ torch.linalg.inv(intrinsics).T
    points_cam = rays * center_depth.to(centers_uv).unsqueeze(-1)
    points_cam_h = torch.cat([points_cam, ones], dim=-1)
    points_lidar_h = points_cam_h @ camera_to_lidar.T
    return points_lidar_h[:, :3]


class IPFPFeatureBackProjector(nn.Module):
    """Minimal IPFP feature aggregation and 3D back-projection module."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 32,
        patch_size: int = 9,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        discard_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregator = PatchAffinityAggregator(
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            patch_size=patch_size,
        )
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.discard_probability = discard_probability

    def forward(
        self,
        image_chw: Tensor,
        metric_depth_hw: Tensor,
        points_lidar: Tensor,
        intrinsics: Tensor,
        lidar_to_camera: Tensor,
        num_centers: int,
        relative_depth_hw: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        image_hw = _image_size_hw(metric_depth_hw)
        projection = project_lidar_to_image(points_lidar, intrinsics, lidar_to_camera, image_hw)
        projected_uv = projection.uv[projection.valid]
        projected_depth = projection.depth[projection.valid]

        if relative_depth_hw is not None:
            metric_depth_hw, scale, shift = fit_metric_depth_from_sparse_projection(
                relative_depth_hw, projected_uv, projected_depth
            )
        else:
            scale = shift = None

        centers_uv = sample_depth_constrained_centers(
            metric_depth_hw,
            projected_depth,
            num_centers=num_centers,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            generator=generator,
        )
        center_depth = sample_map_at_uv(metric_depth_hw, centers_uv)
        image_features = self.aggregator(image_chw, centers_uv)
        coord = backproject_pixels_to_lidar(
            centers_uv,
            center_depth,
            intrinsics,
            lidar_to_camera,
        )

        keep = torch.ones(coord.shape[0], dtype=torch.bool, device=coord.device)
        if self.training and self.discard_probability > 0:
            keep = torch.rand(coord.shape[0], device=coord.device, generator=generator) >= self.discard_probability
            coord = coord[keep]
            image_features = image_features[keep]
            centers_uv = centers_uv[keep]
            center_depth = center_depth[keep]

        return {
            "coord": coord,
            "feat": image_features,
            "uv": centers_uv,
            "depth": center_depth,
            "keep_mask": keep,
            "metric_depth": metric_depth_hw,
            "scale": scale,
            "shift": shift,
        }
