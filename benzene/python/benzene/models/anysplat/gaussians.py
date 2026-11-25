"""AnySplat's Gaussians: what its encoder makes of the network's predictions (src/model/encoder/anysplat.py,
common/gaussian_adapter.py of the AnySplat checkout), and the PLY file the board writes.

  cameras         the pose encoding decoded as AnySplat's copy of vggt's pose_encoding_to_extri_intri: focal lengths
                  (size / 2) / (tan(fov / 2) + 1e-3)
  unproject       the depth map through the cameras to world points (pixel (u, v) at its integer index)
  voxelize        the pixels fused per voxel (VOXEL_SIZE): points and features weighted by the confidences' softmax
                  in the voxel
  adapt           UnifiedGaussianAdapter (sh_degree 4): opacity sigmoid (the opacity mapping at step 0 is the
                  identity), scale 0.001 softplus clamped to 0.3, unit quaternion (x, y, z, w), SH times the degree
                  mask
  write_ply       the 3DGS layout: x y z nx ny nz f_dc_0..2 f_rest_0..71 opacity scale_0..2 rot_0..3 (float32), with
                  opacity as its logit, log scales, the quaternion (w, x, y, z), the SH channel-major (all 4 degrees)

Per-Gaussian 3x3 products are broadcast sums, not batched matmuls: on the ROCm GPUs here a batched matmul or inverse of
more than 65536 matrices is wrong past the 65536th.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .network import (RAW_CHANNELS, RAW_CONFIDENCE, RAW_FEATURES, RAW_HARMONICS, RAW_OPACITY, RAW_ROTATION, RAW_SCALES,
                      SH_BASES, SH_DEGREE)

VOXEL_SIZE = 0.002


@dataclass
class Gaussians:
    means: torch.Tensor  # [N, 3]
    scales: torch.Tensor  # [N, 3]
    rotations: torch.Tensor  # [N, 4] unit quaternions (x, y, z, w)
    harmonics: torch.Tensor  # [N, 3, SH_BASES]
    opacities: torch.Tensor  # [N]

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device) -> Gaussians:
        return Gaussians(self.means.to(device), self.scales.to(device), self.rotations.to(device),
                         self.harmonics.to(device), self.opacities.to(device))

    def covariances(self) -> torch.Tensor:
        """R S S^T R^T [N, 3, 3] (AnySplat's build_covariance)."""
        RS = quaternion_to_matrix(self.rotations) * self.scales[:, None, :]
        return (RS[:, :, None, :] * RS[:, None, :, :]).sum(-1)


def quaternion_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation matrices of quaternions (x, y, z, w) (AnySplat's quaternion_to_matrix: pytorch3d's in scipy order)."""
    i, j, k, r = torch.unbind(q, dim=-1)
    two_s = 2 / ((q * q).sum(dim=-1) + eps)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))


def sh_mask(device=None) -> torch.Tensor:
    """The adapter's SH weight per basis: 1 for the DC term, 0.1 * 0.25^degree above."""
    mask = torch.ones(SH_BASES, device=device)
    for degree in range(1, SH_DEGREE + 1):
        mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
    return mask


def cameras(pose_enc: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
    """pose_enc [S, 9] -> world-to-camera extrinsics [S, 3, 4] and intrinsics [S, 3, 3] as AnySplat decodes them."""
    from vggt.utils.rotation import quat_to_mat

    H, W = hw
    T, quat, fov_h, fov_w = pose_enc[..., :3], pose_enc[..., 3:7], pose_enc[..., 7], pose_enc[..., 8]
    extrinsics = torch.cat([quat_to_mat(quat), T[..., None]], dim=-1)
    intrinsics = torch.zeros(pose_enc.shape[:-1] + (3, 3), device=pose_enc.device, dtype=pose_enc.dtype)
    intrinsics[..., 0, 0] = (W / 2.0) / (torch.tan(fov_w / 2.0) + 1e-3)
    intrinsics[..., 1, 1] = (H / 2.0) / (torch.tan(fov_h / 2.0) + 1e-3)
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def camera_to_world(extrinsics: torch.Tensor) -> torch.Tensor:
    """[S, 3, 4] world-to-camera -> [S, 4, 4] camera-to-world (vggt closed_form_inverse_se3)."""
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3:]
    out = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).repeat(extrinsics.shape[0], 1, 1)
    out[:, :3, :3] = R.transpose(1, 2)
    out[:, :3, 3:] = -R.transpose(1, 2) @ t
    return out


def unproject(depth: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """depth [S, H, W] -> world points [S, H, W, 3] (AnySplat's batchify_unproject_depth_map_to_point_map)."""
    S, H, W = depth.shape
    fu, fv = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cu, cv = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    u = torch.arange(W, device=depth.device)[None, None, :].expand(S, H, W)
    v = torch.arange(H, device=depth.device)[None, :, None].expand(S, H, W)
    cam = torch.stack(((u - cu) * depth / fu, (v - cv) * depth / fv, depth), dim=-1)
    homogeneous = torch.cat((cam, torch.ones_like(cam[..., :1])), dim=-1).flatten(1, 2)
    world = torch.bmm(camera_to_world(extrinsics), homogeneous.transpose(1, 2)).transpose(1, 2)[:, :, :3]
    return world.reshape(S, H, W, 3)


def voxelize(points: torch.Tensor, features: torch.Tensor, confidence: torch.Tensor,
             voxel_size: float = VOXEL_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """AnySplat's voxelizaton_with_fusion: points [P, 3], features [P, C], confidence [P] -> per voxel, in the order of
    torch.unique of the integer voxel coordinates round(p / voxel_size), the points and features weighted by
    exp(c - max) / (sum + 1e-6) over the voxel's pixels."""
    voxels = (points / voxel_size).round().int()
    _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
    n = int(inverse.max()) + 1
    peak = torch.full((n,), -torch.inf, device=points.device).scatter_reduce(0, inverse, confidence, "amax")
    e = torch.exp(confidence - peak[inverse])
    total = torch.zeros(n, device=points.device).index_add_(0, inverse, e)
    weights = (e / (total[inverse] + 1e-6))[:, None]
    fused_points = torch.zeros(n, 3, device=points.device).index_add_(0, inverse, points * weights)
    fused_features = torch.zeros(n, features.shape[1], device=points.device).index_add_(0, inverse, features * weights)
    return fused_points, fused_features


def adapt(points: torch.Tensor, features: torch.Tensor) -> Gaussians:
    """Voxel points [N, 3] and features [N, RAW_CHANNELS - 1] (RAW_FEATURES) -> Gaussians."""
    density = features[:, RAW_OPACITY].sigmoid()
    opacities = 0.5 * (1 - (1 - density) + density)  # map_pdf_to_opacity with exponent 2^0
    scales = (0.001 * F.softplus(features[:, RAW_SCALES])).clamp_max(0.3)
    rotations = features[:, RAW_ROTATION]
    rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
    harmonics = features[:, RAW_HARMONICS].reshape(-1, 3, SH_BASES) * sh_mask(features.device)
    return Gaussians(points.float(), scales.float(), rotations.float(), harmonics.float(), opacities.float())


def make_gaussians(pose_enc: torch.Tensor, depth: torch.Tensor, raw: torch.Tensor,
                   voxel_size: float = VOXEL_SIZE) -> Gaussians:
    """The Gaussians of one sample's predictions: pose_enc [S, 9], depth [S, H, W], raw head outputs
    [S, H, W, RAW_CHANNELS] (computed in fp32)."""
    extrinsics, intrinsics = cameras(pose_enc.float(), depth.shape[-2:])
    points = unproject(depth.float(), extrinsics, intrinsics).reshape(-1, 3)
    raw = raw.reshape(-1, RAW_CHANNELS).float()
    fused_points, fused = voxelize(points, raw[:, RAW_FEATURES], raw[:, RAW_CONFIDENCE], voxel_size)
    return adapt(fused_points, fused)


def ply_properties() -> list[str]:
    rest = 3 * (SH_BASES - 1)
    return (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(rest)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def write_ply(path: str | Path, g: Gaussians) -> None:
    n = len(g)
    p = g.opacities.double().clamp(1e-12, 1 - 1e-12)
    columns = [g.means.double(), torch.zeros(n, 3, dtype=torch.float64, device=g.means.device),
               g.harmonics[:, :, 0].double(), g.harmonics[:, :, 1:].reshape(n, -1).double(), (p / (1 - p)).log()[:, None],
               g.scales.double().log(), g.rotations[:, [3, 0, 1, 2]].double()]
    data = torch.cat(columns, dim=1).float().cpu().numpy()
    header = "\n".join(["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
                       + [f"property float {name}" for name in ply_properties()] + ["end_header"]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(data, "<f4").tobytes())


def read_ply(path: str | Path) -> Gaussians:
    """A PLY of write_ply's layout with the SH of any degree up to SH_DEGREE (the board's gaussians.ply, --sh-degree);
    the SH it leaves out are zeros."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode().splitlines()
    names = [line.split()[-1] for line in lines if line.startswith("property")]
    rest = sum(name.startswith("f_rest_") for name in names) // 3  # per channel
    degree = round((rest + 1) ** 0.5) - 1
    assert (degree + 1) ** 2 - 1 == rest and degree <= SH_DEGREE, f"{path}: {3 * rest} f_rest values"
    expected = ply_properties()
    expected = expected[:9] + [f"f_rest_{i}" for i in range(3 * rest)] + expected[9 + 3 * (SH_BASES - 1) :]
    assert names == expected, f"{path}: unexpected properties"
    n = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
    a = torch.from_numpy(np.frombuffer(raw, "<f4", n * len(names), end).reshape(n, len(names)).copy())
    harmonics = torch.zeros(n, 3, SH_BASES)
    harmonics[:, :, 0] = a[:, 6:9]
    harmonics[:, :, 1 : 1 + rest] = a[:, 9 : 9 + 3 * rest].reshape(n, 3, rest)
    o = 9 + 3 * rest
    return Gaussians(a[:, 0:3], a[:, o + 1 : o + 4].exp(), a[:, [o + 5, o + 6, o + 7, o + 4]], harmonics,
                     a[:, o].sigmoid())
