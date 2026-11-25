"""A 3D Gaussian rasterizer in PyTorch with the semantics of gsplat 1.4.0's rasterization() (the CUDA kernels of
fully_fused_projection, isect_tiles and rasterize_to_pixels, spherical_harmonics.cuh) in AnySplat's decoder settings
(src/model/decoder/decoder_splatting_cuda.py): classic mode, eps2d 0.3, near plane 1e-10, far plane 1e10, radius_clip
0.1, 16 x 16 tiles, colours from SH degree 4 (+ 0.5, at least 0), a white background, the image clamped to [0, 1].
gsplat's CUDA build does not run on the ROCm GPUs here.

  projection   camera-space covariance R S R^T, the EWA Jacobian with the projected point clamped to 1.3 fields of view,
               2D covariance + 0.3 I, its inverse (the conic), radius ceil(3 sqrt(largest eigenvalue)); culled: depth
               outside [near, far], det <= 0, radius <= radius_clip, the 3-sigma box outside the image
  tiles        every tile the box [mean +- radius] touches; per tile the Gaussians sorted by depth (stable: ties in
               Gaussian order)
  pixels       at pixel centres (x + 0.5, y + 0.5): alpha = min(0.999, opacity exp(-sigma)), skipped when sigma < 0 or
               alpha < 1/255; front to back the transmittance T, and a pixel ends before the Gaussian that would take
               T to 1e-4 or below; colour = sum alpha T c + T background
Compositing is vectorized per chunk of tiles: a tile's transmittances are its cumulative sums of log(1 - alpha) (fp64).
Per-Gaussian 3-vector and 3x3 products are broadcast sums: on the ROCm GPUs here a matmul of a tall matrix with few
columns ([1M, 3] @ [3, 3]) or a batched matmul of more than 65536 matrices is wrong for many rows.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .gaussians import Gaussians

TILE = 16
NEAR, FAR = 1e-10, 1e10
EPS2D = 0.3
RADIUS_CLIP = 0.1
CHUNK_ENTRIES = 1 << 16  # tile entries composited at once (a tile with more takes a chunk of its own)


def sh_basis(dirs: torch.Tensor) -> torch.Tensor:
    """The 25 SH basis values of degree <= 4 at directions [M, 3] (normalized here), gsplat's signs and order."""
    x, y, z = (dirs * torch.rsqrt((dirs * dirs).sum(-1, keepdim=True))).unbind(-1)
    z2 = z * z
    fC1, fS1 = x * x - y * y, 2 * x * y
    p6 = 0.9461746957575601 * z2 - 0.3153915652525201
    t0b = -1.092548430592079 * z
    t0c = -2.285228997322329 * z2 + 0.4570457994644658
    t1b = 1.445305721320277 * z
    fC2, fS2 = x * fC1 - y * fS1, x * fS1 + y * fC1
    p12 = z * (1.865881662950577 * z2 - 1.119528997770346)
    t0d = z * (-4.683325804901025 * z2 + 2.007139630671868)
    t1c = 3.31161143515146 * z2 - 0.47308734787878
    t2b = -1.770130769779931 * z
    fC3, fS3 = x * fC2 - y * fS2, x * fS2 + y * fC2
    return torch.stack([
        torch.full_like(x, 0.2820947917738781),
        -0.48860251190292 * y, 0.48860251190292 * z, -0.48860251190292 * x,
        0.5462742152960395 * fS1, t0b * y, p6, t0b * x, 0.5462742152960395 * fC1,
        -0.5900435899266435 * fS2, t1b * fS1, t0c * y, p12, t0c * x, t1b * fC1, -0.5900435899266435 * fC2,
        0.6258357354491763 * fS3, t2b * fS2, t1c * fS1, t0d * y, 1.984313483298443 * z * p12 - 1.006230589874905 * p6,
        t0d * x, t1c * fC1, t2b * fC2, 0.6258357354491763 * fC3,
    ], dim=-1)


def _products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [..., n, k] @ b [..., k, m] as a broadcast sum (no batched matmul)."""
    return (a[..., :, :, None] * b[..., None, :, :]).sum(-2)


@torch.no_grad()
def project(g: Gaussians, w2c: torch.Tensor, K: torch.Tensor, height: int, width: int) -> dict:
    """The visible Gaussians of one camera (w2c [3, 4], K [3, 3]): their indices `ids` and means2d [M, 2], conics
    [M, 3] (a, b, c of 0.5 (a dx^2 + c dy^2) + b dx dy), radii [M], depths [M], colors [M, 3], opacities [M]."""
    R, t = w2c[:3, :3].float(), w2c[:3, 3].float()
    mc = (g.means[:, None, :] * R).sum(-1) + t
    x, y, z = mc.unbind(-1)
    covc = _products(_products(R[None], g.covariances()), R.T[None])
    fx, fy, cx, cy = K[0, 0].float(), K[1, 1].float(), K[0, 2].float(), K[1, 2].float()
    tan_fovx, tan_fovy = 0.5 * width / fx, 0.5 * height / fy
    lim_x_pos, lim_x_neg = (width - cx) / fx + 0.3 * tan_fovx, cx / fx + 0.3 * tan_fovx
    lim_y_pos, lim_y_neg = (height - cy) / fy + 0.3 * tan_fovy, cy / fy + 0.3 * tan_fovy
    rz = 1.0 / z
    rz2 = rz * rz
    tx = z * torch.minimum(lim_x_pos, torch.maximum(-lim_x_neg, x * rz))
    ty = z * torch.minimum(lim_y_pos, torch.maximum(-lim_y_neg, y * rz))
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx * rz, zero, -fx * tx * rz2], -1), torch.stack([zero, fy * rz, -fy * ty * rz2], -1)],
                    dim=1)  # [N, 2, 3]
    cov2 = _products(_products(J, covc), J.transpose(1, 2))
    a, b, c = cov2[:, 0, 0] + EPS2D, cov2[:, 0, 1], cov2[:, 1, 1] + EPS2D
    det = a * c - b * b
    half = 0.5 * (a + c)
    radius = torch.ceil(3.0 * torch.sqrt(half + torch.sqrt(torch.clamp_min(half * half - det, 0.01))))
    mx, my = fx * x * rz + cx, fy * y * rz + cy
    visible = ((z >= NEAR) & (z <= FAR) & (det > 0) & (radius > RADIUS_CLIP)
               & ~((mx + radius <= 0) | (mx - radius >= width) | (my + radius <= 0) | (my - radius >= height)))
    ids = visible.nonzero()[:, 0]
    det = det[ids]
    campos = -R.T @ t
    colors = (sh_basis(g.means[ids] - campos)[:, None, :] * g.harmonics[ids]).sum(-1)
    return {"ids": ids, "means2d": torch.stack([mx[ids], my[ids]], -1),
            "conics": torch.stack([c[ids] / det, -b[ids] / det, a[ids] / det], -1), "radii": radius[ids],
            "depths": z[ids], "colors": torch.clamp_min(colors + 0.5, 0.0), "opacities": g.opacities[ids]}


@torch.no_grad()
def render_view(g: Gaussians, w2c: torch.Tensor, K: torch.Tensor, height: int, width: int,
                background=(1.0, 1.0, 1.0)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One camera -> image [3, H, W] in [0, 1], alpha [H, W], accumulated depth [H, W]."""
    device = g.means.device
    p = project(g, w2c, K, height, width)
    tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
    tiles = tw * th
    mx, my = p["means2d"].unbind(-1)
    tile_radius = p["radii"] / TILE
    x0 = torch.floor(mx / TILE - tile_radius).clamp(0, tw).long()
    x1 = torch.ceil(mx / TILE + tile_radius).clamp(0, tw).long()
    y0 = torch.floor(my / TILE - tile_radius).clamp(0, th).long()
    y1 = torch.ceil(my / TILE + tile_radius).clamp(0, th).long()
    counts = (x1 - x0) * (y1 - y0)
    gid = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    local = torch.arange(len(gid), device=device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    span = (x1 - x0)[gid]
    tile = (y0[gid] + local // span) * tw + x0[gid] + local % span
    key = (tile << 32) | p["depths"][gid].view(torch.int32).long()
    key, order = torch.sort(key, stable=True)
    entry_tile, entry_gauss = key >> 32, gid[order]
    tile_counts = torch.bincount(entry_tile, minlength=tiles)
    tile_start = torch.cumsum(tile_counts, 0) - tile_counts

    color = torch.zeros(tiles, TILE * TILE, 3, device=device)
    log_t = torch.zeros(tiles, TILE * TILE, dtype=torch.float64, device=device)
    depth = torch.zeros(tiles, TILE * TILE, device=device)
    lx = (torch.arange(TILE * TILE, device=device) % TILE).float() + 0.5
    ly = (torch.arange(TILE * TILE, device=device) // TILE).float() + 0.5
    ends = torch.cumsum(tile_counts, 0).cpu().numpy()
    first = 0
    while first < tiles:  # chunks of whole tiles
        last = max(first + 1, int(np.searchsorted(ends, ends[first - 1] + CHUNK_ENTRIES if first else CHUNK_ENTRIES,
                                                   "right")))
        last = min(last, tiles)
        e0, e1 = (int(ends[first - 1]) if first else 0), int(ends[last - 1])
        if e1 > e0:
            et, eg = entry_tile[e0:e1], entry_gauss[e0:e1]
            px = ((et % tw) * TILE).float()[:, None] + lx
            py = ((et // tw) * TILE).float()[:, None] + ly
            dx, dy = mx[eg][:, None] - px, my[eg][:, None] - py
            ca, cb, cc = p["conics"][eg].unbind(-1)
            sigma = 0.5 * (ca[:, None] * dx * dx + cc[:, None] * dy * dy) + cb[:, None] * dx * dy
            del dx, dy
            alpha = torch.clamp_max(p["opacities"][eg][:, None] * torch.exp(-sigma), 0.999)
            ok = (sigma >= 0) & (alpha >= 1.0 / 255.0)
            del sigma
            alpha = torch.where(ok, alpha, 0.0)
            log1m = torch.log1p(-alpha.double())
            inclusive = torch.cumsum(log1m, 0)
            start = tile_start[et] - e0
            inclusive -= torch.where(start[:, None] > 0, inclusive[(start - 1).clamp_min(0)], 0.0)
            used = ok & (inclusive > math.log(1e-4))
            weight = torch.where(used, alpha * torch.exp(inclusive - log1m).float(), 0.0)
            del inclusive
            index = et - first
            color[first:last].index_add_(0, index, weight[:, :, None] * p["colors"][eg][:, None, :])
            log_t[first:last].index_add_(0, index, torch.where(used, log1m, 0.0))
            depth[first:last].index_add_(0, index, weight * p["depths"][eg][:, None])
        first = last
    transmittance = torch.exp(log_t).float()
    image = color + transmittance[:, :, None] * torch.tensor(background, device=device)

    def untile(v):  # [tiles, 256, ...] -> [H, W, ...]
        v = v.reshape(th, tw, TILE, TILE, *v.shape[2:]).transpose(1, 2)
        return v.reshape(th * TILE, tw * TILE, *v.shape[4:])[:height, :width]

    return untile(image).permute(2, 0, 1).clamp(0, 1), untile(1 - transmittance), untile(depth)


@torch.no_grad()
def render(g: Gaussians, extrinsics: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int,
           background=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """Images [V, 3, H, W] of cameras extrinsics [V, 3, 4] (world to camera), intrinsics [V, 3, 3] (pixels)."""
    return torch.stack([render_view(g, extrinsics[v], intrinsics[v], height, width, background)[0]
                        for v in range(extrinsics.shape[0])])
