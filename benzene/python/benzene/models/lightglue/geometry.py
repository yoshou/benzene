"""Two-view geometry for scoring matches against ground-truth poses (no OpenCV in the venv).

epipolar_error: SuperGlue's symmetric epipolar distance with a unit-length translation (Mip-NeRF 360 poses are in
COLMAP units); a match is correct below 5e-4. estimate_pose: RANSAC over the 8-point essential matrix in normalized
image coordinates (16384 hypotheses, Sampson distance below 1 pixel, 4 rounds of least-squares refits on the inliers),
the pose by the cheirality test of the four decompositions; deterministic for a given seed. pose_auc as SuperGlue.
"""

from __future__ import annotations

import numpy as np
import torch

EPI_TH = 5e-4


def _norm(k: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return (k - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]


def _skew(t: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=t.dtype)
    return torch.stack([torch.stack([z, -t[2], t[1]]), torch.stack([t[2], z, -t[0]]), torch.stack([-t[1], t[0], z])])


def epipolar_error(k0: torch.Tensor, k1: torch.Tensor, K0, K1, T: torch.Tensor) -> torch.Tensor:
    x0 = torch.cat([_norm(k0.double(), K0.double()), torch.ones(len(k0), 1, dtype=torch.float64)], 1)
    x1 = torch.cat([_norm(k1.double(), K1.double()), torch.ones(len(k1), 1, dtype=torch.float64)], 1)
    t = T[:3, 3].double()
    E = _skew(t / t.norm()) @ T[:3, :3].double()
    Ep0, Etp1 = x0 @ E.T, x1 @ E
    p1Ep0 = (x1 * Ep0).sum(-1)
    return p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))


def _rows(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:  # x1^T E x0 = 0, E row-major
    u0, v0, u1, v1 = x0[..., 0], x0[..., 1], x1[..., 0], x1[..., 1]
    one = torch.ones_like(u0)
    return torch.stack([u1 * u0, u1 * v0, u1, v1 * u0, v1 * v0, v1, u0, v0, one], -1)


def _essential(A: torch.Tensor) -> torch.Tensor:
    """Least-squares E of the rows A [..., m, 9], projected to the essential manifold (singular values 1, 1, 0)."""
    _, _, Vh = torch.linalg.svd(A, full_matrices=True)
    E = Vh[..., -1, :].reshape(*A.shape[:-2], 3, 3)
    U, _, Vh = torch.linalg.svd(E)
    d = torch.tensor([1.0, 1.0, 0.0], dtype=E.dtype)
    return U @ torch.diag_embed(d.expand(*E.shape[:-2], 3)) @ Vh


def _sampson(E: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    h0 = torch.cat([x0, torch.ones_like(x0[:, :1])], 1)  # [n, 3]
    h1 = torch.cat([x1, torch.ones_like(x1[:, :1])], 1)
    Ex0 = torch.einsum("...ij,nj->...ni", E, h0)
    Etx1 = torch.einsum("...ji,nj->...ni", E, h1)
    num = (h1 * Ex0).sum(-1)
    den = Ex0[..., 0] ** 2 + Ex0[..., 1] ** 2 + Etx1[..., 0] ** 2 + Etx1[..., 1] ** 2
    return num ** 2 / den.clamp_min(1e-30)


def _triangulate_depths(R, t, x0, x1) -> tuple[torch.Tensor, torch.Tensor]:
    P0 = torch.cat([torch.eye(3, dtype=R.dtype), torch.zeros(3, 1, dtype=R.dtype)], 1)
    P1 = torch.cat([R, t[:, None]], 1)
    A = torch.stack([x0[:, :1] * P0[2] - P0[0], x0[:, 1:2] * P0[2] - P0[1],
                     x1[:, :1] * P1[2] - P1[0], x1[:, 1:2] * P1[2] - P1[1]], 1)  # [m, 4, 4]
    X = torch.linalg.svd(A)[2][:, -1]
    X = X[:, :3] / X[:, 3:4]
    return X[:, 2], (X @ R.T + t)[:, 2]


def estimate_pose(k0, k1, K0, K1, thr_px: float = 1.0, iters: int = 16384, seed: int = 0):
    """(R, t) of camera 1 relative to camera 0 from matched pixels, or None."""
    n = len(k0)
    if n < 8:
        return None
    K0, K1 = K0.double(), K1.double()
    x0, x1 = _norm(k0.double(), K0), _norm(k1.double(), K1)
    thr = (thr_px / torch.cat([K0[[0, 1], [0, 1]], K1[[0, 1], [0, 1]]]).mean()) ** 2
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(iters, n, generator=g).argsort(1)[:, :8]
    E = _essential(_rows(x0[idx], x1[idx]))
    inl = _sampson(E, x0, x1) < thr  # [iters, n]
    cnt = inl.sum(1)
    b = int(cnt.argmax())
    best_E, best = E[b], inl[b]
    for _ in range(4):
        if best.sum() < 8:
            break
        E2 = _essential(_rows(x0[best], x1[best]))
        m2 = _sampson(E2, x0, x1) < thr
        if m2.sum() < best.sum():
            break
        best_E, best = E2, m2
    if best.sum() < 5:
        return None
    U, _, Vh = torch.linalg.svd(best_E)
    if torch.det(U) < 0:
        U = -U
    if torch.det(Vh) < 0:
        Vh = -Vh
    W = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    a, c = x0[best], x1[best]
    sol, top = None, -1
    for R in (U @ W @ Vh, U @ W.T @ Vh):
        for t in (U[:, 2], -U[:, 2]):
            d0, d1 = _triangulate_depths(R, t, a, c)
            good = int(((d0 > 0) & (d1 > 0)).sum())
            if good > top:
                sol, top = (R, t), good
    return sol


def pose_error(R, t, T: torch.Tensor) -> float:
    Rg, tg = T[:3, :3].double(), T[:3, 3].double()
    er = torch.rad2deg(torch.arccos(((torch.trace(R.T @ Rg) - 1) / 2).clamp(-1, 1))).item()
    et = torch.rad2deg(torch.arccos((t @ tg / (t.norm() * tg.norm())).clamp(-1, 1))).item()
    return max(er, min(et, 180 - et))  # translation sign ambiguity of E


def pose_auc(errors, thresholds=(5, 10, 20)) -> list[float]:
    e = np.sort(np.asarray(errors, np.float64))
    rec = (np.arange(len(e)) + 1) / len(e)
    e, rec = np.r_[0.0, e], np.r_[0.0, rec]
    out = []
    for th in thresholds:
        i = np.searchsorted(e, th)
        r, x = np.r_[rec[:i], rec[i - 1]], np.r_[e[:i], th]
        out.append(float(np.trapezoid(r, x=x) / th))
    return out
