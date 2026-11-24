"""The decode GEMVs (benzene/dsp/kernels/gemv): the Q4_0 weight stream. (The int8 GEMV reads the W8A8 GEMM's weight
stream, gemm_w8a8.py.)"""

from __future__ import annotations

import numpy as np
import torch


def q4_codes(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cached q4_0 weight record -> (codes int8 [N, K] in [-8, 7], block scales fp16 [N, K/32])."""
    w = rec["w"].float()
    n, k = w.shape
    assert k % 64 == 0 and rec["fmt"] == "q4_0" and rec["cols_padded"] == k
    sc = rec["scales"].float().reshape(n, k // 32)
    se = sc.repeat_interleave(32, 1)
    q = torch.where(se != 0, torch.round(w / torch.where(se != 0, se, 1)), 0)
    assert torch.equal(q * se, w), "weights are not integer multiples of the block scales"
    assert q.min() >= -8 and q.max() <= 7
    s16 = sc.numpy().astype(np.float16)
    assert np.array_equal(s16.astype(np.float32), sc.numpy()), "block scales are not fp16"
    return q.to(torch.int8).numpy(), s16


def gemv_q4_stream(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """int8 [N, K] in [-8, 7] + fp16 [N, K/32] -> the Q4_0 GEMV stream (benzene dsp/kernels/gemv WeightFormat::kQ4_0):
    per 32-row group, per 32-k block 512 B of codes + 8 (byte (kk/8)*128 + r*4 + kk%4: low nibble kk%8 < 4, high
    nibble kk%8 >= 4), then K/64 vectors of block scales (fp16 lane 2r + (b&1) of vector b/2)."""
    n, k = codes.shape
    assert n % 32 == 0 and k % 64 == 0
    c = (codes.astype(np.int16) + 8).astype(np.uint8).reshape(n // 32, 32, k // 32, 4, 2, 4)  # g, r, b, h, half, j
    by = (c[:, :, :, :, 0, :] | (c[:, :, :, :, 1, :] << 4)).transpose(0, 2, 3, 1, 4)  # g, b, h, r, j
    sc = scales.astype(np.float16).reshape(n // 32, 32, k // 64, 2).transpose(0, 2, 1, 3)  # g, bp, r, parity
    out = []
    for g in range(n // 32):
        out.append(np.ascontiguousarray(by[g]).tobytes())
        out.append(np.ascontiguousarray(sc[g]).tobytes())
    return b"".join(out)
