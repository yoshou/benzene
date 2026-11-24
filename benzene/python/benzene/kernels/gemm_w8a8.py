"""The W8A8 GEMM (benzene/dsp/kernels/gemm_w8a8): its weight stream.

Weight stream, per 32-output-column block:
  K/32 int8 weight tiles of 1 KB, byte (k/4)*128 + c*4 + k%4 (k = input index within the tile,
  c = output column within the block), then 4 bias blocks of 256 B: 32 store-scale words
  (0x6000, 0x4000, 0x2000, 0x0400: the four 8-bit slices of the int32 accumulator) followed by
  32 bias2 words = -128 * column sum (removes the +128 offset of the uint8 activations).
Column scales (fp32, one per output) travel separately.

The activation side (what the producer kernels emit) is per-token symmetric int8 of x * inv_s with
the device's rounding: d = amax / 127, q = floor(x * (1 / d) + 128.5) clamped to [1, 255], minus 128.
"""

from __future__ import annotations

import numpy as np
import torch

SLICE_SCALE = (0x6000, 0x4000, 0x2000, 0x0400)


def int8_codes(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cached int8_pc weight record (recipes.quantize_weight) -> (codes int8 [N, K], scales fp32 [N])."""
    w = rec["w"].float()
    sc = rec["scales"].float().reshape(w.shape[0], -1)[:, 0]
    # all-zero rows (e.g. dead fc1 neurons) carry scale 0
    q = torch.where(sc[:, None] > 0, torch.round(w / torch.where(sc > 0, sc, 1)[:, None]), 0)
    assert torch.equal(q * sc[:, None], w), "weights are not integer multiples of the row scales"
    assert q.abs().max() <= 127
    return q.to(torch.int8).numpy(), sc.numpy().astype(np.float32)


def gemm_w8a8_stream(codes: np.ndarray) -> bytes:
    """int8 [N, K] (N, K multiples of 32) -> the GEMM weight stream bytes."""
    n, k = codes.shape
    assert n % 32 == 0 and k % 32 == 0
    out = []
    for cb in range(n // 32):
        blk = codes[cb * 32 : (cb + 1) * 32]  # [c, K]
        tiles = blk.reshape(32, k // 32, 8, 4).transpose(1, 2, 0, 3)  # kt, k/4, c, k%4
        out.append(np.ascontiguousarray(tiles).astype(np.int8).tobytes())
        colsum = blk.astype(np.int64).sum(1)
        bias2 = ((-128 * colsum) & 0xFFFFFFFF).astype(np.uint32)
        for s in SLICE_SCALE:
            out.append(np.concatenate([np.full(32, s, np.uint32), bias2]).tobytes())
    return b"".join(out)
