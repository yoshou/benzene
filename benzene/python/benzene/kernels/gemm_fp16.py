"""The fp16 HMX GEMM (benzene/dsp/kernels/gemm_fp16): its weight stream."""

from __future__ import annotations

import numpy as np


def gemm_fp16_stream(w: np.ndarray, b: np.ndarray) -> bytes:
    """fp32 weight [N, K] (N, K multiples of 32) + bias [N] -> the fp16 HMX GEMM stream (benzene dsp/kernels/gemm_fp16): per
    32-column block K/32 weight tiles of 2 KB, byte (k/2)*128 + c*4 + (k%2)*2, then a 2 KB store block whose first 32
    words are (fp16 scale 1.0 | fp16 bias << 16)."""
    n, k = w.shape
    assert n % 32 == 0 and k % 32 == 0
    h = w.astype(np.float16)
    out = []
    for cb in range(n // 32):
        blk = h[cb * 32 : (cb + 1) * 32]  # [c, K]
        tiles = blk.reshape(32, k // 32, 16, 2).transpose(1, 2, 0, 3)  # kt, k/2, c, k%2
        out.append(np.ascontiguousarray(tiles).tobytes())
        words = np.zeros(512, np.uint32)
        bb = b[cb * 32 : (cb + 1) * 32].astype(np.float16).view(np.uint16).astype(np.uint32)
        words[:32] = 0x3C00 | (bb << 16)
        out.append(words.tobytes())
    return b"".join(out)
