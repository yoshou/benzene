"""The int8 HMX convolution (benzene/dsp/kernels/convolution_int8): its weight stream.

Weight stream, per 32-output-channel block:
  KT = ceil(K / 32) weight tiles of 2 KB: 1 KB int8, byte (k/4)*128 + c*4 + k%4, then 1 KB of zeros (weight.b:deep)
  a 2 KB block (every HMX operand on a 2 KB boundary): the 4 accumulator slice store blocks of 256 B (32 scale words
  0x6000, 0x4000, 0x2000, 0x0400 + 32 bias2 words: -zero_point * column sum, which makes the stored sum that of
  (code - zero point)), then
     uint8 output: 4 multiplier + 4 bias vectors (fp32) in the word order of a reassembled tile line (lane i of vector
                   k: column 8k + i/4): q = trunc(clamp(fp32(sum) * m + b, 0, 255.5)), m = s_acc / s_y, b = bias / s_y + 0.5
     fp32 output:  32 column scales + 32 biases: y = fp32(sum) * scale + bias
K is tap-major (k = tap * C + channel; C = 1: k = tap, padded to 32).
"""

from __future__ import annotations

import numpy as np

from benzene.kernels.gemm_w8a8 import SLICE_SCALE

TILE_BYTES = 2048


def convolution_int8_stream_bytes(K: int, N: int) -> int:
    return (N // 32) * (-(-K // 32) + 1) * TILE_BYTES


def convolution_int8_stream(codes: np.ndarray, u8: tuple | None = None, f32: tuple | None = None,
                 zero_point: int | np.ndarray = 0) -> bytes:
    """codes int8 [N, K] (device K order); u8 = (m [N], b [N]) or f32 = (scale [N], bias [N]), fp32; zero_point of the
    input codes: 0, 128 for signed inputs, or one per k [K] (ALIKED's per-channel zero points). A uint8 output requires
    |sum * m + b| < 2^30 for every input (the device converts it to int32 without a clamp; its packs saturate to
    [0, 255])."""
    assert (u8 is None) != (f32 is None)
    N, K = codes.shape
    KT = -(-K // 32)
    zero_points = np.broadcast_to(np.asarray(zero_point, np.int64), (K,))
    assert N % 32 == 0 and np.abs(codes.astype(np.int32)).max() <= 127
    assert np.isscalar(zero_point) and zero_point in (0, 128) or (zero_points.min() >= 0 and zero_points.max() <= 255)
    if u8 is not None:
        c = codes.astype(np.int64)
        largest_input = int(np.maximum(zero_points, 255 - zero_points).max())  # |code - zero point|
        amax = np.maximum(np.clip(c, 0, None).sum(1), -np.clip(c, None, 0).sum(1)) * largest_input
        bound = amax * np.abs(np.asarray(u8[0], np.float64)) + np.abs(np.asarray(u8[1], np.float64))
        if bound.max() >= 2.0**30:
            raise ValueError(f"requantization out of the int32 range: max |sum m + b| {bound.max():.3g}")
    cp = np.zeros((N, KT * 32), np.int8)
    cp[:, :K] = codes
    lane_col = np.arange(4)[:, None] * 8 + np.arange(32)[None] // 4  # [k][i] -> column within the block
    out = []
    for cb in range(N // 32):
        sl = slice(cb * 32, (cb + 1) * 32)
        tiles = cp[sl].reshape(32, KT, 8, 4).transpose(1, 2, 0, 3).reshape(KT, 1024)
        deep = np.zeros((KT, 2048), np.int8)
        deep[:, :1024] = tiles
        out.append(deep.tobytes())
        blocks = np.zeros((4, 64), np.uint32)
        bias2 = ((-(cp[sl, :K].astype(np.int64) * zero_points[None]).sum(1)) & 0xFFFFFFFF).astype(np.uint32)
        for i, sc in enumerate(SLICE_SCALE):
            blocks[i, :32] = sc
            blocks[i, 32:] = bias2
        out.append(blocks.tobytes())
        if u8 is not None:
            m, b = (np.asarray(v, np.float32)[sl] for v in u8)
            tail = np.concatenate([m[lane_col].ravel(), b[lane_col].ravel()]).astype(np.float32)  # 8 x 32 floats
        else:
            tail = np.zeros(256, np.float32)
            tail[:32], tail[32:64] = np.asarray(f32[0], np.float32)[sl], np.asarray(f32[1], np.float32)[sl]
        out.append(tail.tobytes())
    data = b"".join(out)
    assert len(data) == convolution_int8_stream_bytes(K, N)
    return data


def read_convolution_int8_stream(stream: bytes, K: int, N: int, u8: bool):
    """Codes [N, K] (device K order) and (m, b) / (scale, bias) of a convolution_int8 weight stream (inverse of
    convolution_int8_stream)."""
    KT = -(-K // 32)
    cb_bytes = (KT + 1) * TILE_BYTES
    codes = np.zeros((N, KT * 32), np.int8)
    p, q = np.zeros(N, np.float32), np.zeros(N, np.float32)
    for cb in range(N // 32):
        o = cb * cb_bytes
        blk = np.frombuffer(stream[o : o + KT * TILE_BYTES], np.int8).reshape(KT, TILE_BYTES)[:, :1024]
        codes[cb * 32 : (cb + 1) * 32] = blk.reshape(KT, 8, 32, 4).transpose(2, 0, 1, 3).reshape(32, KT * 32)
        tail = np.frombuffer(stream[o + KT * TILE_BYTES + 1024 : o + (KT + 1) * TILE_BYTES], np.float32)
        if u8:  # expanded: lane i of vector k = column 8k + i/4
            p[cb * 32 : (cb + 1) * 32] = tail[:128].reshape(4, 32)[:, ::4].reshape(-1)
            q[cb * 32 : (cb + 1) * 32] = tail[128:].reshape(4, 32)[:, ::4].reshape(-1)
        else:
            p[cb * 32 : (cb + 1) * 32], q[cb * 32 : (cb + 1) * 32] = tail[:32], tail[32:64]
    return codes[:, :K], p, q
