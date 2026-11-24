"""Weight formats, RTN / GPTQ weight quantization, and SmoothQuant + int8 activation fake-quant.

All weight formats reproduce llama.cpp block layouts bit-exactly in their dequantized
values, so the Hexagon (HVX/HMX) kernels written with hexagon-clang can be validated
against this file:

  q8_0   : 32-element blocks, fp16 d,  x = d * q,          q in [-127, 127]
  q4_0   : 32-element blocks, fp16 d,  x = d * (q - 8),    q in [0, 15]
  iq4_nl : 32-element blocks, fp16 d,  x = d * K[q],       K = IQ4_NL non-linear codebook
  int8_pc / int4_pc : per-output-channel symmetric (for an integer-activation path)

Activation fake-quant (A8) is per-token dynamic symmetric int8 after an optional
per-input-channel SmoothQuant division (fusable into the quantize pass on HVX).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_CHUNK = 1 << 24  # max elements per scale-search / quantization temporary (64 MB fp32)

IQ4NL_K = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32
)


def _fp16(x: torch.Tensor) -> torch.Tensor:
    """Round scales to fp16 as stored in the block (zero-safe)."""
    return x.to(torch.float16).to(torch.float32)


@dataclass(frozen=True)
class Format:
    name: str
    group: int  # 0 = per output channel (whole row)
    bits: int

    # ---- scale selection -------------------------------------------------
    def base_scale(self, w: torch.Tensor) -> torch.Tensor:
        """llama.cpp reference scale for a [rows, g] block. Signed for q4_0/iq4_nl."""
        if self.name in ("q8_0", "int8_pc"):
            return w.abs().amax(-1) / 127.0
        if self.name == "int4_pc":
            return w.abs().amax(-1) / 7.0
        idx = w.abs().argmax(-1, keepdim=True)
        mx = w.gather(-1, idx).squeeze(-1)  # signed extreme
        if self.name == "q4_0":
            return mx / -8.0
        if self.name == "iq4_nl":
            return mx / IQ4NL_K[0].item()  # map the extreme onto -127
        raise ValueError(self.name)

    def quant(self, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Dequantized value of w with per-row scale d (broadcast over the last dim)."""
        d = d.unsqueeze(-1) if d.dim() == w.dim() - 1 else d
        safe = torch.where(d == 0, torch.ones_like(d), d)
        x = w / safe
        if self.name in ("q8_0", "int8_pc"):
            q = x.round().clamp(-127, 127)
        elif self.name == "int4_pc":
            q = x.round().clamp(-7, 7)
        elif self.name == "q4_0":
            q = x.round().clamp(-8, 7)
        elif self.name == "iq4_nl":
            k = IQ4NL_K.to(w.device)
            mid = (k[1:] + k[:-1]) / 2
            q = k[torch.bucketize(x.contiguous(), mid)]
        else:
            raise ValueError(self.name)
        return torch.where(d == 0, torch.zeros_like(w), q * d)

    def find_scale(self, w: torch.Tensor, h: torch.Tensor | None = None, grid: int = 24) -> torch.Tensor:
        """Pick a per-row fp16 scale minimizing (diag-Hessian weighted) block error."""
        if w.dim() == 2 and w.numel() > _CHUNK:  # rows are independent: bound the temporaries
            step = max(1, _CHUNK // w.shape[1])
            return torch.cat([self.find_scale(w[r : r + step], h, grid) for r in range(0, w.shape[0], step)])
        d0 = self.base_scale(w)
        if grid <= 1:
            return _fp16(d0)
        wt = h if h is not None else torch.ones(w.shape[-1], device=w.device)
        best_d = _fp16(d0)
        best_e = ((self.quant(w, best_d) - w).pow(2) * wt).sum(-1)
        for r in torch.linspace(0.70, 1.10, grid).tolist():
            cands = [d0 * r]
            if self.name in ("q4_0", "iq4_nl"):
                cands.append(-d0 * r * (7 / 8 if self.name == "q4_0" else 113 / 127))
            for c in cands:
                d = _fp16(c)
                e = ((self.quant(w, d) - w).pow(2) * wt).sum(-1)
                better = e < best_e
                best_e = torch.where(better, e, best_e)
                best_d = torch.where(better, d, best_d)
        return best_d

    def bits_per_weight(self) -> float:
        return self.bits + (16 / self.group if self.group else 0)


FORMATS = {
    "q8_0": Format("q8_0", 32, 8),
    "q4_0": Format("q4_0", 32, 4),
    "iq4_nl": Format("iq4_nl", 32, 4),
    "int8_pc": Format("int8_pc", 0, 8),
    "int4_pc": Format("int4_pc", 0, 4),
}


def _group(fmt: Format, cols: int) -> int:
    return fmt.group or cols


# --------------------------------------------------------------------------- RTN
@torch.no_grad()
def rtn(w: torch.Tensor, fmt: Format, hdiag: torch.Tensor | None = None,
        grid: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    g = _group(fmt, cols)
    assert cols % g == 0, f"{cols} not divisible by {g}"
    wf = w.float().reshape(rows, cols // g, g)
    hd = None if hdiag is None else hdiag.float().reshape(cols // g, g)
    out = torch.empty_like(wf)
    scales = torch.empty(rows, cols // g, device=w.device)
    step = max(1, _CHUNK // g)
    for j in range(cols // g):
        for r in range(0, rows, step):
            blk = wf[r : r + step, j]
            d = fmt.find_scale(blk, None if hd is None else hd[j], grid)
            out[r : r + step, j] = fmt.quant(blk, d)
            scales[r : r + step, j] = d
    return out.reshape(rows, cols), scales


# --------------------------------------------------------------------------- GPTQ
@torch.no_grad()
def _cholesky(A: torch.Tensor, upper: bool = False) -> tuple[torch.Tensor, int]:
    """Cholesky factor of A and its info (0: success). A failure is retried on the same matrix first: ROCm's
    factorization can fail spuriously on a positive definite matrix on a GPU under load; a successful attempt's factor
    is the same either way."""
    for _ in range(4):
        L, info = torch.linalg.cholesky_ex(A, upper=upper)
        if int(info) == 0:
            break
    return L, int(info)


def gptq(
    w: torch.Tensor,
    H: torch.Tensor,
    fmt: Format,
    percdamp: float = 0.01,
    blocksize: int = 128,
    act_order: bool = True,
    grid: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ (Frantar et al.) with contiguous fp16-scale blocks.

    With act_order the column order follows diag(H) but block scales are fixed up
    front from the original weights ("static groups"), so the result stays in the
    contiguous llama.cpp layout.
    """
    W = w.float().clone()
    rows, cols = W.shape
    H = H.float().clone()
    g = _group(fmt, cols)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    hdiag = torch.diag(H).clone()

    scales = None
    if act_order or g == cols:
        scales = torch.stack(
            [fmt.find_scale(W[:, j * g : (j + 1) * g], hdiag[j * g : (j + 1) * g], grid)
             for j in range(cols // g)], dim=1)  # [rows, ngroups]
    perm = torch.argsort(hdiag, descending=True) if act_order else torch.arange(cols, device=W.device)
    W = W[:, perm]
    H = H[perm][:, perm]
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(cols, device=H.device) * damp
    for attempt in range(5):
        L, info = _cholesky(H)
        if info == 0:
            break
        H += torch.eye(cols, device=H.device) * damp * (10 ** attempt)
    if info != 0:
        raise torch.linalg.LinAlgError(f"GPTQ: H not positive definite after damping (info {info})")
    Hinv, info = _cholesky(torch.cholesky_inverse(L), upper=True)
    if info != 0:  # fp32 rounding of an ill-conditioned inverse (dead inputs, few samples): in float64 on the CPU
        H64 = H.double().cpu()
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H64)), upper=True)
        Hinv = Hinv.float().to(H.device)
    Q = torch.zeros_like(W)
    group_of = perm // g  # group of each processed column
    static = scales is not None
    if not static:
        scales = torch.zeros(rows, cols // g, device=W.device)

    cur_scale = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            wcol = W1[:, i]
            dd = Hinv1[i, i]
            if static:
                s = scales[:, group_of[col]]
            else:
                if col % g == 0:  # dynamic group scale from the error-updated weights
                    blk = torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :g]
                    cur_scale = fmt.find_scale(blk, hdiag[perm[col : col + g]], grid)
                    scales[:, col // g] = cur_scale
                s = cur_scale
            q = fmt.quant(wcol, s)
            Q1[:, i] = q
            err = (wcol - q) / dd
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    inv = torch.argsort(perm)
    return Q[:, inv], scales


# --------------------------------------------------------------------------- activations
class ActQuantLinear(torch.nn.Module):
    """Linear with 8-bit activation fake-quant and optional SmoothQuant vector.

    y = Qa(x / s) @ Qw(W * s)^T + b   (s = 1 when smoothing is disabled)

    Activation modes (what the Hexagon kernel would do):
      "tok"    per-token dynamic symmetric int8
      "tok16"  per-token dynamic 16-bit (two uint8 planes on integer HMX, llama.cpp PR 29740)
      "blk32"  dynamic symmetric int8 per 32-element block with fp16 scale
               (llama.cpp HVX path: activations requantized to Q8_0, vrmpy dot products)
      "static" static per-tensor asymmetric uint8 from calibration min/max
               (HMX int8 path: activation.ub tiles, only per-column output scaling)
    """

    def __init__(self, lin: torch.nn.Linear, w_q: torch.Tensor, smooth: torch.Tensor | None,
                 mode: str = "tok", ch_range: tuple[torch.Tensor, torch.Tensor] | None = None):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = torch.nn.Parameter(w_q.to(lin.weight.dtype), requires_grad=False)
        self.bias = lin.bias
        self.register_buffer("inv_s", None if smooth is None else (1.0 / smooth).to(lin.weight.dtype))
        self.mode = mode
        if mode == "static":
            lo, hi = (t.float().to(w_q.device) for t in ch_range)
            if smooth is not None:
                lo, hi = lo / smooth, hi / smooth
            lo, hi = min(lo.min().item(), 0.0), max(hi.max().item(), 0.0)
            self.a_scale = max(hi - lo, 1e-8) / 255.0
            self.a_zp = float(round(-lo / self.a_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_s is not None:
            x = x * self.inv_s
        xf = x.float()
        if self.mode == "tok":
            d = _fp16(xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127)
            xq = (xf / d).round().clamp(-127, 127) * d
        elif self.mode == "tok16":  # llama.cpp PR 29740 integer HMX: 16-bit offset-binary per token
            d = xf.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 32767
            xq = torch.floor(xf / d + 0.5).clamp(-32768, 32767) * d
        elif self.mode == "blk32":
            k = xf.shape[-1]
            pad = padded_cols(k) - k
            xb = torch.nn.functional.pad(xf, (0, pad)).unflatten(-1, (-1, 32))
            d = _fp16(xb.abs().amax(-1, keepdim=True) / 127)
            d = torch.where(d == 0, torch.ones_like(d), d)
            xq = ((xb / d).round().clamp(-127, 127) * d).flatten(-2)[..., :k]
        elif self.mode == "static":
            q = (xf / self.a_scale).round() + self.a_zp
            xq = (q.clamp(0, 255) - self.a_zp) * self.a_scale
        else:
            raise ValueError(self.mode)
        return torch.nn.functional.linear(xq.to(x.dtype), self.weight, self.bias)


# Mixed prefill / decode emulation. On the device the prompt is one prefill (HMX GEMM) and every
# generated token is a decode step (HVX GEMV), possibly with different weight formats. In a
# teacher-forced pass (prompt + forced answer in one forward) positions < split are the prompt.
_MIXED_SPLIT: list[int | None] = [None]


def set_mixed_split(split: int | None) -> None:
    _MIXED_SPLIT[0] = split


class MixedLinear(torch.nn.Module):
    """Prompt positions (< split) through ``prefill``, later positions through ``decode``.

    For LLM layers only (input [batch, tokens, features]). Without a split (free-running
    generation) a multi-token call is the prefill and a single-token call a decode step.
    """

    def __init__(self, prefill: torch.nn.Module, decode: torch.nn.Module):
        super().__init__()
        self.prefill, self.decode = prefill, decode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split, t = _MIXED_SPLIT[0], x.shape[-2]
        if split is None:
            return self.prefill(x) if t > 1 else self.decode(x)
        if t <= split:
            return self.prefill(x)
        return torch.cat([self.prefill(x[..., :split, :]), self.decode(x[..., split:, :])], dim=-2)


def hmx_kchunk(k: int, max_tiles: int = 128) -> int:
    """K elements per weight-scale group of the PR 29740 integer HMX matmul (balanced k-chunks)."""
    nkt = padded_cols(k) // 32
    nkc = math.ceil(nkt / max_tiles)
    return math.ceil(nkt / nkc) * 32


@torch.no_grad()
def q4_0_to_hmx_int8(w_dq: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Emulate PR 29740's Q4_0 -> int8 HMX weight conversion.

    Per output column and k-chunk: dmax = max|d|, F = round(d * 32768 / dmax),
    w8 = round(q * 16 * F / 32768), effective weight = w8 * dmax / 16.
    w_dq: dequantized Q4_0 weight [rows, K_padded]; scales: fp16-exact block scales [rows, K_padded/32].
    """
    rows, kp = w_dq.shape
    kc = hmx_kchunk(kp)
    d = scales.float()
    safe = torch.where(d == 0, torch.ones_like(d), d)
    q = (w_dq.float().view(rows, -1, 32) / safe[..., None]).round()      # Q4_0 codes - 8
    out = torch.empty_like(w_dq, dtype=torch.float32)
    for c0 in range(0, kp, kc):
        b0, b1 = c0 // 32, min(kp, c0 + kc) // 32
        dc = d[:, b0:b1]
        dmax = _fp16(dc.abs().amax(1, keepdim=True))
        F = torch.where(dmax > 0, (dc * 32768 / dmax.clamp_min(1e-30)).round().clamp(-32768, 32767),
                        torch.zeros_like(dc))
        w8 = torch.floor(q[:, b0:b1] * 16 * F[..., None] / 32768 + 0.5).clamp(-128, 127)
        out[:, c0 : b1 * 32] = (w8 * (dmax / 16)[..., None]).reshape(rows, -1)
    return out


def smooth_vector(act_absmax: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    wmax = w.float().abs().amax(0).clamp_min(1e-5)
    amax = act_absmax.float().to(w.device).clamp_min(1e-5)
    s = amax.pow(alpha) / wmax.pow(1 - alpha)
    return s.clamp(1e-2, 1e3)


def padded_cols(cols: int, g: int = 32) -> int:
    return int(math.ceil(cols / g) * g)


# --------------------------------------------------------------------------- static activation clips
def _centers(top: torch.Tensor, nb: int) -> torch.Tensor:
    return (torch.arange(nb, dtype=torch.float64) + 0.5)[None] / nb * top.double()[:, None]


def clip_values(hist: torch.Tensor, top: torch.Tensor, how: str, levels: int = 255) -> torch.Tensor:
    """Clip per histogram row (hist [R, nb] of the non-zero magnitudes over [0, top[r]]) for a uniform grid of
    `levels` steps up to the clip (255: uint8 of non-negative values; 127: int8 of signed ones, by magnitude):
    "max" the maximum, "p4" the 99.99th percentile, "mse" the clip minimizing the squared error (rounding below,
    clipping above)."""
    h = hist.double()
    nb = h.shape[1]
    if how == "max":
        return top.double().clone()
    v = _centers(top, nb)
    if how == "p4":
        cum = h.cumsum(1) / h.sum(1, keepdim=True).clamp_min(1)
        i = (cum < 0.9999).sum(1).clamp(max=nb - 1)
        return v.gather(1, i[:, None])[:, 0] + top.double() / nb / 2
    assert how == "mse", how
    best_c, best_e = top.double().clone(), torch.full_like(top.double(), float("inf"))
    for r in torch.logspace(-1.5, 0, 120, dtype=torch.float64):  # clip = r * top
        c = (top.double() * r)[:, None]
        d = c / levels
        q = torch.where(v > c, c, (v / d).round() * d)
        e = (h * (v - q) ** 2).sum(1)
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        best_c = torch.where(better, c[:, 0], best_c)
    return best_c


# --------------------------------------------------------------------------- static per-channel uint8 activations
def _bin_centers(low: torch.Tensor, high: torch.Tensor, nb: int) -> torch.Tensor:
    return low.double()[:, None] + (torch.arange(nb, dtype=torch.float64)[None] + 0.5) / nb * (high - low).double()[
        :, None]


def _affine_error(values: torch.Tensor, counts: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Squared error per row of the histogram (bin values / counts [R, nb]) under x = s (clamp(round(x / s) + z) - z)."""
    q = (values / scale[:, None] + zero[:, None]).round().clamp(0, 255)
    return (counts * (values - scale[:, None] * (q - zero[:, None])) ** 2).sum(1)


def activation_quantizer(stats: dict, clip: str, zp: str, signed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(scale, zero point) fp64 per channel of a quantization point from its calibration statistics."""
    low = stats["min"].double().clamp(max=0.0)  # (zero representable: the padding / the sampling's outside)
    high = stats["max"].double().clamp(min=0.0)
    counts = stats["hist"].double()
    values = _bin_centers(stats["min"], stats["max"], counts.shape[1])
    if clip == "max":
        clip_low, clip_high = low, high
    elif clip == "p4":
        cumulative = counts.cumsum(1) / counts.sum(1, keepdim=True).clamp_min(1)
        index = (cumulative < 0.9999).sum(1).clamp(max=counts.shape[1] - 1)
        clip_high = values.gather(1, index[:, None])[:, 0].clamp(min=0.0)
        clip_low = low
        if signed:
            index = (cumulative < 0.0001).sum(1).clamp(max=counts.shape[1] - 1)
            clip_low = values.gather(1, index[:, None])[:, 0].clamp(max=0.0)
    else:
        assert clip == "mse", clip
        best = torch.full_like(low, float("inf"))
        clip_low, clip_high = low.clone(), high.clone()
        for r in torch.logspace(-1.5, 0, 120, dtype=torch.float64):
            a = low * r if signed else low
            b = high * r
            scale = ((b - a) / 255).clamp_min(1e-12)
            zero = (-a / scale).round().clamp(0, 255)
            error = _affine_error(values, counts, scale, zero)
            better = error < best
            best = torch.where(better, error, best)
            clip_low = torch.where(better, a, clip_low)
            clip_high = torch.where(better, b, clip_high)
    if zp == "128":
        scale = (torch.maximum(-clip_low, clip_high) / 127).clamp_min(1e-12)
        return scale, torch.full_like(scale, 128.0)
    if zp == "layer":
        best_total, best = float("inf"), None
        for z in range(1, 255):
            scale = torch.maximum(-clip_low / z, clip_high / (255 - z)).clamp_min(1e-12)
            zero = torch.full_like(scale, float(z))
            total = float(_affine_error(values, counts, scale, zero).sum())
            if total < best_total:
                best_total, best = total, (scale, zero)
        return best
    assert zp == "channel", zp
    scale = ((clip_high - clip_low) / 255).clamp_min(1e-12)
    return scale, (-clip_low / scale).round().clamp(0, 255)


def quantize_activation(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, channel_dim: int) -> torch.Tensor:
    shape = [1] * x.dim()
    shape[channel_dim] = -1
    s = scale.to(x).reshape(shape)
    z = zero.to(x).reshape(shape)
    return s * ((x / s + z).round().clamp(0, 255) - z)


def int8_weight_codes(weight: torch.Tensor, s_in: torch.Tensor, H: torch.Tensor, method: str, device: str):
    """int8 codes [N, K] and per-output-channel scales [N] (fp64) of weight [N, K] (fp64) with the per-K input scales
    folded in (GPTQ with the Hessian H of the unscaled input)."""
    Wm = weight * s_in[None]
    fmt = FORMATS["int8_pc"]
    if method == "gptq":
        f = 1.0 / s_in
        Hs = H.double() * f[:, None] * f[None, :]
        Q, ws = gptq(Wm.float().to(device), Hs.float().to(device), fmt)
    else:
        Q, ws = rtn(Wm.float(), fmt)
    Q, ws = Q.double().cpu(), ws.double().cpu().reshape(-1)
    codes = torch.where(ws[:, None] > 0, (Q / torch.where(ws > 0, ws, 1)[:, None]).round(), 0)
    assert codes.abs().max() <= 127
    return codes, torch.where(ws > 0, ws, torch.ones_like(ws))
