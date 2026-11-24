"""Convolutions as W8A8 GEMMs (fake-quant) for the VGGT DPT heads - the int8 alternative the accuracy evaluation
measures (the device runs the DPT heads in fp16, benzene dsp/kernels/gemm_fp16).

A convolution as a GEMM has output pixels as rows: a k x k convolution reads the k*k input pixels
of the receptive field (K = k*k*Cin, tap-major on the device), a 1 x 1 convolution one pixel, a transposed convolution
with kernel = stride (the DPT resize layers) one input pixel for k*k*Cout output columns. Weights are int8 per output
column (an output channel, for the transposed convolution an (output channel, tap) pair). Activation modes:
  "tok"   per GEMM row: one int8 scale per output pixel over its whole receptive field (k*k*Cin values) - exactly the
          per-token int8 the GEMM input stage computes when it gathers the receptive field
  "tens"  one dynamic scale per frame over the whole input map (the map is quantized once, the GEMM input stage only
          gathers int8 rows)
SmoothQuant vectors are per input channel (all taps of a channel share it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benzene.quantization import _fp16


def weight_matrix(mod: torch.nn.Module) -> torch.Tensor:
    """The GEMM weight [rows = output columns, cols = K] of a Linear / Conv2d / ConvTranspose2d (PyTorch K order:
    Conv2d (ci, ky, kx) as F.unfold; ConvTranspose2d rows (co, ky, kx), cols ci)."""
    w = mod.weight
    if isinstance(mod, torch.nn.ConvTranspose2d):
        return w.permute(1, 2, 3, 0).reshape(-1, w.shape[0])
    return w.reshape(w.shape[0], -1)


def weight_from_matrix(mod: torch.nn.Module, m: torch.Tensor) -> torch.Tensor:
    w = mod.weight
    if isinstance(mod, torch.nn.ConvTranspose2d):
        ci, co, kh, kw = w.shape
        return m.reshape(co, kh, kw, ci).permute(3, 0, 1, 2).contiguous()
    return m.reshape(w.shape).contiguous()


def taps(mod: torch.nn.Module) -> int:
    return 1 if isinstance(mod, (torch.nn.Linear, torch.nn.ConvTranspose2d)) else mod.kernel_size[0] * mod.kernel_size[1]


def channel_amax(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per input channel |x| max of a module input (Linear: last dim, conv: dim 1)."""
    if isinstance(mod, torch.nn.Linear):
        return x.detach().abs().reshape(-1, x.shape[-1]).amax(0)
    return x.detach().abs().transpose(0, 1).reshape(x.shape[1], -1).amax(1)


def gemm_rows(mod: torch.nn.Module, x: torch.Tensor, max_rows: int = 1 << 15):
    """The GEMM input rows (fp32, K in weight_matrix order) of a module input, in chunks of <= max_rows rows."""
    x = x.detach()
    if isinstance(mod, torch.nn.Linear):
        r = x.reshape(-1, x.shape[-1])
        for i in range(0, r.shape[0], max_rows):
            yield r[i : i + max_rows].float()
    elif isinstance(mod, torch.nn.ConvTranspose2d) or mod.kernel_size == (1, 1) and mod.stride == (1, 1):
        r = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        for i in range(0, r.shape[0], max_rows):
            yield r[i : i + max_rows].float()
    else:
        for b in range(x.shape[0]):  # one map at a time: the unfolded rows of a 148 x 148 x 256 map are 200 MB
            u = F.unfold(x[b : b + 1].float(), mod.kernel_size, mod.dilation, mod.padding, mod.stride)[0].T
            for i in range(0, u.shape[0], max_rows):
                yield u[i : i + max_rows]


def _scale(amax: torch.Tensor) -> torch.Tensor:
    """fp16 int8 scale amax / 127; 1 where it is 0 (all-zero rows after a ReLU: codes 0, as the device's scale-0 rows)."""
    d = _fp16(amax / 127)
    return torch.where(d > 0, d, torch.ones_like(d))


class ActQuantConv(torch.nn.Module):
    """Conv2d / ConvTranspose2d with a fake-quantized weight (dequantized values, any format) and int8 activations."""

    def __init__(self, conv: torch.nn.Module, w_q: torch.Tensor, smooth: torch.Tensor | None, mode: str = "tok"):
        super().__init__()
        self.conv = conv
        self.weight = torch.nn.Parameter(weight_from_matrix(conv, w_q).to(conv.weight.dtype), requires_grad=False)
        self.bias = conv.bias
        self.register_buffer("inv_s", None if smooth is None else (1.0 / smooth).to(conv.weight.dtype))
        self.mode = mode
        self.transposed = isinstance(conv, torch.nn.ConvTranspose2d)

    def _conv(self, x, w):
        c = self.conv
        if self.transposed:
            return F.conv_transpose2d(x, w, None, c.stride, c.padding, c.output_padding, c.groups, c.dilation)
        return F.conv2d(x, w, None, c.stride, c.padding, c.dilation, c.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_s is not None:
            x = x * self.inv_s.to(x.dtype)[None, :, None, None]
        xf, w = x.float(), self.weight.to(x.dtype)
        c = self.conv
        if self.mode == "tens":
            d = _scale(xf.abs().amax((1, 2, 3), keepdim=True))
            y = self._conv(((xf / d).round().clamp(-127, 127) * d).to(x.dtype), w)
        elif self.transposed or c.kernel_size == (1, 1) and c.stride == (1, 1):  # GEMM row = input pixel
            d = _scale(xf.abs().amax(1, keepdim=True))
            y = self._conv(((xf / d).round().clamp(-127, 127) * d).to(x.dtype), w)
        else:  # GEMM row = output pixel: the scale is the max over its receptive field; one 1 x 1 conv per tap
            kh, kw = c.kernel_size
            (sh, sw), (ph, pw) = c.stride, c.padding
            am = F.max_pool2d(xf.abs().amax(1, keepdim=True), (kh, kw), (sh, sw), (ph, pw))
            d = _scale(am)  # [B, 1, Ho, Wo]
            xp = F.pad(xf, (pw, pw, ph, ph))
            Ho, Wo = d.shape[-2:]
            y = None
            for ky in range(kh):
                for kx in range(kw):
                    xt = xp[:, :, ky : ky + sh * (Ho - 1) + 1 : sh, kx : kx + sw * (Wo - 1) + 1 : sw]
                    q = ((xt / d).round().clamp(-127, 127) * d).to(x.dtype)
                    t = F.conv2d(q, w[:, :, ky : ky + 1, kx : kx + 1])
                    y = t if y is None else y + t
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)[None, :, None, None]
        return y
