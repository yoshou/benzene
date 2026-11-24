"""The exported SuperPoint in PyTorch: the device model file's convolution weights and constants with the device's
integer arithmetic, the reference of the device checks (checks/end_to_end.py)."""

from __future__ import annotations

import numpy as np

from benzene.kernels.convolution_int8 import read_convolution_int8_stream
from benzene.model_file import load_export


def exported_superpoint(sp, path) -> None:
    """Replace SuperPoint's convolutions by the exported device model's, in its integer arithmetic: activations are uint8
    codes (the image * 255 into conv1a), exact integer sums (fp64, rounded), uint8 outputs trunc(clamp(fp32(acc) * m + b,
    0, 255.5)) - ReLU and max pool of the forward act on the codes unchanged - and the heads' fp32 acc * scale + bias.
    The device's own weights and constants, so the comparison does not depend on quantizing again (GPTQ on a GPU)."""
    import torch

    from .recipes import integer_convolution

    ex = load_export(path)

    class ExportedConvolution(torch.nn.Module):
        def __init__(self, conv, codes, multipliers, biases, u8, first, columns=None):
            super().__init__()
            co, ci, kh, kw = conv.weight.shape
            K = kh * kw * ci
            n = columns or co
            w = torch.from_numpy(codes[:n, :K].astype(np.float64)).reshape(-1, kh, kw, ci).permute(0, 3, 1, 2)
            dev = conv.weight.device
            self.register_buffer("w", w.contiguous().to(dev))
            self.register_buffer("multipliers", torch.from_numpy(multipliers[:n]).to(dev)[None, :, None, None])
            self.register_buffer("biases", torch.from_numpy(biases[:n]).to(dev)[None, :, None, None])
            self.u8, self.first, self.pad = u8, first, conv.padding

        def forward(self, x):
            codes = (x.double() * 255).round() if self.first else x.double()
            return integer_convolution(codes, self.w, 1, self.pad, self.multipliers, self.biases, self.u8)

    def conv_parts(name):
        data, meta = ex[name]
        return data, meta["K"], meta["N"]

    for name in ("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b"):
        data, K, N = conv_parts(f"sp.{name}")
        codes, m, b = read_convolution_int8_stream(data, K, N, True)
        setattr(sp, name, ExportedConvolution(getattr(sp, name), codes, m, b, True, name == "conv1a"))
    data, K, N = conv_parts("sp.convPaDa")
    codes, m, b = read_convolution_int8_stream(data, K, N, True)
    sp.convPa = ExportedConvolution(sp.convPa, codes[:256], m[:256], b[:256], True, False)
    sp.convDa = ExportedConvolution(sp.convDa, codes[256:], m[256:], b[256:], True, False)
    for name, cols in (("convPb", 65), ("convDb", 256)):
        data, K, N = conv_parts(f"sp.{name}")
        codes, sc, bi = read_convolution_int8_stream(data, K, N, False)
        setattr(sp, name, ExportedConvolution(getattr(sp, name), codes, sc, bi, False, False, cols))
