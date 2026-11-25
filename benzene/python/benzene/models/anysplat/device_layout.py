"""Device (Hexagon) layouts of AnySplat's weights for the benzene runtime: VGGT's blobs (the VGGT port's
device_layout.Layout) made of AnySplat's checkpoint and quantized weights, and the Gaussian head's own blob
(benzene protocol.h anysplat::gaussian_head_part: norm, the image merger and the output convolution as fp16 GEMM
streams).
"""

from __future__ import annotations

import numpy as np
import torch

from benzene.kernels.gemm_fp16 import gemm_fp16_stream
from benzene.model_file import pack_parts

from ..vggt.device_layout import Layout
from .network import RAW_CHANNELS
from .paths import MODEL_DIR

MERGER_KERNEL = 7  # protocol.h anysplat::kMergerKernel
MERGER_STRIP = 64  # anysplat::kMergerStrip: a pixel's 3 x 7 values of a window row, zero-padded
OUTPUT_CHANNELS = 96  # anysplat::kGaussianOutputChannels: RAW_CHANNELS zero-padded

_CKPT = None


def ckpt(name: str) -> np.ndarray:
    """A tensor of AnySplat's checkpoint by its module name in network.load_network's model (the checkpoint's without
    "encoder."), as fp32 numpy (the aggregator's are stored in bf16)."""
    global _CKPT
    if _CKPT is None:
        from safetensors import safe_open

        _CKPT = safe_open(str(MODEL_DIR / "model.safetensors"), "pt")
    return _CKPT.get_tensor("encoder." + name).float().numpy()


def _rec(recipe: str, name: str) -> dict:
    from .recipes import RECIPES, cache_path, spec_of

    spec = spec_of(RECIPES[recipe][1], name)
    assert spec is not None and spec.get("a8"), f"{name}: not a W8A8 layer of {recipe}"
    return torch.load(cache_path(name, spec)[0])


ANYSPLAT = Layout(ckpt, _rec)


def blob_gaussian_head() -> tuple[bytes, np.ndarray]:
    """norm weight / bias; input_merger.0 with K = (window row, strip value): per window row the 3 x 7 weights in
    (channel, column) order, zero-padded to MERGER_STRIP; output_conv2.2 with its columns zero-padded to
    OUTPUT_CHANNELS."""
    p = "gaussian_param_head."
    w = ckpt(p + "input_merger.0.weight")  # [128, 3, 7, 7]
    merger = np.zeros((128, MERGER_KERNEL, MERGER_STRIP), np.float32)
    merger[:, :, : 3 * MERGER_KERNEL] = w.transpose(0, 2, 1, 3).reshape(128, MERGER_KERNEL, 3 * MERGER_KERNEL)
    merger = merger.reshape(128, MERGER_KERNEL * MERGER_STRIP)
    out = np.pad(ckpt(p + "scratch.output_conv2.2.weight").reshape(RAW_CHANNELS, 128),
                 ((0, OUTPUT_CHANNELS - RAW_CHANNELS), (0, 0)))
    out_bias = np.pad(ckpt(p + "scratch.output_conv2.2.bias"), (0, OUTPUT_CHANNELS - RAW_CHANNELS))
    return pack_parts([ckpt(p + "norm.weight"), ckpt(p + "norm.bias"),
                       gemm_fp16_stream(np.ascontiguousarray(merger), ckpt(p + "input_merger.0.bias")),
                       gemm_fp16_stream(np.ascontiguousarray(out), out_bias)])
