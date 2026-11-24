"""LightGlue's tensors in a device model file (benzene dsp/models/lightglue), fp16
(benzene.kernels.gemm_fp16.gemm_fp16_stream: weights and biases rounded to fp16), for D descriptor dimensions in H heads
of d = D / H. The device keeps D-wide rows in whole vectors of 64 (96 -> 128, the rest zero): the GEMMs whose input is
such a row (the projections, the out projection, the final projection) take K = D rounded up to 64, the extra weights
zero.
  lg.input           input_proj (input_dim -> D), only when the model has one (input_dim != D)
  lg.<l>.self.qkv    Wqkv with the rows reordered to [q | k | v] (head-major, d each; PyTorch: (head, dim, 3))
  lg.<l>.self.out, .self.ffn1 (2D -> 2D), .self.ln (fp32 gamma | beta, 2D each), .self.ffn2 (2D -> D)
  lg.<l>.cross.qkv   [to_qk | to_v] (N = 2D), .cross.out, .cross.ffn1, .cross.ln, .cross.ffn2
  lg.<l>.final       log_assignment[l].final_proj / D^0.25 (the scaling of both sides folded in), every layer: with
                     adaptive inference the last layer run matches
  lg.<l>.match       log_assignment[l].matchability (weight [D], bias; fp16 values in fp32, as LightGlue's fp16 Linear)
  lg.<l>.conf        token_confidence[l] for l < layers - 1 (weight [D], bias; fp16 values in fp32): the early stop
                     (variants with adaptive inference: token_confidences)
  lg.posenc          Wr [d / 2][2] (fp16 values in fp32: the device computes the rotary tables as LightGlue's fp16 Linear)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from benzene.kernels.gemm_fp16 import gemm_fp16_stream
from benzene.model_file import Writer


def lightglue_header(lg) -> dict:
    """The model file header entries of a LightGlue: layers, heads, dimensions, and the input dimensions when the
    model projects its input descriptors (lg.input)."""
    c = lg.conf
    header = {"n_layers": c.n_layers, "heads": c.num_heads, "dim": c.descriptor_dim}
    return header if c.input_dim == c.descriptor_dim else {**header, "input_dim": c.input_dim}


def lightglue_tensors(lg, token_confidences: bool = True) -> dict[str, bytes | np.ndarray]:
    """{name: weight stream bytes or fp32 array} in file order; token_confidences: the variant has the adaptive
    inference's heads (protocol::lightglue::token_confidences)."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in lg.state_dict().items()}
    heads, dim = lg.conf.num_heads, lg.conf.descriptor_dim
    head_dim = dim // heads
    row = -(-dim // 64) * 64  # a D-wide row's halfwords on the device

    def wide(w: np.ndarray) -> np.ndarray:  # K padded to the row width
        return np.pad(w, ((0, 0), (0, row - w.shape[1]))) if w.shape[1] < row else w

    out: dict[str, bytes | np.ndarray] = {}
    if "input_proj.weight" in sd:
        out["lg.input"] = gemm_fp16_stream(sd["input_proj.weight"], sd["input_proj.bias"])
    perm = np.array([h * 3 * head_dim + d * 3 + i for i in range(3) for h in range(heads) for d in range(head_dim)])
    for l in range(lg.conf.n_layers):
        p = f"transformers.{l}."
        for blk in ("self_attn", "cross_attn"):
            q = p + blk + "."
            tag = f"lg.{l}.{blk.split('_')[0]}."
            if blk == "self_attn":
                w, b = sd[q + "Wqkv.weight"][perm], sd[q + "Wqkv.bias"][perm]
                wo, bo = sd[q + "out_proj.weight"], sd[q + "out_proj.bias"]
            else:
                w = np.concatenate([sd[q + "to_qk.weight"], sd[q + "to_v.weight"]])
                b = np.concatenate([sd[q + "to_qk.bias"], sd[q + "to_v.bias"]])
                wo, bo = sd[q + "to_out.weight"], sd[q + "to_out.bias"]
            out[tag + "qkv"] = gemm_fp16_stream(wide(w), b)
            out[tag + "out"] = gemm_fp16_stream(wide(wo), bo)
            out[tag + "ffn1"] = gemm_fp16_stream(sd[q + "ffn.0.weight"], sd[q + "ffn.0.bias"])
            out[tag + "ln"] = np.concatenate([sd[q + "ffn.1.weight"], sd[q + "ffn.1.bias"]])
            out[tag + "ffn2"] = gemm_fp16_stream(sd[q + "ffn.3.weight"], sd[q + "ffn.3.bias"])
    hf = lambda *a: np.concatenate(a).astype(np.float16).astype(np.float32)  # noqa: E731
    scale = dim**0.25
    for l in range(lg.conf.n_layers):
        la = f"log_assignment.{l}."
        out[f"lg.{l}.final"] = gemm_fp16_stream(wide(sd[la + "final_proj.weight"] / scale),
                                            sd[la + "final_proj.bias"] / scale)
        out[f"lg.{l}.match"] = hf(sd[la + "matchability.weight"][0], sd[la + "matchability.bias"])
        if token_confidences and l + 1 < lg.conf.n_layers:
            tc = f"token_confidence.{l}.token.0."
            out[f"lg.{l}.conf"] = hf(sd[tc + "weight"][0], sd[tc + "bias"])
    out["lg.posenc"] = sd["posenc.Wr.weight"].astype(np.float16).astype(np.float32)
    return out


def write_model_file(path: Path, header: dict, extractor_tensors: dict, lg, start: float) -> None:
    """A matcher's model file: the extractor's tensors (data, or (data, metadata)), then LightGlue's; the header plus
    lightglue_header. Prints its size and the time since start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Writer(path)
    for name, value in extractor_tensors.items():
        data, meta = value if isinstance(value, tuple) else (value, {})
        w.raw(name, data, **meta)
    for name, v in lightglue_tensors(lg).items():
        if isinstance(v, bytes):
            w.raw(name, v)
        else:
            w.f32(name, v)
    size = w.finish({**header, **lightglue_header(lg)})
    print(f"wrote {path} ({size / 2**20:.1f} MiB, {len(w.tensors)} tensors) in {time.time() - start:.0f}s")
