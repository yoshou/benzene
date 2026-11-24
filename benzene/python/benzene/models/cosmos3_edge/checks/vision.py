"""The whole vision encoder (27 layers) + post norm + projector on the DSP (benzene cosmos3_edge_vision).

Input: the captured encoder input (patch + position embeddings, all images of the sample) and image boundaries;
output compared with the projector output of the recipe's fake-quant model, and for the first two images with
the unquantized fp32 stack (numpy).

  python -m benzene.models.cosmos3_edge.checks.vision w8a8_sq05s_gptq
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from math import erf
from pathlib import Path

import numpy as np
import torch

from benzene import board

from ..device_layout import ckpt
from ..paths import BOARD_DIR, OUT_DIR
from ..recipes import DEVICE_RECIPE

PJ = "model.projector."


def fp32_layer(layer: int, x: np.ndarray, cu: np.ndarray) -> np.ndarray:
    """The unquantized layer from the checkpoint weights (fp32 numpy): the common reference."""
    pre = f"model.visual.encoder.layers.{layer}."

    def ln(h, n):
        mu, var = h.mean(1, keepdims=True), h.var(1, keepdims=True)
        return (h - mu) / np.sqrt(var + 1e-6) * ckpt(pre + n + ".weight") + ckpt(pre + n + ".bias")

    def lin(h, n):
        return h @ ckpt(pre + n + ".weight").T + ckpt(pre + n + ".bias")

    h = ln(x, "layer_norm1")
    q, k, v = (lin(h, f"self_attn.{p}_proj") for p in "qkv")
    o = np.zeros_like(q)
    for i in range(len(cu) - 1):
        s0, s1 = cu[i], cu[i + 1]
        Q, K, V = (a[s0:s1].reshape(s1 - s0, 16, 72) for a in (q, k, v))
        S = np.einsum("qhd,khd->hqk", Q, K) / np.sqrt(72)
        P = np.exp(S - S.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        o[s0:s1] = np.einsum("hqk,khd->qhd", P, V).reshape(s1 - s0, -1)
    x1 = x + lin(o, "self_attn.out_proj")
    z = lin(ln(x1, "layer_norm2"), "mlp.fc1")
    return x1 + lin(0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))), "mlp.fc2")


def fp32_tail(x: np.ndarray) -> np.ndarray:
    def ln(h, w, b):
        mu, var = h.mean(1, keepdims=True), h.var(1, keepdims=True)
        return (h - mu) / np.sqrt(var + 1e-6) * w + b

    z = ln(x, ckpt("model.visual.post_layernorm.weight"), ckpt("model.visual.post_layernorm.bias"))
    u = ln(z, ckpt(PJ + "norm.weight"), ckpt(PJ + "norm.bias")).reshape(-1, 4 * 1152)
    h = u @ ckpt(PJ + "linear_fc1.weight").T + ckpt(PJ + "linear_fc1.bias")
    h = 0.5 * h * (1 + np.vectorize(erf)(h / np.sqrt(2)))
    return h @ ckpt(PJ + "linear_fc2.weight").T + ckpt(PJ + "linear_fc2.bias")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--fp32-images", type=int, default=2, help="images compared with the unquantized fp32 model")
    args = parser.parse_args()
    acts = OUT_DIR / "acts" / args.recipe
    c0 = torch.load(acts / "model.visual.encoder.layers.0.layer.pt")
    tail = torch.load(acts / "vision_tail.layer.pt")
    x = c0["x"].numpy().astype(np.float32)
    cu = c0["cu_seqlens"].numpy().astype(np.int32)
    y_fq = tail["y"].numpy()
    N = int(cu[-1])
    model = f"{BOARD_DIR}/cosmos3_edge-{args.recipe}"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bundle").mkdir()
        (d / "bundle" / "x.f32").write_bytes(x.tobytes())
        (d / "bundle" / "image_offsets.i32").write_bytes(cu.tobytes())
        work = board.empty_directory("pr")
        board.upload(f"{td}/bundle/", f"{work}/bundle/")
        r = board.run_benzene(f"cosmos3_edge_vision {model}.hexmodel pr/bundle pr/out")
        print(r.stdout.strip(), r.stderr.strip())
        if r.returncode:
            sys.exit(f"board run failed ({r.returncode})")
        board.download(f"{work}/out/y.f32", f"{td}/")
        y = np.frombuffer((d / "y.f32").read_bytes(), np.float32).reshape(N // 4, -1)

    def err(a, b):
        return np.sqrt(((a - b).astype(np.float64) ** 2).mean()) / np.sqrt((b.astype(np.float64) ** 2).mean())

    ni = min(args.fp32_images, len(cu) - 1)
    n2 = int(cu[ni])
    h = x[:n2].astype(np.float64)
    for l in range(27):
        h = fp32_layer(l, h, cu[: ni + 1])
    y32 = fp32_tail(h)
    print(f"vision ({len(cu) - 1} images, {N} patches -> {N // 4} tokens): device vs PyTorch fake-quant {err(y, y_fq):.3e}; "
          f"first {ni} images vs unquantized fp32: device {err(y[: n2 // 4], y32):.3e}, fake-quant {err(y_fq[: n2 // 4], y32):.3e}")


if __name__ == "__main__":
    main()
