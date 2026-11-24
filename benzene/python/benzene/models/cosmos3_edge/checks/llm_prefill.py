"""All 28 LLM layers (prefill) on the DSP from rpcmem-resident exported weights (benzene cosmos3_edge_llm_prefill).

Inputs come from capture_activations.py (layer 0 input hidden states = embeddings with vision tokens merged,
M-RoPE cos / sin, layer 27 output of the recipe's fake-quant model). The model file must be on the board
(export_hexmodel.py, copied to paths.BOARD_DIR). The first --tokens prompt tokens run (causal: exact for them).

  python -m benzene.models.cosmos3_edge.checks.llm_prefill w8a8_sq05s_gptq --tokens 1024
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board

from ..paths import BOARD_DIR, OUT_DIR
from ..recipes import DEVICE_RECIPE


def fp32_llm(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, layers: int = 28) -> np.ndarray:
    """Unquantized LLM layers (checkpoint weights, float32 numpy, causal GQA attention)."""
    from ..device_layout import ckpt

    T = x.shape[0]
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)

    def rot(a):
        return a * cos[:, None] + np.concatenate([-a[..., 64:], a[..., :64]], -1) * sin[:, None]

    for l in range(layers):
        pre = f"model.language_model.layers.{l}."
        w = lambda n: ckpt(pre + n + ".weight")  # noqa: E731
        rms = lambda h, n: h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5) * w(n)  # noqa: E731
        h = rms(x, "input_layernorm")
        q = rot((h @ w("self_attn.q_proj").T).reshape(T, 16, 128))
        k = np.repeat(rot((h @ w("self_attn.k_proj").T).reshape(T, 8, 128)), 2, axis=1)
        v = np.repeat((h @ w("self_attn.v_proj").T).reshape(T, 8, 128), 2, axis=1)
        S = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(128) + mask
        Pm = np.exp(S - S.max(-1, keepdims=True))
        Pm /= Pm.sum(-1, keepdims=True)
        x = x + np.einsum("hqk,khd->qhd", Pm, v).reshape(T, -1) @ w("self_attn.o_proj").T
        x = x + np.maximum(rms(x, "post_attention_layernorm") @ w("mlp.fc1").T, 0) ** 2 @ w("mlp.fc2").T
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--worst", type=int, default=0, help="print the N tokens farthest from the unquantized model")
    args = parser.parse_args()
    acts = OUT_DIR / "acts" / args.recipe
    c0 = torch.load(acts / "model.language_model.layers.0.layer.pt")
    c27 = torch.load(acts / "model.language_model.layers.27.layer.pt")
    P = min(args.tokens, c0["x"].shape[0])
    x = c0["x"][:P].numpy().astype(np.float32)
    rope = np.concatenate([c0["cos"][:P, :64].numpy(), c0["sin"][:P, :64].numpy()]).astype(np.float32)
    y_ref = c27["y"][:P].numpy()
    model = f"{BOARD_DIR}/cosmos3_edge-{args.recipe}"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bundle").mkdir()
        (d / "bundle" / "x.f32").write_bytes(x.tobytes())
        (d / "bundle" / "rope.f32").write_bytes(rope.tobytes())
        work = board.empty_directory("pr")
        board.upload(f"{td}/bundle/", f"{work}/bundle/")
        r = board.run_benzene(f"cosmos3_edge_llm_prefill {model}.hexmodel pr/bundle pr/out")
        print(r.stdout.strip(), r.stderr.strip())
        if r.returncode:
            sys.exit(f"board run failed ({r.returncode})")
        board.download(f"{work}/out/y.f32", f"{td}/")
        y = np.frombuffer((d / "y.f32").read_bytes(), np.float32).reshape(P, -1)
    def err(a, b):
        return np.sqrt(((a - b).astype(np.float64) ** 2).mean()) / np.sqrt((b.astype(np.float64) ** 2).mean())

    def cosmin(a, b):
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
        return c.min(), np.median(c)

    y32 = fp32_llm(x, c0["cos"][:P].numpy(), c0["sin"][:P].numpy())
    print(f"LLM prefill (28 layers), {P} tokens, final hidden states: device vs PyTorch fake-quant {err(y, y_ref):.3e}; "
          f"vs unquantized fp32: device {err(y, y32):.3e} (cos min {cosmin(y, y32)[0]:.4f}), "
          f"fake-quant {err(y_ref, y32):.3e} (cos min {cosmin(y_ref, y32)[0]:.4f})")
    if args.worst:  # the tokens whose final hidden state is farthest from the unquantized one, for both
        def cosines(a, b):
            return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)

        device_cos, fake_quant_cos = cosines(y, y32), cosines(y_ref, y32)
        for token in np.argsort(device_cos)[: args.worst]:
            print(f"  token {token}: cos vs fp32 device {device_cos[token]:.4f}, fake-quant {fake_quant_cos[token]:.4f}, "
                  f"|fp32| {np.linalg.norm(y32[token]):.1f}")


if __name__ == "__main__":
    main()
