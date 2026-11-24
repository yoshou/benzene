"""Device layouts of the Cosmos 3 Edge weights for the benzene runtime (benzene/dsp/models/cosmos3_edge): the layer
blobs of the model file, in the part order of benzene/include/benzene/protocol.h (cosmos3_edge::llm_layer_part,
vision_layer_part, vision_tail_part), built from the checkpoint and the cached quantized weights
(recipes.quantize_weight).
"""

from __future__ import annotations

import numpy as np
import torch

from benzene.kernels.gemm_w8a8 import int8_codes, gemm_w8a8_stream
from benzene.kernels.gemv import q4_codes, gemv_q4_stream
from benzene.model_file import pack_parts

_CKPT_INDEX: dict | None = None

# The language model's tensors are in the omni transformer's checkpoint (transformer/, shared with the generator tower)
# without the "model.language_model." prefix and with that checkpoint's projection names; the vision encoder, the
# projector and the lm_head keep their module names.
LANGUAGE_MODEL_PREFIX = "model.language_model."
LANGUAGE_MODEL_PROJECTIONS = {
    ".self_attn.q_proj.": ".self_attn.to_q.",
    ".self_attn.k_proj.": ".self_attn.to_k.",
    ".self_attn.v_proj.": ".self_attn.to_v.",
    ".self_attn.o_proj.": ".self_attn.to_out.",
    ".mlp.fc1.": ".mlp.up_proj.",
    ".mlp.fc2.": ".mlp.down_proj.",
}


def checkpoint_key(name: str) -> str:
    """The safetensors key of a Cosmos3EdgeForConditionalGeneration parameter (module-style name)."""
    if not name.startswith(LANGUAGE_MODEL_PREFIX):
        return name
    key = name[len(LANGUAGE_MODEL_PREFIX):]
    for module_name, checkpoint_name in LANGUAGE_MODEL_PROJECTIONS.items():
        key = key.replace(module_name, checkpoint_name)
    return key


def ckpt(name: str) -> np.ndarray:
    """A checkpoint tensor (module-style name) as fp32 numpy."""
    import json

    from safetensors import safe_open

    from .paths import MODEL_DIR

    global _CKPT_INDEX
    if _CKPT_INDEX is None:
        _CKPT_INDEX = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    key = checkpoint_key(name)
    with safe_open(str(MODEL_DIR / _CKPT_INDEX[key]), "pt") as f:
        return f.get_tensor(key).float().numpy()


def _rec(recipe: str, name: str) -> dict:
    from .recipes import RECIPES, _cache_path, spec_of

    return torch.load(_cache_path(name, spec_of(RECIPES[recipe], name))[0])


def _rec_dec(recipe: str, name: str) -> dict:
    """The decode-side record of a mixed (prefill / decode) recipe; the layer's own record otherwise."""
    from .recipes import RECIPES, _cache_path, spec_of

    spec = spec_of(RECIPES[recipe], name)
    return torch.load(_cache_path(name, spec.get("decode", spec))[0])


def w8a8_linear(recipe: str, names: list[str], rows: int | None = None, cols: int | None = None,
                perm: np.ndarray | None = None):
    """Fused W8A8 stream of Linear layers (rows concatenated, optionally reordered by perm), optionally zero-padded
    -> (bytes, scales)."""
    cs, ss = zip(*(int8_codes(_rec(recipe, n)) for n in names))
    c, s = np.concatenate(cs), np.concatenate(ss).astype(np.float32)
    if perm is not None:
        c, s = c[perm], s[perm]
    if rows and c.shape[0] < rows:
        s = np.pad(s, (0, rows - c.shape[0]))
        c = np.pad(c, ((0, rows - c.shape[0]), (0, 0)))
    if cols and c.shape[1] < cols:
        c = np.pad(c, ((0, 0), (0, cols - c.shape[1])))
    return gemm_w8a8_stream(c), s


def inv_smooth(recipe: str, name: str, size: int | None = None) -> np.ndarray:
    v = (1.0 / _rec(recipe, name)["smooth"].float()).numpy().astype(np.float32)
    return np.pad(v, (0, (size or len(v)) - len(v)))


def rope_dim_perm(heads: int, d: int = 128) -> np.ndarray:
    """Q / K output rows of the decoder layers: each head's dims as [0..31, 64..95, 32..63, 96..127], so that the
    device's qkv GEMM finds the rotate_half RoPE pairs (i, i + 64) in neighbouring 32-column blocks."""
    h = np.concatenate([np.arange(0, 32), np.arange(64, 96), np.arange(32, 64), np.arange(96, 128)])
    return (np.arange(heads)[:, None] * d + h[None, :]).reshape(-1)


QKV_PERM = np.concatenate([rope_dim_perm(16), 2048 + rope_dim_perm(8), 3072 + np.arange(1024)])  # 16 q / 8 kv heads


def blob_decoder_layer(recipe: str, pre: str) -> tuple[bytes, np.ndarray]:
    """Language model decoder layer (protocol.h cosmos3_edge::llm_layer_part parts); pre e.g.
    'model.language_model.layers.5.'. Q / K rows in
    the rope_dim_perm order."""
    qkv, qkv_s = w8a8_linear(recipe, [pre + f"self_attn.{p}_proj" for p in "qkv"], perm=QKV_PERM)
    o, o_s = w8a8_linear(recipe, [pre + "self_attn.o_proj"])
    fc1, fc1_s = w8a8_linear(recipe, [pre + "mlp.fc1"])
    fc2, fc2_s = w8a8_linear(recipe, [pre + "mlp.fc2"])
    g1 = ckpt(pre + "input_layernorm.weight") * inv_smooth(recipe, pre + "self_attn.q_proj")
    g2 = ckpt(pre + "post_attention_layernorm.weight") * inv_smooth(recipe, pre + "mlp.fc1")
    return pack_parts([qkv, qkv_s, o, o_s, fc1, fc1_s, fc2, fc2_s, g1, inv_smooth(recipe, pre + "self_attn.o_proj"),
                       g2, inv_smooth(recipe, pre + "mlp.fc2")])


VIS_MLP_PAD = 4320  # vision mlp 4304 -> 4320
VIS_HD, VIS_H = 72, 16  # vision head dim, heads
# fused q|k|v rows, per projection each head's dims 0..63 (two 32-column blocks), then dims 64..71 of heads 4m .. 4m+3
# in block 32 + m: the qkv GEMM's column blocks map to the attention's 32-dim tiles (the device writes them directly)
VIS_QKV_PERM = np.concatenate([
    np.concatenate([h * VIS_HD + np.arange(64) for h in range(VIS_H)] + [h * VIS_HD + 64 + np.arange(8) for h in range(VIS_H)])
    + p * VIS_H * VIS_HD for p in range(3)])


def blob_vision_layer(recipe: str, pre: str) -> tuple[bytes, np.ndarray]:
    """Vision encoder layer (cosmos3_edge::vision_layer_part parts); pre e.g. 'model.visual.encoder.layers.3.'"""
    F = VIS_MLP_PAD

    def bias(names, n=None):
        b = np.concatenate([ckpt(pre + nm + ".bias") for nm in names]).astype(np.float32)
        return np.pad(b, (0, (n or len(b)) - len(b)))

    qkv_names = [f"self_attn.{p}_proj" for p in "qkv"]
    qkv, qkv_s = w8a8_linear(recipe, [pre + n for n in qkv_names], perm=VIS_QKV_PERM)
    qkv_b = bias(qkv_names)[VIS_QKV_PERM].astype(np.float32)
    o, o_s = w8a8_linear(recipe, [pre + "self_attn.out_proj"])
    fc1, fc1_s = w8a8_linear(recipe, [pre + "mlp.fc1"], rows=F)
    fc2, fc2_s = w8a8_linear(recipe, [pre + "mlp.fc2"], cols=F)
    i_qkv, i_fc1 = inv_smooth(recipe, pre + "self_attn.q_proj"), inv_smooth(recipe, pre + "mlp.fc1")
    return pack_parts([qkv, qkv_s, qkv_b, o, o_s, bias(["self_attn.out_proj"]),
                       fc1, fc1_s, bias(["mlp.fc1"], F), fc2, fc2_s, bias(["mlp.fc2"]),
                       ckpt(pre + "layer_norm1.weight") * i_qkv, ckpt(pre + "layer_norm1.bias") * i_qkv,
                       inv_smooth(recipe, pre + "self_attn.out_proj"),
                       ckpt(pre + "layer_norm2.weight") * i_fc1, ckpt(pre + "layer_norm2.bias") * i_fc1,
                       inv_smooth(recipe, pre + "mlp.fc2", F)])


def blob_vision_tail(recipe: str) -> tuple[bytes, np.ndarray]:
    """Vision post_layernorm + projector (cosmos3_edge::vision_tail_part parts)."""
    pj = "model.projector."
    inv1 = inv_smooth(recipe, pj + "linear_fc1")
    fc1, fc1_s = w8a8_linear(recipe, [pj + "linear_fc1"])
    fc2, fc2_s = w8a8_linear(recipe, [pj + "linear_fc2"])
    return pack_parts([ckpt("model.visual.post_layernorm.weight"), ckpt("model.visual.post_layernorm.bias"),
                       np.tile(ckpt(pj + "norm.weight"), 4) * inv1, np.tile(ckpt(pj + "norm.bias"), 4) * inv1,
                       fc1, fc1_s, ckpt(pj + "linear_fc1.bias"), inv_smooth(recipe, pj + "linear_fc2"),
                       fc2, fc2_s, ckpt(pj + "linear_fc2.bias")])


def q4_linear(recipe: str, names: list[str], perm: np.ndarray | None = None) -> bytes:
    """Fused Q4_0 stream of the decode-side weights of Linear layers (rows concatenated, optionally reordered)."""
    cs, ss = zip(*(q4_codes(_rec_dec(recipe, n)) for n in names))
    c, s = np.concatenate(cs), np.concatenate(ss)
    if perm is not None:
        c, s = c[perm], s[perm]
    return gemv_q4_stream(c, s)


def blob_decoder_layer_dec4(recipe: str, base: str, pre: str) -> tuple[bytes, np.ndarray]:
    """The four Q4_0 decode streams (qkv, o, fc1, fc2) of an LLM layer of the mixed recipe `recipe`. The device feeds
    them the activations it prepares for the int8 copy of `base`, so both must use the same smoothing vectors."""
    for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.fc1", "mlp.fc2"):
        assert torch.equal(_rec_dec(recipe, pre + n)["smooth"], _rec(base, pre + n)["smooth"]), f"{pre + n}: smoothing differs"
    return pack_parts([q4_linear(recipe, [pre + f"self_attn.{p}_proj" for p in "qkv"], perm=QKV_PERM),
                       q4_linear(recipe, [pre + "self_attn.o_proj"]), q4_linear(recipe, [pre + "mlp.fc1"]),
                       q4_linear(recipe, [pre + "mlp.fc2"])])
