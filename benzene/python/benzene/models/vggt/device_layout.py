"""Device (Hexagon) layouts of the VGGT weights for the benzene runtime (dsp/models/vggt).

The weight streams are the kernels' (benzene.kernels), the part packing the model file's (benzene.model_file). Layout
makes the blobs of a VGGT-architecture checkpoint from two sources: its tensors by VGGT's module names and the cached
quantized weights of a recipe (recipes.quantize_weight); the module functions are those of VGGT's own checkpoint and
recipes (ports of other VGGT-architecture models make their own Layout, benzene.models.anysplat.device_layout).

Block blob (benzene protocol.h vision_transformer::block_part): fused qkv stream (rows q | k | v, each head's 64 dims in
two 32-column blocks, the order the qkv GEMM's kAttentionTiles64 output stage maps to the attention tiles), its scales
and bias; proj and fc2 with the LayerScale vectors ls1 / ls2 multiplied into their column scales and biases; fc1; the
norm affines times the smoothing inverses of the GEMMs they feed; 1 / s of proj and fc2 (applied by their input stages);
the q / k norm weights and biases of aggregator blocks (zeros for the DINOv2 blocks).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from benzene.kernels.gemm_fp16 import gemm_fp16_stream
from benzene.kernels.gemm_w8a8 import int8_codes, gemm_w8a8_stream
from benzene.model_file import pack_parts

from .paths import MODEL_DIR, OUT_DIR

# DPT head convolutions in benzene protocol.h dpt_head::convolution order (fp16 on the device: the device recipe's
# heads precision)
DPT_CONVS = (["projects.0", "projects.1", "projects.2", "projects.3", "resize_layers.0", "resize_layers.1",
              "resize_layers.3", "scratch.layer1_rn", "scratch.layer2_rn", "scratch.layer3_rn", "scratch.layer4_rn",
              "scratch.refinenet4.resConfUnit2.conv1", "scratch.refinenet4.resConfUnit2.conv2", "scratch.refinenet4.out_conv"]
             + [f"scratch.refinenet{k}.{m}" for k in (3, 2, 1)
                for m in ("resConfUnit1.conv1", "resConfUnit1.conv2", "resConfUnit2.conv1", "resConfUnit2.conv2", "out_conv")]
             + ["scratch.output_conv1", "scratch.output_conv2.0"])
TRANSPOSED = ("resize_layers.0", "resize_layers.1")


def block_names() -> list[tuple[str, int]]:
    """(prefix, kind) of the runtime's blocks in order: 24 DINOv2 blocks, then frame / global pairs."""
    out = [(f"aggregator.patch_embed.blocks.{i}.", 0) for i in range(24)]
    for i in range(24):
        out += [(f"aggregator.frame_blocks.{i}.", 1), (f"aggregator.global_blocks.{i}.", 2)]
    return out


def conv_gemm_matrix(w: np.ndarray, transposed: bool) -> np.ndarray:
    """A convolution's weight as the fp16 GEMM's [N, K]: K tap-major (tap, input channel); a transposed convolution
    (kernel = stride) has its columns in (tap, output channel) order and K = its input channels."""
    if transposed:  # [ci, co, kh, kw] -> rows (tap, co), columns ci
        ci, co, kh, kw = w.shape
        return w.transpose(2, 3, 1, 0).reshape(kh * kw * co, ci)
    co, ci, kh, kw = w.shape  # [co, ci, kh, kw] -> columns (tap, ci)
    return w.transpose(0, 2, 3, 1).reshape(co, kh * kw * ci)


class Layout:
    """The blobs of a VGGT-architecture checkpoint. tensor(name): a tensor by VGGT's module name as fp32 numpy (raises
    when there is none); record(recipe, name): the cached quantized weight of a W8A8 layer of the recipe."""

    def __init__(self, tensor: Callable[[str], np.ndarray], record: Callable[[str, str], dict]):
        self.tensor, self.record = tensor, record

    def w8a8_linear(self, recipe: str, name: str, col_mul: np.ndarray | None = None) -> tuple[bytes, np.ndarray]:
        """W8A8 stream of one Linear and its fp32 column scales (times col_mul, e.g. a LayerScale vector)."""
        c, s = int8_codes(self.record(recipe, name))
        s = s.astype(np.float32)
        if col_mul is not None:
            s = (s * col_mul).astype(np.float32)
        return gemm_w8a8_stream(c), s

    def inv_smooth(self, recipe: str, name: str) -> np.ndarray:
        """1 / s of the layer's SmoothQuant vector (ones without smoothing)."""
        r = self.record(recipe, name)
        if r["smooth"] is None:
            return np.ones(r["w"].shape[1] if r["w"].dim() == 2 else 1, np.float32)
        return (1.0 / r["smooth"].float()).numpy().astype(np.float32)

    def blob_block(self, recipe: str, pre: str, agg: bool) -> tuple[bytes, np.ndarray]:
        """One transformer block; pre e.g. 'aggregator.frame_blocks.3.' or 'aggregator.patch_embed.blocks.3.'"""
        t = self.tensor
        ls1, ls2 = t(pre + "ls1.gamma"), t(pre + "ls2.gamma")
        qkv, qkv_s = self.w8a8_linear(recipe, pre + "attn.qkv")
        o, o_s = self.w8a8_linear(recipe, pre + "attn.proj", ls1)
        fc1, fc1_s = self.w8a8_linear(recipe, pre + "mlp.fc1")
        fc2, fc2_s = self.w8a8_linear(recipe, pre + "mlp.fc2", ls2)
        i_qkv, i_fc1 = self.inv_smooth(recipe, pre + "attn.qkv"), self.inv_smooth(recipe, pre + "mlp.fc1")
        qk = (np.concatenate([t(pre + f"attn.{n}_norm.{p}") for n in "qk" for p in ("weight", "bias")]) if agg
              else np.zeros(256, np.float32))
        return pack_parts([qkv, qkv_s, t(pre + "attn.qkv.bias"),
                           o, o_s, t(pre + "attn.proj.bias") * ls1,
                           fc1, fc1_s, t(pre + "mlp.fc1.bias"),
                           fc2, fc2_s, t(pre + "mlp.fc2.bias") * ls2,
                           t(pre + "norm1.weight") * i_qkv, t(pre + "norm1.bias") * i_qkv,
                           self.inv_smooth(recipe, pre + "attn.proj"),
                           t(pre + "norm2.weight") * i_fc1, t(pre + "norm2.bias") * i_fc1,
                           self.inv_smooth(recipe, pre + "mlp.fc2"), qk])

    def blob_tail(self) -> tuple[bytes, np.ndarray]:
        """DINOv2 final norm, aggregator camera / register tokens, the DINOv2 patch embedding as the fp16 GEMM stream (K
        = 3 x 14 x 14 = 588 inputs zero-padded to 640; protocol.h vggt::tail_part)."""
        t = self.tensor
        pw = t("aggregator.patch_embed.patch_embed.proj.weight").reshape(1024, 588)
        return pack_parts([t("aggregator.patch_embed.norm.weight"), t("aggregator.patch_embed.norm.bias"),
                           t("aggregator.camera_token").reshape(2, 1024),
                           t("aggregator.register_token").reshape(2, 4, 1024),
                           gemm_fp16_stream(np.pad(pw, ((0, 0), (0, 640 - 588))),
                                            t("aggregator.patch_embed.patch_embed.proj.bias"))])

    def blob_cam_head(self, recipe: str) -> tuple[bytes, np.ndarray]:
        """Camera head outside its trunk blocks (benzene protocol.h vggt::camera_part)."""
        t, ch = self.tensor, "camera_head."
        mod, mod_s = self.w8a8_linear(recipe, ch + "poseLN_modulation.1")
        pb1, pb1_s = self.w8a8_linear(recipe, ch + "pose_branch.fc1")
        i_pb1 = self.inv_smooth(recipe, ch + "pose_branch.fc1")
        return pack_parts([t(ch + "token_norm.weight"), t(ch + "token_norm.bias"),
                           t(ch + "embed_pose.weight"), t(ch + "embed_pose.bias"), t(ch + "empty_pose_tokens").reshape(9),
                           mod, mod_s, t(ch + "poseLN_modulation.1.bias"),
                           self.inv_smooth(recipe, ch + "poseLN_modulation.1"),
                           t(ch + "trunk_norm.weight") * i_pb1, t(ch + "trunk_norm.bias") * i_pb1,
                           pb1, pb1_s, t(ch + "pose_branch.fc1.bias"),
                           t(ch + "pose_branch.fc2.weight"), t(ch + "pose_branch.fc2.bias")])

    def blob_dpt_conv(self, head: str, conv: str) -> tuple[bytes, np.ndarray]:
        """One convolution of the DPT head <head>_head (protocol.h dpt_head::convolution_part) as the fp16 GEMM: K
        tap-major (tap, input channel), the columns of a transposed convolution (tap, output channel), the bias per
        column in the store blocks."""
        name = f"{head}_head.{conv}"
        w = self.tensor(name + ".weight")
        try:
            b = self.tensor(name + ".bias")
        except Exception:  # noqa: BLE001 - layer<k>_rn have no bias
            b = np.zeros(w.shape[1] if conv in TRANSPOSED else w.shape[0], np.float32)
        if conv in TRANSPOSED:
            b = np.tile(b, w.shape[2] * w.shape[3])
        return pack_parts([gemm_fp16_stream(np.ascontiguousarray(conv_gemm_matrix(w, conv in TRANSPOSED)), b)])

    def blob_dpt_misc(self, head: str) -> tuple[bytes, np.ndarray]:
        """norm weight / bias, output_conv2.2 weight [c][32] and bias (protocol.h vggt::dpt_head_part)."""
        t, p = self.tensor, f"{head}_head."
        w = t(p + "scratch.output_conv2.2.weight")
        return pack_parts([t(p + "norm.weight"), t(p + "norm.bias"), w.reshape(w.shape[0], -1),
                           t(p + "scratch.output_conv2.2.bias")])


_CKPT = None


def ckpt(name: str) -> np.ndarray:
    """A tensor of the VGGT-1B checkpoint as fp32 numpy."""
    global _CKPT
    if _CKPT is None:
        from safetensors import safe_open

        _CKPT = safe_open(str(MODEL_DIR / "model.safetensors"), "pt")
    return _CKPT.get_tensor(name).float().numpy()


def _rec(recipe: str, name: str) -> dict:
    from .recipes import RECIPES, cache_path, spec_of

    spec = spec_of(RECIPES[recipe][1], name)
    assert spec is not None and spec.get("a8"), f"{name}: not a W8A8 layer of {recipe}"
    return torch.load(cache_path(name, spec, OUT_DIR)[0])


VGGT = Layout(ckpt, _rec)
