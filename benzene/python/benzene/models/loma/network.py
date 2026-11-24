"""LoMa's inference networks, written after the davnords/LoMa checkout (commit ecc947d; paths.SRC_DIR) with its
parameter names, so that its checkpoints load with strict=True: the frozen DaD detector (loma/detector/dad.py), the
frozen DeDoDe descriptor (loma/descriptor/dedode.py, DINOv2 from loma/descriptor/transformer) and the LoMa matcher
(loma/loma.py, after LightGlue). The checkout's own modules are not used: they run in autocast with the device and AMP
dtype chosen at import time, and the recipes need explicit precision.

  Detector    DaD (dedode_detector_S): VGG-11-BN features before each max pool (64 / 128 / 256 / 512 channels at 1, 1/2,
              1/4, 1/8), a ConvRefiner per scale from 1/8 up (1 logit + context; the logits summed with bicubic
              upsampling, the context upsampled bilinearly), ImageNet-normalized input
  Descriptor  DeDoDe-G: VGG-19-BN features as above plus DINOv2 ViT-L/14 patch tokens (1/14, 1024 channels), a
              ConvRefiner per scale from 1/14 up (description + context, both upsampled bilinearly), unnormalized
              input; DeDoDe-B (VGG only, from 1/8) for LoMa-B128 / B64
  Matcher     9 LightGlue layers (self attention with rotary encoding of the normalized keypoints, cross attention, FFN
              2D -> 2D -> LayerNorm -> GELU -> D), the last layer's dual-softmax assignment softmax(S, 1) softmax(S, 0)
              (the matchability head is only trained), mutual nearest neighbours above 0.1
Keypoints are normalized to [-1, 1] per axis with pixel centres at -1 + (2 i + 1) / size (the checkout's convention).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VGG_LAYERS = {"11": 22, "13": 28, "19": 40}  # torchvision features[:n]: up to the fourth max pool


class VGGEncoder(nn.Module):
    """torchvision's VGG-BN features up to the fourth max pool; forward returns the maps before each max pool."""

    def __init__(self, size: str) -> None:
        super().__init__()
        constructor = {"11": tvm.vgg11_bn, "13": tvm.vgg13_bn, "19": tvm.vgg19_bn}[size]
        self.layers = nn.ModuleList(constructor(weights=None).features[: VGG_LAYERS[size]])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)
            x = layer(x)
        return features


def conv_block(in_dim: int, out_dim: int, depthwise: bool, kernel_size: int) -> nn.Sequential:
    """ConvRefiner.create_block: conv (depthwise or full), BatchNorm, ReLU, 1x1 conv."""
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, 1, kernel_size // 2, groups=in_dim if depthwise else 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, 1, 1, 0),
    )


class ConvRefiner(nn.Module):
    """block1 (1x1 conv block), hidden_blocks (5x5 depthwise conv blocks), (x + x0) / 1.4, out_conv (1x1)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden_blocks: int) -> None:
        super().__init__()
        self.block1 = conv_block(in_dim, hidden_dim, False, 1)
        self.hidden_blocks = nn.Sequential(*[conv_block(hidden_dim, hidden_dim, True, 5) for _ in range(hidden_blocks)])
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.block1(x)
        x = self.hidden_blocks(x0)
        return self.out_conv((x + x0) / 1.4)


class Decoder(nn.Module):
    """The ConvRefiners per scale (layers[scale]); forward splits the output into (delta, context) at `split`."""

    def __init__(self, layers: dict[str, ConvRefiner], split: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.split = split
        self.scales = sorted(layers, key=int, reverse=True)

    def forward(self, features: torch.Tensor, context: torch.Tensor | None, scale: str):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        out = self.layers[scale](features)
        return out[:, : self.split], out[:, self.split :]


class Detector(nn.Module):
    """DaD: forward(images [B, 3, H, W] in [0, 1]) -> logits [B, 1, H, W] (fp32)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = VGGEncoder("11")
        self.decoder = Decoder({"8": ConvRefiner(512, 512, 256 + 1, 3), "4": ConvRefiner(256 + 256, 256, 128 + 1, 3),
                                "2": ConvRefiner(128 + 128, 64, 32 + 1, 3), "1": ConvRefiner(64 + 32, 32, 1 + 1, 3)},
                               split=1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None], persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder((images - self.mean) / self.std)
        logits, context = None, None
        for index, (feature_map, scale) in enumerate(zip(reversed(features), self.decoder.scales)):
            delta, context = self.decoder(feature_map, context, scale)
            logits = delta.float() if logits is None else logits + delta.float()
            if index < len(features) - 1:
                size = features[-(index + 2)].shape[-2:]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        return logits.float()


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, dim: int = 1024) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] % self.patch_size == 0 and x.shape[-1] % self.patch_size == 0, x.shape
        return self.proj(x).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    """DINOv2 ViT-L/14 (dinov2_vitl14_pretrain: no registers, 518 px position grid of 37 x 37); forward -> the
    normalized patch tokens as a map [B, 1024, H / 14, W / 14]."""

    def __init__(self, dim: int = 1024, depth: int = 24, heads: int = 16, grid: int = 37) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(14, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # (training only)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def position_embedding(self, rows: int, columns: int) -> torch.Tensor:
        """interpolate_pos_encoding: the 37 x 37 grid resized bicubically (scale factors (n + 0.1) / 37) to rows x
        columns, the class token's entry first; [1, 1 + rows columns, dim] fp32."""
        pos_embed = self.pos_embed.float()
        grid = int(math.sqrt(pos_embed.shape[1] - 1))
        if rows == grid and columns == grid:
            return pos_embed
        patch = pos_embed[:, 1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, scale_factor=((rows + 0.1) / grid, (columns + 0.1) / grid), mode="bicubic")
        assert patch.shape[-2:] == (rows, columns), patch.shape
        return torch.cat((pos_embed[:, :1], patch.permute(0, 2, 3, 1).reshape(1, rows * columns, -1)), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        rows, columns = H // 14, W // 14
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1).to(tokens.dtype), tokens), dim=1)
        tokens = tokens + self.position_embedding(rows, columns).to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)[:, 1:]
        return tokens.permute(0, 2, 1).reshape(B, -1, rows, columns)


class FrozenDINOv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dinov2_vitl14 = DinoVisionTransformer()


class VGGDINOv2Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = VGGEncoder("19")
        self.frozen_dinov2 = FrozenDINOv2()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.vgg(x) + [self.frozen_dinov2.dinov2_vitl14(x)]


class Descriptor(nn.Module):
    """DeDoDe: forward(images [B, 3, H, W] in [0, 1]) -> descriptions [B, D, H, W]; arch "dedode_g" (VGG-19 + DINOv2,
    H, W multiples of 14) or "dedode_b" (VGG of vgg_size)."""

    def __init__(self, arch: str, descriptor_dim: int, vgg_size: str = "19", imagenet_normalization: bool = False,
                 hidden_blocks: int = 5) -> None:
        super().__init__()
        d = descriptor_dim
        layers = {"8": ConvRefiner(512, 512, 256 + d, hidden_blocks), "4": ConvRefiner(256 + 256, 256, 128 + d,
                  hidden_blocks), "2": ConvRefiner(128 + 128, 64, 32 + d, hidden_blocks),
                  "1": ConvRefiner(64 + 32, 32, 1 + d, hidden_blocks)}
        if arch == "dedode_g":
            layers = {"14": ConvRefiner(1024, 768, 512 + d, hidden_blocks),
                      "8": ConvRefiner(512 + 512, 512, 256 + d, hidden_blocks), **{k: v for k, v in layers.items()
                                                                                   if k != "8"}}
            self.encoder = VGGDINOv2Encoder()
        else:
            self.encoder = VGGEncoder(vgg_size)
        self.decoder = Decoder(layers, split=d)
        self.imagenet_normalization = imagenet_normalization
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[None, :, None, None], persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.imagenet_normalization:
            images = (images - self.mean) / self.std
        features = self.encoder(images)
        descriptions, context = None, None
        for index, (feature_map, scale) in enumerate(zip(reversed(features), self.decoder.scales)):
            delta, context = self.decoder(feature_map, context, scale)
            descriptions = delta if descriptions is None else descriptions + delta
            if index < len(features) - 1:
                size = features[-(index + 2)].shape[-2:]
                descriptions = F.interpolate(descriptions, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return descriptions


def sample_descriptions(descriptions: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    """describe_keypoints: the description map [B, D, H, W] sampled bilinearly at the normalized keypoints [B, N, 2]
    (align_corners False, zeros outside) -> [B, N, D] fp32."""
    return F.grid_sample(descriptions.float(), keypoints[:, None], mode="bilinear", align_corners=False)[:, :, 0].mT


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary(encoding: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return t * encoding[0] + rotate_half(t) * encoding[1]


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.Wr = nn.Linear(2, head_dim // 2, bias=False)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """[2 (cos, sin), B, 1, N, head_dim], each frequency twice (interleaved pairs)."""
        projected = self.Wr(keypoints)
        return torch.stack([torch.cos(projected), torch.sin(projected)], 0).unsqueeze(-3).repeat_interleave(2, dim=-1)


def feed_forward(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim, elementwise_affine=True), nn.GELU(),
                         nn.Linear(2 * dim, dim))


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = feed_forward(dim)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv = self.Wqkv(x).unflatten(-1, (self.heads, -1, 3)).transpose(1, 2)  # channel h 3 hd + 3 d + (q, k, v)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2].contiguous()
        context = F.scaled_dot_product_attention(apply_rotary(encoding, q), apply_rotary(encoding, k), v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = feed_forward(dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def heads(t):
            return t.unflatten(-1, (self.heads, -1)).transpose(1, 2)

        qk0, qk1, v0, v1 = heads(self.to_qk(x0)), heads(self.to_qk(x1)), heads(self.to_v(x0)), heads(self.to_v(x1))
        m0 = F.scaled_dot_product_attention(qk0, qk1, v1).transpose(1, 2).flatten(start_dim=-2)
        m1 = F.scaled_dot_product_attention(qk1, qk0, v0).transpose(1, 2).flatten(start_dim=-2)
        return x0 + self.ffn(torch.cat([x0, self.to_out(m0)], -1)), x1 + self.ffn(torch.cat([x1, self.to_out(m1)], -1))


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(dim, heads)
        self.cross_attn = CrossBlock(dim, heads)

    def forward(self, desc0, desc1, encoding0, encoding1):
        return self.cross_attn(self.self_attn(desc0, encoding0), self.self_attn(desc1, encoding1))


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.matchability = nn.Linear(dim, 1)  # (trained, unused at inference)
        self.final_proj = nn.Linear(dim, dim)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        """softmax(S, 2) softmax(S, 1) of S = (P d0 / D^1/4)(P d1 / D^1/4)^T."""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        d = mdesc0.shape[-1]
        sim = torch.einsum("bmd,bnd->bmn", mdesc0 / d**0.25, mdesc1 / d**0.25)
        return F.softmax(sim, dim=2) * F.softmax(sim, dim=1)


def filter_matches(scores: torch.Tensor, threshold: float):
    """Mutual nearest neighbours of the assignment above threshold: (m0, m1, mscores0, mscores1), -1 = unmatched."""
    max0, max1 = scores.max(2), scores.max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    mscores0 = torch.where(mutual0, max0.values, max0.values.new_tensor(0))
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), mscores0.new_tensor(0))
    valid0 = mutual0 & (mscores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, m1)
    return torch.where(valid0, m0, -1), torch.where(valid1, m1, -1), mscores0, mscores1


class LoMaNetwork(nn.Module):
    """A LoMa checkpoint's networks: the matcher (input_proj, posenc, transformers, log_assignment) at the top level,
    _detector and _descriptor below, as in the checkout's LoMa module."""

    def __init__(self, input_dim: int = 256, embed_dim: int = 256, heads: int = 4, layers: int = 9,
                 descriptor: str = "dedode_g", vgg_size: str = "19", imagenet_normalization: bool = False) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.posenc = LearnableFourierPositionalEncoding(embed_dim // heads)
        self.transformers = nn.ModuleList([TransformerLayer(embed_dim, heads) for _ in range(layers)])
        self.log_assignment = nn.ModuleList([MatchAssignment(embed_dim) for _ in range(layers)])
        self._detector = Detector()
        self._descriptor = Descriptor(descriptor, input_dim, vgg_size, imagenet_normalization)

    def match(self, kpts0: torch.Tensor, kpts1: torch.Tensor, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        """The assignment scores [B, N0, N1] of normalized keypoints [B, N, 2] and descriptions [B, N, input_dim]."""
        desc0, desc1 = self.input_proj(desc0), self.input_proj(desc1)
        encoding0, encoding1 = self.posenc(kpts0), self.posenc(kpts1)
        for layer in self.transformers:
            desc0, desc1 = layer(desc0, desc1, encoding0, encoding1)
        return self.log_assignment[len(self.transformers) - 1](desc0, desc1)


VARIANTS = {  # the checkout's configurations: (constructor arguments, checkpoint file)
    "loma_b": ({}, "loma_B.pt"),
    "loma_b128": ({"input_dim": 128, "descriptor": "dedode_b"}, "loma_B128.pth"),
    "loma_b64": ({"input_dim": 64, "descriptor": "dedode_b", "vgg_size": "11", "imagenet_normalization": True},
                 "loma_B64.pth"),
}


def load_network(variant: str, weights_dir, device: str = "cuda:0") -> LoMaNetwork:
    """The variant's network with its checkpoint (all parameters fp32: DINOv2's bf16 weights converted exactly)."""
    arguments, file = VARIANTS[variant]
    net = LoMaNetwork(**arguments)
    state = torch.load(f"{weights_dir}/{file}", map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=True)
    return net.eval().to(device)
