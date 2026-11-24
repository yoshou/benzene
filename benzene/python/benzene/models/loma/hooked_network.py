"""LoMa's convolutional networks - the DaD detector and DeDoDe-G's VGG and refiners - written out step by step with
BatchNorm folded in, so calibration and the recipes can look at and replace every tensor the device quantizes; without
hooks it computes what network.Detector / network.Descriptor compute (up to float rounding of the folds).

HookedDetector(net._detector) / HookedDescriptor(net._descriptor) take hooks (both optional):
  activation(name, x)          the tensor at quantization point `name` -> the tensor the network goes on with
  convolution(name, x, layer)  replaces convolution `name` of input x (layer: Convolution with the folded weight and
                               bias); None: the float convolution

The parts follow the device: a refiner's block outputs are linear (no ReLU: signed quantization points), its residual
(x + x0) / 1.4 is out_conv applied to the concatenation [x, x0] with both weight halves divided by 1.4 (one integer
GEMM on the device), block1's input is the concatenation [feature, context] (the context upsampled from the coarser
scale), and the out convolution's first `split` channels are the logits / description deltas (fp32 on the device)
and the rest the context. The detector's image enters normalized ((x - mean) / std, zero padding), the descriptor's
as x / 255. DINOv2 (the descriptor's 1/14 features) is one step: hook "desc.dino" sees its output map.

Names: <net>.enc.<i> (VGG convolution i with BatchNorm and ReLU), <net>.dec<s>.{block1.conv1, block1.conv2,
hidden<j>.dw, hidden<j>.pw, out} (refiner of scale s); activation points <net>.enc.<i> (ReLU outputs, the pools read
them), <net>.dec<s>.{context (the upsampled context entering scale s), block1.mid (ReLU), x0, hidden<j>.mid (ReLU),
hidden<j>.out}, desc.dino.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .network import IMAGENET_MEAN, IMAGENET_STD, ConvRefiner, Descriptor, Detector


@dataclass
class Convolution:
    weight: torch.Tensor  # [N, K / groups, kh, kw] fp32
    bias: torch.Tensor  # [N] fp32
    groups: int = 1

    @property
    def padding(self) -> int:
        return self.weight.shape[-1] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=self.padding, groups=self.groups)


def fold(conv: nn.Conv2d, bn: nn.BatchNorm2d | None) -> Convolution:
    """The convolution with the (eval) BatchNorm after it folded in."""
    weight = conv.weight.detach().float()
    bias = conv.bias.detach().float() if conv.bias is not None else torch.zeros(weight.shape[0], device=weight.device)
    if bn is not None:
        g = bn.weight.detach().float() / torch.sqrt(bn.running_var.float() + bn.eps)
        weight = weight * g[:, None, None, None]
        bias = (bias - bn.running_mean.float()) * g + bn.bias.detach().float()
    return Convolution(weight.contiguous(), bias.contiguous(), conv.groups)


def vgg_convolutions(layers: nn.ModuleList, prefix: str) -> tuple[dict[str, Convolution], list[bool]]:
    """The VGG features' convolutions with their BatchNorm (each followed by ReLU) and, per convolution, whether a max
    pool follows it (its output is a feature map)."""
    convolutions, pooled = {}, []
    modules = list(layers)
    for index, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            assert isinstance(modules[index + 1], nn.BatchNorm2d) and isinstance(modules[index + 2], nn.ReLU)
            convolutions[f"{prefix}.enc.{len(convolutions)}"] = fold(module, modules[index + 1])
            pooled.append(index + 3 < len(modules) and isinstance(modules[index + 3], nn.MaxPool2d))
    return convolutions, pooled


def refiner_convolutions(refiner: ConvRefiner, prefix: str) -> dict[str, Convolution]:
    out = {f"{prefix}.block1.conv1": fold(refiner.block1[0], refiner.block1[1]),
           f"{prefix}.block1.conv2": fold(refiner.block1[3], None)}
    for j, block in enumerate(refiner.hidden_blocks):
        out[f"{prefix}.hidden{j}.dw"] = fold(block[0], block[1])
        out[f"{prefix}.hidden{j}.pw"] = fold(block[3], None)
    final = fold(refiner.out_conv, None)
    out[f"{prefix}.out"] = Convolution(torch.cat([final.weight, final.weight], 1) / 1.4, final.bias)
    return out


class HookedConvNet:
    """The shared part of HookedDetector / HookedDescriptor: the convolutions by name, the hooks, the refiner chain."""

    activation_hook: Callable | None = None
    convolution_hook: Callable | None = None

    def __init__(self, prefix: str, encoder_layers: nn.ModuleList, refiners: dict[str, ConvRefiner], split: int,
                 activation=None, convolution=None) -> None:
        self.prefix = prefix
        self.split = split
        self.convolutions, self.pooled = vgg_convolutions(encoder_layers, prefix)
        self.scales = sorted(refiners, key=int, reverse=True)
        self.hidden_blocks = {s: len(refiners[s].hidden_blocks) for s in self.scales}
        for s in self.scales:
            self.convolutions.update(refiner_convolutions(refiners[s], f"{prefix}.dec{s}"))
        self.activation_hook = activation
        self.convolution_hook = convolution

    def act(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.activation_hook(name, x) if self.activation_hook is not None else x

    def conv(self, name: str, x: torch.Tensor) -> torch.Tensor:
        layer = self.convolutions[name]
        if self.convolution_hook is not None:
            y = self.convolution_hook(name, x, layer)
            if y is not None:
                return y
        return layer(x)

    def encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for i, pooled in enumerate(self.pooled):
            x = self.act(f"{self.prefix}.enc.{i}", F.relu(self.conv(f"{self.prefix}.enc.{i}", x)))
            if pooled:
                features.append(x)
                x = F.max_pool2d(x, 2, 2)
        return features

    def refine(self, scale: str, feature: torch.Tensor, context: torch.Tensor | None):
        """Refiner `scale` on its feature map and the upsampled context: (deltas, context) at the feature's size."""
        p = f"{self.prefix}.dec{scale}"
        x = feature if context is None else torch.cat([feature, self.act(f"{p}.context", context)], 1)
        x0 = self.act(f"{p}.x0", self.conv(f"{p}.block1.conv2",
                                           self.act(f"{p}.block1.mid", F.relu(self.conv(f"{p}.block1.conv1", x)))))
        h = x0
        for j in range(self.hidden_blocks[scale]):
            h = self.act(f"{p}.hidden{j}.out", self.conv(f"{p}.hidden{j}.pw", self.act(
                f"{p}.hidden{j}.mid", F.relu(self.conv(f"{p}.hidden{j}.dw", h)))))
        out = self.conv(f"{p}.out", torch.cat([h, x0], 1))
        return out[:, : self.split], out[:, self.split :]


class HookedDetector(HookedConvNet):
    """DaD: __call__(images [B, 3, H, W] in [0, 1]) -> logits [B, 1, H, W] fp32."""

    def __init__(self, detector: Detector, activation=None, convolution=None) -> None:
        super().__init__("det", detector.encoder.layers, dict(detector.decoder.layers), 1, activation, convolution)
        device = detector.mean.device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None]
        self.std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder((images - self.mean) / self.std)
        logits, context = None, None
        for index, scale in enumerate(self.scales):
            delta, context = self.refine(scale, features[-(index + 1)], context)
            logits = delta.float() if logits is None else logits + delta.float()
            if index < len(self.scales) - 1:
                size = features[-(index + 2)].shape[-2:]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return logits


class HookedDescriptor(HookedConvNet):
    """DeDoDe-G: __call__(images [B, 3, H, W] in [0, 1], H and W multiples of 14) -> descriptions [B, D, H, W]."""

    def __init__(self, descriptor: Descriptor, activation=None, convolution=None) -> None:
        assert not descriptor.imagenet_normalization and "14" in descriptor.decoder.layers, "DeDoDe-G only"
        super().__init__("desc", descriptor.encoder.vgg.layers, dict(descriptor.decoder.layers),
                         descriptor.decoder.split, activation, convolution)
        self.dinov2 = descriptor.encoder.frozen_dinov2.dinov2_vitl14

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images) + [self.act("desc.dino", self.dinov2(images))]
        descriptions, context = None, None
        for index, scale in enumerate(self.scales):
            delta, context = self.refine(scale, features[-(index + 1)], context)
            descriptions = delta if descriptions is None else descriptions + delta
            if index < len(self.scales) - 1:
                size = features[-(index + 2)].shape[-2:]
                descriptions = F.interpolate(descriptions, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return descriptions
