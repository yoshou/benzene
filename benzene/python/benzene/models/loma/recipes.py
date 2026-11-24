"""Quantization recipes for LoMa. A recipe name is "<cnn>+<dino>+<matcher>" (CNN_RECIPES, DINO_RECIPES,
MATCHER_RECIPES); "fp32" / "fp16" alone mean the same for all three.

CNN (the DaD detector and DeDoDe-G's VGG-19 and refiners, hooked_network):
  fp16  autocast fp16
  fq*   int8 fake quantization (the device's int8 networks, float arithmetic otherwise): every
        convolution with BatchNorm folded in, int8 weights per output channel (w: "gptq", or "rtn"; the depthwise 5x5
        ones always by RTN per channel) with the input's per-channel scales folded in; every activation point uint8 per
        channel x = s (q - z) with a static scale and zero point from the calibration histograms
        (benzene.quantization.activation_quantizer: clip "mse" / "p4" / "max", zp "channel" / "layer" / "128"); the
        signed points (block outputs, the upsampled context, DINOv2's output) shrink both ends by one factor. The
        images are exact (codes = pixels: the detector's normalization and the descriptor's / 255 are scales).
        Not quantized: the logits and description deltas (fp32 outputs), their upsampled sums.
DINOv2 (dinov2_quantization): fp32, fp16 (autocast), w8a8 (every block Linear W8A8 as the VGGT / Cosmos 3 Edge
device path: per-channel int8 GPTQ weights, per-token int8 activations, SmoothQuant 0.5; the rest fp16).
Matcher: fp32, fp16 (autocast; attention in fp16 as the device's flash attention).
"""

from __future__ import annotations

import types

import torch

from benzene.quantization import FORMATS, activation_quantizer, int8_weight_codes, quantize_activation, rtn

from ..lightglue.matcher import autocast
from .calibrate import load_calibration
from .hooked_network import Convolution, HookedDescriptor, HookedDetector

CNN_RECIPES: dict[str, dict | str | None] = {
    "fp32": None,
    "fp16": "fp16",
    "fq": dict(w="gptq", clip="mse", zp="channel"),
    "fq_rtn": dict(w="rtn", clip="mse", zp="channel"),
    "fq_p4": dict(w="gptq", clip="p4", zp="channel"),
    "fq_zl": dict(w="gptq", clip="mse", zp="layer"),
    # the device recipe: fq's quantization on the gray device input in the device's arithmetic
    # (device_model.DeviceLoMa from device_records.quantize_dev)
    "dev": dict(w="gptq", clip="mse", zp="channel", dev=True),
    # one network quantized, the other fp32
    "fq_det": dict(w="gptq", clip="mse", zp="channel", only="det"),
    "fq_desc": dict(w="gptq", clip="mse", zp="channel", only="desc"),
}
DINO_RECIPES = {"fp32": None, "fp16": "fp16", "w8a8": "w8a8"}
MATCHER_RECIPES = {"fp32": None, "fp16": "fp16"}
SIGNED_SUFFIXES = (".x0", ".out", ".context", "desc.dino")  # no ReLU before them


def split(recipe: str) -> tuple[str, str, str]:
    if recipe in ("fp32", "fp16"):
        return recipe, recipe, recipe
    parts = recipe.split("+")
    if len(parts) != 3 or parts[0] not in CNN_RECIPES or parts[1] not in DINO_RECIPES or parts[2] not in MATCHER_RECIPES:
        raise ValueError(f"recipe {recipe!r}: <{'|'.join(CNN_RECIPES)}>+<{'|'.join(DINO_RECIPES)}>+"
                         f"<{'|'.join(MATCHER_RECIPES)}>")
    return parts[0], parts[1], parts[2]


def signed(name: str) -> bool:
    return name.endswith(SIGNED_SUFFIXES)


def input_points(name: str, net) -> list[str]:
    """The activation points a convolution reads, in its K order ([] for an image input)."""
    p, _, layer = name.rpartition(".")
    if ".enc." in name:
        index = int(layer)
        return [] if index == 0 else [f"{net.prefix}.enc.{index - 1}"]
    scale_prefix, part = name.split(".dec")[0], name.split(".dec")[1]
    scale, _, rest = part.partition(".")
    d = f"{scale_prefix}.dec{scale}"
    if rest == "block1.conv1":
        feature = feature_point(net, scale)
        return [feature] if scale == net.scales[0] else [feature, f"{d}.context"]
    if rest == "block1.conv2":
        return [f"{d}.block1.mid"]
    if rest == "out":
        return [f"{d}.hidden{net.hidden_blocks[scale] - 1}.out", f"{d}.x0"]
    j = int(rest.split(".")[0][6:])
    if rest.endswith(".dw"):
        return [f"{d}.x0" if j == 0 else f"{d}.hidden{j - 1}.out"]
    return [f"{d}.hidden{j}.mid"]


def feature_point(net, scale: str) -> str:
    """The activation point of the feature map refiner `scale` reads (the encoder output before a pool, DINOv2)."""
    if scale == "14":
        return "desc.dino"
    pooled = [i for i, p in enumerate(net.pooled) if p]  # encoder convolutions whose outputs are the feature maps
    return f"{net.prefix}.enc.{pooled[{'1': 0, '2': 1, '4': 2, '8': 3}[scale]]}"


def image_scales(net) -> torch.Tensor:
    """Per-channel scale of the image codes (pixels 0..255) as the first convolution's input."""
    if net.prefix == "det":
        return (1.0 / (255.0 * net.std.double().cpu().reshape(-1)))
    return torch.full((3,), 1.0 / 255, dtype=torch.float64)


def fake_quantize(net, spec: dict, calibration: dict, device: str, verbose: bool = False):
    """Sets net's hooks to spec's int8 fake quantization (module docstring)."""
    activations = calibration["activations"]
    scales = {name: activation_quantizer(activations[name], spec["clip"], spec["zp"], signed(name))
              for name in activations if name.startswith(net.prefix)}
    weights: dict[str, Convolution] = {}
    for name, layer in net.convolutions.items():
        points = input_points(name, net)
        s_in = torch.cat([scales[p][0] for p in points]) if points else image_scales(net)
        if layer.groups > 1:  # depthwise: per channel its 25 taps, scales folded per channel
            w = layer.weight.double().cpu().reshape(layer.weight.shape[0], -1) * s_in[:, None]
            q, ws = rtn(w.float(), FORMATS["int8_pc"])
            dequantized = (q.double() / s_in[:, None]).reshape(layer.weight.shape)
        else:
            taps = layer.weight.shape[-1] ** 2
            matrix = layer.weight.double().cpu().reshape(layer.weight.shape[0], -1)
            codes, s_w = int8_weight_codes(matrix, s_in.repeat_interleave(taps), calibration["hessians"][name]["H"],
                                           spec["w"], device)
            dequantized = (codes * s_w[:, None] / s_in.repeat_interleave(taps)[None]).reshape(layer.weight.shape)
        weights[name] = Convolution(dequantized.float().to(layer.weight.device), layer.bias, layer.groups)
        if verbose:
            print(f"  {name}: {tuple(layer.weight.shape)}")
    net.convolution_hook = lambda name, x, layer: weights[name](x)

    def activation(name: str, x: torch.Tensor) -> torch.Tensor:
        scale, zero = scales[name]
        return quantize_activation(x, scale, zero, 1)

    net.activation_hook = activation


def apply_recipe(net, recipe: str, settings, device: str, verbose: bool = True):
    """An object with LoMa's interface for runner.extract / match (_detector, _descriptor, match) running `recipe`,
    and the autocast dtypes (extract, match)."""
    cnn, dino, matcher_recipe = split(recipe)
    match_dtype = torch.float16 if MATCHER_RECIPES[matcher_recipe] == "fp16" else None
    spec = CNN_RECIPES[cnn]
    if isinstance(spec, dict) and spec.get("dev"):
        from .device_model import DeviceLoMa
        from .device_records import quantize_dev
        from .dinov2_quantization import quantized_dinov2

        dinov2 = net._descriptor.encoder.frozen_dinov2.dinov2_vitl14
        dinov2 = quantized_dinov2(dinov2, dino, settings, device, verbose) if dino != "fp32" else dinov2
        return DeviceLoMa(quantize_dev(net, settings, device, verbose), net, settings, device, dinov2), None, match_dtype
    detector, descriptor = HookedDetector(net._detector), HookedDescriptor(net._descriptor)
    if isinstance(spec, dict):
        calibration = load_calibration(settings)
        for part in (detector, descriptor):
            if spec.get("only", part.prefix) == part.prefix:
                fake_quantize(part, spec, calibration, device, verbose)
    cnn_dtype = torch.float16 if spec == "fp16" else None
    if dino != "fp32":
        from .dinov2_quantization import quantized_dinov2

        descriptor.dinov2 = quantized_dinov2(net._descriptor.encoder.frozen_dinov2.dinov2_vitl14, dino, settings,
                                             device, verbose)

    def run_detector(images):
        with autocast(cnn_dtype):
            return detector(images).float()

    def run_descriptor(images):
        with autocast(cnn_dtype):
            return descriptor(images).float()

    model = types.SimpleNamespace(_detector=run_detector, _descriptor=run_descriptor, match=net.match)
    return model, None, match_dtype
