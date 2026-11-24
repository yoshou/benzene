"""Quantization recipes for XFeat + LightGlue. A recipe name is "<xfeat>+<lg>" (XFEAT_RECIPES, LG_RECIPES); "fp32" and
"fp16" alone mean the same for both networks.

XFeat:
  fp16  autocast fp16
  dev*  the device's int8 network (device_model.DeviceXFeat): BatchNorm folded into the convolutions, int8 weights
        per output channel (w: "gptq" or "rtn") with the per-channel input scales folded in, static per-channel
        activation scales from the calibration histograms (clip: "mse", "p4", "max"; uint8 for the non-negative inputs,
        int8 by magnitude with zero point 128 for the signed ones), dead channels (never non-zero in calibration) with
        zero weights and the median scale. block1.0 and keypoint.0 read the raw pixels (the normalization folded into
        their requantization per image), the skip branch and the fusion's sum stay in fp32 up to the quantization of
        their sums, the 1 x 1 heads and the descriptor map put out fp32.
LightGlue: "fp16" (LightGlue's own mixed precision, conf.mp; attention in fp16 as the device's flash attention),
"fp16a" (fp16 with LighterGlue's adaptive inference: point pruning at width_confidence 0.95, no early stop).
"""

from __future__ import annotations

import numpy as np
import torch

from benzene.quantization import FORMATS, clip_values, gptq, rtn

from ..lightglue import matcher
from .calibrate import X5, load_calibration
from .device_model import LAYERS, SIGNED_INPUTS, DeviceXFeat, border_tap_sums
from .runner import ADAPTIVE

XFEAT_RECIPES: dict[str, dict | str | None] = {
    "fp32": None,
    "fp16": "fp16",
    "dev": dict(w="gptq", clip="mse"),
    "dev_rtn": dict(w="rtn", clip="mse"),
    "dev_p4": dict(w="gptq", clip="p4"),
    "dev_max": dict(w="gptq", clip="max"),
}
LG_RECIPES: dict[str, str | None] = {"fp32": None, "fp16": "fp16", "fp16a": "fp16a"}
DEVICE_RECIPE = "dev+fp16"  # the recipe the device runs: export_hexmodel's and the checks' default

# device layer -> XFeat module (a BasicLayer's Sequential, or the plain Conv2d of the last layers)
MODULES = {name: name.replace("fusion.", "block_fusion.").replace("heatmap.", "heatmap_head.")
           .replace("keypoint.", "keypoint_head.") for name in LAYERS}
# the consumer whose input scale a uint8 output takes (x3, x4: the next block's; x5: its own statistics)
CONSUMER = {"block1.0": "block1.1", "block1.1": "block1.2", "block1.2": "block1.3", "block2.0": "block2.1",
            "block2.1": "block3.0", "block3.0": "block3.1", "block3.1": "block3.2", "block3.2": "block4.0",
            "block4.0": "block4.1", "block4.1": "block4.2", "block4.2": "block5.0", "block5.0": "block5.1",
            "block5.1": "block5.2", "block5.2": "block5.3", "block5.3": X5, "fusion.0": "fusion.1",
            "fusion.1": "fusion.2", "heatmap.0": "heatmap.1", "heatmap.1": "heatmap.2", "keypoint.0": "keypoint.1",
            "keypoint.1": "keypoint.2", "keypoint.2": "keypoint.3"}
RAW_INPUTS = ("block1.0", "keypoint.0")  # the raw pixels (normalization folded per image)


def split(recipe: str) -> tuple[str, str]:
    if "+" not in recipe:
        return recipe, recipe
    xfeat, lg = recipe.split("+")
    return xfeat, lg


def autocast_dtypes(recipe: str):
    """XFeat's autocast dtype (LightGlue's fp16 runs through its own mixed-precision switch, conf.mp)."""
    xfeat, _ = split(recipe)
    return torch.float16 if XFEAT_RECIPES[xfeat] == "fp16" else None, None


def folded(net, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(weight [N, C, kh, kw], bias [N]) fp64 of layer `name` with its BatchNorm (affine=False) folded in."""
    module = net.get_submodule(MODULES[name])
    if isinstance(module, torch.nn.Conv2d):
        return module.weight.detach().double().cpu(), module.bias.detach().double().cpu()
    conv, bn = module.layer[0], module.layer[1]
    inv = 1.0 / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)
    return conv.weight.detach().double().cpu() * inv[:, None, None, None], -bn.running_mean.detach().double().cpu() * inv


def statistics_key(name: str) -> str:
    """The calibration key of a layer's input (its Conv2d's name)."""
    module = MODULES[name]
    return module + ".layer.0" if name not in ("fusion.2", "heatmap.2", "keypoint.3") else module


def input_scales(stats: dict, clip: str) -> torch.Tensor:
    """Static per-channel scale of an input (uint8 of non-negative values, int8 of signed ones by magnitude); dead
    channels at the median of the live ones."""
    levels = 127 if stats["signed"] else 255
    s = clip_values(stats["hist_c"], stats["amax"], clip, levels).clamp_min(1e-12) / levels
    dead = stats["amax"].double() <= 0
    if dead.any():
        s = torch.where(dead, s[~dead].median(), s)
    return s


def quantize_layer(W: torch.Tensor, s_in: torch.Tensor | None, stats: dict, spec: dict, device: str):
    """int8 codes [N, C, kh, kw] and their per-output-channel scales s_w (fp64) of the folded weight W with the input
    scales s_in folded in (None: raw pixels, scale 1); dead input channels get zero weights."""
    N, C, kh, kw = W.shape
    s = torch.ones(C, dtype=torch.float64) if s_in is None else s_in
    dead = stats["amax"].double() <= 0
    Wm = (W * s[None, :, None, None] * (~dead).double()[None, :, None, None]).reshape(N, -1)
    f = (1.0 / s).repeat_interleave(kh * kw)  # unfold order (ci, ky, kx)
    H = stats["H"].double() * f[:, None] * f[None, :]
    fmt = FORMATS["int8_pc"]
    if spec["w"] == "gptq":
        Q, ws = gptq(Wm.float().to(device), H.float().to(device), fmt)
    else:
        Q, ws = rtn(Wm.float(), fmt)
    Q, s_w = Q.double().cpu(), ws.double().cpu().reshape(-1)
    codes = torch.where(s_w[:, None] > 0, (Q / torch.where(s_w > 0, s_w, 1)[:, None]).round(), 0)
    assert codes.abs().max() <= 127
    s_w = torch.where(s_w > 0, s_w, torch.ones_like(s_w))  # (dead output channels: codes 0)
    return codes.to(torch.int8).reshape(N, C, kh, kw), s_w


def as_f32(x: torch.Tensor) -> np.ndarray:
    return x.double().numpy().astype(np.float32)


def quantize_dev(net, spec: dict, calibration: dict, device: str, verbose: bool = True) -> dict:
    """The device records of every layer (device_model.DeviceXFeat, export_hexmodel)."""
    scales = {}  # per input: the static scale
    for name in LAYERS:
        if name not in RAW_INPUTS:
            scales[name] = input_scales(calibration[statistics_key(name)], spec["clip"])
    scales[X5] = input_scales(calibration[X5], spec["clip"])
    records = {}
    for name in LAYERS:
        W, bias = folded(net, name)
        stats = calibration[statistics_key(name)]
        codes, s_w = quantize_layer(W, scales.get(name), stats, spec, device)
        record = {"codes": codes, "zero_point": 128 if name in SIGNED_INPUTS else 0}
        if name in RAW_INPUTS:
            s_y = scales[CONSUMER[name]]
            sums = border_tap_sums(codes) if name == "block1.0" else codes.to(torch.int64).sum((1, 2, 3))[:, None]
            record.update(k=as_f32(s_w / s_y), beta=as_f32(bias / s_y + 0.5), tap_sums=sums.to(torch.int32).numpy())
        elif name == "block1.3":
            skip = net.skip1[1]
            ws = skip.weight.detach().double().cpu().reshape(-1)
            record.update(scale=as_f32(s_w), bias=as_f32(bias), skip_weight=as_f32(ws), skip_weight16=as_f32(ws / 16),
                          skip_bias=as_f32(skip.bias.detach().double().cpu()),
                          inverse_scale=as_f32(1.0 / scales["block2.0"]))
        elif name in CONSUMER:
            s_y = scales[CONSUMER[name]]
            record.update(m=as_f32(s_w / s_y), b=as_f32(bias / s_y + 0.5))
        else:  # fp32 outputs
            record.update(scale=as_f32(s_w), bias=as_f32(bias))
            if name == "fusion.2":
                record.update(inverse_scale=as_f32(1.0 / scales["heatmap.0"]))
        records[name] = record
        if verbose:
            sv = "raw pixels" if name in RAW_INPUTS else f"input scale {scales[name].min():.3g}..{scales[name].max():.3g}"
            print(f"  {name}: {spec['w']} {tuple(W.shape)}, {sv}")
    records["fusion"] = {"k3": as_f32(scales["block4.0"] / scales["fusion.0"]),
                         "k4": as_f32(scales["block5.0"] / scales["fusion.0"]),
                         "k5": as_f32(scales[X5] / scales["fusion.0"])}
    return records


def apply_recipe(net, lg, recipe: str, device: str, verbose: bool = True):
    """The XFeat network of the recipe (DeviceXFeat for "dev*", else net itself) with LightGlue's precision set;
    returns (net, records or None)."""
    x, l = split(recipe)
    xs, ls = XFEAT_RECIPES[x], LG_RECIPES[l]
    lg.conf.mp = ls in ("fp16", "fp16a")
    matcher.set_adaptive(lg, ADAPTIVE if ls == "fp16a" else None)
    if isinstance(xs, dict):
        records = quantize_dev(net, xs, load_calibration("cpu"), device, verbose)
        return DeviceXFeat(records, device), records
    return net, None
