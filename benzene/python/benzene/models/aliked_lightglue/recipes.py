"""Quantization recipes for ALIKED + LightGlue. A recipe name is "<aliked>+<lg>" (ALIKED_RECIPES, LG_RECIPES); "fp32"
and "fp16" alone mean the same for both networks.

ALIKED:
  fp16  autocast fp16
  fq*   int8 fake quantization (the device's int8 network, float arithmetic otherwise): every
        convolution (network.CONVOLUTIONS) with BatchNorm folded in, int8 weights per output channel (w: "gptq" or
        "rtn") with the input's per-channel scales folded in; every activation at a quantization point
        (network.ACTIVATIONS: SELU outputs, their averages, the descriptor head's patches / samples) uint8 per channel
        x = s (q - z) with a static scale and zero point from the calibration histograms:
          zp "channel"  per channel: [low, clip] with low the channel's minimum (SELU outputs: -1.758, dense there),
                        the clip by "mse" (the squared error of the histogram, rounding below and clipping above),
                        "p4" (99.99th percentile) or "max"; the descriptor head's patches / samples (normalized
                        features, both signs) shrink both ends by one factor
          zp "layer"    one zero point for all channels of a tensor (the one of least total squared error)
          zp "128"      int8 by magnitude (zero point 128)
        the deformable layers' bilinear samples requantized to their input's codes (the device's HMX reads codes);
        the image is exact (codes = gray, scale 1 / 255). Not quantized: the score head's sum of its four parts, the
        sigmoid, the feature map's normalization, the offsets (fp32 outputs of the offset convolutions), the
        descriptors. "<recipe>@a,b" quantizes only convolutions a, b (and their inputs), "<recipe>~a,b" all but them;
        w16: layers whose weights stay float (16-bit weights on the device's HVX layers), their inputs quantized.
  dev   the device's network (device_model.DeviceAliked from quantize_dev's records): fq_w16's quantization with
        block1.conv1's weights in int16, each ResBlock's conv2 and downsample as one integer GEMM (concatenated K,
        one scale per output channel), every step in the device's arithmetic.
LightGlue: "fp16" (LightGlue's own mixed precision, conf.mp; attention in fp16 as the device's flash attention),
"fp16a" (fp16 with LightGlue's adaptive inference: early stop at depth_confidence 0.95, point pruning at
width_confidence 0.99).
"""

from __future__ import annotations

import torch

from benzene.quantization import activation_quantizer, int8_weight_codes, quantize_activation

from ..lightglue import matcher
from .calibrate import load_calibration
from .hooked_network import ACTIVATIONS, CONVOLUTIONS, SELU_ALPHA, SELU_LAMBDA, HookedAliked
from .runner import ADAPTIVE

ALIKED_RECIPES: dict[str, dict | str | None] = {
    "fp32": None,
    "fp16": "fp16",
    "fq": dict(w="gptq", clip="mse", zp="channel"),
    "fq_rtn": dict(w="rtn", clip="mse", zp="channel"),
    "fq_max": dict(w="gptq", clip="max", zp="channel"),
    "fq_p4": dict(w="gptq", clip="p4", zp="channel"),
    "fq_zl": dict(w="gptq", clip="mse", zp="layer"),
    "fq_z128": dict(w="gptq", clip="mse", zp="128"),
    # the HVX layers' weights in 16 bits (as float here): block1.conv1 (raw pixels), and the score head's 3x3 layers
    "fq_w16": dict(w="gptq", clip="mse", zp="channel", w16=["block1.conv1"]),
    "fq_w16s": dict(w="gptq", clip="mse", zp="channel",
                    w16=["block1.conv1", "score_head.2", "score_head.4", "score_head.6"]),
    # the device recipe: fq_w16 in the device's arithmetic (device_model.DeviceAliked), each ResBlock's second
    # convolution and its downsample quantized together (one accumulator, one scale per output channel)
    "dev": dict(w="gptq", clip="mse", zp="channel", w16=["block1.conv1"], dev=True),
}
LG_RECIPES: dict[str, str | None] = {"fp32": None, "fp16": "fp16", "fp16a": "fp16a"}
DEVICE_RECIPE = "dev+fp16"  # the recipe the device runs: export_hexmodel's and the checks' default
SIGNED = ("desc.patch", "desc.samples")  # normalized features: both signs (the rest: SELU outputs and averages)


def split(recipe: str) -> tuple[str, str]:
    if "+" not in recipe:
        return recipe, recipe
    aliked, lg = recipe.split("+")
    return aliked, lg


def aliked_spec(name: str):
    """ALIKED_RECIPES entry; "<recipe>@a,b" quantizes only convolutions a, b, "<recipe>~a,b" all but a, b."""
    for separator, key in (("@", "only"), ("~", "skip")):
        if separator in name:
            base, layers = name.split(separator)
            return {**ALIKED_RECIPES[base], key: layers.split(",")}
    return ALIKED_RECIPES[name]


def autocast_dtypes(recipe: str):
    """ALIKED's autocast dtype (LightGlue's fp16 runs through its own mixed-precision switch, conf.mp)."""
    aliked, _ = split(recipe)
    return torch.float16 if aliked_spec(aliked) == "fp16" else None, None


# ---------------------------------------------------------------------------------------------- activations
# ---------------------------------------------------------------------------------------------- weights
def input_scales(name: str, scales: dict) -> torch.Tensor:
    """Per-K scale [K] of convolution `name`'s GEMM input (the image: 1 / 255)."""
    source, kind = CONVOLUTIONS[name]
    s = torch.full((1,), 1.0 / 255, dtype=torch.float64) if source == "image" else scales[source][0]
    if kind in ("3x3", "deformable", "patch"):
        return s.repeat_interleave(9)
    if kind == "aggregation":
        return s.repeat(16)
    return s


# ---------------------------------------------------------------------------------------------- fake quantization
def fake_quantized_network(net, spec: dict, calibration: dict, device: str, verbose: bool = True) -> HookedAliked:
    """HookedAliked with spec's int8 fake quantization (module docstring)."""
    layers = [name for name in CONVOLUTIONS if name in spec.get("only", [name]) and name not in spec.get("skip", [])]
    quantized_inputs = {CONVOLUTIONS[name][0] for name in layers} - {"image"}
    activations = calibration["activations"]
    scales = {name: activation_quantizer(activations[name], spec["clip"], spec["zp"], name in SIGNED)
              for name in ACTIVATIONS}
    wrapped = HookedAliked(net)
    for name in layers:
        if name in spec.get("w16", []):  # (16-bit weights: their rounding is negligible here)
            continue
        weight, bias = wrapped.weights[name]
        weight64 = weight.double().cpu().reshape(weight.shape[0], -1)
        s_in = input_scales(name, scales)
        codes, s_w = int8_weight_codes(weight64, s_in, calibration["hessians"][name]["H"], spec["w"], device)
        dequantized = (codes * s_w[:, None] / s_in[None]).reshape(weight.shape)
        wrapped.weights[name] = (dequantized.float().to(weight.device), bias)
        if verbose:
            print(f"  {name}: {spec['w']} {tuple(weight.shape)}")

    def activation(name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in quantized_inputs:
            return x
        scale, zero = scales[name]
        return quantize_activation(x, scale, zero, 1 if name == "desc.patch" else 0)

    def deformable(name: str, x: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        source = CONVOLUTIONS[name][0]
        if name not in layers:
            return columns
        scale, zero = scales[source]
        return quantize_activation(columns, scale.repeat_interleave(9), zero.repeat_interleave(9), 0)

    wrapped.activation_hook = activation
    wrapped.deformable_hook = deformable
    return wrapped


# ---------------------------------------------------------------------------------------------- the device recipe
# The device's layers: name -> (input activation, kind, output activation or None for fp32 rows); a ResBlock's conv2
# carries its downsample (DEVICE_RESIDUALS) in the same accumulator.
DEVICE_LAYERS = {
    "block1.conv1": ("image", "3x3", "block1.mid"),
    "block1.conv2": ("block1.mid", "3x3", "x1"),
    "block2.conv1": ("x1.pool", "3x3", "block2.mid"),
    "block2.conv2": ("block2.mid", "3x3", "x2"),
    "block3.conv1.offset": ("x2.pool", "3x3", None),
    "block3.conv1": ("x2.pool", "deformable", "block3.mid"),
    "block3.conv2.offset": ("block3.mid", "3x3", None),
    "block3.conv2": ("block3.mid", "deformable", "x3"),
    "block4.conv1.offset": ("x3.pool", "3x3", None),
    "block4.conv1": ("x3.pool", "deformable", "block4.mid"),
    "block4.conv2.offset": ("block4.mid", "3x3", None),
    "block4.conv2": ("block4.mid", "deformable", "x4"),
    "conv1": ("x1", "1x1", "agg1"),
    "conv2": ("x2", "1x1", "agg2"),
    "conv3": ("x3", "1x1", "agg3"),
    "conv4": ("x4", "1x1", "agg4"),
    "score_head.0.1": ("agg1", "1x1", None),
    "score_head.0.2": ("agg2", "1x1", None),
    "score_head.0.3": ("agg3", "1x1", None),
    "score_head.0.4": ("agg4", "1x1", None),
    "score_head.2": ("score1", "3x3", "score2"),
    "score_head.4": ("score2", "3x3", "score3"),
    "score_head.6": ("score3", "3x3", None),
    "desc.offset0": ("desc.patch", "patch", "desc.hidden"),
    "desc.offset2": ("desc.hidden", "1x1", None),
    "desc.sf": ("desc.samples", "1x1", "desc.features"),
    "desc.aggregation": ("desc.features", "aggregation", None),
}
DEVICE_RESIDUALS = {"block2.conv2": "block2.downsample", "block3.conv2": "block3.downsample",
                    "block4.conv2": "block4.downsample"}
POOLS = {"x1.pool": ("x1", 2), "x2.pool": ("x2", 4), "x3.pool": ("x3", 4)}  # pooled tensor -> (source, window)
INT16_LEVELS = 32767


def input_zero_points(name: str, quantizers: dict) -> torch.Tensor:
    """The zero point [K] of each k of layer `name`'s GEMM input (PyTorch's K order; the image: 0)."""
    source, kind = CONVOLUTIONS[name]
    z = torch.zeros(1, dtype=torch.float64) if source == "image" else quantizers[source][1]
    if kind in ("3x3", "deformable", "patch"):
        return z.repeat_interleave(9)
    if kind == "aggregation":
        return z.repeat(16)
    return z


def selu_constants(S: torch.Tensor, B: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> dict:
    """The SELU requantization's constants (device_model.selu_codes) of a layer with pre-activation v = acc S + B
    whose output codes have the quantizer (scale, zero): fp32 Sp = S lambda / s, Bp = B lambda / s + z + 0.5,
    St = S log2 e, Bt = B log2 e, Q_i = A c_i (A = lambda alpha / s, c_i: device_model.EXP2_F32), Dn = z + 0.5 - A."""
    from .device_model import EXP2_F32  # (device_model imports this module)

    s, z = scale.double(), zero.double()
    S64, B64 = S.double(), B.double()
    A = SELU_LAMBDA * SELU_ALPHA / s
    log2e = 1.4426950408889634
    out = {"Sp": (S64 * SELU_LAMBDA / s).float(), "Bp": (B64 * SELU_LAMBDA / s + z + 0.5).float(),
           "St": (S64 * log2e).float(), "Bt": (B64 * log2e).float(), "Dn": (z + 0.5 - A).float()}
    for i, c in enumerate(EXP2_F32):
        out[f"Q{i}"] = (A * float(c)).float()
    return out


def quantize_dev(net, spec: dict, calibration: dict, device: str, verbose: bool = True) -> dict:
    """The device records (device_model.DeviceAliked, export_hexmodel): per activation its scale / zero point, per
    layer its integer codes [N, C, kh, kw] (a residual layer also its downsample's [N, C', 1, 1]), weight scales and
    output constants, per pooled tensor its requantization."""
    activations = calibration["activations"]
    quantizers = {name: activation_quantizer(activations[name], spec["clip"], spec["zp"], name in SIGNED)
                  for name in ACTIVATIONS}
    quantizers["image"] = (torch.full((1,), 1.0 / 255, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    wrapped = HookedAliked(net)
    weights = {name: tuple(t.double().cpu() for t in wrapped.weights[name]) for name in CONVOLUTIONS}
    hessians = calibration["hessians"]
    scales = {name: q for name, q in quantizers.items()}
    records = {"activations": {name: {"scale": s, "zero": z} for name, (s, z) in quantizers.items()}, "layers": {},
               "pools": {}}
    for name, (source, kind, output) in DEVICE_LAYERS.items():
        weight, bias = weights[name]
        N = weight.shape[0]
        matrix = weight.reshape(N, -1)
        s_in = input_scales(name, scales)
        H = hessians[name]["H"].double()
        residual = DEVICE_RESIDUALS.get(name)
        if residual is not None:  # [conv2 | downsample] in one accumulator, a block-diagonal Hessian
            r_weight, r_bias = weights[residual]
            r_matrix = r_weight.reshape(N, -1)
            K1 = matrix.shape[1]
            matrix = torch.cat([matrix, r_matrix], 1)
            s_in = torch.cat([s_in, input_scales(residual, scales)])
            H = torch.block_diag(H, hessians[residual]["H"].double())
            bias = bias + r_bias
        if name in spec.get("w16", []):
            Wm = matrix * s_in[None]
            s_w = (Wm.abs().amax(1) / INT16_LEVELS).clamp_min(1e-30)
            codes = (Wm / s_w[:, None]).round()
        else:
            codes, s_w = int8_weight_codes(matrix, s_in, H, spec["w"], device)
        record = {"codes": codes[:, : weight[0].numel()].reshape(weight.shape).to(torch.int32), "weight_scale": s_w,
                  "input": source, "kind": kind, "output": output,
                  "S": s_w.float(), "B": bias.float()}
        # the zero points' part of the sum, -sum(w z) (the device's store bias / correction): acc = sum(w q) + bias2
        zero_k = input_zero_points(name, quantizers)
        if residual is not None:
            record["residual_codes"] = codes[:, K1:].reshape(r_weight.shape).to(torch.int32)
            record["residual_input"] = CONVOLUTIONS[residual][0]
            zero_k = torch.cat([zero_k, quantizers[CONVOLUTIONS[residual][0]][1]])
        record["bias2"] = -(codes.to(torch.int64) * zero_k.round().to(torch.int64)[None]).sum(1)
        if kind == "3x3":  # the padding codes (x = 0)
            record["padding"] = quantizers[source][1].clone()
        if kind == "deformable":  # the sampled input's zero points (outside the map: x = 0)
            record["input_zero"] = quantizers[source][1].clone()
        if output is not None:
            record.update(selu_constants(record["S"], record["B"], *quantizers[output]))
        records["layers"][name] = record
        if verbose:
            print(f"  {name}: {'int16' if name in spec.get('w16', []) else spec['w']} {tuple(matrix.shape)}"
                  f"{' + ' + residual if residual else ''}")
    one, nothing = torch.ones(8), torch.zeros(8)  # (the score head's sum is the pre-activation itself)
    records["score1"] = selu_constants(one, nothing, *quantizers["score1"])
    records["dequantize"] = {f"agg{k}": {"scale": quantizers[f"agg{k}"][0].float(), "zero": quantizers[f"agg{k}"][1].float()}
                             for k in range(1, 5)}
    for pooled, (source, window) in POOLS.items():
        s_in, z_in = quantizers[source]
        s_out, z_out = quantizers[pooled]
        records["pools"][pooled] = {"source": source, "window": window,
                                    "K": (s_in / (window * window * s_out)).float(),
                                    "Z": (z_out - z_in * s_in / s_out + 0.5).float()}
    for name in ("desc.patch", "desc.samples"):  # fp32 values -> codes: q = trunc(x R + (z + 0.5))
        s, z = quantizers[name]
        records[name] = {"R": (1.0 / s).float(), "Z": (z + 0.5).float()}
    return records


def apply_recipe(net, lg, recipe: str, device: str, verbose: bool = True):
    """The ALIKED network of the recipe (a wrapped network for "fq*" / "dev*", else net itself) with LightGlue's
    precision set; returns (net, records or None)."""
    a, l = split(recipe)
    spec, lg_spec = aliked_spec(a), LG_RECIPES[l]
    lg.conf.mp = lg_spec in ("fp16", "fp16a")
    matcher.set_adaptive(lg, ADAPTIVE if lg_spec == "fp16a" else None)
    if isinstance(spec, dict) and spec.get("dev"):
        from .device_model import DeviceAliked

        records = quantize_dev(net, spec, load_calibration("cpu"), device, verbose)
        return DeviceAliked(records, net.dkd, device), records
    if isinstance(spec, dict):
        return fake_quantized_network(net, spec, load_calibration("cpu"), device, verbose), None
    return net, None
