"""Quantization recipes for SuperPoint + LightGlue. A recipe name is "<sp>+<lg>" (SP_RECIPES, LG_RECIPES); "fp32" and
"fp16" alone mean the same for both networks.

SuperPoint convolutions as the device runs them in int8 (benzene dsp/kernels/convolution_int8): weights int8 per output channel
(RTN or GPTQ), activations uint8 with zero point 0 (every convolution input is a ReLU / max-pool output, conv1a's the
image gray / 255, which uint8 holds exactly) and a static scale from the calibration statistics:
  a="t"   one scale per tensor
  a="c"   one scale per input channel: the device folds it into the weights (W * s_c, quantized per output channel
          afterwards) and into the producing layer's per-output-channel requantization, so it costs nothing there
  clip    "max" calibration maximum, "p4" the 99.99th percentile of the non-zero values, "mse" the clip minimizing
          the squared error of the non-zero values (rounding below, clipping above)
  a="tok" dynamic uint8 per output pixel over its receptive field (the per-row scale of the GEMM input stage):
          a reference only, the device's requantized uint8 outputs do not fit it
  heads=False  convPb / convDb (1 x 1 detector / descriptor heads) and their inputs stay in floating point
Quantization happens where the device does it: at the input of every quantized convolution (a uint8 requantized
output then a 2 x 2 max pool is the same as quantizing after the pool: the pool commutes with a monotone per-channel
map).

LightGlue: "fp16" (autocast; attention in fp16 as the device's flash attention), "fp16a" (fp16 with LightGlue's adaptive
inference, benzene.models.lightglue.matcher.set_adaptive) or W8A8 as the VGGT / Cosmos 3 Edge transformers (per-token
int8 activations, int8 GPTQ / RTN weights per output channel, SmoothQuant) on every Linear of the transformer layers and
the final projection of the match assignment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benzene.quantization import FORMATS, ActQuantLinear, clip_values, gptq, rtn, smooth_vector

from ..lightglue import matcher
from .calibrate import lg_linears, load_calibration, sp_convs
from .runner import ADAPTIVE

SP_HEADS = ("convPb", "convDb")
SP_RECIPES: dict[str, dict | str | None] = {
    "fp32": None,
    "fp16": "fp16",
    "w8": dict(w="rtn"),
    "w8g": dict(w="gptq"),
    "a8t_max": dict(w="rtn", a="t", clip="max"),
    "a8t_p4": dict(w="rtn", a="t", clip="p4"),
    "a8t_mse": dict(w="rtn", a="t", clip="mse"),
    "a8c_max": dict(w="rtn", a="c", clip="max"),
    "a8c_p4": dict(w="rtn", a="c", clip="p4"),
    "a8c_mse": dict(w="rtn", a="c", clip="mse"),
    "a8t_mse_g": dict(w="gptq", a="t", clip="mse"),
    "a8c_mse_g": dict(w="gptq", a="c", clip="mse"),
    "a8c_p4_g": dict(w="gptq", a="c", clip="p4"),
    "a8c_mse_hf": dict(w="rtn", a="c", clip="mse", heads=False),
    "a8tok": dict(w="rtn", a="tok"),
    # the device recipe: a8c_mse_g with the arithmetic of the device convolution emulated exactly - integer sums, then per output
    # channel q = clamp(trunc(fp32(acc) * m + b), 0, 255) in fp32 (m = s_acc / s_y, b = bias / s_y + 0.5) for the
    # consumer's calibrated per-channel scale s_y; the 1x1 heads put out fp32 acc * s_acc + bias
    "dev": dict(w="gptq", a="c", clip="mse", dev=True),
}
SP_ORDER = ("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b", "convPa", "convDa", "convPb",
            "convDb")
SP_CONSUMER = {"conv1a": "conv1b", "conv1b": "conv2a", "conv2a": "conv2b", "conv2b": "conv3a", "conv3a": "conv3b",
               "conv3b": "conv4a", "conv4a": "conv4b", "conv4b": "convPa", "convPa": "convPb", "convDa": "convDb"}
SP_PRODUCER = {**{v: k for k, v in SP_CONSUMER.items()}, "convDa": "conv4b"}
LG_RECIPES: dict[str, dict | str | None] = {
    "fp32": None,
    "fp16": "fp16",
    "fp16a": "fp16a",
    "w8a8": dict(w="gptq", smooth=0.5),
    "w8a8_rtn": dict(w="rtn", smooth=0.5),
    "w8a8_sq0": dict(w="gptq", smooth=None),
}
DEVICE_RECIPE = "dev+fp16"  # the recipe the device runs: export_hexmodel's and the checks' default


def split(recipe: str) -> tuple[str, str]:
    if "+" not in recipe:
        return recipe, recipe
    sp, lg = recipe.split("+")
    return sp, lg


def sp_spec(name: str):
    """SP_RECIPES entry; "<recipe>@a,b" quantizes only layers a, b, "<recipe>~a,b" all but a, b (sensitivity runs)."""
    for sep, key in (("@", "only"), ("~", "skip")):
        if sep in name:
            base, layers = name.split(sep)
            return {**SP_RECIPES[base], key: layers.split(",")}
    return SP_RECIPES[name]


def autocast_dtypes(recipe: str):
    """SuperPoint's autocast dtype (LightGlue's fp16 runs through its own mixed-precision switch, conf.mp: its
    forward re-enters autocast with enabled=conf.mp, which would turn an outer autocast off)."""
    sp, lg = split(recipe)
    return torch.float16 if sp_spec(sp) == "fp16" else None, None


def act_scales(statistics: dict, spec: dict, first: bool) -> torch.Tensor:
    """Static uint8 scale per input channel ([Cin]; equal values for a="t")."""
    C = statistics["amax"].numel()
    if first:  # the image: gray / 255 exactly
        return torch.full((C,), 1.0 / 255, dtype=torch.float64)
    if spec["a"] == "t":
        top = statistics["amax"].max().reshape(1)
        c = clip_values(statistics["hist_t"][None], top, spec["clip"])
        return (c / 255).expand(C).clone()
    return clip_values(statistics["hist_c"], statistics["amax"], spec["clip"]).clamp_min(1e-12) / 255


# ---------------------------------------------------------------------------------------------- SuperPoint
class FakeQuantizedConvolution(torch.nn.Module):
    """Conv2d with dequantized int8 weights and uint8 static (or dynamic "tok") input activations."""

    def __init__(self, conv: torch.nn.Conv2d, w_q: torch.Tensor, a: str | None, s: torch.Tensor | None):
        super().__init__()
        self.conv = conv
        self.weight = torch.nn.Parameter(w_q.reshape(conv.weight.shape).to(conv.weight), requires_grad=False)
        self.bias = conv.bias
        self.a = a
        self.register_buffer("s", None if s is None else s.float().to(conv.weight.device)[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        xf = x.float()
        if self.a in ("t", "c"):
            xf = (xf / self.s).round().clamp(0, 255) * self.s
        elif self.a == "tok":
            kh, kw = c.kernel_size
            ph, pw = c.padding
            d = F.max_pool2d(xf.amax(1, keepdim=True), (kh, kw), 1, (ph, pw)) / 255
            d = torch.where(d > 0, d, torch.ones_like(d))
            xp = F.pad(xf, (pw, pw, ph, ph))
            H, W = xf.shape[-2:]
            y = None
            for ky in range(kh):
                for kx in range(kw):
                    q = (xp[:, :, ky : ky + H, kx : kx + W] / d).round().clamp(0, 255) * d
                    t = F.conv2d(q, self.weight[:, :, ky : ky + 1, kx : kx + 1].float())
                    y = t if y is None else y + t
            return y + self.bias.float()[None, :, None, None]
        return F.conv2d(xf, self.weight.float(), self.bias.float(), c.stride, c.padding)


def integer_convolution(codes: torch.Tensor, weights: torch.Tensor, stride, padding, multipliers: torch.Tensor,
                        biases: torch.Tensor, u8: bool) -> torch.Tensor:
    """benzene's int8 convolution (dsp/kernels/convolution_int8) on uint8 codes and int8 weights, both fp64: the exact
    sums (|acc| < 2^53), rounded so that a GPU convolution algorithm with fp64 rounding errors (FFT / Winograd style)
    cannot change them, converted to fp32; then the uint8 codes trunc(clamp(acc * m + b, 0, 255.5)) (two fp32
    roundings, as HVX) or the heads' acc * scale + bias."""
    acc = F.conv2d(codes, weights, None, stride, padding).round().float()
    if u8:
        return torch.trunc((acc * multipliers + biases).clamp(0, 255.5))
    return acc * multipliers + biases


class DeviceConvolution(torch.nn.Module):
    """The device recipe's convolution: input codes x / s_in (x is codes * s_in) through integer_convolution, the uint8
    codes returned as q * s_out."""

    def __init__(self, conv: torch.nn.Conv2d, rec: dict):
        super().__init__()
        self.conv = conv
        dev = conv.weight.device
        self.register_buffer("w", rec["codes"].double().to(dev).reshape(conv.weight.shape))
        self.register_buffer("s_in", rec["s_in"].double().to(dev)[None, :, None, None])
        self.u8 = rec["m"] is not None
        if self.u8:
            self.register_buffer("m", rec["m"].float().to(dev)[None, :, None, None])
            self.register_buffer("b", rec["b"].float().to(dev)[None, :, None, None])
            self.register_buffer("s_out", rec["s_out"].double().to(dev)[None, :, None, None])
        else:
            self.register_buffer("scale", rec["s_acc"].float().to(dev)[None, :, None, None])
            self.register_buffer("bias_f", conv.bias.detach().float()[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        codes = (x.double() / self.s_in).round().clamp(0, 255)
        if self.u8:
            q = integer_convolution(codes, self.w, c.stride, c.padding, self.m, self.b, True)
            return (q.double() * self.s_out).float()
        return integer_convolution(codes, self.w, c.stride, c.padding, self.scale, self.bias_f, False)


def dev_scales(statistics: dict, spec: dict, first: bool) -> torch.Tensor:
    """act_scales with the dead channels (never positive in calibration) at the median scale of the live ones: their
    requantization multipliers stay moderate (the consumer ignores them: zero weights)."""
    s = act_scales(statistics, spec, first).double()
    dead = statistics["amax"].double() <= 0
    if dead.any() and not first:
        s = torch.where(dead, s[~dead].median(), s)
    return s


def quantize_sp_dev(sp, spec: dict, calib: dict, device: str, verbose: bool = True) -> dict:
    """The device recipe: layers in network order; each consumer's per-channel input scale (calibrated) is folded into
    its weights and is its producer's requantization target."""
    fmt = FORMATS["int8_pc"]
    convs = sp_convs(sp)
    records: dict[str, dict] = {}
    for name in SP_ORDER:
        conv = convs[name]
        W = conv.weight.detach().double().cpu()
        Cout, Cin, kh, kw = W.shape
        s_in = dev_scales(calib[name], spec, name == "conv1a")
        if name in SP_PRODUCER:  # the producer's target is this input scale (convDa shares convPa's: same input)
            assert torch.allclose(records[SP_PRODUCER[name]]["s_out"], s_in)
        dead = calib[name]["amax"].double() <= 0  # input channels never positive in calibration: their weights zero
        Wm = (W * s_in[None, :, None, None] * (~dead).double()[None, :, None, None]).reshape(Cout, -1)  # folded
        f = (1.0 / s_in).repeat_interleave(kh * kw)  # unfold order (ci, ky, kx)
        H = calib[name]["H"].double() * f[:, None] * f[None, :]
        Q, ws = gptq(Wm.float().to(device), H.float().to(device), fmt) if spec["w"] == "gptq" else rtn(Wm.float(), fmt)
        Q, s_acc = Q.double().cpu(), ws.double().cpu().reshape(-1)
        codes = torch.where(s_acc[:, None] > 0, (Q / torch.where(s_acc > 0, s_acc, 1)[:, None]).round(), 0)
        assert codes.abs().max() <= 127
        s_acc = torch.where(s_acc > 0, s_acc, torch.ones_like(s_acc))  # (dead output channels: codes 0)
        rec = {"codes": codes.to(torch.int8).reshape(Cout, Cin, kh, kw), "s_in": s_in, "s_acc": s_acc, "m": None}
        cons = SP_CONSUMER.get(name)
        if cons is not None:
            s_y = dev_scales(calib[cons], spec, False)
            bias = conv.bias.detach().double().cpu()
            rec.update(s_out=s_y, m=(s_acc / s_y).float(), b=(bias / s_y + 0.5).float())
        records[name] = rec
        setattr(sp, name, DeviceConvolution(conv, rec))
        if verbose:
            print(f"  {name}: {spec['w']} {tuple(W.shape)}, input scale {s_in.min():.3g}..{s_in.max():.3g}")
    return records


def quantize_sp(sp, spec: dict, calib: dict, device: str, verbose: bool = True) -> dict:
    """Replace SuperPoint's convolutions per spec; returns the per-layer quantization records (scales, codes) the
    exporter writes."""
    fmt = FORMATS["int8_pc"]
    records = {}
    for name, conv in sp_convs(sp).items():
        if spec.get("heads", True) is False and name in SP_HEADS:
            continue
        if name not in spec.get("only", [name]) or name in spec.get("skip", []):
            continue
        statistics = calib[name]
        a = spec.get("a")
        s = act_scales(statistics, spec, name == "conv1a") if a in ("t", "c") else None
        W = conv.weight.detach().double().cpu()
        Cout, Cin, kh, kw = W.shape
        fold = s if a == "c" else None  # per input channel scale folded into the weights
        Wm = (W * fold[None, :, None, None]).reshape(Cout, -1) if fold is not None else W.reshape(Cout, -1)
        if spec["w"] == "gptq":
            H = statistics["H"].double()
            if fold is not None:
                f = (1.0 / fold).repeat_interleave(kh * kw)  # unfold order (ci, ky, kx)
                H = H * f[:, None] * f[None, :]
            Q, ws = gptq(Wm.float().to(device), H.float().to(device), fmt)
            Q, ws = Q.double().cpu(), ws.double().cpu()
        else:
            Q, ws = rtn(Wm.float(), fmt)
            Q, ws = Q.double(), ws.double()
        w_q = (Q.reshape(Cout, Cin, kh, kw) / fold[None, :, None, None]) if fold is not None else Q.reshape(W.shape)
        setattr(sp, name, FakeQuantizedConvolution(conv, w_q.float(), a, s))
        records[name] = {"s_in": s, "w_codes": (Q / ws.reshape(-1, 1)).round().to(torch.int8),
                         "w_scale": ws.reshape(-1), "fold": fold}
        if verbose:
            sv = "" if s is None else f", input scale {s.min().item():.3g}..{s.max().item():.3g}"
            print(f"  {name}: {spec['w']} weights {tuple(W.shape)}, activations {a}{sv}")
    return records


# ---------------------------------------------------------------------------------------------- LightGlue
def quantize_lg(lg, spec: dict, calib: dict, verbose: bool = True) -> int:
    fmt = FORMATS["int8_pc"]
    n = 0
    for name, lin in lg_linears(lg).items():
        if name.endswith("matchability"):
            continue
        statistics = calib[name]
        W = lin.weight.detach().float()
        sm = smooth_vector(statistics["amax"], W, spec["smooth"]).to(W.device) if spec["smooth"] else None
        Wm = W * sm[None] if sm is not None else W
        if spec["w"] == "gptq":
            H = statistics["H"].float().to(W.device)
            if sm is not None:
                H = H / sm[:, None] / sm[None, :]
            Q, _ = gptq(Wm, H, fmt)
        else:
            Q, _ = rtn(Wm, fmt)
        parent = lg.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], ActQuantLinear(lin, Q.to(W.dtype), sm, "tok"))
        n += 1
    if verbose:
        print(f"  LightGlue: {n} Linears W8A8 ({spec})")
    return n


def apply_recipe(sp, lg, recipe: str, device: str, verbose: bool = True):
    """SuperPoint with the recipe's convolutions (GPTQ on device) and LightGlue's precision set; returns (sp, records
    or None)."""
    s, l = split(recipe)
    sps, lgs = sp_spec(s), LG_RECIPES[l]
    lg.conf.mp = lgs in ("fp16", "fp16a")
    matcher.set_adaptive(lg, ADAPTIVE if lgs == "fp16a" else None)
    if not isinstance(sps, dict) and not isinstance(lgs, dict):
        return sp, None
    sp_calibration, lg_calibration = load_calibration("cpu")
    records = None
    if isinstance(sps, dict):
        records = (quantize_sp_dev if sps.get("dev") else quantize_sp)(sp, sps, sp_calibration, device, verbose)
    if isinstance(lgs, dict):
        quantize_lg(lg, lgs, lg_calibration, verbose)
    return sp, records


__all__ = ["SP_RECIPES", "LG_RECIPES", "apply_recipe", "autocast_dtypes", "split", "sp_spec"]
