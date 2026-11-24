"""Quantization recipes for VGGT and their application to the model.

A recipe is a precision setting for everything left unquantized (agg: autocast dtype of the aggregator, head: of the
camera / depth / point heads, None = fp32) plus an ordered list of (module regex, spec); the first match wins,
unmatched Linear / Conv layers keep their weights. spec keys (as in cosmos3_edge/recipes.py):
  fmt        int8_pc (per output column int8) | q8_0 | q4_0 | iq4_nl | int4_pc
  method     rtn | gptq
  act_order  GPTQ column order by diag(H)                                       [True]
  a8         per-token int8 activation fake-quant (Linear: per token; convolution: vgq.ActQuantConv)  [False]
  amode      convolutions: "tok" (one scale per GEMM row = output pixel over its receptive field) | "tens" (per frame)
  smooth     SmoothQuant alpha (None = off)
  skip       leave the module as it is (to exempt a layer from a later, broader regex)

The device recipe w8a8_tfc quantizes every Linear of the transformer (DINOv2 patch embedding blocks, frame and global
blocks) and of the camera head as the Cosmos 3 Edge W8A8 path (per-channel int8 GPTQ weights, per-token int8
activations, SmoothQuant 0.5 folded into the preceding norm / output stage); the DPT heads stay fp16 (the device runs
them on the fp16 HMX GEMM). The int8 DPT recipes (w8a8_all, only_dpt*) quantize the DPT heads as well.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import torch

from benzene.quantization import FORMATS, ActQuantLinear, gptq, rtn

from .convolution_quantization import ActQuantConv, taps, weight_matrix
from .paths import OUT_DIR

PATCH = r"^aggregator\.patch_embed\.patch_embed\.proj$"
DINO = r"^aggregator\.patch_embed\.blocks\."
FRAME = r"^aggregator\.frame_blocks\."
GLOBAL = r"^aggregator\.global_blocks\."
CAM = r"^camera_head\.(trunk\.|poseLN_modulation\.|pose_branch\.fc1$)"
DPT = r"^(depth|point)_head\.(projects|resize_layers|scratch)\."
DPT_LAST = r"^(depth|point)_head\.scratch\.output_conv2\.2$"
TF = [DINO, FRAME, GLOBAL]
QUANTIZABLE = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)

F16 = dict(agg=torch.float16, head=torch.float16)


def _r(regexes, **spec):
    return [(rx, spec) for rx in regexes]


W8A8S = dict(fmt="int8_pc", method="gptq", a8=True, smooth=0.5)
SKIP_LAST = [(DPT_LAST, dict(skip=True))]

RECIPES: dict[str, tuple[dict, list]] = {
    "fp32": (dict(agg=None, head=None), []),             # the reference
    "bf16": (dict(agg=torch.bfloat16, head=None), []),   # official mixed precision
    "fp16": (F16, []),                                  # the device's float floor
    # ---- W8A8 as the HMX int8 GEMM runs it
    "w8a8_tf": (F16, _r(TF, **W8A8S)),
    "w8a8_tfc": (F16, _r(TF + [CAM], **W8A8S)),
    "w8a8_tfc_dfc2": (F16, [(DINO + r".*\.mlp\.fc2$", dict(skip=True))] + _r(TF + [CAM], **W8A8S)),  # DINOv2 fc2 fp16
    "w8a8_all": (F16, _r(TF + [CAM], **W8A8S) + SKIP_LAST + _r([DPT], **W8A8S, amode="tok")),
    "w8a8_all_tens": (F16, _r(TF + [CAM], **W8A8S) + SKIP_LAST + _r([DPT], **W8A8S, amode="tens")),
    "w8a8_all_last": (F16, _r(TF + [CAM, DPT], **W8A8S, amode="tok")),
    "w8a8_all_patch": (F16, _r(TF + [CAM, PATCH], **W8A8S) + SKIP_LAST + _r([DPT], **W8A8S, amode="tok")),
    # ---- ablations of the transformer part
    "w8a8_tf_rtn": (F16, _r(TF, fmt="int8_pc", method="rtn", a8=True, smooth=0.5)),
    "w8a8_tf_sq0": (F16, _r(TF, fmt="int8_pc", method="gptq", a8=True)),
    "w8_tf": (F16, _r(TF, fmt="int8_pc", method="gptq")),  # weights only
    # ---- one part at a time
    "only_dino": (F16, _r([DINO], **W8A8S)),
    "only_frame": (F16, _r([FRAME], **W8A8S)),
    "only_global": (F16, _r([GLOBAL], **W8A8S)),
    "only_cam": (F16, _r([CAM], **W8A8S)),
    "only_dino_qkv": (F16, _r([DINO + r".*\.attn\.qkv$"], **W8A8S)),
    "only_dino_proj": (F16, _r([DINO + r".*\.attn\.proj$"], **W8A8S)),
    "only_dino_fc1": (F16, _r([DINO + r".*\.mlp\.fc1$"], **W8A8S)),
    "only_dino_fc2": (F16, _r([DINO + r".*\.mlp\.fc2$"], **W8A8S)),
    "only_dpt": (F16, SKIP_LAST + _r([DPT], **W8A8S, amode="tok")),
    "only_dpt_tens": (F16, SKIP_LAST + _r([DPT], **W8A8S, amode="tens")),
    # ---- without calibration data (RTN, no smoothing): the DPT activation modes
    "only_dpt_rtn0": (F16, SKIP_LAST + _r([DPT], fmt="int8_pc", method="rtn", a8=True, amode="tok")),
    "only_dpt_rtn0_tens": (F16, SKIP_LAST + _r([DPT], fmt="int8_pc", method="rtn", a8=True, amode="tens")),
}
DEVICE_RECIPE = "w8a8_tfc"  # the recipe the device runs: export_hexmodel's and the checks' default


def spec_of(recipe: list, name: str) -> dict | None:
    for rx, spec in recipe:
        if re.search(rx, name):
            return None if spec.get("skip") else spec
    return None


def _cache_path(name: str, spec: dict):
    key = {k: spec.get(k) for k in ("fmt", "method", "act_order", "smooth")}
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return OUT_DIR / "qweights" / h / f"{name}.pt", key


def smooth_vector(mod: torch.nn.Module, amax: torch.Tensor, alpha: float) -> torch.Tensor:
    """SmoothQuant vector per input channel: amax^a / wmax^(1-a), wmax over every weight reading the channel."""
    w = weight_matrix(mod).float()
    if isinstance(mod, torch.nn.Conv2d):
        w = w.reshape(w.shape[0], -1, taps(mod)).abs().amax(2)  # (ci, ky, kx) columns -> ci
    wmax = w.abs().amax(0).clamp_min(1e-5)
    s = amax.float().to(w.device).clamp_min(1e-5).pow(alpha) / wmax.pow(1 - alpha)
    return s.clamp(1e-2, 1e3)


@torch.no_grad()
def quantize_weight(name: str, mod: torch.nn.Module, spec: dict, smooth: torch.Tensor | None) -> torch.Tensor:
    """Dequantized fp32 GEMM weight (weight_matrix layout, times the smoothing vector); cached on disk."""
    from .calibrate import load_hessian

    path, key = _cache_path(name, spec)
    if path.exists():
        return torch.load(path, map_location=mod.weight.device)["w"].float()
    fmt = FORMATS[spec["fmt"]]
    w = weight_matrix(mod).float()
    H = load_hessian(name, w.device)["H"].float() if spec["method"] == "gptq" or fmt.group else None
    if smooth is not None:
        ks = smooth.repeat_interleave(taps(mod)) if isinstance(mod, torch.nn.Conv2d) else smooth
        w = w * ks[None, :]
        if H is not None:
            H = H / ks[:, None] / ks[None, :]
    assert fmt.group == 0 or w.shape[1] % fmt.group == 0, f"{name}: K {w.shape[1]} not a multiple of the block"
    if spec["method"] == "gptq":
        q, scales = gptq(w, H, fmt, act_order=spec.get("act_order", True))
    else:
        q, scales = rtn(w, fmt, None if H is None else torch.diag(H))
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "spec.json").write_text(json.dumps(key))
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save({"w": q.cpu(), "scales": scales.cpu(), "fmt": fmt.name, "smooth": None if smooth is None else smooth.cpu()},
               tmp)
    os.replace(tmp, path)
    return q


@torch.no_grad()
def apply_recipe(model, recipe_name: str, verbose: bool = True) -> dict:
    from .calibrate import load_hessian

    prec, recipe = RECIPES[recipe_name]
    t0 = time.time()
    names = [n for n, m in model.named_modules() if isinstance(m, QUANTIZABLE)]
    replaced, params = 0, 0
    for name in names:
        spec = spec_of(recipe, name)
        if spec is None:
            continue
        mod = model.get_submodule(name)
        smooth = None
        if spec.get("a8") and spec.get("smooth") is not None:
            smooth = smooth_vector(mod, load_hessian(name, mod.weight.device)["amax"], spec["smooth"])
        q = quantize_weight(name, mod, spec, smooth)
        if spec.get("a8"):
            new = (ActQuantLinear(mod, q, smooth, mode="tok") if isinstance(mod, torch.nn.Linear)
                   else ActQuantConv(mod, q, smooth, mode=spec.get("amode", "tok")))
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, new)
        else:
            from .convolution_quantization import weight_from_matrix

            mod.weight.data.copy_(weight_from_matrix(mod, q).to(mod.weight.dtype))
        replaced += 1
        params += mod.weight.numel()
    if verbose:
        print(f"[{recipe_name}] quantized {replaced} layers ({params / 1e6:.0f}M weights) in {time.time() - t0:.0f}s",
              flush=True)
    return {"replaced": replaced, "params": params, "agg": str(prec["agg"]), "head": str(prec["head"])}
