"""Quantization recipes and their application to an (fp16) Cosmos 3 Edge reasoner (Cosmos3EdgeForConditionalGeneration).

A recipe is an ordered list of (module regex, spec); the first match wins and
unmatched Linear layers stay fp16. spec keys:
  fmt        : q8_0 | q4_0 | iq4_nl | int8_pc | int4_pc
  method     : rtn | gptq
  act_order  : GPTQ column order by diag(H) (static block scales)   [True]
  a8         : activation fake-quant: False | "tok" (per-token int8, =True) |
               "blk32" (Q8_0 blocks, llama.cpp HVX) | "static" (per-tensor uint8, HMX int8)
  smooth     : SmoothQuant alpha for the a8 path (None = off)         [None]
  smooth_qkv : q/k/v projections share one smoothing vector (from their stacked weights), so it
               folds into the preceding norm and one quantized activation feeds a fused qkv GEMM
  decode     : spec for generated-token positions (HVX GEMV on the device); the spec itself
               then covers the prompt (prefill, HMX GEMM). LLM layers only (qlib.MixedLinear)
"""

from __future__ import annotations

import hashlib
import os
import json
import re
import time

import torch

from benzene.quantization import FORMATS, ActQuantLinear, MixedLinear, gptq, padded_cols, rtn, smooth_vector

from .paths import OUT_DIR

VISION = r"^model\.visual\.encoder\.layers\."
PROJ = r"^model\.projector\."
LLM = r"^model\.language_model\.layers\."
HEAD = r"^lm_head$"
ALL = [VISION, PROJ, LLM, HEAD]
BODY = [VISION, PROJ, LLM]  # everything but lm_head


def _r(regexes, **spec):
    return [(rx, spec) for rx in regexes]


QKV = re.compile(r"\.(q|k|v)_proj$")
LLM_FC2 = r"^model\.language_model\.layers\.\d+\.mlp\.fc2$"
# W8A8 as the HMX int8 GEMM runs it: per-channel int8 weights (GPTQ), per-token int8 activations,
# SmoothQuant 0.5 shared by q/k/v
W8A8S = dict(fmt="int8_pc", method="gptq", a8=True, smooth=0.5, smooth_qkv=True)
# decode on HVX like llama.cpp: IQ4_NL (GPTQ) weights x Q8_0-requantized activations
IQ4BLK = dict(fmt="iq4_nl", method="gptq", a8="blk32")
HEAD8BLK = dict(fmt="q8_0", method="rtn", a8="blk32")


RECIPES: dict[str, list] = {
    "bf16": [],
    "fp16": [],
    # ---- weight-only, fp16 activations (HMX fp16 / HVX qf16 path) --------------
    "w8_rtn": _r(ALL, fmt="q8_0", method="rtn"),
    "w4_rtn": _r(BODY, fmt="q4_0", method="rtn") + _r([HEAD], fmt="q8_0", method="rtn"),
    "w4_gptq": _r(BODY, fmt="q4_0", method="gptq") + _r([HEAD], fmt="q8_0", method="rtn"),
    "w4_gptq_noact": _r(BODY, fmt="q4_0", method="gptq", act_order=False)
    + _r([HEAD], fmt="q8_0", method="rtn"),
    "iq4_rtn": _r(BODY, fmt="iq4_nl", method="rtn") + _r([HEAD], fmt="q8_0", method="rtn"),
    "iq4_gptq": _r(BODY, fmt="iq4_nl", method="gptq") + _r([HEAD], fmt="q8_0", method="rtn"),
    "w4_gptq_head4": _r(ALL, fmt="q4_0", method="gptq"),
    # component sensitivity: only one part at 4 bit (GPTQ q4_0)
    "only_vision_w4": _r([VISION], fmt="q4_0", method="gptq"),
    "only_proj_w4": _r([PROJ], fmt="q4_0", method="gptq"),
    "only_llm_w4": _r([LLM], fmt="q4_0", method="gptq"),
    "only_head_w4": _r([HEAD], fmt="q4_0", method="gptq"),
    # ---- integer activations (HVX int8 vrmpy / HMX int8 path) ---------------------
    "w8a8_rtn": _r(ALL, fmt="int8_pc", method="rtn", a8=True),
    "w8a8_sq05": _r(ALL, fmt="int8_pc", method="rtn", a8=True, smooth=0.5),
    "w8a8_sq08": _r(ALL, fmt="int8_pc", method="rtn", a8=True, smooth=0.8),
    "w8a8_sq05_gptq": _r(ALL, fmt="int8_pc", method="gptq", a8=True, smooth=0.5),
    "w8a8_sq05s_gptq": _r(ALL, **W8A8S),
    "w4a8_sq05_gptq": _r(BODY, fmt="int4_pc", method="gptq", a8=True, smooth=0.5)
    + _r([HEAD], fmt="int8_pc", method="rtn", a8=True, smooth=0.5),
    # ---- HVX int8 like llama.cpp: weight blocks x Q8_0-requantized activations ------
    "w8a8blk": _r(ALL, fmt="q8_0", method="rtn", a8="blk32"),
    "w4a8blk_gptq": _r(BODY, fmt="q4_0", method="gptq", a8="blk32")
    + _r([HEAD], fmt="q8_0", method="rtn", a8="blk32"),
    "iq4a8blk_gptq": _r(BODY, fmt="iq4_nl", method="gptq", a8="blk32")
    + _r([HEAD], fmt="q8_0", method="rtn", a8="blk32"),
    # ---- HMX int8: static per-tensor uint8 activations x per-channel int8/int4 -------
    # ---- llama.cpp PR 29740 integer HMX: Q4_0 -> per-column/k-chunk int8, per-token activations
    "hmxint_w4a16_gptq": _r(BODY, fmt="q4_0", method="gptq", a8="tok16", hmxint=True)
    + _r([HEAD], fmt="q8_0", method="rtn"),
    "hmxint_w4a8_gptq": _r(BODY, fmt="q4_0", method="gptq", a8="tok", hmxint=True)
    + _r([HEAD], fmt="q8_0", method="rtn"),
    "hmx_w8a8": _r(ALL, fmt="int8_pc", method="rtn", a8="static"),
    "hmx_w8a8_sq05": _r(ALL, fmt="int8_pc", method="rtn", a8="static", smooth=0.5),
    "hmx_w8a8_sq08": _r(ALL, fmt="int8_pc", method="rtn", a8="static", smooth=0.8),
    "hmx_w8a8_sq05_gptq": _r(ALL, fmt="int8_pc", method="gptq", a8="static", smooth=0.5),
    "hmx_w4a8_sq05_gptq": _r(BODY, fmt="int4_pc", method="gptq", a8="static", smooth=0.5)
    + _r([HEAD], fmt="int8_pc", method="rtn", a8="static", smooth=0.5),
    # ---- where the W8A8 error comes from (device recipe minus one part; a rule with spec None keeps fp16)
    "w8a8_only_llm": _r([LLM, HEAD], **W8A8S),
    "w8a8_only_vision": _r([VISION, PROJ], **W8A8S),
    "w8a8_llm_fc2_fp16": [(LLM_FC2, None)] + _r(ALL, **W8A8S),
    "w8a8_head_fp16": [(HEAD, None)] + _r(ALL, **W8A8S),
    # ---- device configurations: prompt = prefill (HMX GEMM), generated tokens = decode (HVX GEMV).
    # The first answer token comes from the last prompt position, i.e. the prefill weights, and the
    # lm_head always runs as a GEMV. All W8A8 (one weight set) is w8a8_sq05s_gptq.
    "dev_w8a8_iq4": _r([VISION, PROJ], **W8A8S) + _r([LLM], **W8A8S, decode=IQ4BLK)
    + _r([HEAD], **HEAD8BLK),
    # w8a8_sq05s_gptq with the decode copy of the LLM and the lm_head as 4-bit blocks (Q4_0, GPTQ) on the same
    # per-token int8 activations and the same smoothing vectors as the int8 copy, so the device feeds both the same
    # scaled activations (benzene cosmos3_edge_generate --dec4; export_hexmodel w8a8_sq05s_gptq --dec4 dev_w8a8_q4hs)
    "dev_w8a8_q4hs": _r([VISION, PROJ], **W8A8S)
    + _r([LLM], **W8A8S, decode=dict(fmt="q4_0", method="gptq", a8=True, smooth=0.5, smooth_qkv=True))
    + _r([HEAD], fmt="q4_0", method="gptq", a8=True, smooth=0.5),
}
DEVICE_RECIPE = "w8a8_sq05s_gptq"  # the recipe the device runs: export_hexmodel's and the checks' default


def spec_of(recipe: list, name: str) -> dict | None:
    for rx, spec in recipe:
        if re.search(rx, name):
            return spec
    return None


def _cache_path(name: str, spec: dict):
    key = {k: spec.get(k) for k in ("fmt", "method", "act_order", "smooth")}
    if spec.get("smooth_qkv") and QKV.search(name):  # other layers share the plain cache
        key["smooth_qkv"] = True
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return OUT_DIR / "qweights" / h / f"{name}.pt", key


@torch.no_grad()
def quantize_weight(name: str, lin: torch.nn.Linear, spec: dict, smooth: torch.Tensor | None):
    """Return the dequantized (fp32) weight for ``lin`` under ``spec``; cached on disk."""
    from .calibrate import load_hessian

    path, key = _cache_path(name, spec)
    if path.exists():
        return torch.load(path, map_location=lin.weight.device)["w"].float()
    fmt = FORMATS[spec["fmt"]]
    dev = lin.weight.device
    w = lin.weight.float()
    hs = None
    if spec["method"] == "gptq" or fmt.group:
        hs = load_hessian(name, dev)
    if smooth is not None:
        w = w * smooth[None, :]
    cols = w.shape[1]
    g = fmt.group or cols
    pc = padded_cols(cols, g)
    H = None
    if hs is not None:
        H = hs["H"].float()
        if smooth is not None:  # H of x/s
            inv = 1.0 / smooth
            H = H * inv[:, None] * inv[None, :]
    if pc != cols:  # pad K to the 32-element block (zero weights, dead inputs)
        w = torch.nn.functional.pad(w, (0, pc - cols))
        if H is not None:
            H = torch.nn.functional.pad(H, (0, pc - cols, 0, pc - cols))
    if spec["method"] == "gptq":
        q, scales = gptq(w, H, fmt, act_order=spec.get("act_order", True))
    else:
        q, scales = rtn(w, fmt, None if H is None else torch.diag(H))
    q = q[:, :cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "spec.json").write_text(json.dumps(key))
    # w: dequantized fp32 [rows, cols]; scales: fp16-exact per (row, 32-block) incl. padding;
    # smooth: SmoothQuant vector already multiplied into w (activation must be divided by it)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save({"w": q.cpu(), "scales": scales.cpu(), "fmt": fmt.name, "cols_padded": pc,
                "smooth": None if smooth is None else smooth.cpu()}, tmp)
    os.replace(tmp, path)
    return q


class _Smoother:
    """SmoothQuant vectors from the original weights. With smooth_qkv the q/k/v group vector is
    computed when the group's first member is seen, i.e. before any member is replaced."""

    def __init__(self, model):
        self.model, self.group = model, {}

    def __call__(self, name: str, mod: torch.nn.Linear, spec: dict) -> torch.Tensor | None:
        from .calibrate import load_hessian

        if not spec.get("a8") or spec.get("smooth") is None:
            return None
        amax = load_hessian(name, mod.weight.device)["amax"]  # q/k/v share the q_proj statistics
        if not (spec.get("smooth_qkv") and QKV.search(name)):
            return smooth_vector(amax, mod.weight, spec["smooth"])
        parent = name.rpartition(".")[0]
        key = (parent, spec["smooth"])
        if key not in self.group:
            ws = [self.model.get_submodule(f"{parent}.{p}_proj").weight for p in "qkv"]
            assert all(isinstance(self.model.get_submodule(f"{parent}.{p}_proj"), torch.nn.Linear)
                       for p in "qkv"), f"{parent}: q/k/v already replaced"
            self.group[key] = smooth_vector(amax, torch.cat(ws, 0), spec["smooth"])
        return self.group[key]


@torch.no_grad()
def _quantized(name: str, mod: torch.nn.Linear, spec: dict, smoother, inplace: bool):
    """Module computing ``mod`` under ``spec``; weight-only specs update ``mod`` when inplace."""
    smooth = smoother(name, mod, spec)
    q = quantize_weight(name, mod, spec, smooth)
    if spec.get("hmxint"):
        from benzene.quantization import q4_0_to_hmx_int8
        c = torch.load(_cache_path(name, spec)[0], map_location=q.device)
        assert c["fmt"] == "q4_0"
        wp = torch.nn.functional.pad(c["w"].float(), (0, c["cols_padded"] - c["w"].shape[1]))
        q = q4_0_to_hmx_int8(wp, c["scales"])[:, : q.shape[1]]
    if spec.get("a8"):
        mode = "tok" if spec["a8"] is True else spec["a8"]
        rng = None
        if mode == "static":
            from .calibrate import load_actrange
            rng = load_actrange(name)
        return ActQuantLinear(mod, q, smooth, mode=mode, ch_range=rng)
    if inplace:
        mod.weight.data.copy_(q.to(mod.weight.dtype))
        return mod
    new = torch.nn.Linear(mod.in_features, mod.out_features, bias=False,
                          device=mod.weight.device, dtype=mod.weight.dtype)
    new.weight.data.copy_(q.to(mod.weight.dtype))
    new.bias = mod.bias
    return new


@torch.no_grad()
def apply_recipe(model, recipe_name: str, verbose: bool = True) -> dict:
    recipe = RECIPES[recipe_name]
    t0 = time.time()
    replaced, bits = 0, {}
    smoother = _Smoother(model)
    # names only: holding the module list would keep every replaced fp16 weight alive
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    for name in names:
        mod = model.get_submodule(name)
        spec = spec_of(recipe, name)
        if spec is None:
            continue
        if spec.get("decode"):
            new = MixedLinear(_quantized(name, mod, spec, smoother, inplace=False),
                              _quantized(name, mod, spec["decode"], smoother, inplace=False))
            bits[spec["decode"]["fmt"]] = bits.get(spec["decode"]["fmt"], 0) + mod.weight.numel()
        else:
            new = _quantized(name, mod, spec, smoother, inplace=True)
        if new is not mod:
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child, new)
        replaced += 1
        bits[spec["fmt"]] = bits.get(spec["fmt"], 0) + mod.weight.numel()
    if verbose:
        print(f"[{recipe_name}] quantized {replaced} Linear layers in {time.time() - t0:.0f}s "
              f"{ {k: f'{v / 1e6:.0f}M' for k, v in bits.items()} }", flush=True)
    return {"replaced": replaced, "params": bits}
