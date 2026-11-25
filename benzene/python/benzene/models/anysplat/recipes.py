"""Quantization recipes for AnySplat: the VGGT port's (benzene.models.vggt.recipes: specs, the W8A8 path, GPTQ,
SmoothQuant) on AnySplat's network, with its own quantized weights and Hessians under OUT_DIR.

A recipe is the precision of everything left unquantized (agg: autocast dtype of the aggregator, head: of the camera,
depth and Gaussian heads, None = fp32) plus the ordered (module regex, spec) list of VGGT's recipes.

  fp32       the reference
  bf16       AnySplat's own inference: the aggregator in bf16 (its weights are stored in bf16), the heads in fp32
  fp16       the device's float floor
  w8a8_tf    every Linear of the transformer (DINOv2 blocks, frame and global blocks) as the device's W8A8 GEMM path
             (per-channel int8 GPTQ weights, per-token int8 activations, SmoothQuant 0.5), the rest fp16
  w8a8_tfc   w8a8_tf and the camera head's Linears: the device recipe (the DPT and Gaussian heads in fp16 on the fp16 HMX
             GEMM, as VGGT's)
  w8a8_tfc_rtn0  w8a8_tfc with round-to-nearest weights and no smoothing (no calibration data; rtn0 as in the VGGT
                 port's recipe names)
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..vggt import recipes as vggt_recipes
from ..vggt.recipes import CAM, F16, TF, W8A8S, spec_of  # noqa: F401 - spec_of re-exported
from .paths import OUT_DIR


def _r(regexes, **spec):
    return [(rx, spec) for rx in regexes]


RECIPES: dict[str, tuple[dict, list]] = {
    "fp32": (dict(agg=None, head=None), []),
    "bf16": (dict(agg=torch.bfloat16, head=None), []),
    "fp16": (F16, []),
    "w8a8_tf": (F16, _r(TF, **W8A8S)),
    "w8a8_tfc": (F16, _r(TF + [CAM], **W8A8S)),
    "w8a8_tfc_rtn0": (F16, _r(TF + [CAM], fmt="int8_pc", method="rtn", a8=True)),
}
DEVICE_RECIPE = "w8a8_tfc"  # the recipe the device runs: export_hexmodel's and the checks' default


def cache_path(name: str, spec: dict) -> tuple[Path, dict]:
    """The VGGT port's cache_path under AnySplat's OUT_DIR."""
    return vggt_recipes.cache_path(name, spec, OUT_DIR)


def apply_recipe(model, recipe_name: str, verbose: bool = True) -> dict:
    """The VGGT port's apply_recipe with AnySplat's RECIPES and OUT_DIR."""
    return vggt_recipes.apply_recipe(model, recipe_name, verbose, recipes=RECIPES, out_dir=OUT_DIR)
