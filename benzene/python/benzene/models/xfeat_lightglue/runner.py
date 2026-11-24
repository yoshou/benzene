"""XFeat + LighterGlue (XFeat's LightGlue) and their inference on the evaluation pairs with controlled precision.

load_models() builds:
  XFeat       XFeatModel of the verlab/accelerated_features checkout in paths.SRC_DIR (commit e92685f), weights
              weights/xfeat.pt - the extractor XFeat.match_lighterglue uses (xfeat-lighterglue.pt also holds an
              extractor, with other BatchNorm statistics: the official matcher loads it with strict=False and ignores it)
  LighterGlue cvg/LightGlue's LightGlue (benzene.models.lightglue.source) with the configuration of
              modules/lighterglue.py - input_dim 64, descriptor_dim 96, 6 layers, 1 head, filter_threshold 0.1 - and
              the matcher.* weights of weights/xfeat-lighterglue.pt, renamed as it renames them and loaded strictly;
              all 6 layers on every keypoint (its default adaptive inference, point pruning at width_confidence 0.95
              without early stop, is the recipe "fp16a"), explicit attention (flash off: the flash path computes
              attention in fp16)
The checkout's modules/ package (an __init__ with only a docstring) is imported as "modules", as its own files import
each other.

load_pair(): a pair of benzene.models.lightglue.prepare_data with both images centre-cropped to multiples of 32 -
XFeat's input size (its own preprocess_tensor resizes instead; the crop keeps the uint8 pixels the device reads) -
and the intrinsics shifted with them.

extract(): XFeat.detectAndCompute on the gray image as is (XFeat averages the colour channels and normalizes the
image itself; the pairs hold luminance gray, the device's input), with a deterministic order: keypoints by descending
score, ties in raster order (the official argsort is not stable). Outputs as SuperPoint's: kpts{0,1} [k, 2] (x, y
pixel), scores{0,1} [k], desc{0,1} [k, 64] fp32, matches0 [k0], mscores0 [k0] (CPU).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from ..lightglue import evaluation, matcher
from ..lightglue.matcher import autocast
from ..lightglue.source import modules as lightglue_modules
from .paths import SRC_DIR, WEIGHTS

MAX_KPTS = 1024  # keypoints per image (detectAndCompute top_k; the official default 4096)
DESCRIPTOR_DIMENSIONS = 64
DETECTION_THRESHOLD = 0.05  # detectAndCompute's default
LIGHTERGLUE = {"input_dim": 64, "descriptor_dim": 96, "n_layers": 6, "num_heads": 1, "filter_threshold": 0.1}
ADAPTIVE = (-1, 0.95)  # LighterGlue's depth_confidence (early stop off), width_confidence (point pruning)
INPUT_ALIGNMENT = 32  # XFeat's image sizes are multiples of 32


def xfeat_modules():
    """(modules.model, modules.xfeat, modules.interpolator) of the checkout."""
    if "modules.xfeat" not in sys.modules:
        package = types.ModuleType("modules")
        package.__path__ = [str(SRC_DIR / "modules")]
        sys.modules["modules"] = package
    return (importlib.import_module("modules.model"), importlib.import_module("modules.xfeat"),
            importlib.import_module("modules.interpolator"))


def lighterglue_state_dict(path: Path) -> dict:
    """The matcher.* weights of xfeat-lighterglue.pt with LighterGlue's renames (matcher. dropped, self_attn.<i> /
    cross_attn.<i> -> transformers.<i>.self_attn / cross_attn)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k[len("matcher."):]: v for k, v in state.items() if k.startswith("matcher.")}
    for i in range(LIGHTERGLUE["n_layers"]):
        for block in ("self_attn", "cross_attn"):
            state = {k.replace(f"{block}.{i}", f"transformers.{i}.{block}"): v for k, v in state.items()}
    return state


def load_models(device: str = "cuda:0"):
    """(XFeatModel, LightGlue) in eval mode on device, fp32."""
    model, _, _ = xfeat_modules()
    net = model.XFeatModel()
    net.load_state_dict(torch.load(WEIGHTS / "xfeat.pt", map_location="cpu", weights_only=True))
    _, lgm = lightglue_modules()
    lg = lgm.LightGlue(features=None, flash=False, depth_confidence=-1, width_confidence=-1, **LIGHTERGLUE)
    incompatible = lg.load_state_dict(lighterglue_state_dict(WEIGHTS / "xfeat-lighterglue.pt"), strict=False)
    # every parameter from the file: only the buffer LightGlue computes itself may be missing
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {"confidence_thresholds"}:
        raise RuntimeError(f"xfeat-lighterglue.pt does not fit LighterGlue: {incompatible}")
    return net.eval().to(device), lg.eval().to(device)


def load_pair(path: str | Path) -> dict:
    """An evaluation / calibration pair with both images centre-cropped to multiples of 32 (intrinsics shifted)."""
    rec = evaluation.load_pair(path)
    for i in (0, 1):
        image = rec[f"image{i}"]
        height, width = image.shape
        crop_height = height // INPUT_ALIGNMENT * INPUT_ALIGNMENT
        crop_width = width // INPUT_ALIGNMENT * INPUT_ALIGNMENT
        y0, x0 = (height - crop_height) // 2, (width - crop_width) // 2
        rec[f"image{i}"] = image[y0 : y0 + crop_height, x0 : x0 + crop_width].contiguous()
        K = rec[f"K{i}"].clone()
        K[0, 2] -= x0
        K[1, 2] -= y0
        rec[f"K{i}"] = K
    return rec


@torch.inference_mode()
def extract(net, img: torch.Tensor, device: str, ac=None, top_k: int = MAX_KPTS,
            threshold: float = DETECTION_THRESHOLD) -> dict:
    """XFeat.detectAndCompute of a uint8 gray image [H, W] (H, W multiples of 32): keypoints [k, 2], keypoint_scores
    [k], descriptors [k, 64], keypoints by descending score (ties in raster order)."""
    _, xfeat, interpolator = xfeat_modules()
    x = img.to(device).float()[None, None]
    height, width = x.shape[-2:]
    with autocast(ac):
        features, logits, reliability = net(x)
    features, logits, reliability = features.float(), logits.float(), reliability.float()
    features = F.normalize(features, dim=1)
    heatmap = xfeat.XFeat.get_kpts_heatmap(None, logits)
    keypoints = xfeat.XFeat.NMS(None, heatmap, threshold=threshold, kernel_size=5)  # [1, n, 2] raster order
    nearest = interpolator.InterpolateSparse2d("nearest")
    bilinear = interpolator.InterpolateSparse2d("bilinear")
    scores = (nearest(heatmap, keypoints, height, width) * bilinear(reliability, keypoints, height, width)).squeeze(-1)
    scores[torch.all(keypoints == 0, dim=-1)] = -1
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :top_k]
    keypoints = torch.gather(keypoints, 1, order[..., None].expand(-1, -1, 2))
    scores = torch.gather(scores, 1, order)
    descriptors = interpolator.InterpolateSparse2d("bicubic")(features, keypoints, height, width)
    descriptors = F.normalize(descriptors, dim=-1)
    valid = scores[0] > 0
    return {"keypoints": keypoints[0][valid].float(), "keypoint_scores": scores[0][valid],
            "descriptors": descriptors[0][valid]}


def run_pair(net, lg, rec: dict, device: str, xfeat_ac=None, lg_ac=None) -> dict:
    return matcher.run_pair(extract, net, lg, rec, device, xfeat_ac, lg_ac)
