"""SuperPoint + LightGlue models and their inference on the evaluation pairs (benzene.models.lightglue.prepare_data)
with controlled precision.

load_models() builds the models from the cvg/LightGlue checkout (benzene.models.lightglue.source). Weights are the
LightGlue v0.1_arxiv release files in paths.WEIGHTS.

run_pair() extracts SuperPoint features of both images (input gray / 255, exactly the device input) and matches them
with LightGlue; `sp_ac` / `lg_ac` are autocast dtypes for the two networks (None = fp32). Outputs are on the CPU:
kpts{0,1} [k, 2] (x, y pixel centres), scores{0,1} [k], desc{0,1} [k, 256] fp32, matches0 [k0] (index into image 1
or -1), mscores0 [k0].
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..lightglue import evaluation, matcher
from ..lightglue.matcher import autocast
from ..lightglue.source import modules
from .paths import WEIGHTS

MAX_KPTS = 1024  # keypoints per image (SuperPoint max_num_keypoints)
ADAPTIVE = (0.95, 0.99)  # LightGlue's default depth_confidence (early stop), width_confidence (point pruning)
DESCRIPTOR_DIMENSIONS = 256


def _local_weights(url: str, *args, **kwargs):
    return torch.load(WEIGHTS / Path(url).name, map_location="cpu", weights_only=True)


def load_pair(path: str | Path) -> dict:
    """An evaluation pair, its images as they are (the accuracy evaluation's input)."""
    return evaluation.load_pair(path)


def load_models(device: str = "cuda:0", max_kpts: int = MAX_KPTS):
    """(SuperPoint, LightGlue) in eval mode on device, fp32. LightGlue runs all 9 layers (no early stop, no point
    pruning) with the explicit attention path (flash off: the official flash path computes attention in fp16)."""
    spm, lgm = modules()
    hub = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = _local_weights
    try:
        sp = spm.SuperPoint(max_num_keypoints=max_kpts)
        lg = lgm.LightGlue(features="superpoint", flash=False, depth_confidence=-1, width_confidence=-1)
    finally:
        torch.hub.load_state_dict_from_url = hub
    return sp.eval().to(device), lg.eval().to(device)


@torch.inference_mode()
def extract(sp, img: torch.Tensor, device: str, ac=None) -> dict:
    x = img.to(device).float()[None, None] / 255.0
    with autocast(ac):
        f = sp({"image": x})
    return {k: v[0].float() for k, v in f.items()}


def run_pair(sp, lg, rec: dict, device: str, sp_ac=None, lg_ac=None) -> dict:
    return matcher.run_pair(extract, sp, lg, rec, device, sp_ac, lg_ac)
