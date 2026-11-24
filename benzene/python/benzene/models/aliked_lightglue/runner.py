"""ALIKED + LightGlue and their inference on the evaluation pairs with controlled precision.

load_models() builds:
  ALIKED     lightglue/aliked.py of the cvg/LightGlue checkout (benzene.models.lightglue.source.aliked_module), model
             aliked-n16 (paths.WEIGHTS/aliked-n16.pth), detection threshold 0.2 and NMS radius 2 (LightGlue's
             defaults for it), at most MAX_KPTS keypoints per image
  LightGlue  features="aliked": the release weights aliked_lightglue.pth (input_dim 128 projected to 256 dimensions, 4
             heads, 9 layers), all layers on every keypoint (adaptive inference is the recipe "fp16a"), explicit
             attention (flash off: the flash path computes attention in fp16)

load_pair(): a pair of benzene.models.lightglue.prepare_data with both images centre-cropped to multiples of 32 -
ALIKED's network works on multiples of 32 (its own InputPadder pads by replication; the crop keeps the uint8 pixels the
device reads) - and the intrinsics shifted with them.

extract(): the gray image as ALIKED's RGB input (gray / 255 in all three channels: what its forward does with a
one-channel image), extract_dense_map -> (feature map, score map), then DKD with a deterministic order - the score
map's NMS maxima (radius 2) above the threshold outside the 2-pixel border, by descending score with ties in raster
order (DKD sorts with an unstable sort, and only when there are more than the limit), the first MAX_KPTS - refined by
DKD's soft-argmax and scored by its bilinear sample, then SDDH's descriptors. Outputs as SuperPoint's: kpts{0,1}
[k, 2] (x, y pixel centres), scores{0,1} [k], desc{0,1} [k, 128] fp32, matches0 [k0], mscores0 [k0] (CPU).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..lightglue import evaluation, matcher
from ..lightglue.matcher import autocast
from ..lightglue.source import aliked_module, modules
from .paths import WEIGHTS

MODEL_NAME = "aliked-n16"
MAX_KPTS = 1024  # keypoints per image
DESCRIPTOR_DIMENSIONS = 128
DETECTION_THRESHOLD = 0.2  # ALIKED's default
NMS_RADIUS = 2
INPUT_ALIGNMENT = 32  # ALIKED's network: sizes divisible by 2^5
ADAPTIVE = (0.95, 0.99)  # LightGlue's default depth_confidence (early stop), width_confidence (point pruning)


def _local_weights(url: str, *args, **kwargs):
    return torch.load(WEIGHTS / Path(url).name, map_location="cpu", weights_only=True)


def load_models(device: str = "cuda:0"):
    """(ALIKED, LightGlue) in eval mode on device, fp32."""
    aliked = aliked_module()
    _, lgm = modules()
    hub = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = _local_weights
    try:
        net = aliked.ALIKED(model_name=MODEL_NAME, max_num_keypoints=MAX_KPTS, detection_threshold=DETECTION_THRESHOLD,
                            nms_radius=NMS_RADIUS)
        lg = lgm.LightGlue(features="aliked", flash=False, depth_confidence=-1, width_confidence=-1)
    finally:
        torch.hub.load_state_dict_from_url = hub
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


def rgb_input(img: torch.Tensor, device: str) -> torch.Tensor:
    """ALIKED's input of a uint8 gray image [H, W]: [1, 3, H, W] gray / 255 (grayscale_to_rgb of the gray image)."""
    x = img.to(device).float()[None, None] / 255.0
    return x.expand(-1, 3, -1, -1).contiguous()


def select_keypoints(score_map: torch.Tensor, max_kpts: int = MAX_KPTS, threshold: float = DETECTION_THRESHOLD,
                     radius: int = NMS_RADIUS) -> torch.Tensor:
    """DKD's keypoint pixels of a score map [1, 1, H, W] as flat indices y W + x: the NMS maxima above the threshold
    (none: above the map's mean, as DKD; summed in fp64) outside the border, by descending score, ties in raster
    order, at most max_kpts."""
    aliked = aliked_module()
    scores = score_map.detach()
    nms_scores = aliked.simple_nms(scores, radius)
    nms_scores[:, :, :radius, :] = 0
    nms_scores[:, :, :, :radius] = 0
    nms_scores[:, :, -radius:, :] = 0
    nms_scores[:, :, :, -radius:] = 0
    mask = nms_scores > threshold
    if mask.sum() == 0:  # (the mean in fp64, rounded to fp32: the device's)
        mask = nms_scores > scores.reshape(-1).double().mean().float()
    indices = mask.reshape(-1).nonzero()[:, 0]
    order = torch.sort(scores.reshape(-1)[indices], descending=True, stable=True).indices[:max_kpts]
    return indices[order]


def refine_keypoints(dkd, score_map: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DKD's sub-pixel keypoints (normalized to [-1, 1]) and scores of the keypoint pixels `indices`: the soft-argmax
    over the (2r + 1)^2 window (temperature 0.1), the score map's bilinear sample there (align_corners True)."""
    _, _, h, w = score_map.shape
    patches = dkd.unfold(score_map)[0].t()  # [H W, (2r + 1)^2]
    hw_grid = dkd.hw_grid.to(score_map)
    patch_scores = patches[indices]
    keypoints_xy_nms = torch.stack([indices % w, torch.div(indices, w, rounding_mode="trunc")], dim=1)
    max_v = patch_scores.max(dim=1).values[:, None]
    x_exp = ((patch_scores - max_v) / dkd.temperature).exp()
    xy_residual = x_exp @ hw_grid / x_exp.sum(dim=1)[:, None]
    wh = torch.tensor([w - 1, h - 1], device=score_map.device)
    keypoints_xy = (keypoints_xy_nms + xy_residual) / wh * 2 - 1
    scores = F.grid_sample(score_map, keypoints_xy.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0, :]
    return keypoints_xy, scores


@torch.inference_mode()
def extract(net, img: torch.Tensor, device: str, ac=None, max_kpts: int = MAX_KPTS) -> dict:
    """ALIKED's features of a uint8 gray image [H, W] (H, W multiples of 32): keypoints [k, 2] (pixels), keypoint_scores
    [k], descriptors [k, 128]; net is the fp32 ALIKED (ac: autocast dtype of its networks), a network with its
    interface (hooked_network.HookedAliked) or the device model."""
    from .device_model import DeviceAliked

    if isinstance(net, DeviceAliked):  # its own arithmetic throughout
        return net.extract(img, max_kpts)
    x = rgb_input(img, device)
    height, width = x.shape[-2:]
    with autocast(ac):
        feature_map, score_map = net.extract_dense_map(x)
    feature_map, score_map = feature_map.float(), score_map.float()
    indices = select_keypoints(score_map, max_kpts)
    keypoints, scores = refine_keypoints(net.dkd, score_map, indices)
    with autocast(ac):
        descriptors, _ = net.desc_head(feature_map, [keypoints])
    wh = torch.tensor([width - 1, height - 1], device=x.device)
    return {"keypoints": wh * (keypoints + 1) / 2.0, "keypoint_scores": scores, "descriptors": descriptors[0].float()}


def run_pair(net, lg, rec: dict, device: str, aliked_ac=None, lg_ac=None) -> dict:
    return matcher.run_pair(extract, net, lg, rec, device, aliked_ac, lg_ac)
