"""LoMa and its inference on the evaluation pairs with controlled precision and input settings.

load_pair(): a pair of benzene.models.lightglue.prepare_data plus its colour frames (rgb0, rgb1: uint8 [H, W, 3] read
from the raw datasets with prepare_data's crop; their grayscale must reproduce the stored images).

Settings (what the networks see; the defaults: LoMa's own match()):
  detector_size    "native" (the image as it is) or the long side the detector input is resized to (aspect kept,
                   both sides floored to multiples of 8: DaD.load_image; LoMa's default 1024)
  descriptor_size  "native14" / "native56" (each side resized to the nearest multiple of 14 / 56: DINOv2's patches /
                   and VGG's three halvings) or (W, H) (LoMa: 784 x 784, aspect not kept: DeDoDeDescriptor.read_image)
  keypoints        DaD keypoints per image (LoMa's default 2048; its paper evaluates 4096)
  color            "rgb" or "gray" (the stored grayscale image in all three channels)
Resizing is PIL's default (bicubic) on the uint8 image, as LoMa reads image files.

extract(): DaD's logits -> softmax over all pixels -> 3 x 3 NMS (non-maxima zeroed) -> the `keypoints` largest by
descending probability, ties in raster order (the checkout's torch.topk leaves the order of ties open) -> sub-pixel
offsets from the softmax of the 3 x 3 logits / 0.5 (zero logits outside the map); DeDoDe's description map sampled
bilinearly there. run_pair(): the matcher on both images' normalized keypoints and descriptions, its dual-softmax
assignment, mutual nearest neighbours above 0.1. Outputs (CPU): kpts{0,1} [k, 2] pixel coordinates of the original
image (pixel centres at integers, the intrinsics' convention), scores{0,1} [k] (keypoint probabilities), desc{0,1}
[k, D] fp32, kpts_normalized{0,1}, matches0 [k0], mscores0 [k0].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from benzene.workspace import MIPNERF360, TUM_RGBD

from ..lightglue import evaluation
from ..lightglue.matcher import autocast
from .network import filter_matches, load_network, sample_descriptions
from .paths import WEIGHTS

VARIANT = "loma_b"
FILTER_THRESHOLD = 0.1  # LoMa.Cfg.filter_threshold
NMS_SIZE = 3
SUBPIXEL_TEMPERATURE = 0.5


@dataclass(frozen=True)
class Settings:
    detector_size: str | int = 1024
    descriptor_size: str | tuple[int, int] = (784, 784)
    keypoints: int = 2048
    color: str = "rgb"

    @property
    def name(self) -> str:
        desc = self.descriptor_size if isinstance(self.descriptor_size, str) else "x".join(map(str, self.descriptor_size))
        return f"det{self.detector_size}-desc{desc}-k{self.keypoints}-{self.color}"


def load_model(variant: str = VARIANT, device: str = "cuda:0"):
    return load_network(variant, WEIGHTS, device)


def _frame_path(rec: dict, i: int) -> Path:
    if rec["dataset"] == "tum":
        return TUM_RGBD / rec["scene"] / rec["names"][i]
    return MIPNERF360 / rec["scene"] / "images_8" / rec["names"][i]


def _rgb_frame(rec: dict, i: int) -> np.ndarray:
    a = np.array(Image.open(_frame_path(rec, i)).convert("RGB"))
    height, width = rec[f"image{i}"].shape
    y0, x0 = (a.shape[0] - height) // 2, (a.shape[1] - width) // 2
    a = np.ascontiguousarray(a[y0 : y0 + height, x0 : x0 + width])
    f = a.astype(np.float64)
    gray = np.clip(np.round(f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114), 0, 255).astype(np.uint8)
    assert np.array_equal(gray, rec[f"image{i}"].numpy()), f"{rec['name']}: frame {i} does not match its gray image"
    return a


def load_pair(path: str | Path) -> dict:
    """An evaluation / calibration pair with its colour frames rgb0, rgb1."""
    rec = evaluation.load_pair(path)
    for i in (0, 1):
        rec[f"rgb{i}"] = torch.from_numpy(_rgb_frame(rec, i))
    return rec


def detector_shape(width: int, height: int, size: str | int) -> tuple[int, int]:
    if size == "native":
        return width, height
    scale = size / max(width, height)
    return int((scale * width) // 8 * 8), int((scale * height) // 8 * 8)


def descriptor_shape(width: int, height: int, size: str | tuple[int, int]) -> tuple[int, int]:
    if size in ("native14", "native56"):
        multiple = int(size[6:])
        return max(multiple, round(width / multiple) * multiple), max(multiple, round(height / multiple) * multiple)
    return size


def network_input(rgb: np.ndarray, shape: tuple[int, int], device: str) -> torch.Tensor:
    """[1, 3, H, W] in [0, 1] of the uint8 image resized (PIL bicubic) to shape (W, H) if it differs."""
    if (rgb.shape[1], rgb.shape[0]) != tuple(shape):
        rgb = np.asarray(Image.fromarray(rgb).resize(tuple(shape)))
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


def network_codes(rgb: np.ndarray, shape: tuple[int, int], device: str) -> torch.Tensor:
    """The device's input: uint8 gray codes [H, W] of an image whose three channels are equal (image_rgb of a gray
    setting), resized like network_input (PIL treats the channels alike)."""
    gray = rgb[..., 0]
    if (gray.shape[1], gray.shape[0]) != tuple(shape):
        gray = np.array(Image.fromarray(gray).resize(tuple(shape)))
    return torch.from_numpy(np.ascontiguousarray(gray)).to(device)


def image_rgb(rec: dict, i: int, settings: Settings) -> np.ndarray:
    if settings.color == "gray":
        return np.repeat(rec[f"image{i}"].numpy()[..., None], 3, axis=2)
    return rec[f"rgb{i}"].numpy()


def select_keypoints(logits: torch.Tensor, keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DaD's keypoints of logits [1, 1, H, W]: normalized positions [k, 2] and probabilities [k] (sample_keypoints with
    NMS 3, top-k, sub-pixel temperature 0.5; ties by raster order)."""
    _, _, H, W = logits.shape
    probs = logits.reshape(1, H * W).softmax(dim=-1).reshape(1, 1, H, W)
    probs = probs * (probs == F.max_pool2d(probs, NMS_SIZE, stride=1, padding=NMS_SIZE // 2))
    flat = probs.reshape(-1)
    indices = torch.sort(flat, descending=True, stable=True).indices[:keypoints]
    device = logits.device
    xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
    positions = torch.stack((xs[indices % W], ys[indices // W]), dim=-1)
    offsets_x = torch.linspace(-1 + 1 / NMS_SIZE, 1 - 1 / NMS_SIZE, NMS_SIZE, device=device) * NMS_SIZE / W
    offsets_y = torch.linspace(-1 + 1 / NMS_SIZE, 1 - 1 / NMS_SIZE, NMS_SIZE, device=device) * NMS_SIZE / H
    offsets = torch.stack(torch.meshgrid(offsets_y, offsets_x, indexing="ij")[::-1], dim=-1).reshape(-1, 2)
    patches = F.unfold(logits.reshape(1, 1, H, W), NMS_SIZE, padding=NMS_SIZE // 2)[0][:, indices]  # [9, k]
    weights = (patches / SUBPIXEL_TEMPERATURE).softmax(dim=0)
    return positions + torch.einsum("kn,kd->nd", weights, offsets), flat[indices]


@torch.inference_mode()
def extract(net, rgb: np.ndarray, settings: Settings, device: str, ac=None) -> dict:
    """LoMa's features of a uint8 image [H, W, 3]: keypoints_normalized [k, 2], keypoints [k, 2] (pixels, centres at
    integers), keypoint_scores [k], descriptors [k, D] fp32; ac: autocast dtype of the networks (None: fp32)."""
    from .device_model import DeviceLoMa

    if isinstance(net, DeviceLoMa):  # its own arithmetic throughout
        return net.extract(rgb, settings)
    height, width = rgb.shape[:2]
    detector_image = network_input(rgb, detector_shape(width, height, settings.detector_size), device)
    descriptor_image = network_input(rgb, descriptor_shape(width, height, settings.descriptor_size), device)
    with autocast(ac):
        logits = net._detector(detector_image).float()
    keypoints, scores = select_keypoints(logits, settings.keypoints)
    with autocast(ac):
        descriptions = net._descriptor(descriptor_image)
    descriptors = sample_descriptions(descriptions, keypoints[None])[0]
    size = torch.tensor([width, height], device=keypoints.device, dtype=torch.float32)
    return {"keypoints_normalized": keypoints, "keypoints": size * (keypoints + 1) / 2 - 0.5,
            "keypoint_scores": scores, "descriptors": descriptors.float()}


@torch.inference_mode()
def match(net, f0: dict, f1: dict, ac=None) -> dict:
    with autocast(ac):
        scores = net.match(f0["keypoints_normalized"][None], f1["keypoints_normalized"][None],
                           f0["descriptors"][None], f1["descriptors"][None]).float()
    m0, _, mscores0, _ = filter_matches(scores, FILTER_THRESHOLD)
    return {"matches0": m0[0], "mscores0": mscores0[0]}


def run_pair(net, rec: dict, settings: Settings, device: str, extract_ac=None, match_ac=None) -> dict:
    f0 = extract(net, image_rgb(rec, 0, settings), settings, device, extract_ac)
    f1 = extract(net, image_rgb(rec, 1, settings), settings, device, extract_ac)
    m = match(net, f0, f1, match_ac)
    out = {"kpts0": f0["keypoints"], "kpts1": f1["keypoints"], "scores0": f0["keypoint_scores"],
           "scores1": f1["keypoint_scores"], "desc0": f0["descriptors"], "desc1": f1["descriptors"],
           "kpts_normalized0": f0["keypoints_normalized"], "kpts_normalized1": f1["keypoints_normalized"], **m}
    return {k: v.cpu() for k, v in out.items()}
