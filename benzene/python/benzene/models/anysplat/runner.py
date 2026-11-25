"""AnySplat model loading and inference with controlled precision.

predict() runs the aggregator under `agg` autocast (None = fp32) and the camera / depth / Gaussian heads under `head`
autocast (None = fp32); outputs are fp32 on the model's device: pose_enc [S, 9], depth / depth_conf [S, H, W], raw
[S, H, W, RAW_CHANNELS] (the Gaussian head's raw outputs, network.RAW_*; gaussians.make_gaussians makes the Gaussians).
camera_poses() runs only the aggregator and the camera head (the evaluation's joint context + target pass).

The aggregator's fp32 attention takes PyTorch's ROCm flash attention (AOTriton), experimental on the GPUs here and
enabled by TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (set below unless given): for 8 frames of 448 x 448 its tokens are
the math path's to 2e-6 (relative), at 15 s instead of 23 s and 5 GiB instead of 14 GiB of GPU memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .network import load_network
from .paths import MODEL_DIR

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # read at the first attention call


def load_model(device: str = "cuda:0"):
    return load_network(MODEL_DIR, device)


def load_sample(path: str | Path, device: str) -> dict[str, Any]:
    """A cached sample (prepare_data) with x [8, 3, H, W] and xt [2, 3, H, W]: its contexts and targets / 255."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["x"] = rec["images"].to(device).float() / 255.0
    rec["xt"] = rec["targets"].to(device).float() / 255.0
    rec["name"] = Path(path).stem
    return rec


def _ac(dtype):
    return torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


@torch.inference_mode()
def predict(model, x: torch.Tensor, agg=None, head=None, frames_chunk_size: int = 2) -> dict[str, torch.Tensor]:
    """x: [S, 3, H, W] in [0, 1]. The DPT heads run frames_chunk_size frames at a time (each frame on its own: the
    same values; 8 frames of 448 x 448 at once take ~5 GiB in fp32)."""
    images = x[None]
    with _ac(agg):
        tokens, ps = model.aggregator(images)
    with _ac(head):
        pose_enc = model.camera_head(tokens)[-1][0]
        d, dc = model.depth_head(tokens, images=images, patch_start_idx=ps, frames_chunk_size=frames_chunk_size)
        raw = model.gaussian_param_head(tokens, images=images, patch_start_idx=ps,
                                        frames_chunk_size=frames_chunk_size)
    return {"pose_enc": pose_enc.float(), "depth": d[0, ..., 0].float(), "depth_conf": dc[0].float(),
            "raw": raw[0].permute(0, 2, 3, 1).float()}


@torch.inference_mode()
def camera_poses(model, x: torch.Tensor, agg=None, head=None) -> torch.Tensor:
    """pose_enc [S, 9] of frames x [S, 3, H, W] (aggregator and camera head only)."""
    with _ac(agg):
        tokens, _ = model.aggregator(x[None])
    with _ac(head):
        return model.camera_head(tokens)[-1][0].float()
