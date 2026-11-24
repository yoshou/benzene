"""VGGT model loading and inference with controlled precision.

predict() runs the aggregator under `agg` autocast (None = fp32) and the camera / depth / point heads under `head`
autocast (None = fp32, as the official forward, which disables autocast for the heads). Outputs are fp32 on the CPU:
pose_enc [S, 9], depth / depth_conf / world_points_conf [S, H, W], world_points [S, H, W, 3].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import MODEL_DIR


def load_model(device: str = "cuda:0", track: bool = False):
    from safetensors.torch import load_file

    from vggt.models.vggt import VGGT

    model = VGGT(enable_track=track)
    sd = load_file(str(MODEL_DIR / "model.safetensors"))
    if not track:
        sd = {k: v for k, v in sd.items() if not k.startswith("track_head.")}
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_sample(path: str | Path, device: str) -> dict[str, Any]:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["x"] = rec["images"].to(device).float() / 255.0  # exactly the loader's tensor
    rec["name"] = Path(path).stem
    return rec


def _ac(dtype):
    return torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


@torch.inference_mode()
def predict(model, x: torch.Tensor, agg=None, head=None, keep_tokens: bool = False) -> dict[str, torch.Tensor]:
    """x: [S, 3, H, W] in [0, 1]."""
    images = x[None]
    with _ac(agg):
        tokens, ps = model.aggregator(images)
    out: dict[str, torch.Tensor] = {}
    with _ac(head):
        out["pose_enc"] = model.camera_head(tokens)[-1][0]
        d, dc = model.depth_head(tokens, images=images, patch_start_idx=ps)
        p, pc = model.point_head(tokens, images=images, patch_start_idx=ps)
    out.update(depth=d[0, ..., 0], depth_conf=dc[0], world_points=p[0], world_points_conf=pc[0])
    out = {k: v.float().cpu() for k, v in out.items()}
    if keep_tokens:
        out["tokens"] = tokens
    return out
