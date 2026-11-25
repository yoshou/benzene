"""Build cached multi-view samples for the accuracy evaluation of the quantized AnySplat and for its calibration.

A sample is 10 frames of one scene, preprocessed exactly as AnySplat's process_image (src/utils/image.py: the shorter
side resized to 448 by PIL's default bicubic filter, then a centre crop of 448 x 448) and stored as the exact uint8
values (AnySplat's input is these / 255). Of the 10 frames, in capture order, the 3rd and the 8th are novel-view
targets and the other 8 the context the model sees (AnySplat's eval_nvs holds out views the same way):
  images    [8, 3, 448, 448] uint8 contexts      targets   [2, 3, 448, 448] uint8
  gt_extri  [8, 3, 4] world-to-camera (OpenCV axes), gt_intri [8, 3, 3] for the crop; target_extri, target_intri the same
            for the targets
  gt_depth  [8, 448, 448] fp16 metres of the contexts (0 = no measurement; TUM only, nearest-resampled)

Sources, the calibration / evaluation scene split and the frames' ground truth are the VGGT port's
(benzene.models.vggt.prepare_data, its Tum and Mip360 with this preprocess): TUM RGB-D (10 frames from a random start
with a random stride of 8..32 frames) and Mip-NeRF 360 (even samples: 10 of the 24 views nearest to a random anchor; odd
samples: 10 random views of the scene).

  python -m benzene.models.anysplat.prepare_data --split calib --count 64
  python -m benzene.models.anysplat.prepare_data --split eval --count 96
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..vggt.prepare_data import SPLITS, Mip360, Tum
from .paths import DATA_DIR

SIZE = 448
FRAMES = 10
TARGETS = (2, 7)  # positions of the targets among the sample's frames


def preprocess(path: Path) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """process_image for one image -> (uint8 [3, 448, 448], (sx, sy) resize factors, (left, top) crop offsets)."""
    img = Image.open(path).convert("RGB")  # (process_image drops an alpha channel)
    w, h = img.size
    if w > h:
        nw, nh = int(w * (SIZE / h)), SIZE
    else:
        nw, nh = SIZE, int(h * (SIZE / w))
    img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    a = np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.uint8).transpose(2, 0, 1)
    return np.ascontiguousarray(a), (nw / w, nh / h), (left, top)


def build(split: str, count: int, seed: int) -> None:
    out_dir = DATA_DIR / "cache" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [("tum", n) for n in SPLITS[split]["tum"]] + [("mip360", n) for n in SPLITS[split]["mip360"]]
    loaded: dict = {}
    for k in range(count):
        ds, scene = sources[k % len(sources)]  # round-robin over the scenes
        out = out_dir / f"{k:03d}_{ds}_{scene.replace('rgbd_dataset_', '')}.pt"
        sub = np.random.default_rng([seed, k])  # per sample: a larger --count keeps the existing samples
        if out.exists():
            continue
        if scene not in loaded:
            loaded[scene] = Tum(scene) if ds == "tum" else Mip360(scene)
        src = loaded[scene]
        idx = src.pick(sub, FRAMES) if ds == "tum" else src.pick(sub, FRAMES, local=(k // len(sources)) % 2 == 0)
        fr = [src.frame(i, preprocess) for i in idx]
        ctx = [f for j, f in enumerate(fr) if j not in TARGETS]
        tgt = [fr[j] for j in TARGETS]
        stack = lambda frames, key: torch.from_numpy(np.stack([f[key] for f in frames]))  # noqa: E731
        rec = {"dataset": ds, "scene": scene, "frames": [f["name"] for f in ctx],
               "target_frames": [f["name"] for f in tgt],
               "images": stack(ctx, "img"), "targets": stack(tgt, "img"),
               "gt_extri": stack(ctx, "extri"), "gt_intri": stack(ctx, "intri"),
               "target_extri": stack(tgt, "extri"), "target_intri": stack(tgt, "intri"),
               "gt_depth": None if fr[0]["depth"] is None else stack(ctx, "depth")}
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        torch.save(rec, tmp)
        os.replace(tmp, out)
        print(f"{out.name} contexts {rec['frames'][0]} .. {rec['frames'][-1]}, targets {rec['target_frames']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    build(args.split, args.count, args.seed if args.seed is not None else {"calib": 1, "eval": 2}[args.split])


if __name__ == "__main__":
    main()
