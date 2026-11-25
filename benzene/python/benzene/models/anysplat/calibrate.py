"""Calibration statistics for AnySplat's W8A8 layers (the VGGT port's collect_hessians): per Linear the Hessian
X^T X / n of its input and the per-input-channel |x| maximum, over the contexts of the calibration samples.

The model runs on --device with the aggregator under fp16 autocast and the camera head in fp32; the statistics
accumulate in fp32 on --hessian-device and go to OUT_DIR/hessians/<module>.pt.

  python -m benzene.models.anysplat.calibrate --group all     # both in one pass
  python -m benzene.models.anysplat.calibrate --group tf      # DINOv2 blocks, frame / global blocks
  python -m benzene.models.anysplat.calibrate --group heads   # camera head
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

from ..vggt.calibrate import collect_hessians
from .paths import DATA_DIR, OUT_DIR
from .recipes import CAM, TF
from .runner import camera_poses, load_model, load_sample

GROUPS = {"all": TF + [CAM], "tf": TF, "heads": [CAM]}


def calibration_files(count: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache/calib/*.pt")))
    return files[:count] if count else files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS), required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hessian-device", default="cuda:1")
    args = parser.parse_args()

    def run(model, f):
        camera_poses(model, load_sample(f, args.device)["x"], agg=torch.float16)

    collect_hessians(load_model(args.device), GROUPS[args.group], calibration_files(args.count), run,
                     args.hessian_device, OUT_DIR, args.group)
    sys.stdout.flush()
    os._exit(0)  # skip ROCm teardown (it can segfault after all work is saved)


if __name__ == "__main__":
    main()
