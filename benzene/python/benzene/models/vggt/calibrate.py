"""Calibration statistics for VGGT: per-layer Hessians H = X^T X / n of the GEMM inputs (convolutions: the unfolded
receptive fields, convolution_quantization.gemm_rows) and the per-input-channel |x| max (SmoothQuant).

The model runs on --device (aggregator under fp16 autocast, heads in fp32 as the official forward); the statistics
accumulate in fp32 on --hessian-device and go to OUT_DIR/hessians/<module>.pt.

  python -m benzene.models.vggt.calibrate --group tf      # DINOv2 blocks, frame / global blocks, patch embedding
  python -m benzene.models.vggt.calibrate --group heads   # camera head, DPT heads
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import torch

from .convolution_quantization import channel_amax, gemm_rows, weight_matrix
from .paths import DATA_DIR, OUT_DIR
from .recipes import CAM, DPT, PATCH, QUANTIZABLE, TF
from .runner import load_model, load_sample, predict

HESSIAN_DIR = OUT_DIR / "hessians"
GROUPS = {"tf": TF + [PATCH], "heads": [CAM, DPT]}


def calibration_files(count: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache/calib/*.pt")))
    return files[:count] if count else files


def load_hessian(name: str, device) -> dict:
    return torch.load(HESSIAN_DIR / f"{name}.pt", map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS), required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hessian-device", default="cuda:1")
    args = parser.parse_args()
    model = load_model(args.device)
    pats = [re.compile(p) for p in GROUPS[args.group]]
    HESSIAN_DIR.mkdir(parents=True, exist_ok=True)
    acc: dict[str, dict] = {}

    def hook(name):
        def fn(mod, inp, out):
            a = acc[name]
            for r in gemm_rows(mod, inp[0]):
                r = r.to(args.hessian_device)
                a["H"].addmm_(r.T, r)
                a["n"] += r.shape[0]
            a["amax"] = torch.maximum(a["amax"], channel_amax(mod, inp[0]).float().to(args.hessian_device))
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, QUANTIZABLE) and any(p.search(name) for p in pats):
            k = weight_matrix(mod).shape[1]
            cin = mod.in_features if isinstance(mod, torch.nn.Linear) else mod.in_channels
            acc[name] = {"H": torch.zeros(k, k, device=args.hessian_device), "n": 0,
                         "amax": torch.zeros(cin, device=args.hessian_device)}
            mod.register_forward_hook(hook(name))
    gib = sum(a["H"].numel() * 4 for a in acc.values()) / 2**30
    print(f"{args.group}: {len(acc)} layers, {gib:.1f} GiB of Hessians on {args.hessian_device}", flush=True)
    t0 = time.time()
    files = calibration_files(args.count)
    for i, f in enumerate(files):
        s = load_sample(f, args.device)
        predict(model, s["x"], agg=torch.float16, head=None)
        if i % 8 == 0:
            print(f"  {time.strftime('%H:%M:%S')} {i + 1}/{len(files)} samples {time.time() - t0:.0f}s", flush=True)
    for name, a in acc.items():
        torch.save({"H": (a["H"] / a["n"]).cpu(), "n": a["n"], "amax": a["amax"].cpu()}, HESSIAN_DIR / f"{name}.pt")
    (HESSIAN_DIR / f".done_{args.group}").write_text(str(len(files)))
    print(f"saved {len(acc)} to {HESSIAN_DIR} ({time.time() - t0:.0f}s)", flush=True)
    sys.stdout.flush()
    os._exit(0)  # skip ROCm teardown (it can segfault after all work is saved)


if __name__ == "__main__":
    main()
