"""Calibration statistics of the fp32 ALIKED over the calibration pairs (benzene.models.lightglue.prepare_data --split
calib, cropped by runner.load_pair; both images of every pair, the descriptor head at their keypoints), through
hooked_network.HookedAliked's hooks.

Per quantization point (network.ACTIVATIONS):
  min, max  per channel
  hist      per channel histogram of the values, NB bins over [min, max]
Per convolution (network.CONVOLUTIONS):
  H         Hessian X^T X / n of its GEMM input (the unfolded 3x3 receptive fields in PyTorch's K order (c, ky, kx),
            the deformable layers' sampled columns, the descriptor head's patches / aggregation inputs)
Two passes (ranges first, then histograms and Hessians); results in OUT_DIR/calib/aliked.pt. LightGlue stays fp16 on
the device: no statistics.

  python -m benzene.models.aliked_lightglue.calibrate
"""

from __future__ import annotations

import argparse
import time

import torch

from ..lightglue.evaluation import calibration_files
from .hooked_network import ACTIVATIONS, CONVOLUTIONS, HookedAliked, unfold
from .paths import OUT_DIR
from .runner import extract, load_models, load_pair

CALIBRATION_DIR = OUT_DIR / "calib"
NB = 2048


def load_calibration(device: str = "cpu") -> dict:
    return torch.load(CALIBRATION_DIR / "aliked.pt", map_location=device)


def channel_rows(name: str, x: torch.Tensor) -> torch.Tensor:
    """An activation as [channels, values]."""
    if name == "desc.patch":  # [N, C, 3, 3]
        return x.transpose(0, 1).reshape(x.shape[1], -1)
    return x.reshape(x.shape[0], -1)


def gemm_input(name: str, x: torch.Tensor) -> torch.Tensor:
    """A convolution's GEMM input [K, n] from what the convolution hook receives."""
    kind = CONVOLUTIONS[name][1]
    if kind == "deformable":  # (the hook receives the sampled columns)
        return x
    if kind in ("3x3",):
        return unfold(kind, x)
    return x.reshape(x.shape[0], -1)  # 1x1, patch, aggregation: [K, n, 1]


def _histogram(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, nb: int) -> torch.Tensor:
    """Histogram per row of x [R, L] with nb bins over [low[r], high[r]]."""
    R = x.shape[0]
    span = (high - low).clamp_min(1e-30)
    index = ((x - low[:, None]) / span[:, None] * nb).long().clamp_(0, nb - 1)
    index += torch.arange(R, device=x.device)[:, None] * nb
    return torch.zeros(R * nb, device=x.device).scatter_add_(0, index.flatten(), torch.ones_like(x).flatten()).reshape(
        R, nb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device
    net, _ = load_models(device)
    records = [load_pair(f) for f in calibration_files(args.count)]
    stats = {name: {} for name in ACTIVATIONS}
    hessians: dict[str, dict] = {}
    phase = [1]

    def activation(name: str, x: torch.Tensor) -> torch.Tensor:
        rows = channel_rows(name, x.detach().float())
        s = stats[name]
        if phase[0] == 1:
            low, high = rows.amin(1), rows.amax(1)
            s["min"] = low if "min" not in s else torch.minimum(s["min"], low)
            s["max"] = high if "max" not in s else torch.maximum(s["max"], high)
        else:
            s["hist"] += _histogram(rows, s["min"], s["max"], NB)
        return x

    def convolution(name: str, x: torch.Tensor, weights) -> None:
        if phase[0] == 2:
            u = gemm_input(name, x.detach().float())
            h = hessians.setdefault(name, {"H": torch.zeros(u.shape[0], u.shape[0], device=device), "n": 0})
            h["H"].addmm_(u, u.T)
            h["n"] += u.shape[1]
        return None

    wrapped = HookedAliked(net, activation=activation, convolution=convolution)
    t0 = time.time()
    for p in (1, 2):
        phase[0] = p
        if p == 2:
            for s in stats.values():
                s["hist"] = torch.zeros(len(s["min"]), NB, device=device)
        for rec in records:
            for i in (0, 1):
                extract(wrapped, rec[f"image{i}"], device)
        print(f"pass {p}: {2 * len(records)} images {time.time() - t0:.0f}s", flush=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = {"activations": {name: {k: v.cpu() for k, v in s.items()} for name, s in stats.items()},
           "hessians": {name: {"H": (h["H"] / h["n"]).cpu(), "n": h["n"]} for name, h in hessians.items()}}
    torch.save(out, CALIBRATION_DIR / "aliked.pt")
    print(f"done: {len(stats)} activations, {len(hessians)} convolutions ({len(records)} pairs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
