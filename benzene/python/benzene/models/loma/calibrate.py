"""Calibration statistics of the fp32 LoMa networks over the calibration pairs (benzene.models.lightglue.prepare_data
--split calib with their colour frames; both images of every pair at the evaluation's settings, evaluate.SETTINGS),
through the hooks of hooked_network.HookedDetector / HookedDescriptor.

Per quantization point (every activation hook name):
  min, max  per channel
  hist      per channel histogram of the values, NB bins over [min, max]
Per convolution:
  H         Hessian X^T X / n of its GEMM input: the unfolded 3x3 receptive fields in PyTorch's K order (c, ky, kx),
            the input channels of a 1x1 convolution, per channel the 25 taps of a depthwise 5x5 one ([C, 25, 25])
Two passes (ranges first, then histograms and Hessians); results in OUT_DIR/calib/<settings name>.pt. DINOv2 and the
matcher are not covered (their own recipes).

  python -m benzene.models.loma.calibrate [--settings det...-desc...-k...-rgb]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from ..lightglue.evaluation import calibration_files
from .hooked_network import HookedDescriptor, HookedDetector
from .paths import OUT_DIR
from .runner import descriptor_shape, detector_shape, image_rgb, load_model, load_pair, network_input

CALIBRATION_DIR = OUT_DIR / "calib"
NB = 2048


def calibration_path(settings):
    return CALIBRATION_DIR / f"{settings.name}.pt"


def load_calibration(settings, device: str = "cpu") -> dict:
    return torch.load(calibration_path(settings), map_location=device)


def gemm_input(x: torch.Tensor, layer) -> torch.Tensor:
    """A convolution's GEMM input from its input map [1, C, H, W]: [K, n] (3x3, 1x1) or [C, 25, n] (depthwise)."""
    kernel = layer.weight.shape[-1]
    if layer.groups > 1:
        return F.unfold(x, kernel, padding=kernel // 2)[0].reshape(x.shape[1], kernel * kernel, -1)
    if kernel > 1:
        return F.unfold(x, kernel, padding=kernel // 2)[0]
    return x[0].reshape(x.shape[1], -1)


def histogram(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, nb: int) -> torch.Tensor:
    """Histogram per row of x [R, L] with nb bins over [low[r], high[r]]."""
    R = x.shape[0]
    span = (high - low).clamp_min(1e-30)
    index = ((x - low[:, None]) / span[:, None] * nb).long().clamp_(0, nb - 1)
    index += torch.arange(R, device=x.device)[:, None] * nb
    return torch.zeros(R * nb, device=x.device).scatter_add_(0, index.flatten(), torch.ones_like(x).flatten()).reshape(
        R, nb)


def main() -> None:
    from .evaluate import SETTINGS, parse_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device
    settings = parse_settings(args.settings) if args.settings else SETTINGS
    net = load_model(device=device)
    records = [load_pair(f) for f in calibration_files(args.count)]
    stats: dict[str, dict] = {}
    hessians: dict[str, dict] = {}
    phase = [1]

    def activation(name: str, x: torch.Tensor) -> torch.Tensor:
        rows = x.detach().float()[0].reshape(x.shape[1], -1)
        s = stats.setdefault(name, {})
        if phase[0] == 1:
            low, high = rows.amin(1), rows.amax(1)
            s["min"] = low if "min" not in s else torch.minimum(s["min"], low)
            s["max"] = high if "max" not in s else torch.maximum(s["max"], high)
        else:
            if "hist" not in s:
                s["hist"] = torch.zeros(len(s["min"]), NB, device=device)
            s["hist"] += histogram(rows, s["min"], s["max"], NB)
        return x

    def convolution(name: str, x: torch.Tensor, layer) -> None:
        if phase[0] == 2:
            u = gemm_input(x.detach().float(), layer)
            if u.dim() == 3:
                h = hessians.setdefault(name, {"H": torch.zeros(u.shape[0], u.shape[1], u.shape[1], device=device),
                                               "n": 0})
                h["H"] += torch.einsum("ckn,cln->ckl", u, u)
            else:
                h = hessians.setdefault(name, {"H": torch.zeros(u.shape[0], u.shape[0], device=device), "n": 0})
                h["H"].addmm_(u, u.T)
            h["n"] += u.shape[-1]
        return None

    detector = HookedDetector(net._detector, activation, convolution)
    descriptor = HookedDescriptor(net._descriptor, activation, convolution)
    t0 = time.time()
    for p in (1, 2):
        phase[0] = p
        for rec in records:
            for i in (0, 1):
                rgb = image_rgb(rec, i, settings)
                height, width = rgb.shape[:2]
                with torch.inference_mode():
                    detector(network_input(rgb, detector_shape(width, height, settings.detector_size), device))
                    descriptor(network_input(rgb, descriptor_shape(width, height, settings.descriptor_size), device))
        print(f"pass {p}: {2 * len(records)} images {time.time() - t0:.0f}s", flush=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = {"settings": settings.name,
           "activations": {name: {k: v.cpu() for k, v in s.items()} for name, s in stats.items()},
           "hessians": {name: {"H": (h["H"] / h["n"]).cpu(), "n": h["n"]} for name, h in hessians.items()}}
    torch.save(out, calibration_path(settings))
    print(f"done: {len(stats)} activations, {len(hessians)} convolutions ({len(records)} pairs, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
