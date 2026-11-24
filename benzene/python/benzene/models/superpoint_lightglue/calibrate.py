"""Calibration statistics of the fp32 models over the calibration pairs (benzene.models.lightglue.prepare_data --split
calib).

SuperPoint (both images of every pair), per convolution input:
  amax   per input channel |x| max
  hist_c per input channel histogram of the non-zero values, NB_C bins over [0, amax_c]
  hist_t per tensor histogram of the non-zero values, NB_T bins over [0, max amax]
  H      Hessian X^T X / n of the unfolded receptive fields (K in PyTorch order (ci, ky, kx))
LightGlue, per Linear of the transformer layers and the final match assignment: H and the per-channel |x| max.
Two passes (maxima first, then histograms and Hessians); results in OUT_DIR/calib/{sp,lg}.pt.

  python -m benzene.models.superpoint_lightglue.calibrate
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from ..lightglue.evaluation import calibration_files, load_pair
from ..lightglue.matcher import match
from .paths import OUT_DIR
from .runner import extract, load_models

CALIBRATION_DIR = OUT_DIR / "calib"
NB_C, NB_T = 2048, 16384


def lg_linears(lg) -> dict[str, torch.nn.Linear]:
    """The Linears a LightGlue recipe may quantize: transformer layers and the last layer's match assignment."""
    last = f"log_assignment.{lg.conf.n_layers - 1}."
    return {n: m for n, m in lg.named_modules() if isinstance(m, torch.nn.Linear)
            and (n.startswith("transformers.") or n.startswith(last))}


def sp_convs(sp) -> dict[str, torch.nn.Conv2d]:
    return {n: m for n, m in sp.named_modules() if isinstance(m, torch.nn.Conv2d)}


def load_calibration(device="cpu") -> tuple[dict, dict]:
    return (torch.load(CALIBRATION_DIR / "sp.pt", map_location=device),
            torch.load(CALIBRATION_DIR / "lg.pt", map_location=device))


def _hist(x: torch.Tensor, top: torch.Tensor, nb: int) -> torch.Tensor:
    """Histogram per row of x [R, L] (non-zero values only) with nb bins over [0, top[r]]."""
    R = x.shape[0]
    idx = (x / top[:, None].clamp_min(1e-30) * nb).long().clamp_(0, nb - 1)
    idx += torch.arange(R, device=x.device)[:, None] * nb
    w = (x > 0).float()
    return torch.zeros(R * nb, device=x.device).scatter_add_(0, idx.flatten(), w.flatten()).reshape(R, nb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dev = args.device
    sp, lg = load_models(dev)
    files = calibration_files(args.count)
    records = [load_pair(f) for f in files]
    convs, lins = sp_convs(sp), lg_linears(lg)
    statistics = {n: {"amax": torch.zeros(m.in_channels, device=dev)} for n, m in convs.items()}
    phase = [1]

    def sp_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()[0]  # [C, H, W]
            C = x.shape[0]
            xr = x.reshape(C, -1)
            s = statistics[name]
            if phase[0] == 1:
                s["amax"] = torch.maximum(s["amax"], xr.amax(1))
                return
            s["hist_c"] += _hist(xr, s["amax"], NB_C)
            s["hist_t"] += _hist(xr.reshape(1, -1), s["amax"].max().reshape(1), NB_T)[0]
            u = F.unfold(x[None], mod.kernel_size, mod.dilation, mod.padding, mod.stride)[0]  # [K, L]
            s["H"].addmm_(u, u.T)
            s["n"] += u.shape[1]
        return fn

    lst = {n: {"H": torch.zeros(m.in_features, m.in_features, device=dev), "n": 0,
               "amax": torch.zeros(m.in_features, device=dev)} for n, m in lins.items()}

    def lg_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            s = lst[name]
            s["H"].addmm_(x.T, x)
            s["n"] += x.shape[0]
            s["amax"] = torch.maximum(s["amax"], x.abs().amax(0))
        return fn

    for n, m in convs.items():
        m.register_forward_hook(sp_hook(n))
    t0 = time.time()
    for r in records:  # pass 1: maxima
        extract(sp, r["image0"], dev), extract(sp, r["image1"], dev)
    print(f"pass 1: {2 * len(records)} images {time.time() - t0:.0f}s", flush=True)
    for n, m in convs.items():
        k = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        statistics[n].update(hist_c=torch.zeros(m.in_channels, NB_C, device=dev), hist_t=torch.zeros(NB_T, device=dev),
                     H=torch.zeros(k, k, device=dev), n=0)
    for n, m in lins.items():
        m.register_forward_hook(lg_hook(n))
    phase[0] = 2
    for i, r in enumerate(records):  # pass 2: histograms, Hessians; LightGlue on the pair
        f0, f1 = extract(sp, r["image0"], dev), extract(sp, r["image1"], dev)
        match(lg, f0, f1, (r["image0"].shape[1], r["image0"].shape[0]), (r["image1"].shape[1], r["image1"].shape[0]))
        if i % 16 == 0:
            print(f"  {time.strftime('%H:%M:%S')} pass 2: {i + 1}/{len(records)} pairs {time.time() - t0:.0f}s",
                  flush=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    sp_out = {n: {"amax": s["amax"].cpu(), "hist_c": s["hist_c"].cpu(), "hist_t": s["hist_t"].cpu(),
                  "H": (s["H"] / s["n"]).cpu(), "n": s["n"]} for n, s in statistics.items()}
    lg_out = {n: {"amax": s["amax"].cpu(), "H": (s["H"] / s["n"]).cpu(), "n": s["n"]} for n, s in lst.items()}
    torch.save(sp_out, CALIBRATION_DIR / "sp.pt")
    torch.save(lg_out, CALIBRATION_DIR / "lg.pt")
    print(f"done: {len(sp_out)} convolutions, {len(lg_out)} Linears, {len(records)} pairs, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
