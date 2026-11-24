"""Calibration statistics of the fp32 XFeat over the calibration pairs (benzene.models.lightglue.prepare_data --split
calib, cropped by runner.load_pair; both images of every pair).

Per convolution input (XFeat's Conv2d modules by name, e.g. block2.0.layer.0) and for x5 (block5's output, which
reaches the fusion only through the upsampling):
  amax   per channel |x| max
  signed whether the input had negative values (then the histograms are of |x|)
  hist_c per channel histogram of the non-zero |x|, NB_C bins over [0, amax_c]
  H      Hessian X^T X / n of the unfolded receptive fields (K in PyTorch order (ci, ky, kx)); not for x5
Two passes (maxima first, then histograms and Hessians); results in OUT_DIR/calib/xfeat.pt. LighterGlue stays fp16 on
the device: no statistics.

  python -m benzene.models.xfeat_lightglue.calibrate
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from ..lightglue.evaluation import calibration_files
from .paths import OUT_DIR
from .runner import load_models, load_pair

CALIBRATION_DIR = OUT_DIR / "calib"
NB_C = 2048
X5 = "x5"  # block5's output (the fusion's third summand)


def xfeat_convolutions(net) -> dict[str, torch.nn.Conv2d]:
    return {n: m for n, m in net.named_modules() if isinstance(m, torch.nn.Conv2d)}


def load_calibration(device="cpu") -> dict:
    return torch.load(CALIBRATION_DIR / "xfeat.pt", map_location=device)


def _histogram(x: torch.Tensor, top: torch.Tensor, nb: int) -> torch.Tensor:
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
    net, _ = load_models(dev)
    records = [load_pair(f) for f in calibration_files(args.count)]
    convs = xfeat_convolutions(net)
    channels = {n: m.in_channels for n, m in convs.items()} | {X5: net.block5[-1].layer[0].out_channels}
    stats = {n: {"amax": torch.zeros(c, device=dev), "signed": False} for n, c in channels.items()}
    phase = [1]

    def observe(name, x, module):
        x = x.detach().float()[0]  # [C, H, W]
        s = stats[name]
        xr = x.reshape(x.shape[0], -1)
        if phase[0] == 1:
            s["amax"] = torch.maximum(s["amax"], xr.abs().amax(1))
            s["signed"] = s["signed"] or bool((xr < 0).any())
            return
        s["hist_c"] += _histogram(xr.abs(), s["amax"], NB_C)
        if module is not None:
            u = F.unfold(x[None], module.kernel_size, module.dilation, module.padding, module.stride)[0]  # [K, L]
            s["H"].addmm_(u, u.T)
            s["n"] += u.shape[1]

    for n, m in convs.items():
        m.register_forward_hook(lambda mod, inp, out, n=n: observe(n, inp[0], mod))
    net.block5.register_forward_hook(lambda mod, inp, out: observe(X5, out, None))
    t0 = time.time()
    with torch.inference_mode():
        for r in records:  # pass 1: maxima
            for i in (0, 1):
                net(r[f"image{i}"].to(dev).float()[None, None])
        print(f"pass 1: {2 * len(records)} images {time.time() - t0:.0f}s", flush=True)
        for n, c in channels.items():
            stats[n]["hist_c"] = torch.zeros(c, NB_C, device=dev)
            if n in convs:
                m = convs[n]
                k = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                stats[n].update(H=torch.zeros(k, k, device=dev), n=0)
        phase[0] = 2
        for r in records:  # pass 2: histograms, Hessians
            for i in (0, 1):
                net(r[f"image{i}"].to(dev).float()[None, None])
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, s in stats.items():
        out[n] = {"amax": s["amax"].cpu(), "signed": s["signed"], "hist_c": s["hist_c"].cpu()}
        if "H" in s:
            out[n].update(H=(s["H"] / s["n"]).cpu(), n=s["n"])
    torch.save(out, CALIBRATION_DIR / "xfeat.pt")
    signed = [n for n, s in out.items() if s["signed"]]
    print(f"done: {len(out)} inputs ({len(records)} pairs, {time.time() - t0:.0f}s); signed: {', '.join(signed)}")


if __name__ == "__main__":
    main()
