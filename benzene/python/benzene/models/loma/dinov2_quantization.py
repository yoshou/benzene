"""DeDoDe-G's DINOv2 ViT-L/14 in the precision of a recipe (recipes.DINO_RECIPES):
  fp16  autocast fp16
  w8a8  every block Linear (attn.qkv, attn.proj, mlp.fc1, mlp.fc2) as the device's W8A8 GEMM path of the VGGT /
        Cosmos 3 Edge vision blocks: SmoothQuant 0.5 (s = amax^0.5 / wmax^0.5 per input channel; on the device folded
        into the norms / output stages), per-channel int8 weights by GPTQ on W s, per-token dynamic int8 activations
        with fp16 scales (benzene.quantization.ActQuantLinear "tok"); the rest (patch embedding, attention, norms,
        GELU) in fp16 under autocast
Calibration: per Linear the Hessian X^T X / n of its input and the per-channel |x| maximum, over both images of
every calibration pair at the descriptor's input size of the settings -> OUT_DIR/calib/dinov2-<settings>.pt; the
quantized weights are cached under OUT_DIR/qweights/dinov2-<settings>/.

  python -m benzene.models.loma.dinov2_quantization [--settings det...-desc...-k...-rgb]
"""

from __future__ import annotations

import argparse
import copy
import time

import torch

from benzene.quantization import FORMATS, ActQuantLinear, gptq, smooth_vector

from ..lightglue.evaluation import calibration_files
from ..lightglue.matcher import autocast
from .paths import OUT_DIR

LINEARS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
SMOOTH_ALPHA = 0.5


def calibration_path(settings):
    return OUT_DIR / "calib" / f"dinov2-{settings.name}.pt"


def linear_names(dinov2) -> list[str]:
    return [f"blocks.{i}.{part}" for i in range(len(dinov2.blocks)) for part in LINEARS]


@torch.inference_mode()
def calibrate(settings, count: int | None, device: str) -> None:
    from .runner import descriptor_shape, image_rgb, load_model, load_pair, network_input

    net = load_model(device=device)
    dinov2 = net._descriptor.encoder.frozen_dinov2.dinov2_vitl14
    stats: dict[str, dict] = {}

    def observe(name):
        def hook(module, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            s = stats.setdefault(name, {"H": torch.zeros(x.shape[1], x.shape[1], device=device), "n": 0,
                                        "amax": torch.zeros(x.shape[1], device=device)})
            s["H"].addmm_(x.T, x)
            s["n"] += x.shape[0]
            s["amax"] = torch.maximum(s["amax"], x.abs().amax(0))
        return hook

    handles = [dinov2.get_submodule(name).register_forward_pre_hook(observe(name)) for name in linear_names(dinov2)]
    t0 = time.time()
    records = [load_pair(f) for f in calibration_files(count)]
    for rec in records:
        for i in (0, 1):
            rgb = image_rgb(rec, i, settings)
            dinov2(network_input(rgb, descriptor_shape(rgb.shape[1], rgb.shape[0], settings.descriptor_size), device))
    for handle in handles:
        handle.remove()
    calibration_path(settings).parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: {"H": (s["H"] / s["n"]).cpu(), "amax": s["amax"].cpu(), "n": s["n"]} for name, s in stats.items()},
               calibration_path(settings))
    print(f"dinov2 calibration: {len(stats)} linears, {2 * len(records)} images, {time.time() - t0:.0f}s", flush=True)


def quantized_weight(name: str, linear: torch.nn.Linear, stats: dict, cache_dir) -> tuple[torch.Tensor, torch.Tensor]:
    """(dequantized int8 weight of W s, smoothing vector s) of a block Linear; cached."""
    path = cache_dir / f"{name}.pt"
    if path.exists():
        record = torch.load(path, map_location=linear.weight.device)
        return record["w"].float(), record["smooth"].float()
    w = linear.weight.detach().float()
    smooth = smooth_vector(stats["amax"], w, SMOOTH_ALPHA)
    H = stats["H"].float().to(w.device) / smooth[:, None] / smooth[None, :]
    q, scales = gptq(w * smooth[None, :], H, FORMATS["int8_pc"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"w": q.cpu(), "scales": scales.cpu(), "smooth": smooth.cpu()}, path)
    return q, smooth


def weight_record(dinov2, name: str, settings, device: str) -> dict:
    """The cached W8A8 record {"w", "scales", "smooth"} of block Linear `name` (quantized now if not cached)."""
    cache_dir = OUT_DIR / "qweights" / f"dinov2-{settings.name}"
    path = cache_dir / f"{name}.pt"
    if not path.exists():
        stats = torch.load(calibration_path(settings), map_location=device)[name]
        linear = dinov2.get_submodule(name)
        quantized_weight(name, linear, stats, cache_dir)
    return torch.load(path, map_location="cpu")


class Fp16Dinov2(torch.nn.Module):
    """A DINOv2 module run under autocast fp16, its output fp32."""

    def __init__(self, dinov2) -> None:
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(torch.float16):
            return self.dinov2(x).float()


def quantized_dinov2(dinov2, recipe: str, settings, device: str, verbose: bool = True) -> torch.nn.Module:
    if recipe == "fp16":
        return Fp16Dinov2(dinov2)
    assert recipe == "w8a8", recipe
    stats = torch.load(calibration_path(settings), map_location=device)
    model = copy.deepcopy(dinov2)
    t0 = time.time()
    for name in linear_names(model):
        linear = model.get_submodule(name)
        q, smooth = quantized_weight(name, linear, stats[name], OUT_DIR / "qweights" / f"dinov2-{settings.name}")
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, ActQuantLinear(linear, q, smooth, mode="tok"))
    if verbose:
        print(f"  dinov2 w8a8: {len(LINEARS) * len(model.blocks)} linears ({time.time() - t0:.0f}s)", flush=True)
    return Fp16Dinov2(model)


def main() -> None:
    from .evaluate import SETTINGS, parse_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    calibrate(parse_settings(args.settings) if args.settings else SETTINGS, args.count, args.device)


if __name__ == "__main__":
    main()
