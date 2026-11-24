"""The device recipe's records ("dev", recipes.py): what device_model.DeviceLoMa computes with and the export writes
into the model file, from the fp32 networks and the calibration statistics of the settings (gray images: the device
input). Only the convolutional networks (hooked_network); DINOv2 and the matcher keep their own recipes.

records = {
  "settings":    the runner.Settings name
  "activations": {point: {"scale", "zero"}}  fp64 [C] per channel (benzene.quantization.activation_quantizer, clip
                 "mse", zero point per channel; signed points shrink both ends)
  "layers":      {convolution: {
                   "kind"     "gray" (the first convolution of a network on gray codes: one input channel, the RGB
                              weights summed), "3x3", "1x1", "dw" (5x5 depthwise)
                   "codes"    int8 weights [N, Cin, k, k] ([C, 1, 5, 5] depthwise) - the per-channel input scales
                              folded in
                   "bias2"    int64 [N]: -sum(w z) over the input zero points (depthwise: the initial sum)
                   "padding"  uint8 [Cin]: the padding codes (the input zero points)
                   "output"   "u8": codes of the output point, q = trunc(clamp(f32(f32(acc) m) + b, 0, 255.5)) (ReLU
                              points: zero point 0, the clamp is the ReLU); "f32": rows v = f32(f32(acc) S) + B
                   "m", "b" / "S", "B"  fp32 [N]
                 }}
}
The detector's image: gray codes g with the normalization folded into the first convolution - a[n, t] = sum_c W[n, c,
t] / (255 std_c) on u = g - z_image, its constant sum_t (a z_image - sum_c W mean_c / std_c) in the bias; padding
pixels read z_image, the integer that minimizes the border's error (the reference pads every normalized channel with
0: pixel values 255 mean_c, one per channel); the descriptor's: x = g / 255, padding 0 (exact). Depthwise weights by
RTN per channel, the others by GPTQ (the gray convolutions' Hessians from the calibration pairs' gray images).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from benzene.quantization import FORMATS, activation_quantizer, int8_weight_codes, rtn

from ..lightglue.evaluation import calibration_files
from .calibrate import load_calibration
from .hooked_network import HookedDescriptor, HookedDetector
from .network import IMAGENET_MEAN, IMAGENET_STD
from .recipes import input_points, signed
from .runner import descriptor_shape, detector_shape, image_rgb, load_pair, network_codes


def output_point(name: str) -> str | None:
    """The activation point convolution `name` writes (None: fp32 rows)."""
    if name.endswith(".out"):
        return None
    if ".enc." in name:
        return name
    if name.endswith("block1.conv1"):
        return name[: -len("conv1")] + "mid"
    if name.endswith("block1.conv2"):
        return name[: -len("block1.conv2")] + "x0"
    if name.endswith(".dw"):
        return name[: -len("dw")] + "mid"
    return name[: -len("pw")] + "out"


def gray_hessian(settings, network: str, z_image: int, device: str) -> torch.Tensor:
    """H [9, 9] of the first convolution's gray input over the calibration pairs: u = g - z_image (detector, padding
    z_image: u = 0) or x = g / 255 (descriptor, padding 0)."""
    H = torch.zeros(9, 9, dtype=torch.float64, device=device)
    n = 0
    for f in calibration_files():
        rec = load_pair(f)
        for i in (0, 1):
            rgb = image_rgb(rec, i, settings)
            height, width = rgb.shape[:2]
            shape = (detector_shape(width, height, settings.detector_size) if network == "det"
                     else descriptor_shape(width, height, settings.descriptor_size))
            g = network_codes(rgb, shape, device)[None, None].double()
            x = (g - z_image) if network == "det" else g / 255.0
            u = F.unfold(x, 3, padding=1)[0]
            H += u @ u.T
            n += u.shape[1]
    return (H / n).float().cpu()


def detector_image_zero(weight: torch.Tensor) -> int:
    """The padding code z of the detector's gray image minimizing sum over n, t of (sum_c W[n, c, t] (mean_c - z / 255)
    / std_c)^2 (the error of a padding tap against the reference's zero-padded normalized channels)."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float64)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float64)
    W = weight.double().cpu()  # [N, 3, 3, 3]
    errors = [float(((W * ((mean - z / 255.0) / std)[None, :, None, None]).sum(1) ** 2).sum()) for z in range(256)]
    return int(np.argmin(errors))


DEAD_SCALE = 1e-9  # activation_quantizer's scale of a channel that was 0 in all calibration data


def output_constants(record: dict, S: torch.Tensor, B: torch.Tensor, point: str | None, quantizers: dict) -> None:
    """The output constants of a layer with weight scales S and biases B (fp64 [N]); a dead output channel (scale
    below DEAD_SCALE: m would be ~1e12) writes its zero point, the value 0, as the fake quantization's codes do."""
    if point is None:
        record.update(output="f32", S=S.float(), B=B.float())
        return
    scale, zero = quantizers[point]
    dead = scale < DEAD_SCALE
    m = torch.where(dead, torch.zeros_like(S), S / scale)
    b = torch.where(dead, zero + 0.5, B / scale + zero + 0.5)
    record.update(output="u8", m=m.float(), b=b.float(), dead=int(dead.sum()))


def network_records(part, calibration: dict, settings, device: str, verbose: bool) -> tuple[dict, dict]:
    quantizers = {name: activation_quantizer(stats, "mse", "channel", signed(name))
                  for name, stats in calibration["activations"].items() if name.startswith(part.prefix)}
    layers = {}
    for name, layer in part.convolutions.items():
        weight = layer.weight.double().cpu()
        bias = layer.bias.double().cpu()
        points = input_points(name, part)
        record: dict = {}
        if not points:  # the gray image
            if part.prefix == "det":
                z = detector_image_zero(weight)
                mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float64)
                std = torch.tensor(IMAGENET_STD, dtype=torch.float64)
                a = (weight / (255.0 * std)[None, :, None, None]).sum(1).reshape(weight.shape[0], 9)
                constant = bias + (a * z - (weight * (mean / std)[None, :, None, None]).sum(1).reshape(-1, 9)).sum(1)
                s_in = torch.ones(9, dtype=torch.float64)
            else:
                z = 0
                a = weight.sum(1).reshape(weight.shape[0], 9)
                constant = bias
                s_in = torch.full((9,), 1.0 / 255, dtype=torch.float64)
            H = gray_hessian(settings, part.prefix, z, device)
            codes, s_w = int8_weight_codes(a, s_in, H, "gptq", device)
            record.update(kind="gray", codes=codes.to(torch.int8).reshape(-1, 1, 3, 3),
                          padding=torch.tensor([z], dtype=torch.uint8),
                          bias2=(-z * codes.sum(1)).to(torch.int64))
            output_constants(record, s_w, constant, output_point(name), quantizers)
        else:
            s_in = torch.cat([quantizers[p][0] for p in points])
            z_in = torch.cat([quantizers[p][1] for p in points])
            if layer.groups > 1:  # depthwise 5x5, RTN per channel
                q, s_w = rtn((weight.reshape(weight.shape[0], 25) * s_in[:, None]).float(), FORMATS["int8_pc"])
                s_w = s_w.double().reshape(-1)
                codes = torch.where(s_w[:, None] > 0, (q.double() / torch.where(s_w > 0, s_w, 1)[:, None]).round(), 0)
                s_w = torch.where(s_w > 0, s_w, torch.ones_like(s_w))
                record.update(kind="dw", codes=codes.to(torch.int8).reshape(-1, 1, 5, 5),
                              bias2=(-z_in * codes.sum(1)).round().to(torch.int64))
            else:
                taps = weight.shape[-1] ** 2
                codes, s_w = int8_weight_codes(weight.reshape(weight.shape[0], -1), s_in.repeat_interleave(taps),
                                               calibration["hessians"][name]["H"], "gptq", device)
                record.update(kind="3x3" if taps == 9 else "1x1", codes=codes.to(torch.int8).reshape(weight.shape),
                              bias2=(-(codes * z_in.repeat_interleave(taps)[None]).sum(1)).round().to(torch.int64))
            record["padding"] = z_in.round().to(torch.uint8)
            output_constants(record, s_w, bias, output_point(name), quantizers)
        assert record["codes"].abs().max() <= 127
        layers[name] = record
        if verbose:
            print(f"  {name}: {record['kind']} {tuple(record['codes'].shape)} -> {record['output']}", flush=True)
    activations = {name: {"scale": s, "zero": z} for name, (s, z) in quantizers.items()}
    return activations, layers


def quantize_dev(net, settings, device: str, verbose: bool = True) -> dict:
    assert settings.color == "gray", "the device reads gray images"
    calibration = load_calibration(settings)
    records = {"settings": settings.name, "activations": {}, "layers": {}}
    for part in (HookedDetector(net._detector), HookedDescriptor(net._descriptor)):
        activations, layers = network_records(part, calibration, settings, device, verbose)
        records["activations"].update(activations)
        records["layers"].update(layers)
    return records
