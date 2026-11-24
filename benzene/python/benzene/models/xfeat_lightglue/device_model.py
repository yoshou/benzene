"""XFeat as the device computes it (the recipe "dev"): integer convolutions with fp32 requantization, every step in the
device's arithmetic, so the device's outputs can be checked against it bit for bit.

Activations between convolutions are uint8 codes per channel scale: of ReLU outputs with zero point 0, of the two
signed convolution inputs (block2.0: block1's output plus the skip branch; heatmap.0: the descriptor map) with zero
point 128 (padding = 128). A convolution sums its int8 weight codes times (code - zero point) exactly (the per-channel
input scales are folded into the weights), then
  uint8 output  q = clamp(trunc(fp32(fp32(acc) * m) + b), 0, 255)      (m = s_w / s_y, b = bias / s_y + 0.5, s_y the
                                                                       consumer's input scale; ReLU by the clamp)
  fp32 output   y = fp32(fp32(acc) * scale) + bias                     (the 1 x 1 heads, the descriptor map)
Every fp32 operation is rounded on its own (the device never fuses a multiply and an add). The stages that are not one
convolution:
  normalization   XFeat's InstanceNorm of the raw gray image x: n = H W, S1 = sum x, S2 = sum x^2 (exact integers),
                  mean = f32(S1 / n), inv_std = f32(1 / sqrt((n S2 - S1^2) / n^2 + 1e-5)) in fp64. block1.0 and
                  keypoint.0 take the raw pixels (codes = x, zero point 0) and fold the normalization into their
                  requantization per image: m = f32(k inv_std), b = f32(beta - f32(f32(mean S) m)) with k = s_w / s_y,
                  beta = bias / s_y + 0.5 and S the sum of the weights of the taps inside the image - for block1.0's 3x3
                  zero padding one of 9 border classes (top / middle / bottom row x left / middle / right column),
                  for keypoint.0's 1x1 convolution over the 8 x 8 cells (unfold2d) all 64
  block1.3 + skip block1.3's fp32 output x1 = max(fp32(fp32(acc) scale) + bias, 0) plus the skip branch
                  skip = f32(f32(sum16 u) + v) (sum16: the 4 x 4 block sum of x, the 1 x 1 convolution of the average-
                  pooled normalized image with u = f32(Ws / 16 inv_std), v = f32(bs - f32(f32(Ws mean) inv_std))),
                  t = f32(x1 + skip), quantized for block2.0: q = clamp(trunc(f32(f32(t r) + 128.5)), 0, 255), r = 1 / s_y
  fusion          x3 + up2(x4) + up4(x5) from the codes of x3 (block3.2's output at block4.0's input scale), x4 (block4.2's,
                  at block5.0's), x5 (block5.3's, at its own): up = bilinear (align_corners False) of the codes, exact in
                  fp32 (dyadic weights); q = clamp(trunc(f32(f32(f32(f32(x3 k3) + f32(u4 k4)) + f32(u5 k5)) + 0.5)),
                  0, 255) with k = s / s_fusion
  heatmap input   the descriptor map (fusion.2, fp32) quantized: q = clamp(trunc(f32(f32(feats r) + 128.5)), 0, 255)
  reliability     sigmoid of heatmap.2's fp32 output
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ZERO_POINT = 128  # of the signed inputs
LAYERS = ("block1.0", "block1.1", "block1.2", "block1.3", "block2.0", "block2.1", "block3.0", "block3.1", "block3.2",
          "block4.0", "block4.1", "block4.2", "block5.0", "block5.1", "block5.2", "block5.3", "fusion.0", "fusion.1",
          "fusion.2", "heatmap.0", "heatmap.1", "heatmap.2", "keypoint.0", "keypoint.1", "keypoint.2", "keypoint.3")
SIGNED_INPUTS = ("block2.0", "heatmap.0")
STRIDE2 = ("block1.1", "block1.3", "block3.0", "block4.0", "block5.0")


def image_statistics(x: torch.Tensor) -> tuple[float, float]:
    """(mean, inv_std) fp32 values of the uint8 image [H, W] (the device's normalization)."""
    n = x.numel()
    values = x.to(torch.int64)
    s1 = int(values.sum())
    s2 = int((values * values).sum())
    variance = float(n * s2 - s1 * s1) / float(n * n)
    mean = float(torch.tensor(float(s1) / float(n), dtype=torch.float32))
    inv_std = float(torch.tensor(1.0 / math.sqrt(variance + 1e-5), dtype=torch.float32))
    return mean, inv_std


def border_classes(height: int, width: int, device) -> torch.Tensor:
    """[H, W] class of a 3x3 / pad 1 output pixel: 3 row class (0 top, 1 middle, 2 bottom) + column class."""
    rows = torch.ones(height, dtype=torch.long, device=device)
    rows[0], rows[-1] = 0, 2
    columns = torch.ones(width, dtype=torch.long, device=device)
    columns[0], columns[-1] = 0, 2
    return rows[:, None] * 3 + columns[None, :]


def border_tap_sums(codes: torch.Tensor) -> torch.Tensor:
    """[N, 9] sums of a 3x3 single-channel kernel's codes [N, 1, 3, 3] over the taps inside the image per border
    class (border_classes)."""
    w = codes.to(torch.int64)[:, 0]
    sums = torch.zeros(w.shape[0], 9, dtype=torch.int64)
    for row_class in range(3):
        for column_class in range(3):
            rows = [ty for ty in range(3) if not (row_class == 0 and ty == 0) and not (row_class == 2 and ty == 2)]
            columns = [tx for tx in range(3) if not (column_class == 0 and tx == 0) and not (column_class == 2 and tx == 2)]
            sums[:, 3 * row_class + column_class] = w[:, rows][:, :, columns].sum((1, 2))
    return sums


def f32(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


class DeviceXFeat(torch.nn.Module):
    """forward(x [1, 1, H, W], the raw gray values) -> (feats [1, 64, h, w], logits [1, 65, h, w], reliability
    [1, 1, h, w]) fp32 like XFeatModel.forward (h, w = H / 8, W / 8), from the device records (recipes.quantize_dev,
    or exported_model.records read back from a model file)."""

    def __init__(self, records: dict, device: str):
        super().__init__()
        self.records = {name: {k: (torch.as_tensor(v).to(device) if not isinstance(v, (int, float)) else v)
                               for k, v in record.items()} for name, record in records.items()}
        self.device = device

    def convolution(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        """The exact integer sums [N, h, w] of layer `name` over input codes [C, H, W] (int, zero point removed)."""
        record = self.records[name]
        w = record["codes"]
        stride = 2 if name in STRIDE2 else 1
        padding = w.shape[-1] // 2
        acc = F.conv2d(codes.double()[None], w.double(), None, stride, padding)[0]
        return acc.round().float()

    def uint8_output(self, name: str, acc: torch.Tensor) -> torch.Tensor:
        record = self.records[name]
        v = acc * record["m"][:, None, None]
        v = v + record["b"][:, None, None]
        return torch.trunc(v).clamp(0, 255)

    def f32_output(self, name: str, acc: torch.Tensor) -> torch.Tensor:
        record = self.records[name]
        return acc * record["scale"][:, None, None] + record["bias"][:, None, None]

    def normalized_input(self, name: str, acc: torch.Tensor, mean: float, inv_std: float,
                         classes: torch.Tensor | None) -> torch.Tensor:
        """uint8 output of a convolution over raw pixels with the normalization folded in (block1.0, keypoint.0)."""
        record = self.records[name]
        m = record["k"] * f32(inv_std)
        sums = record["tap_sums"].float()  # [N, classes]
        b = record["beta"][:, None] - (f32(mean) * sums) * m[:, None]
        bias = b[:, classes] if classes is not None else b[:, 0, None, None]
        v = acc * m[:, None, None]
        v = v + bias
        return torch.trunc(v).clamp(0, 255)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        r = self.records
        image = x[0, 0].round().to(torch.uint8)
        height, width = image.shape
        mean, inv_std = image_statistics(image.cpu())
        pixels = image.float()[None]  # [1, H, W] codes
        # block1
        acc = self.convolution("block1.0", pixels)
        q = self.normalized_input("block1.0", acc, mean, inv_std, border_classes(height, width, pixels.device))
        q = self.uint8_output("block1.1", self.convolution("block1.1", q))
        q = self.uint8_output("block1.2", self.convolution("block1.2", q))
        rb = r["block1.3"]
        x1 = self.f32_output("block1.3", self.convolution("block1.3", q)).clamp_min(0)
        sum16 = F.avg_pool2d(pixels.double()[None], 4, 4)[0].mul(16).round().float()  # [1, H/4, W/4]
        u = rb["skip_weight16"] * f32(inv_std)
        v = rb["skip_bias"] - (rb["skip_weight"] * f32(mean)) * f32(inv_std)
        skip = sum16 * u[:, None, None] + v[:, None, None]
        t = x1 + skip
        q = torch.trunc(t * rb["inverse_scale"][:, None, None] + 128.5).clamp(0, 255)
        # block2 .. block5
        q = self.uint8_output("block2.0", self.convolution("block2.0", q - ZERO_POINT))
        for name in ("block2.1", "block3.0", "block3.1", "block3.2"):
            q = self.uint8_output(name, self.convolution(name, q))
        x3 = q
        for name in ("block4.0", "block4.1", "block4.2"):
            q = self.uint8_output(name, self.convolution(name, q))
        x4 = q
        for name in ("block5.0", "block5.1", "block5.2", "block5.3"):
            q = self.uint8_output(name, self.convolution(name, q))
        x5 = q
        # fusion
        size = x3.shape[-2:]
        u4 = F.interpolate(x4[None], size, mode="bilinear", align_corners=False)[0]
        u5 = F.interpolate(x5[None], size, mode="bilinear", align_corners=False)[0]
        rf = r["fusion"]
        fused = x3 * rf["k3"][:, None, None] + u4 * rf["k4"][:, None, None]
        fused = fused + u5 * rf["k5"][:, None, None]
        q = torch.trunc(fused + 0.5).clamp(0, 255)
        q = self.uint8_output("fusion.0", self.convolution("fusion.0", q))
        q = self.uint8_output("fusion.1", self.convolution("fusion.1", q))
        feats = self.f32_output("fusion.2", self.convolution("fusion.2", q))
        # heads
        q = torch.trunc(feats * r["fusion.2"]["inverse_scale"][:, None, None] + 128.5).clamp(0, 255)
        q = self.uint8_output("heatmap.0", self.convolution("heatmap.0", q - ZERO_POINT))
        q = self.uint8_output("heatmap.1", self.convolution("heatmap.1", q))
        reliability = torch.sigmoid(self.f32_output("heatmap.2", self.convolution("heatmap.2", q)))
        cells = F.pixel_unshuffle(pixels[None], 8)[0]  # [64, H/8, W/8], channel ky * 8 + kx (unfold2d's order)
        q = self.normalized_input("keypoint.0", self.convolution("keypoint.0", cells), mean, inv_std, None)
        q = self.uint8_output("keypoint.1", self.convolution("keypoint.1", q))
        q = self.uint8_output("keypoint.2", self.convolution("keypoint.2", q))
        logits = self.f32_output("keypoint.3", self.convolution("keypoint.3", q))
        return feats[None], logits[None], reliability[None]
