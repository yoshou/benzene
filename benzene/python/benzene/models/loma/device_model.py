"""LoMa as the device computes it (the recipe "dev"): the convolutional networks' integer convolutions and fp32 steps
in the device's arithmetic (records: device_records.quantize_dev), so the device's keypoints and the descriptor
stages can be checked against it bit for bit (benzene/dsp/models/loma); DINOv2 (W8A8 GEMMs with fp16 attention) and
the matcher (fp16) are the recipes' fake-quantized / autocast models - their device versions are not bit-exact.

Activations between convolutions are uint8 codes per channel, x = s (q - z). A convolution sums its integer weight
codes times (q - z) exactly (padding pixels read z: x = 0; the gray image's first convolution reads the image codes
with z_image), then either the output point's codes q = trunc(clamp(f32(f32(acc) m) + b, 0, 255.5)) or fp32 rows
v = f32(f32(acc) S) + B. Every fp32 operation is rounded on its own (the device never fuses a multiply and an add).
  pools          2x2 max of codes
  refiner        block1 over [feature, context] (one K: both codes with their zero points), the hidden blocks
                 (depthwise 5x5, 1x1), out over [x, x0] (the residual's / 1.4 in its weights): fp32 rows, the first
                 `split` channels the logit / description deltas, the rest the context
  context        upsampled bilinearly to the next scale (below), quantized to that scale's context codes:
                 q = trunc(clamp(f32(f32(v R) + Zq), 0, 255.5)), R = f32(1 / s), Zq = f32(z + 0.5)
  bilinear       align_corners False, per output index i of n from N: scale = f32(N / n), src = max(f32(f32(f32(i +
                 0.5) scale) - 0.5), 0), i0 = floor(src), i1 = min(i0 + 1, N - 1), l1 = f32(src - i0), l0 = f32(1 -
                 l1); columns first: T = f32(f32(l0 x[i0]) + f32(l1 x[i1])), then rows the same on T
  bicubic        (the logits; align_corners False, A = -0.75) src = f32(f32(f32(i + 0.5) scale) - 0.5), i0 =
                 floor(src), t = f32(src - i0), PyTorch's four coefficients in fp32 from t, taps i0 - 1 .. i0 + 2
                 clamped to the map; columns first, the four products summed in tap order, then rows
  logits         L = bicubic(... bicubic(L8) + L4 ...) + L1 (each fp32 add on its own)
  keypoints      e = exp2_f32(f32(f32(L - max L) log2 e)) (vector_math::exp2); NMS: e equal to its 3x3
                 neighbourhood's maximum (pixels outside ignored), others 0; the `keypoints` largest values, ties in
                 raster order; the total: per row 32 lane sums (lane l: the row's pixels 32 v + l in order) reduced
                 by the rotate-and-add tree (lanes + lanes 16 up, 8, 4, 2, 1), the row totals added in float64 in row
                 order, rounded to fp32; score = f32(value / total); the position x_n = f32(f32(2 x + 1) / W) - 1
                 (likewise y), plus the sub-pixel offset: z_k = f32(2 l_k) of the 3x3 logits (0 outside), w_k =
                 exp2_f32(f32(f32(z_k - max z) log2 e)), sum = w_0 + .. + w_8 in order, offset = f32(f32(sum_k
                 f32(w_k d_k)) / sum) with d = -u, 0, u (u = f32(2 / W)), position = f32(x_n + offset)
  descriptions   D_s the description deltas of scale s; F = D14 upsampled to 1/8 plus D8, upsampled to 1/4 plus D4,
                 to 1/2 plus D2, to full resolution plus D1 (the device: dense to 1/4, at the keypoints' pixels below);
                 the keypoint's sample as grid_sample (align_corners False, zero outside): ix = f32(f32(f32(x_n + 1)
                 W) - 1) / 2 (likewise iy), x0 = floor(ix), wx1 = f32(ix - x0), wx0 = f32(1 - wx1), the corners nw,
                 ne, sw, se weighted f32(wy wx) and summed in that order
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benzene.vector_math import exp2_f32, f32

LOG2E = 1.4426950408889634
CUBIC_A = -0.75
NMS_SIZE = 3


# ---------------------------------------------------------------------------------------------- resampling
def bilinear_taps(N: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(i0, i1, l0, l1) of the n output indices from N inputs."""
    scale = torch.tensor(N, dtype=torch.float32) / torch.tensor(n, dtype=torch.float32)
    i = torch.arange(n, dtype=torch.float32, device=device) + f32(0.5).to(device)
    src = i * scale.to(device)
    src = src - f32(0.5).to(device)
    src = torch.clamp_min(src, 0.0)
    i0 = torch.floor(src)
    l1 = src - i0
    l0 = f32(1.0).to(device) - l1
    i0 = i0.long()
    return i0, torch.clamp_max(i0 + 1, N - 1), l0, l1


def bilinear(x: torch.Tensor, size) -> torch.Tensor:
    """x [C, h, w] fp32 -> [C, H, W] (module docstring)."""
    H, W = int(size[0]), int(size[1])
    c0, c1, a0, a1 = bilinear_taps(x.shape[-1], W, x.device)
    t = x[..., c0] * a0
    t = t + x[..., c1] * a1
    r0, r1, b0, b1 = bilinear_taps(x.shape[-2], H, x.device)
    y = t[..., r0, :] * b0[:, None]
    return y + t[..., r1, :] * b1[:, None]


def cubic_coefficients(t: torch.Tensor) -> list[torch.Tensor]:
    """PyTorch's get_cubic_upsample_coefficients in fp32 (each operation rounded)."""
    A = f32(CUBIC_A).to(t.device)

    def convolution1(x):  # ((A + 2) x - (A + 3)) x x + 1
        r = (A + f32(2.0).to(t.device)) * x
        r = r - (A + f32(3.0).to(t.device))
        r = r * x
        r = r * x
        return r + f32(1.0).to(t.device)

    def convolution2(x):  # ((A x - 5 A) x + 8 A) x - 4 A
        r = A * x
        r = r - f32(5.0).to(t.device) * A
        r = r * x
        r = r + f32(8.0).to(t.device) * A
        r = r * x
        return r - f32(4.0).to(t.device) * A

    one = f32(1.0).to(t.device)
    return [convolution2(t + one), convolution1(t), convolution1(one - t), convolution2(f32(2.0).to(t.device) - t)]


def bicubic_taps(N: int, n: int, device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    scale = torch.tensor(N, dtype=torch.float32) / torch.tensor(n, dtype=torch.float32)
    i = torch.arange(n, dtype=torch.float32, device=device) + f32(0.5).to(device)
    src = i * scale.to(device)
    src = src - f32(0.5).to(device)
    i0 = torch.floor(src)
    t = src - i0
    i0 = i0.long()
    return [torch.clamp(i0 - 1 + k, 0, N - 1) for k in range(4)], cubic_coefficients(t)


def bicubic(x: torch.Tensor, size) -> torch.Tensor:
    """x [C, h, w] fp32 -> [C, H, W] (module docstring)."""
    H, W = int(size[0]), int(size[1])
    columns, cw = bicubic_taps(x.shape[-1], W, x.device)
    t = x[..., columns[0]] * cw[0]
    for k in range(1, 4):
        t = t + x[..., columns[k]] * cw[k]
    rows, rw = bicubic_taps(x.shape[-2], H, x.device)
    y = t[..., rows[0], :] * rw[0][:, None]
    for k in range(1, 4):
        y = y + t[..., rows[k], :] * rw[k][:, None]
    return y


def lane_row_totals(e: torch.Tensor) -> torch.Tensor:
    """Per row of e [H, W] the device's 32-lane sum (module docstring) -> [H] fp32."""
    H, W = e.shape
    vectors = (W + 31) // 32
    padded = F.pad(e, (0, vectors * 32 - W))
    lanes = padded[:, :32].clone()
    for v in range(1, vectors):
        lanes = lanes + padded[:, 32 * v : 32 * v + 32]
    for shift in (16, 8, 4, 2, 1):
        lanes = lanes + torch.roll(lanes, -shift, dims=1)
    return lanes[:, 0]


# ---------------------------------------------------------------------------------------------- the model
class DeviceLoMa:
    """The device's LoMa: detect / describe / extract as runner.extract, match with the matcher in fp16."""

    def __init__(self, records: dict, net, settings, device: str, dinov2=None) -> None:
        from .hooked_network import HookedDescriptor, HookedDetector

        self.device = device
        self.net = net
        self.settings = settings
        self.layers = {name: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in record.items()}
                       for name, record in records["layers"].items()}
        self.activations = {name: {k: v.to(device) for k, v in a.items()} for name, a in records["activations"].items()}
        self.detector_plan = HookedDetector(net._detector)
        self.descriptor_plan = HookedDescriptor(net._descriptor)
        self.dinov2 = dinov2  # the descriptor's DINOv2 (a recipe's module): fp32 RGB image -> fp32 map

    # ------------------------------------------------------------------------------- integer parts
    def accumulate(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        """Exact integer sums [N, H, W] (float64) of layer `name` over input codes [C, H, W]: sum(w q) with the
        padding codes outside, plus bias2."""
        layer = self.layers[name]
        w = layer["codes"].double()
        k = w.shape[-1]
        x = codes.double()
        if k > 1:
            C, H, W = x.shape
            pad = k // 2
            padded = layer["padding"].double()[:, None, None].expand(C, H + 2 * pad, W + 2 * pad).clone()
            padded[:, pad : pad + H, pad : pad + W] = x
            x = padded
        acc = F.conv2d(x[None], w, groups=x.shape[0] if layer["kind"] == "dw" else 1)[0]
        return acc.round() + layer["bias2"].double()[:, None, None]

    def output(self, name: str, acc: torch.Tensor) -> torch.Tensor:
        layer = self.layers[name]
        a = acc.float()
        if layer["output"] == "u8":
            v = a * layer["m"][:, None, None]
            v = v + layer["b"][:, None, None]
            return torch.trunc(v.clamp(0.0, 255.5))
        v = a * layer["S"][:, None, None]
        return v + layer["B"][:, None, None]

    def convolution(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        return self.output(name, self.accumulate(name, codes))

    def quantize(self, point: str, x: torch.Tensor) -> torch.Tensor:
        a = self.activations[point]
        R = (f32(1.0).to(self.device).double() / a["scale"]).float()
        Zq = (a["zero"] + 0.5).float()
        q = x * R[:, None, None]
        q = q + Zq[:, None, None]
        return torch.trunc(q.clamp(0.0, 255.5))

    def encoder(self, plan, gray: torch.Tensor) -> list[torch.Tensor]:
        x = gray.float()[None]
        features = []
        for i, pooled in enumerate(plan.pooled):
            x = self.convolution(f"{plan.prefix}.enc.{i}", x)
            if pooled:
                features.append(x)
                x = F.max_pool2d(x[None], 2, 2)[0]
        return features

    def refine(self, plan, scale: str, feature: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        p = f"{plan.prefix}.dec{scale}"
        x = feature if context is None else torch.cat([feature, context], 0)
        x0 = self.convolution(f"{p}.block1.conv2", self.convolution(f"{p}.block1.conv1", x))
        h = x0
        for j in range(plan.hidden_blocks[scale]):
            h = self.convolution(f"{p}.hidden{j}.pw", self.convolution(f"{p}.hidden{j}.dw", h))
        return self.convolution(f"{p}.out", torch.cat([h, x0], 0))

    # ------------------------------------------------------------------------------- detector
    @torch.inference_mode()
    def detector_logits(self, gray: torch.Tensor) -> torch.Tensor:
        """Gray codes [H, W] at the detector's size -> logits [H, W] fp32."""
        plan = self.detector_plan
        features = self.encoder(plan, gray)
        logits, context = None, None
        for index, scale in enumerate(plan.scales):
            out = self.refine(plan, scale, features[-(index + 1)], context)
            logits = out[:1] if logits is None else logits + out[:1]
            if index < len(plan.scales) - 1:
                size = features[-(index + 2)].shape[-2:]
                logits = bicubic(logits, size)
                context = self.quantize(f"det.dec{plan.scales[index + 1]}.context", bilinear(out[1:], size))
        return logits[0]

    @torch.inference_mode()
    def keypoints(self, logits: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(normalized positions [k, 2], probabilities [k], raster pixels [k]) of the logits [H, W] (module
        docstring)."""
        H, W = logits.shape
        device = logits.device
        d = logits - logits.max()
        e = exp2_f32(d * f32(LOG2E).to(device))
        peak = F.max_pool2d(e[None, None], NMS_SIZE, stride=1, padding=NMS_SIZE // 2)[0, 0]
        values = torch.where(e == peak, e, torch.zeros_like(e)).reshape(-1)
        indices = torch.sort(values, descending=True, stable=True).indices[:count]
        total = 0.0
        for row_total in lane_row_totals(e).double().tolist():  # (in row order, float64)
            total += row_total
        total = f32(total).to(device)
        scores = values[indices] / total
        ys, xs = indices // W, indices % W
        x_n = (f32(2.0).to(device) * xs.float() + f32(1.0).to(device)) / f32(W).to(device)
        x_n = x_n - f32(1.0).to(device)
        y_n = (f32(2.0).to(device) * ys.float() + f32(1.0).to(device)) / f32(H).to(device)
        y_n = y_n - f32(1.0).to(device)
        patches = F.unfold(logits[None, None], NMS_SIZE, padding=NMS_SIZE // 2)[0][:, indices]  # [9, k], 0 outside
        z = patches * f32(2.0).to(device)
        w = exp2_f32((z - z.max(0).values[None]) * f32(LOG2E).to(device))
        total_w = w[0].clone()
        for k in range(1, 9):
            total_w = total_w + w[k]
        ux = f32(2.0).to(device) / f32(W).to(device)
        uy = f32(2.0).to(device) / f32(H).to(device)
        num_x, num_y = None, None
        for k in range(9):
            ky, kx = divmod(k, 3)
            tx = w[k] * (ux * (kx - 1))
            ty = w[k] * (uy * (ky - 1))
            num_x = tx if num_x is None else num_x + tx
            num_y = ty if num_y is None else num_y + ty
        positions = torch.stack([x_n + num_x / total_w, y_n + num_y / total_w], 1)
        return positions, scores, indices

    # ------------------------------------------------------------------------------- descriptor
    @torch.inference_mode()
    def dinov2_codes(self, gray: torch.Tensor) -> torch.Tensor:
        """Gray codes [H, W] at the descriptor's size -> DINOv2's output codes [1024, H / 14, W / 14] (desc.dino)."""
        image = (gray.float() / 255.0)[None, None].expand(1, 3, -1, -1).contiguous()
        return self.quantize("desc.dino", self.dinov2(image)[0])

    def description_map(self, gray: torch.Tensor, dino_codes: torch.Tensor | None = None,
                        stages: dict | None = None) -> torch.Tensor:
        """Gray codes [H, W] at the descriptor's size -> the description map F [D, H, W] fp32 (module docstring).
        dino_codes: DINOv2's codes to use (the device's: its DINOv2 is not bit-exact), else dinov2_codes; stages:
        receives the maps of scales 14, 8 and 4 (F14 = D14, F8, F4 [D, h, w])."""
        plan = self.descriptor_plan
        features = self.encoder(plan, gray)
        features = features + [self.dinov2_codes(gray) if dino_codes is None else dino_codes]
        descriptions, context = None, None
        split = plan.split
        for index, scale in enumerate(plan.scales):
            out = self.refine(plan, scale, features[-(index + 1)], context)
            descriptions = out[:split] if descriptions is None else descriptions + out[:split]
            if stages is not None and scale in ("14", "8", "4"):
                stages[f"F{scale}"] = descriptions
            if index < len(plan.scales) - 1:
                size = features[-(index + 2)].shape[-2:]
                descriptions = bilinear(descriptions, size)
                context = self.quantize(f"desc.dec{plan.scales[index + 1]}.context", bilinear(out[split:], size))
        return descriptions

    @staticmethod
    def sample(description_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """grid_sample of F [D, H, W] at normalized positions [k, 2] (module docstring) -> [k, D]."""
        D, H, W = description_map.shape
        device = description_map.device
        one = f32(1.0).to(device)

        def unnormalize(coordinate, size):
            u = (coordinate + one) * f32(size).to(device)
            return (u - one) / f32(2.0).to(device)

        ix, iy = unnormalize(positions[:, 0], W), unnormalize(positions[:, 1], H)
        x0, y0 = torch.floor(ix), torch.floor(iy)
        wx1, wy1 = ix - x0, iy - y0
        wx0, wy0 = one - wx1, one - wy1
        x0, y0 = x0.long(), y0.long()
        flat = description_map.reshape(D, -1)
        total = None
        for dy, dx, wy, wx in ((0, 0, wy0, wx0), (0, 1, wy0, wx1), (1, 0, wy1, wx0), (1, 1, wy1, wx1)):
            yy, xx = y0 + dy, x0 + dx
            valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            corner = flat[:, (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1))].T
            term = (wy * wx)[:, None] * torch.where(valid[:, None], corner, torch.zeros_like(corner))
            total = term if total is None else total + term
        return total

    # ------------------------------------------------------------------------------- runner interface
    def match(self, kpts0, kpts1, desc0, desc1):
        return self.net.match(kpts0, kpts1, desc0, desc1)

    @torch.inference_mode()
    def extract(self, rgb, settings) -> dict:
        """runner.extract's outputs for a gray image (rgb: its three equal channels)."""
        from .runner import descriptor_shape, detector_shape, network_codes

        height, width = rgb.shape[:2]
        gray_detector = network_codes(rgb, detector_shape(width, height, settings.detector_size), self.device)
        gray_descriptor = network_codes(rgb, descriptor_shape(width, height, settings.descriptor_size), self.device)
        positions, scores, _ = self.keypoints(self.detector_logits(gray_detector), settings.keypoints)
        descriptors = self.sample(self.description_map(gray_descriptor), positions)
        size = torch.tensor([width, height], device=positions.device, dtype=torch.float32)
        return {"keypoints_normalized": positions, "keypoints": size * (positions + 1) / 2 - 0.5,
                "keypoint_scores": scores, "descriptors": descriptors}


__all__ = ["DeviceLoMa", "bicubic", "bilinear", "lane_row_totals"]
