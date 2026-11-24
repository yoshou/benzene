"""ALIKED as the device computes it (the recipe "dev"): integer convolutions, fp32 requantization and the
post-processing, every step in the device's arithmetic, so the device's outputs can be checked against it bit for bit
(benzene/dsp/models/aliked). The records come from recipes.quantize_dev (or exported_model.records, read back from a
model file).

Activations between convolutions are uint8 codes per channel: x = s (q - z) with the channel's scale s and zero point z
(benzene.quantization.activation_quantizer; the image: codes = gray, s = 1 / 255, z = 0). A convolution sums its
integer weight codes times (q - z) exactly (the per-channel input scales folded into the weights; padding pixels read
z: x = 0), then
  v = f32(f32(acc) S) + B                       (S the weight scale, B the folded BatchNorm / conv biases)
and either fp32 rows (the offsets, the score head parts, the logit, the descriptors) or the SELU requantization for
the consumer's codes (selu_codes), in fp32 from the accumulator a = f32(acc):
  positive  y = f32(f32(a Sp) + Bp)            (Sp = S lambda / s_out, Bp = B lambda / s_out + z_out + 0.5)
  t         t = f32(f32(a St) + Bt)            (St = S log2 e, Bt = B log2 e: v log2 e)
  negative  tc = clamp(t, -30, 0); r = f32(tc + 1.5 2^23) (rounds tc to an integer n); n = f32(r - 1.5 2^23);
            f = f32(tc - n); p = A 2^f by the degree-4 polynomial EXP2_F32 with A = lambda alpha / s_out folded into
            its coefficients Q_i = A c_i (Horner); e = p 2^n by adding n << 23 to p's bits; y = f32(e + Dn)
            (Dn = z_out + 0.5 - A)
  q = trunc(clamp(t > 0 ? positive : negative, 0, 255.5))
Every fp32 operation is rounded on its own (the device never fuses a multiply and an add). The other stages:
  pooling        2x2 / 4x4 averages of codes into the pooled tensor's codes: q = clamp(trunc(f32(f32(f32(sum) K) + Z)),
                 0, 255), K = s_in / (n^2 s_out), Z = z_out - z_in s_in / s_out + 0.5
  ResBlocks      conv2 and the downsample in one accumulator: conv2's codes over its input and the downsample's over
                 the block's (pooled) input
  deformable     offsets = clamp(fp32 rows of the offset convolution, +-max(H, W) / 4); per output pixel and tap the
                 position (y - 1 + ky) + dy, (x - 1 + kx) + dx (f32), outside (-1, H) x (-1, W): z; else the four
                 corners' q - z (0 outside the map) weighted hy hx, hy lx, ly hx, ly lx (ly = py - floor(py), hy = 1 -
                 ly), summed in this order, code = clamp(trunc(f32(sum + (z + 0.5))), 0, 255); then the integer GEMM
                 over the columns (tap-major on the device)
  score head     its first 1x1 convolution as four parts, each at its input's resolution (fp32 rows U_k); U_2..4
                 upsampled (bilinear, align_corners True: per output index src = f32(f32(w - 1) / f32(W - 1)) i,
                 c0 = trunc(src), l1 = src - c0, l0 = 1 - l1; rows first: H(x) = f32(f32(l0 T[c0]) + f32(l1 T[c1])),
                 then columns: V(y) = f32(f32(m0 H[r0]) + f32(m1 H[r1]))); s = ((U_1 + V_2) + V_3) + V_4, SELU
                 codes; the 3x3 layers' codes; the logit's sigmoid = reciprocal(1 + exp2(logit * -log2 e)) with the
                 device's vector_math::exp2 (degree-7 Taylor) and vector_math::reciprocal (bit trick + 3 Newton
                 steps)
  feature map    per pixel: agg1's dequantized codes f32(f32(q - z) s) and agg2..4's dequantized codes upsampled as
                 above (128 channels), normalized by the device's 1 / ||x||: the 128 squares summed as 4 x 32 lanes
                 (lane-wise in channel order, then the lanes in a rotate-and-add tree, lane 0), 1 / sqrt by the bit
                 trick and 3 Newton steps (vector_math::precise_reciprocal_square_root), min(., 1e12), times x
  keypoints      DKD: NMS (radius 2), the border, the threshold, by descending score (ties in raster order), at most
                 K; the soft-argmax over the 5x5 window: e_k = exp2_f32(f32(f32(s_k - max) 10 log2 e)), sums in window
                 order, residual = f32(sum dx e / sum e) (IEEE division); the score as grid_sample (align_corners True)
                 with the sums in its order; coordinates through LightGlue's normalizations step by step
  descriptors    SDDH: the feature map's 3x3 patch at the keypoint pixel into desc.patch codes (q = clamp(trunc(f32(
                 f32(x R) + (z + 0.5))), 0, 255)), offset_conv.0 (SELU codes) and .2 (fp32 offsets), 16 bilinear
                 samples (grid_sample's formulas, zero outside) into desc.samples codes, sf_conv (SELU codes), the
                 aggregation (fp32), normalized as the feature map, fp16
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from benzene.vector_math import exp2_f32, f32, reciprocal_f32

from .recipes import DEVICE_LAYERS
from .runner import DETECTION_THRESHOLD, MAX_KPTS, NMS_RADIUS

# 2^f on [-0.5, 0.5], degree 4 (Lawson minimax, fp32 coefficients): the negative branch's error in codes is
# A = lambda alpha / s (up to ~1300) times the polynomial's relative error
EXP2_F32 = (0.9999992847442627, 0.6931217908859253, 0.240247443318367, 0.05591785907745361, 0.009570094756782055)
SELU_CONSTANTS = ("Sp", "Bp", "St", "Bt", "Dn") + tuple(f"Q{i}" for i in range(len(EXP2_F32)))
ROUNDING_CONSTANT_F32 = 12582912.0  # 1.5 * 2^23: adding it rounds |t| < 2^22 to an integer
SOFT_ARGMAX_SCALE = 10.0 * 1.4426950408889634  # 1 / temperature (0.1) * log2 e
N_POSITIONS = 16
# ---------------------------------------------------------------------------------------------- elementwise
def selu_codes(a: torch.Tensor, constants: dict, channel_dim: int = 0) -> torch.Tensor:
    """The SELU requantization (module docstring) of fp32 accumulators a (or pre-activations with S = 1, B = 0) ->
    codes (float values 0 .. 255)."""
    shape = [1] * a.dim()
    shape[channel_dim] = -1
    k = {name: constants[name].to(a.device).reshape(shape) for name in SELU_CONSTANTS}
    positive = a * k["Sp"]
    positive = positive + k["Bp"]
    t = a * k["St"]
    t = t + k["Bt"]
    tc = t.clamp(-30.0, 0.0)
    magic = f32(ROUNDING_CONSTANT_F32)
    r = tc + magic
    n = r - magic
    f = tc - n
    p = f * k[f"Q{len(EXP2_F32) - 1}"]
    p = p + k[f"Q{len(EXP2_F32) - 2}"]
    for i in range(len(EXP2_F32) - 3, -1, -1):
        p = p * f
        p = p + k[f"Q{i}"]
    e = (p.view(torch.int32) + (r.view(torch.int32) << 23)).view(torch.float32)
    negative = e + k["Dn"]
    y = torch.where(t > 0, positive, negative)
    return torch.trunc(y.clamp(0.0, 255.5))


def sigmoid(logit: torch.Tensor) -> torch.Tensor:
    e = exp2_f32(logit * f32(-1.4426950408889634))
    return reciprocal_f32(f32(1.0) + e)


def inverse_norm(squares_sum: torch.Tensor) -> torch.Tensor:
    """1 / max(sqrt(s), 1e-12) as the device computes it: bit-trick start, 3 Newton steps, min(., 1e12)."""
    r = (torch.tensor(0x5F3759DF, dtype=torch.int32) - (squares_sum.view(torch.int32) >> 1)).view(torch.float32)
    half = squares_sum * f32(0.5)
    for _ in range(3):
        r = r * (f32(1.5) - half * (r * r))
    return torch.minimum(r, f32(1e12))


def lane_sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """Sum of squares over dim 0 (128 channels, x [128, ...]) as the device sums it: four 32-lane vectors lane-wise in
    order, then lane 0 of the rotate-and-add tree (lane i += lane i + 16, 8, 4, 2, 1)."""
    squares = x * x
    lanes = squares[0:32] + squares[32:64]
    lanes = lanes + squares[64:96]
    lanes = lanes + squares[96:128]
    for shift in (16, 8, 4, 2, 1):
        lanes = lanes + torch.roll(lanes, -shift, 0)
    return lanes[0]


def normalized(x: torch.Tensor) -> torch.Tensor:
    """x [128, ...] times the device's inverse norm over dim 0."""
    return x * inverse_norm(lane_sum_of_squares(x))


# ---------------------------------------------------------------------------------------------- resampling
def upsample_taps(w: int, W: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """align_corners upsampling from w to W samples: per output index (c0, c1, l0, l1)."""
    scale = f32(float(w - 1)) / f32(float(W - 1))
    src = scale * torch.arange(W, dtype=torch.float32)
    c0 = torch.trunc(src).long()
    l1 = src - c0.float()
    l0 = f32(1.0) - l1
    c1 = torch.where(c0 < w - 1, c0 + 1, c0)
    return c0.to(device), c1.to(device), l0.to(device), l1.to(device)


def upsample(T: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """T [C, h, w] -> [C, H, W] (align_corners True; rows first, then columns, as the device)."""
    _, h, w = T.shape
    c0, c1, l0, l1 = upsample_taps(w, W, T.device)
    rows = l0 * T[:, :, c0] + l1 * T[:, :, c1]  # [C, h, W]: f32(f32(l0 a) + f32(l1 b))
    r0, r1, m0, m1 = upsample_taps(h, H, T.device)
    return m0[:, None] * rows[:, r0] + m1[:, None] * rows[:, r1]


def bilinear_corners(ix: torch.Tensor, iy: torch.Tensor, H: int, W: int):
    """grid_sample's corners and weights (align_corners True coordinates ix, iy in pixels): [(y, x, weight, inside)]
    in its summation order nw, ne, sw, se."""
    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1
    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_nw) * (iy_se - iy)
    sw = (ix_se - ix) * (iy - iy_nw)
    se = (ix - ix_nw) * (iy - iy_nw)
    out = []
    for yy, xx, weight in ((iy_nw, ix_nw, nw), (iy_nw, ix_se, ne), (iy_se, ix_nw, sw), (iy_se, ix_se, se)):
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        out.append((yy.long().clamp(0, H - 1), xx.long().clamp(0, W - 1), weight, inside))
    return out


def grid_sample(x: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
    """x [C, H, W] sampled at pixel coordinates (ix, iy) [n] (zero outside) -> [C, n], the sums in grid_sample's
    order."""
    C, H, W = x.shape
    value = torch.zeros(C, len(ix), dtype=torch.float32, device=x.device)
    for yy, xx, weight, inside in bilinear_corners(ix, iy, H, W):
        term = x[:, yy, xx] * weight
        value = value + torch.where(inside, term, torch.zeros_like(term))
    return value


def unnormalize(coordinate: torch.Tensor, size: int) -> torch.Tensor:
    """grid_sample's align_corners True pixel coordinate of a normalized one: ((c + 1) / 2) (size - 1)."""
    return ((coordinate + f32(1.0)) / f32(2.0)) * f32(float(size - 1))


# ---------------------------------------------------------------------------------------------- the network
class DeviceAliked(torch.nn.Module):
    """extract(img) -> keypoints / scores / descriptors as runner.extract, from the device records (module
    docstring)."""

    def __init__(self, records: dict, dkd, device: str):
        super().__init__()
        self.device = device
        self.dkd = dkd
        self.dequantization = {name: {k: v.float().to(device) for k, v in a.items()}
                               for name, a in records["dequantize"].items()}
        self.layers = {}
        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [to_device(v) for v in value]
            return value

        for name, record in records["layers"].items():
            self.layers[name] = {k: to_device(v) for k, v in record.items()}
        self.pools = {name: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in p.items()}
                      for name, p in records["pools"].items()}
        self.score1 = {k: v.to(device) for k, v in records["score1"].items()}
        self.quantizers = {name: {k: v.to(device) for k, v in records[name].items()}
                           for name in ("desc.patch", "desc.samples")}

    # ------------------------------------------------------------------------------- integer parts
    def accumulate(self, codes: torch.Tensor, name: str, residual: torch.Tensor | None = None) -> torch.Tensor:
        """The exact integer sums [N, ...] of layer `name` over input codes [C, H, W] (or GEMM columns [K, n] for the
        deformable / patch / aggregation kinds; plus the downsample's over `residual` codes): sum(w q) over the codes
        (3x3: padding pixels at the padding codes) plus bias2 = -sum(w z)."""
        layer = self.layers[name]
        _, kind, _ = DEVICE_LAYERS[name]
        w = layer["codes"].double()
        if kind == "3x3":
            C, H, W = codes.shape
            padded = layer["padding"].double()[:, None, None].expand(C, H + 2, W + 2).clone()
            padded[:, 1 : H + 1, 1 : W + 1] = codes.double()
            acc = F.conv2d(padded[None], w)[0]
        elif kind == "1x1":
            acc = F.conv2d(codes.double()[None], w)[0]
        else:  # GEMM columns [K, n], K in PyTorch's (c, ky, kx) / (p, c) order
            acc = w.reshape(w.shape[0], -1) @ codes.double()
        if residual is not None:
            acc = acc + F.conv2d(residual.double()[None], layer["residual_codes"].double())[0].reshape(acc.shape)
        view = [-1] + [1] * (acc.dim() - 1)
        return acc.round() + layer["bias2"].double().reshape(view)

    def output(self, name: str, acc: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
        """SELU codes (a layer with an output activation: selu_codes of the accumulator) or fp32 v = f32(f32(acc) S)
        + B."""
        layer = self.layers[name]
        shape = [1] * acc.dim()
        shape[channel_dim] = -1
        if layer["output"] is not None:
            return selu_codes(acc.float(), layer, channel_dim)
        v = acc.float() * layer["S"].reshape(shape)
        return v + layer["B"].reshape(shape)

    def pool(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        p = self.pools[name]
        n = p["window"]
        sums = F.avg_pool2d(codes.double()[None], n, n)[0].mul(n * n).round().float()
        q = sums * p["K"][:, None, None]
        q = q + p["Z"][:, None, None]
        return torch.trunc(q).clamp(0, 255)

    def deformable_columns(self, codes: torch.Tensor, offset: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        """The deformable layer's sampled codes [C * 9, H W] (K order (c, k)) of codes [C, H, W], offsets [18, H, W]."""
        C, H, W = codes.shape
        ys = torch.arange(H, device=codes.device, dtype=torch.float32)[:, None].expand(H, W)
        xs = torch.arange(W, device=codes.device, dtype=torch.float32)[None, :].expand(H, W)
        d = codes.float() - zero.float()[:, None, None]
        flat = d.reshape(C, -1)
        columns = []
        for k in range(9):
            ky, kx = divmod(k, 3)
            py = (ys + f32(ky - 1)) + offset[2 * k]
            px = (xs + f32(kx - 1)) + offset[2 * k + 1]
            inside = (py > -1) & (py < H) & (px > -1) & (px < W)
            y0 = torch.floor(py)
            x0 = torch.floor(px)
            ly = py - y0
            lx = px - x0
            hy = f32(1.0) - ly
            hx = f32(1.0) - lx
            value = None
            for dy, dx, weight in ((0, 0, hy * hx), (0, 1, hy * lx), (1, 0, ly * hx), (1, 1, ly * lx)):
                yy = y0.long() + dy
                xx = x0.long() + dx
                valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                corner = flat[:, (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).reshape(-1)].reshape(C, H, W)
                term = weight * torch.where(valid, corner, torch.zeros_like(corner))
                value = term if value is None else value + term
            sample = torch.trunc(value + (zero.float()[:, None, None] + f32(0.5))).clamp(0, 255)
            columns.append(torch.where(inside, sample, zero.float()[:, None, None].expand(C, H, W)))
        return torch.stack(columns, 1).reshape(C * 9, H * W)

    def deformable(self, name: str, codes: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        H, W = codes.shape[-2:]
        max_offset = f32(max(H, W) / 4.0)
        offset = self.output(name + ".offset", self.accumulate(codes, name + ".offset"))
        offset = torch.minimum(torch.maximum(offset, -max_offset), max_offset)
        columns = self.deformable_columns(codes, offset, self.layers[name]["input_zero"])
        acc = self.accumulate(columns, name, residual)
        return self.output(name, acc.reshape(-1, H, W))

    def dequantize(self, name: str, codes: torch.Tensor) -> torch.Tensor:
        a = self.dequantization[name]
        d = codes.float() - a["zero"][:, None, None]
        return d * a["scale"][:, None, None]

    # ------------------------------------------------------------------------------- dense maps
    def dense_maps(self, gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """gray uint8 [H, W] -> (score map [H, W] fp32, feature map [128, H, W] fp32)."""
        H, W = gray.shape
        image = gray.float()[None]
        mid = self.output("block1.conv1", self.accumulate(image, "block1.conv1"))
        x1 = self.output("block1.conv2", self.accumulate(mid, "block1.conv2"))
        p1 = self.pool("x1.pool", x1)
        mid = self.output("block2.conv1", self.accumulate(p1, "block2.conv1"))
        x2 = self.output("block2.conv2", self.accumulate(mid, "block2.conv2", residual=p1))
        p2 = self.pool("x2.pool", x2)
        mid = self.deformable("block3.conv1", p2)
        x3 = self.deformable("block3.conv2", mid, residual=p2)
        p3 = self.pool("x3.pool", x3)
        mid = self.deformable("block4.conv1", p3)
        x4 = self.deformable("block4.conv2", mid, residual=p3)
        aggregated = [self.output(f"conv{k}", self.accumulate(x, f"conv{k}")) for k, x in
                      enumerate((x1, x2, x3, x4), 1)]
        hidden = None
        for k, agg in enumerate(aggregated, 1):
            part = self.output(f"score_head.0.{k}", self.accumulate(agg, f"score_head.0.{k}"))
            if k > 1:
                part = upsample(part, H, W)
            hidden = part if hidden is None else hidden + part
        h = selu_codes(hidden, self.score1)
        h = self.output("score_head.2", self.accumulate(h, "score_head.2"))
        h = self.output("score_head.4", self.accumulate(h, "score_head.4"))
        logit = self.output("score_head.6", self.accumulate(h, "score_head.6"))[0]
        score_map = sigmoid(logit)
        parts = [self.dequantize("agg1", aggregated[0])]
        for k in (2, 3, 4):
            parts.append(upsample(self.dequantize(f"agg{k}", aggregated[k - 1]), H, W))
        feature_map = normalized(torch.cat(parts, 0))
        return score_map, feature_map

    # ------------------------------------------------------------------------------- keypoints, descriptors
    def keypoints(self, score_map: torch.Tensor, max_kpts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DKD on the device: (normalized keypoints [k, 2], scores [k], pixel keypoints [k, 2]) on the CPU (its fp32
        division is IEEE's, the device's; the GPU's PyTorch kernels divide less exactly)."""
        from .runner import select_keypoints

        score_map = score_map.cpu()
        H, W = score_map.shape
        indices = select_keypoints(score_map[None, None], max_kpts, DETECTION_THRESHOLD, NMS_RADIUS)
        x0 = indices % W
        y0 = torch.div(indices, W, rounding_mode="trunc")
        r = NMS_RADIUS
        offsets = torch.arange(-r, r + 1, device=score_map.device)
        window = score_map[(y0[:, None, None] + offsets[None, :, None]), (x0[:, None, None] + offsets[None, None, :])]
        window = window.reshape(len(indices), -1)  # [k, 25] in (ky, kx) order
        maximum = window.max(1).values
        e = exp2_f32((window - maximum[:, None]) * f32(SOFT_ARGMAX_SCALE))
        dx = offsets.repeat(2 * r + 1).float()
        dy = offsets.repeat_interleave(2 * r + 1).float()
        total = torch.zeros_like(maximum)
        sum_x = torch.zeros_like(maximum)
        sum_y = torch.zeros_like(maximum)
        for k in range(window.shape[1]):
            total = total + e[:, k]
            sum_x = sum_x + e[:, k] * dx[k]
            sum_y = sum_y + e[:, k] * dy[k]
        kx = x0.float() + sum_x / total
        ky = y0.float() + sum_y / total
        nx = (kx / f32(float(W - 1))) * f32(2.0) - f32(1.0)
        ny = (ky / f32(float(H - 1))) * f32(2.0) - f32(1.0)
        scores = grid_sample(score_map[None], unnormalize(nx, W), unnormalize(ny, H))[0]
        pixels = torch.stack([f32(float(W - 1)) * (nx + f32(1.0)) / f32(2.0),
                              f32(float(H - 1)) * (ny + f32(1.0)) / f32(2.0)], 1)
        return torch.stack([nx, ny], 1), scores, pixels

    def quantize(self, name: str, x: torch.Tensor, channel_dim: int) -> torch.Tensor:
        q = self.quantizers[name]
        shape = [1] * x.dim()
        shape[channel_dim] = -1
        v = x * q["R"].reshape(shape)
        v = v + q["Z"].reshape(shape)
        return torch.trunc(v).clamp(0, 255)

    def descriptors(self, feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """SDDH on the device: fp16 descriptors [k, 128] (as fp32) of the normalized keypoints [k, 2] (their geometry
        on the CPU: IEEE division)."""
        keypoints = keypoints.cpu()
        C, H, W = feature_map.shape
        N = len(keypoints)
        wx, wy = f32(float(W - 1)), f32(float(H - 1))
        kwh_x = ((keypoints[:, 0] / f32(2.0)) + f32(0.5)) * wx
        kwh_y = ((keypoints[:, 1] / f32(2.0)) + f32(0.5)) * wy
        corner_x = (kwh_x.long().float() - 0.5).long().clamp(0, W - 1 - 3)
        corner_y = (kwh_y.long().float() - 0.5).long().clamp(0, H - 1 - 3)
        taps = torch.arange(3)
        ys = (corner_y[:, None, None] + taps[None, :, None]).to(feature_map.device)
        xs = (corner_x[:, None, None] + taps[None, None, :]).to(feature_map.device)
        patch = feature_map[:, ys, xs]  # [C, N, 3, 3]
        patch_codes = self.quantize("desc.patch", patch, 0).permute(0, 2, 3, 1).reshape(C * 9, N)  # (c, ky, kx)
        hidden = self.output("desc.offset0", self.accumulate(patch_codes, "desc.offset0"))  # [32, N]
        offset = self.output("desc.offset2", self.accumulate(hidden[:, :, None], "desc.offset2"))[:, :, 0].cpu()
        max_offset = f32(max(H, W) / 4.0)
        offset = torch.minimum(torch.maximum(offset, -max_offset), max_offset)
        pos_x = kwh_x[None, :] + offset[:N_POSITIONS]  # [16, N]
        pos_y = kwh_y[None, :] + offset[N_POSITIONS:]
        nx = (f32(2.0) * pos_x) / wx - f32(1.0)
        ny = (f32(2.0) * pos_y) / wy - f32(1.0)
        ix = unnormalize(nx.t().reshape(-1), W).to(feature_map.device)  # keypoint-major: n * 16 + p
        iy = unnormalize(ny.t().reshape(-1), H).to(feature_map.device)
        samples = grid_sample(feature_map, ix, iy)  # [C, 16 N]
        sample_codes = self.quantize("desc.samples", samples, 0)
        features = self.output("desc.sf", self.accumulate(sample_codes[:, :, None], "desc.sf"))[:, :, 0]  # [C, 16N]
        columns = features.reshape(C, N, N_POSITIONS).permute(2, 0, 1).reshape(N_POSITIONS * C, N)  # (p, c)
        descriptors = self.output("desc.aggregation", self.accumulate(columns, "desc.aggregation"))  # [128, N]
        return normalized(descriptors).t().to(torch.float16).float()

    @torch.inference_mode()
    def extract(self, img: torch.Tensor, max_kpts: int = MAX_KPTS) -> dict:
        gray = img.to(self.device)
        score_map, feature_map = self.dense_maps(gray)
        keypoints, scores, pixels = self.keypoints(score_map, max_kpts)
        descriptors = self.descriptors(feature_map, keypoints)
        return {"keypoints": pixels.to(self.device), "keypoint_scores": scores.to(self.device),
                "descriptors": descriptors.to(self.device)}


__all__ = ["DeviceAliked", "selu_codes", "sigmoid", "upsample"]
