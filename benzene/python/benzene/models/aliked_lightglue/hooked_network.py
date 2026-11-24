"""ALIKED's network (extract_dense_map) and descriptor head (SDDH) written out step by step, so calibration and the
recipes can look at and replace every tensor the device quantizes; with no hooks it computes what the checkout's modules
compute.

HookedAliked(net, hooks) wraps the fp32 ALIKED (lightglue/aliked.py, aliked-n16) and has its interface
(extract_dense_map, desc_head, dkd). Hooks (all optional):
  activation(name, x)          the tensor at quantization point `name` (ACTIVATIONS) -> the tensor the network goes
                               on with (calibration observes it, a recipe quantizes it)
  convolution(name, x, layer)  replaces layer `name`'s convolution (CONVOLUTIONS) of input x: the BatchNorm-folded
                               weight and bias of folded(net, name), the unfolded input for the GEMM view
                               (unfold(name, x)); None: the float layer
  deformable(name, x, offset)  the sampled input of a deformable layer (deformable_columns) -> the columns used

The order of the parts follows the device (python/benzene/models/aliked_lightglue/device_model.py): the image is gray
(the three RGB weights of block1.conv1 summed: the forward replicates a gray image), the score head's first 1x1
convolution is split into its four 32-channel parts, each at its input's own resolution, upsampled after the
convolution (linear maps commute: the same sum in exact arithmetic).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SELU_LAMBDA = 1.0507009873554804934193349852946
SELU_ALPHA = 1.6732632423543772848170429916717

# the quantization points: name -> where (every one is a SELU output or an average of them, but "image", the
# sampled descriptor features and the feature map patches)
ACTIVATIONS = (
    "block1.mid",    # block1.conv1's output (block1.conv2's input)
    "x1",            # block1's output (conv1's input)
    "x1.pool",       # its 2x2 average (block2.conv1's and block2.downsample's input)
    "block2.mid",
    "x2",
    "x2.pool",       # 4x4 average (block3.conv1's offset convolution, its deformable convolution, downsample)
    "block3.mid",
    "x3",
    "x3.pool",
    "block4.mid",
    "x4",
    "agg1", "agg2", "agg3", "agg4",  # SELU(conv1..4): the score head's input parts at their own resolution
    "score1", "score2", "score3",    # the score head's hidden maps (8, 4, 4 channels)
    "desc.patch",    # the feature map's 3x3 patches around the keypoints (offset_conv.0's input)
    "desc.hidden",   # SELU(offset_conv.0) (offset_conv.2's input)
    "desc.samples",  # the feature map's bilinear samples at the 16 offsets (sf_conv's input)
    "desc.features", # SELU(sf_conv) (the aggregation's input)
)
# the convolutions: name -> (input activation, kind)
CONVOLUTIONS = {
    "block1.conv1": ("image", "3x3"),
    "block1.conv2": ("block1.mid", "3x3"),
    "block2.conv1": ("x1.pool", "3x3"),
    "block2.conv2": ("block2.mid", "3x3"),
    "block2.downsample": ("x1.pool", "1x1"),
    "block3.conv1.offset": ("x2.pool", "3x3"),
    "block3.conv1": ("x2.pool", "deformable"),
    "block3.conv2.offset": ("block3.mid", "3x3"),
    "block3.conv2": ("block3.mid", "deformable"),
    "block3.downsample": ("x2.pool", "1x1"),
    "block4.conv1.offset": ("x3.pool", "3x3"),
    "block4.conv1": ("x3.pool", "deformable"),
    "block4.conv2.offset": ("block4.mid", "3x3"),
    "block4.conv2": ("block4.mid", "deformable"),
    "block4.downsample": ("x3.pool", "1x1"),
    "conv1": ("x1", "1x1"),
    "conv2": ("x2", "1x1"),
    "conv3": ("x3", "1x1"),
    "conv4": ("x4", "1x1"),
    "score_head.0.1": ("agg1", "1x1"),  # score_head.0's input channels 0..31 (x1'), 32..63, 64..95, 96..127
    "score_head.0.2": ("agg2", "1x1"),
    "score_head.0.3": ("agg3", "1x1"),
    "score_head.0.4": ("agg4", "1x1"),
    "score_head.2": ("score1", "3x3"),
    "score_head.4": ("score2", "3x3"),
    "score_head.6": ("score3", "3x3"),
    "desc.offset0": ("desc.patch", "patch"),     # desc_head.offset_conv.0 on the 3x3 patches
    "desc.offset2": ("desc.hidden", "1x1"),      # desc_head.offset_conv.2
    "desc.sf": ("desc.samples", "1x1"),          # desc_head.sf_conv
    "desc.aggregation": ("desc.features", "aggregation"),  # desc_head.agg_weights: [16 x 128] -> 128
}
UPSAMPLING = {"agg2": 2, "agg3": 8, "agg4": 32}  # factor of each score head part's input (align_corners True)


def selu(x: torch.Tensor) -> torch.Tensor:
    return torch.selu(x)


def _batchnorm(bn) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / torch.sqrt(bn.running_var.detach().double().cpu() + bn.eps)
    gain = bn.weight.detach().double().cpu() * inverse
    return gain, bn.bias.detach().double().cpu() - bn.running_mean.detach().double().cpu() * gain


def folded(net, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(weight [N, C, kh, kw], bias [N]) fp64 on the CPU of convolution `name` with its BatchNorm folded in.
    block1.conv1's three input channels are summed (the image is gray); desc.aggregation is [128, 16 * 128] as a
    1x1 convolution over K = p * 128 + c (the 16 samples' features, sample-major)."""
    def conv_weight(conv) -> torch.Tensor:
        return conv.weight.detach().double().cpu()

    def conv_bias(conv, n: int) -> torch.Tensor:
        return conv.bias.detach().double().cpu() if conv.bias is not None else torch.zeros(n, dtype=torch.float64)

    if name == "desc.aggregation":
        agg = net.desc_head.agg_weights.detach().double().cpu()  # [p, c, d]
        weight = agg.permute(2, 0, 1).reshape(agg.shape[2], -1)[:, :, None, None]
        return weight, torch.zeros(agg.shape[2], dtype=torch.float64)
    if name.startswith("desc."):
        conv = {"desc.offset0": net.desc_head.offset_conv[0], "desc.offset2": net.desc_head.offset_conv[2],
                "desc.sf": net.desc_head.sf_conv}[name]
        return conv_weight(conv), conv_bias(conv, conv.out_channels)
    if name.startswith("score_head.0."):
        part = int(name.rsplit(".", 1)[1]) - 1
        w = conv_weight(net.score_head[0])[:, 32 * part : 32 * part + 32]
        return w, torch.zeros(w.shape[0], dtype=torch.float64)
    if name.startswith("score_head.") or name in ("conv1", "conv2", "conv3", "conv4"):
        conv = net.get_submodule(name)
        return conv_weight(conv), conv_bias(conv, conv.out_channels)
    block, layer = name.split(".", 1)
    module = net.get_submodule(block)
    if layer == "downsample":
        conv = module.downsample
        return conv_weight(conv), conv_bias(conv, conv.out_channels)
    if layer.endswith(".offset"):
        conv = module.get_submodule(layer.split(".")[0]).offset_conv
        return conv_weight(conv), conv_bias(conv, conv.out_channels)
    conv = module.get_submodule(layer)
    weight = conv_weight(conv.regular_conv if hasattr(conv, "regular_conv") else conv)
    bias = conv_bias(conv.regular_conv if hasattr(conv, "regular_conv") else conv, weight.shape[0])
    gain, shift = _batchnorm(module.get_submodule("bn" + layer[-1]))
    if name == "block1.conv1":
        weight = weight.sum(1, keepdim=True)
    return weight * gain[:, None, None, None], bias * gain + shift


def bilinear_zero(x: torch.Tensor, py: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    """torchvision deform_conv2d's bilinear sample of x [C, H, W] at positions (py, px) [...]: 0 outside
    (-1, H) x (-1, W), corners outside the map 0 -> [C, ...]."""
    C, H, W = x.shape
    inside = (py > -1) & (py < H) & (px > -1) & (px < W)
    y0 = torch.floor(py)
    x0 = torch.floor(px)
    ly, lx = py - y0, px - x0
    hy, hx = 1 - ly, 1 - lx
    y0, x0 = y0.long(), x0.long()
    flat = x.reshape(C, -1)
    out = torch.zeros((C, *py.shape), dtype=x.dtype, device=x.device)
    for dy, dx, weight in ((0, 0, hy * hx), (0, 1, hy * lx), (1, 0, ly * hx), (1, 1, ly * lx)):
        yy, xx = y0 + dy, x0 + dx
        valid = inside & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        index = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).reshape(-1)
        values = flat[:, index].reshape(C, *py.shape)
        out = out + torch.where(valid, weight, torch.zeros_like(weight)) * values
    return out


def deformable_columns(x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """The deformable 3x3 (pad 1) convolution's input columns: x [C, H, W], offset [18, H, W] ((dy, dx) of kernel
    position k = ky * 3 + kx) -> [C * 9, H * W], K in PyTorch's order (c, ky, kx)."""
    C, H, W = x.shape
    ys = torch.arange(H, device=x.device, dtype=x.dtype)[:, None].expand(H, W)
    xs = torch.arange(W, device=x.device, dtype=x.dtype)[None, :].expand(H, W)
    columns = []
    for k in range(9):
        ky, kx = divmod(k, 3)
        columns.append(bilinear_zero(x, ys - 1 + ky + offset[2 * k], xs - 1 + kx + offset[2 * k + 1]))
    return torch.stack(columns, 1).reshape(C * 9, H * W)


def unfold(kind: str, x: torch.Tensor) -> torch.Tensor:
    """The GEMM view [K, pixels] of a convolution's input x [C, H, W] (3x3: pad 1, K order (c, ky, kx))."""
    if kind == "3x3":
        return F.unfold(x[None], 3, padding=1)[0]
    return x.reshape(x.shape[0], -1)


def get_patches(x: torch.Tensor, keypoints_wh: torch.Tensor) -> torch.Tensor:
    """SDDH's 3x3 patches [N, C, 3, 3] of x [C, H, W] at the keypoints' integer pixels (get_patches: the window's
    corner long(k) - 0.5, clamped into the map)."""
    C, H, W = x.shape
    corner = (keypoints_wh.long() - 3 / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=W - 1 - 3)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=H - 1 - 3)
    offset = torch.arange(3, device=x.device)
    ys = corner[:, 1, None, None] + offset[None, :, None]  # [N, 3 (ky), 1]
    xs = corner[:, 0, None, None] + offset[None, None, :]  # [N, 1, 3 (kx)]
    return x[:, ys, xs].permute(1, 0, 2, 3)  # [N, C, 3, 3]


class HookedAliked(torch.nn.Module):
    """ALIKED's dense network and descriptor head with hooks (module docstring); dkd is the checkout's."""

    def __init__(self, net, activation=None, convolution=None, deformable=None):
        super().__init__()
        self.net = net
        self.dkd = net.dkd
        self.activation_hook = activation
        self.convolution_hook = convolution
        self.deformable_hook = deformable
        self.weights = {name: tuple(t.float().to(next(net.parameters()).device) for t in folded(net, name))
                        for name in CONVOLUTIONS}

    def activation(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.activation_hook(name, x) if self.activation_hook is not None else x

    def convolution(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Layer `name` on input x [C, H, W] (BatchNorm folded, no activation) -> [N, H, W]."""
        if self.convolution_hook is not None:
            y = self.convolution_hook(name, x, self.weights[name])
            if y is not None:
                return y
        weight, bias = self.weights[name]
        kind = CONVOLUTIONS[name][1]
        if kind == "patch":  # (the 3x3 patches as K = 9 C rows)
            weight = weight.reshape(weight.shape[0], -1, 1, 1)
        return F.conv2d(x[None], weight, bias, padding=1 if kind == "3x3" else 0)[0]

    def deformable(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """A deformable layer (offsets from its offset convolution, clamped to max(H, W) / 4) -> [N, H, W]."""
        H, W = x.shape[-2:]
        max_offset = max(H, W) / 4.0
        offset = self.convolution(name + ".offset", x).clamp(-max_offset, max_offset)
        columns = deformable_columns(x, offset)
        if self.deformable_hook is not None:
            columns = self.deformable_hook(name, x, columns)
        if self.convolution_hook is not None:
            y = self.convolution_hook(name, columns, self.weights[name])
            if y is not None:
                return y.reshape(-1, H, W)
        weight, bias = self.weights[name]
        y = weight.reshape(weight.shape[0], -1) @ columns + bias[:, None]
        return y.reshape(-1, H, W)

    def resblock(self, block: str, x: torch.Tensor) -> torch.Tensor:
        deformable = block in ("block3", "block4")
        layer = self.deformable if deformable else self.convolution
        mid = self.activation(f"{block}.mid", selu(layer(f"{block}.conv1", x)))
        out = layer(f"{block}.conv2", mid) + self.convolution(f"{block}.downsample", x)
        return selu(out)

    def extract_dense_map(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """image [1, 3, H, W] (three equal channels) -> (feature map [1, 128, H, W], score map [1, 1, H, W])."""
        x = image[0, :1]
        height, width = x.shape[-2:]
        mid = self.activation("block1.mid", selu(self.convolution("block1.conv1", x)))
        x1 = self.activation("x1", selu(self.convolution("block1.conv2", mid)))
        p = self.activation("x1.pool", F.avg_pool2d(x1, 2, 2))
        x2 = self.activation("x2", self.resblock("block2", p))
        p = self.activation("x2.pool", F.avg_pool2d(x2, 4, 4))
        x3 = self.activation("x3", self.resblock("block3", p))
        p = self.activation("x3.pool", F.avg_pool2d(x3, 4, 4))
        x4 = self.activation("x4", self.resblock("block4", p))
        parts = []
        hidden = None
        for k, x in enumerate((x1, x2, x3, x4), 1):
            agg = self.activation(f"agg{k}", selu(self.convolution(f"conv{k}", x)))
            s = self.convolution(f"score_head.0.{k}", agg)
            if k > 1:
                factor = UPSAMPLING[f"agg{k}"]
                agg = F.interpolate(agg[None], scale_factor=factor, mode="bilinear", align_corners=True)[0]
                s = F.interpolate(s[None], scale_factor=factor, mode="bilinear", align_corners=True)[0]
            parts.append(agg)
            hidden = s if hidden is None else hidden + s
        h = self.activation("score1", selu(hidden))
        h = self.activation("score2", selu(self.convolution("score_head.2", h)))
        h = self.activation("score3", selu(self.convolution("score_head.4", h)))
        score_map = torch.sigmoid(self.convolution("score_head.6", h))
        feature_map = F.normalize(torch.cat(parts, 0), p=2, dim=0)
        return feature_map[None], score_map[None]

    def desc_head(self, feature_map: torch.Tensor, keypoints: list[torch.Tensor]) -> tuple[list, list]:
        """SDDH (feature map [1, C, H, W], keypoints [[N, 2]] in [-1, 1]) -> ([descriptors [N, C]], [offsets])."""
        x = feature_map[0]
        C, H, W = x.shape
        wh = torch.tensor([[W - 1, H - 1]], device=x.device)
        max_offset = max(H, W) / 4.0
        keypoints_wh = (keypoints[0] / 2 + 0.5) * wh
        N = len(keypoints_wh)
        n_pos = self.net.desc_head.n_pos
        patch = self.activation("desc.patch", get_patches(x, keypoints_wh))  # [N, C, 3, 3]
        columns = patch.reshape(N, -1).t()  # [C * 9, N]
        hidden = self.activation("desc.hidden", selu(self.convolution("desc.offset0", columns[:, :, None])))
        offset = self.convolution("desc.offset2", hidden)[:, :, 0].t().clamp(-max_offset, max_offset)  # [N, 2 n_pos]
        offset = offset.view(N, 2, n_pos).permute(0, 2, 1)  # [N, n_pos, 2]
        pos = keypoints_wh.unsqueeze(1) + offset
        pos = 2.0 * pos / wh[None] - 1
        samples = F.grid_sample(x[None], pos.reshape(1, N * n_pos, 1, 2), mode="bilinear", align_corners=True)
        samples = self.activation("desc.samples", samples[0, :, :, 0])  # [C, N n_pos] (keypoint-major)
        features = self.activation("desc.features", selu(self.convolution("desc.sf", samples[:, :, None])))
        features = features[:, :, 0].reshape(C, N, n_pos).permute(2, 0, 1).reshape(n_pos * C, N)  # K = p C + c
        descriptors = self.convolution("desc.aggregation", features[:, :, None])[:, :, 0].t()
        return [F.normalize(descriptors, p=2.0, dim=1)], [offset]
