"""AnySplat's network in PyTorch, as AnySplat's EncoderAnySplat runs it for inference (the InternRobotics/AnySplat
checkout, commit 5f5e208, src/model/encoder/anysplat.py; paths.SRC_DIR): VGGT's aggregator, camera head and depth head
(the vggt checkout) with AnySplat's weights (its aggregator is stored in bf16 and loaded as fp32 here), and its Gaussian
head:

  GaussianHead   heads/vggt_dpt_gs_head.py VGGT_DPT_GS_Head: VGGT's DPT head whose output_conv2 is a 3x3 128 -> 128
                 convolution, ReLU, a 1x1 128 -> 84 convolution, with the image merger (a 7x7 3 -> 128 convolution of the
                 image in [0, 1], ReLU) added to the upsampled features before their position embedding; no activation:
                 RAW_CHANNELS raw values per pixel (the RAW_* channels), which gaussians.make_gaussians turns into
                 Gaussians
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import paths  # noqa: F401 - the vggt checkout on sys.path, for the imports of vggt below

SH_DEGREE = 4
SH_BASES = (SH_DEGREE + 1) ** 2
RAW_CHANNELS = 84
# a pixel's raw channels
RAW_OPACITY = 0  # the density logit
RAW_SCALES = slice(1, 4)
RAW_ROTATION = slice(4, 8)  # quaternion (x, y, z, w), not normalized
RAW_HARMONICS = slice(8, 8 + 3 * SH_BASES)  # [3][SH_BASES], channel-major
RAW_FEATURES = slice(0, RAW_CHANNELS - 1)  # what a voxel averages
RAW_CONFIDENCE = RAW_CHANNELS - 1  # the logit of a voxel's softmax weights


def _dpt_head_class():
    """The vggt checkout's DPTHead (imported here, after the paths import, rather than at the top)."""
    from vggt.heads.dpt_head import DPTHead

    return DPTHead


class GaussianHead(_dpt_head_class()):
    """AnySplat's VGGT_DPT_GS_Head on the vggt checkout's DPTHead."""

    def __init__(self) -> None:
        super().__init__(dim_in=2048, patch_size=14, output_dim=RAW_CHANNELS, activation="norm_exp",
                         conf_activation="expp1", features=256)
        self.input_merger = nn.Sequential(nn.Conv2d(3, 128, 7, 1, 3), nn.ReLU())
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, RAW_CHANNELS, kernel_size=1, stride=1, padding=0))

    def forward(self, aggregated_tokens_list, images: torch.Tensor, patch_start_idx: int,
                frames_chunk_size: int | None = 8) -> torch.Tensor:
        """images [B, S, 3, H, W] in [0, 1] -> raw outputs [B, S, RAW_CHANNELS, H, W]."""
        S = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        return torch.cat([self._forward_impl(aggregated_tokens_list, images, patch_start_idx, s,
                                             min(s + frames_chunk_size, S))
                          for s in range(0, S, frames_chunk_size)], dim=1)

    def _forward_impl(self, aggregated_tokens_list, images, patch_start_idx, frames_start_idx=None,
                      frames_end_idx=None):
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = self.norm(x.reshape(B * S, -1, x.shape[-1]))
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self._apply_pos_embed(self.projects[dpt_idx](x), W, H)
            out.append(self.resize_layers[dpt_idx](x))
        out = self.scratch_forward(out)
        direct_img_feat = self.input_merger(images.flatten(0, 1))
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True) + direct_img_feat
        out = self.scratch.output_conv2(self._apply_pos_embed(out, W, H))
        return out.view(B, S, *out.shape[1:])


def load_network(model_dir, device: str = "cuda:0"):
    """VGGT (camera and depth heads) with AnySplat's weights (model_dir/model.safetensors) and its Gaussian head,
    fp32, eval mode. The module names are the checkpoint's without its "encoder." prefix."""
    from safetensors.torch import load_file

    from vggt.models.vggt import VGGT

    model = VGGT(enable_point=False, enable_track=False)
    model.gaussian_param_head = GaussianHead()
    state = {k.removeprefix("encoder."): v.float() for k, v in load_file(f"{model_dir}/model.safetensors").items()}
    model.load_state_dict(state)
    return model.to(device).eval()
