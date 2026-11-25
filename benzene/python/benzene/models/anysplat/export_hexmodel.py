"""Export a W8A8 recipe of AnySplat to the on-device (Hexagon) model file (.hexmodel + .tab, benzene.model_file.Writer).

VGGT's network under the VGGT port's tensor names and layouts (benzene.models.vggt.export_hexmodel; AnySplat has no
point head), then the Gaussian head:
  vggt.pos_embed, vggt.cls_token, vggt.register_tokens, vggt.blocks.<l>, vggt.tail, vggt.cam.blocks.<i>, vggt.cam.head,
  vggt.dpt.depth.<i>, vggt.dpt.depth.misc
  anysplat.gaussians.<i>    the Gaussian head's convolution i in dpt_head::convolution order (output_conv2.0: 3x3
                            128 -> 128) as the fp16 HMX GEMM's weight stream (dpt_head::convolution_part)
  anysplat.gaussians.head   norm, the merger and the output convolution (protocol.h anysplat::gaussian_head_part)

  python -m benzene.models.anysplat.export_hexmodel w8a8_tfc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benzene.model_file import Writer

from ..vggt.device_layout import DPT_CONVS, block_names
from .device_layout import ANYSPLAT, blob_gaussian_head, ckpt
from .paths import MODEL_DIR, WORK
from .recipes import DEVICE_RECIPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--out", default=str(WORK / "export"))
    parser.add_argument("--device", default="cuda:0", help="where the layers not quantized yet are quantized")
    args = parser.parse_args()
    from .recipes import apply_recipe
    from .runner import load_model

    apply_recipe(load_model(args.device), args.recipe)  # quantizes (and caches) the layers that are not cached yet
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out / f"anysplat-{args.recipe}.hexmodel")
    pe = "aggregator.patch_embed."
    w.f32("vggt.pos_embed", ckpt(pe + "pos_embed")[0])
    w.f32("vggt.cls_token", ckpt(pe + "cls_token").reshape(1024))
    w.f32("vggt.register_tokens", ckpt(pe + "register_tokens").reshape(4, 1024))
    for l, (pre, kind) in enumerate(block_names()):
        w.blob(f"vggt.blocks.{l}", ANYSPLAT.blob_block(args.recipe, pre, kind > 0))
    print("transformer blocks", flush=True)
    w.blob("vggt.tail", ANYSPLAT.blob_tail())
    for i in range(4):
        w.blob(f"vggt.cam.blocks.{i}", ANYSPLAT.blob_block(args.recipe, f"camera_head.trunk.{i}.", False))
    w.blob("vggt.cam.head", ANYSPLAT.blob_cam_head(args.recipe))
    for i, conv in enumerate(DPT_CONVS):
        w.blob(f"vggt.dpt.depth.{i}", ANYSPLAT.blob_dpt_conv("depth", conv))
    w.blob("vggt.dpt.depth.misc", ANYSPLAT.blob_dpt_misc("depth"))
    for i, conv in enumerate(DPT_CONVS):
        w.blob(f"anysplat.gaussians.{i}", ANYSPLAT.blob_dpt_conv("gaussian_param", conv))
    w.blob("anysplat.gaussians.head", blob_gaussian_head())
    print("camera, depth and Gaussian heads", flush=True)
    size = w.finish({"model": "AnySplat", "recipe": args.recipe,
                     "config": json.loads((MODEL_DIR / "config.json").read_text())})
    print(f"wrote {w.path} ({size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
