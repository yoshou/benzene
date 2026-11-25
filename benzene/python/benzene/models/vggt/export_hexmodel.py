"""Export a W8A8 recipe of VGGT to the on-device (Hexagon) model file (.hexmodel + .tab, benzene.model_file.Writer).

Contents (tensor names):
  vggt.pos_embed             DINOv2 position embedding [1370][1024] (cls + 37 x 37 grid), fp32 (resampled by the host)
  vggt.cls_token             [1024], vggt.register_tokens [4][1024] (DINOv2), fp32
  vggt.blocks.<l>            72 transformer blocks in runtime order (benzene protocol.h vision_transformer::block_part):
                             DINOv2 blocks 0..23, then frame block i at 24 + 2 i and global block i at 25 + 2 i
  vggt.tail                  DINOv2 final norm, aggregator camera / register tokens, DINOv2 patch embedding as fp16
                             GEMM (protocol.h vggt::tail_part)
  vggt.cam.blocks.<i>        the camera head's 4 trunk blocks (vision_transformer::block_part, 2048 / 8192 dims)
  vggt.cam.head              the rest of the camera head (vggt::camera_part)
  vggt.dpt.<h>.<i>           DPT heads h = depth, point: convolution i in dpt_head::convolution order as the fp16 HMX
                             GEMM's weight stream (dpt_head::convolution_part)
  vggt.dpt.<h>.misc          norm, output_conv2.2 (vggt::dpt_head_part)

  python -m benzene.models.vggt.export_hexmodel w8a8_tfc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benzene.model_file import Writer

from .device_layout import DPT_CONVS, VGGT, block_names, ckpt
from .paths import MODEL_DIR, WORK
from .recipes import DEVICE_RECIPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--out", default=str(WORK / "export"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from .recipes import apply_recipe
    from .runner import load_model

    apply_recipe(load_model(args.device), args.recipe)  # quantizes (and caches) the layers that are not cached yet
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(out / f"vggt-{args.recipe}.hexmodel")
    pe = "aggregator.patch_embed."
    w.f32("vggt.pos_embed", ckpt(pe + "pos_embed")[0])
    w.f32("vggt.cls_token", ckpt(pe + "cls_token").reshape(1024))
    w.f32("vggt.register_tokens", ckpt(pe + "register_tokens").reshape(4, 1024))
    for l, (pre, kind) in enumerate(block_names()):
        w.blob(f"vggt.blocks.{l}", VGGT.blob_block(args.recipe, pre, kind > 0))
        print("block", l, pre, flush=True)
    w.blob("vggt.tail", VGGT.blob_tail())
    for i in range(4):
        w.blob(f"vggt.cam.blocks.{i}", VGGT.blob_block(args.recipe, f"camera_head.trunk.{i}.", False))
    w.blob("vggt.cam.head", VGGT.blob_cam_head(args.recipe))
    print("camera head", flush=True)
    for h in ("depth", "point"):
        for i, conv in enumerate(DPT_CONVS):
            w.blob(f"vggt.dpt.{h}.{i}", VGGT.blob_dpt_conv(h, conv))
        w.blob(f"vggt.dpt.{h}.misc", VGGT.blob_dpt_misc(h))
        print(f"{h} head", flush=True)
    size = w.finish({"model": "VGGT-1B", "recipe": args.recipe, "config": json.loads((MODEL_DIR / "config.json").read_text())})
    print(f"wrote {w.path} ({size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
