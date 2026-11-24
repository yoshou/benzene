"""Export a W8A8 recipe of the Cosmos 3 Edge reasoner to the on-device (Hexagon) model file.

File layout (all offsets 4096-aligned):
  bytes 0..15   magic "HXMD", u32 version (1), u32 JSON length, u32 0
  JSON          {"recipe", "config", "tensors": {name: {"off", "size", ...}}} (offsets from the file start)
  1 MiB..       layer blobs (W8A8 streams + fp32 vectors, "parts" = offsets within the blob in the order of the benzene
                protocol.h enums cosmos3_edge::llm_layer_part / vision_layer_part / vision_tail_part) and plain fp32
                tensors ("dtype", "shape")
plus <model>.tab, the fixed-size tensor table the device runtime reads (benzene.model_file.Writer).

Contents: the vision patch embedding (fp32; the host computes it) and position embedding grid, 27 vision layers, the
vision tail (post norm + projector), 28 language model layers, the final norm (times inv_s of the lm_head), the lm_head
W8A8 stream (decode GEMV). With --dec4 also "llm.dec4.<l>" (per layer the four Q4_0 decode streams qkv, o, fc1, fc2 of
benzene.kernels.gemv.gemv_q4_stream) and "llm.dec4.head". The token embedding table goes to a separate fp16 file read
by the host.

  python -m benzene.models.cosmos3_edge.export_hexmodel w8a8_sq05s_gptq
  python -m benzene.models.cosmos3_edge.export_hexmodel w8a8_sq05s_gptq --dec4 dev_w8a8_q4hs
      (the same file plus the 4-bit decode weights)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benzene.model_file import Writer

from .device_layout import blob_decoder_layer, blob_vision_layer, blob_vision_tail, ckpt, inv_smooth, w8a8_linear
from .paths import MODEL_DIR, WORK
from .recipes import DEVICE_RECIPE

VISION = "model.visual."
LANGUAGE_MODEL = "model.language_model."
VISION_LAYERS = 27
LANGUAGE_MODEL_LAYERS = 28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--out", default=str(WORK / "export"))
    parser.add_argument("--dec4",
                        help="mixed recipe (e.g. dev_w8a8_q4hs) whose 4-bit decode copy of the LLM layers and "
                             "of the lm_head is added to the file (benzene cosmos3_edge_generate --dec4)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((MODEL_DIR / "config.json").read_text())
    writer = Writer(out / f"cosmos3_edge-{args.recipe}.hexmodel")

    writer.f32("vision.patch_embed.w", ckpt(VISION + "embeddings.patch_embedding.weight"))
    writer.f32("vision.patch_embed.b", ckpt(VISION + "embeddings.patch_embedding.bias"))
    writer.f32("vision.pos_embed", ckpt(VISION + "embeddings.position_embedding.weight"))
    for i in range(VISION_LAYERS):
        writer.blob(f"vision.layers.{i}", blob_vision_layer(args.recipe, f"{VISION}encoder.layers.{i}."))
        print("vision layer", i, flush=True)
    writer.blob("vision.tail", blob_vision_tail(args.recipe))
    for i in range(LANGUAGE_MODEL_LAYERS):
        writer.blob(f"llm.layers.{i}", blob_decoder_layer(args.recipe, f"{LANGUAGE_MODEL}layers.{i}."))
        print("llm layer", i, flush=True)
    writer.f32("llm.norm_g", ckpt(LANGUAGE_MODEL + "norm.weight") * inv_smooth(args.recipe, "lm_head"))
    head, head_scales = w8a8_linear(args.recipe, ["lm_head"])
    writer.raw("llm.lm_head", head, cols=int(head_scales.shape[0]), k=2048)
    writer.f32("llm.lm_head.s", head_scales)
    if args.dec4:  # after the int8 head: a run without --dec4 never reads these
        import torch

        from .device_layout import _rec, _rec_dec, blob_decoder_layer_dec4, q4_linear

        for i in range(LANGUAGE_MODEL_LAYERS):
            writer.blob(f"llm.dec4.{i}", blob_decoder_layer_dec4(args.dec4, args.recipe, f"{LANGUAGE_MODEL}layers.{i}."))
            print("llm dec4 layer", i, flush=True)
        assert torch.equal(_rec_dec(args.dec4, "lm_head")["smooth"], _rec(args.recipe, "lm_head")["smooth"])
        head_4bit = q4_linear(args.dec4, ["lm_head"])
        assert len(head_4bit) == int(head_scales.shape[0]) // 32 * 2048 * 18
        writer.raw("llm.dec4.head", head_4bit, cols=int(head_scales.shape[0]), k=2048)
    size = writer.finish({"model": "Cosmos3-Edge", "recipe": args.recipe, "dec4": args.dec4, "config": config})
    embeddings = ckpt(LANGUAGE_MODEL + "embed_tokens.weight").astype(np.float16)
    embeddings.tofile(out / "embed_tokens.f16")
    print(f"wrote {writer.path} ({size / 2**30:.2f} GiB) and embed_tokens.f16 {embeddings.shape} "
          f"({embeddings.nbytes / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
