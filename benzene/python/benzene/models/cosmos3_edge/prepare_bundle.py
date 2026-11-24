"""Device input bundle of one cached sample for benzene cosmos3_edge_generate (host/cosmos3_edge_commands.cpp
read_prompt_bundle):

  vision_input.f32      [patches][1152]  patch + position embeddings of the images or the video frames (the vision
                                         embeddings module in fp32: the host prepares them, the device runs the encoder)
  image_boundaries.i32  [images + 1]     image i = patches [b[i], b[i + 1]); a video contributes one image per frame
  prompt_ids.i32        [P]              the prompt; vision positions hold the image or video placeholder id
  prompt_rope.f32       [2][P][64]       M-RoPE cos, then sin, of the prompt positions
  metadata.i32          {P, rope_delta, image token id, video token id, end-of-sequence id count, the ids ...}
  prompt.txt                             the prompt text (for reading the outputs)
Generated tokens continue at text positions (sequence index + rope_delta). A prompt holds images or one kind of video
input, not both (the device fills the placeholders with the vision tokens in prompt order).

  python -m benzene.models.cosmos3_edge.prepare_bundle --split eval --sample 0 --out work/cosmos3_edge/bundle/eval_000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from .paths import DATA_DIR
from .runner import load_model, load_sample

HIDDEN = 2048


def end_of_sequence_ids(model) -> list[int]:
    ids = model.generation_config.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids or [])


@torch.no_grad()
def write_bundle(model, sample: dict, out: Path) -> dict:
    """Writes the bundle of `sample` (loaded in fp32 on the CPU, as the model) into `out`; returns its sizes."""
    out.mkdir(parents=True, exist_ok=True)
    inputs = sample["inputs"]
    vision_model = model.model
    ids = inputs["input_ids"]
    prompt_tokens = ids.shape[1]
    has_images = "pixel_values" in inputs
    has_videos = "pixel_values_videos" in inputs
    if has_images and has_videos:
        raise ValueError(f"{sample['name']}: a prompt with images and videos")
    if has_images or has_videos:
        pixels = inputs["pixel_values"] if has_images else inputs["pixel_values_videos"]
        grid = inputs["image_grid_thw"] if has_images else inputs["video_grid_thw"]
        vision_input = vision_model.visual.embeddings(pixels.float(), grid).float()
        frame_patches = [int(h * w) for t, h, w in grid.tolist() for _ in range(int(t))]
        boundaries = np.concatenate([[0], np.cumsum(frame_patches)]).astype(np.int32)
        positions, deltas = vision_model.get_rope_index(
            ids, inputs["mm_token_type_ids"], image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"), attention_mask=inputs["attention_mask"])
        rope_delta = int(deltas.reshape(-1)[0])
    else:
        vision_input = torch.zeros(0, 1152)
        boundaries = np.zeros(1, np.int32)
        positions = torch.arange(prompt_tokens).view(1, 1, -1).expand(3, 1, -1)
        rope_delta = 0
    cos, sin = vision_model.language_model.rotary_emb(torch.zeros(1, 1, HIDDEN), positions)
    eos = end_of_sequence_ids(model)
    metadata = [prompt_tokens, rope_delta, model.config.image_token_id, model.config.video_token_id, len(eos), *eos]
    vision_input.numpy().astype(np.float32).tofile(out / "vision_input.f32")
    boundaries.tofile(out / "image_boundaries.i32")
    ids[0].numpy().astype(np.int32).tofile(out / "prompt_ids.i32")
    np.stack([cos[0, :, :64].numpy(), sin[0, :, :64].numpy()]).astype(np.float32).tofile(out / "prompt_rope.f32")
    np.array(metadata, np.int32).tofile(out / "metadata.i32")
    (out / "prompt.txt").write_text(sample.get("prompt_text", "") + "\n")
    (out / "sample.txt").write_text(sample["name"] + "\n")
    return {"prompt_tokens": prompt_tokens, "images": len(boundaries) - 1, "patches": int(boundaries[-1]),
            "rope_delta": rope_delta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="eval")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = sorted(glob.glob(str(DATA_DIR / "cache" / args.split / "*.pt")))[args.sample]
    model = load_model("cpu", torch.float32)
    sample = load_sample(path, "cpu", torch.float32)
    sizes = write_bundle(model, sample, Path(args.out))
    print(f"{sample['name']}: {sizes['images']} images / {sizes['patches']} patches, prompt {sizes['prompt_tokens']} "
          f"tokens, rope_delta {sizes['rope_delta']} -> {args.out}")


if __name__ == "__main__":
    main()
