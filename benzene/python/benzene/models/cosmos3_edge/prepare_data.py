"""Calibration and evaluation samples of the Cosmos 3 Edge accuracy evaluation as model inputs:
DATA_DIR/cache/<split>/<name>.pt.

Splits (the image sources are split by scene, so no image or clip of calib appears in eval):
  calib  128 prompts - 72 single images (TUM RGB-D frames, Mip-NeRF 360 views) with description / spatial / counting /
         planning / safety instructions, 24 TUM clips as videos (8 frames, 3.75 fps), 32 text-only prompts; a quarter
         of each kind with thinking enabled (the chat template's <think> block left open)
  eval   224 prompts of the same kinds from the other scenes (96 images, 32 clips, 32 text) plus 32 CV-Bench and 32
         RealWorldQA questions (fidelity against the BF16 reference: evaluate.py ref / run)
  bench  all of CV-Bench (2638 multiple-choice questions) and RealWorldQA (765) with their ground truth, without
         thinking (accuracy: evaluate.py bench)
Images and frames are resized to the device budget (DEVICE_IMAGE_MAX_PIXELS: at most 2048 patches each).
A record holds the processor outputs (pixel values as their uint8 codes) and what the prompt was.

  python -m benzene.models.cosmos3_edge.prepare_data --split calib
  python -m benzene.models.cosmos3_edge.prepare_data --split eval
  python -m benzene.models.cosmos3_edge.prepare_data --split bench
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from benzene.workspace import CV_BENCH, MIPNERF360, REALWORLDQA, TUM_RGBD

from .paths import DATA_DIR
from .runner import PIXEL_KEYS, load_processor

# The device's vision encoder runs an image or a video frame as at most 2048 patches of 16 x 16 pixels
# (dsp/models/cosmos3_edge/cosmos3_edge.cpp, kGroupPatches); the evaluation resizes every image to that budget.
DEVICE_IMAGE_MAX_PIXELS = 2048 * 16 * 16
IMAGE_MIN_PIXELS = 256 * 256  # the checkpoint processor's default

TUM_SCENES = {
    "calib": ["freiburg1_360", "freiburg1_desk", "freiburg1_plant", "freiburg2_desk", "freiburg3_structure_texture_far"],
    "eval": ["freiburg1_desk2", "freiburg1_room", "freiburg1_teddy", "freiburg3_long_office_household"],
}
MIPNERF_SCENES = {
    "calib": ["bicycle", "bonsai", "counter", "flowers", "garden"],
    "eval": ["kitchen", "room", "stump", "treehill"],
}
COUNTS = {  # images, videos, text prompts, benchmark questions (per benchmark)
    "calib": (72, 24, 32, 0),
    "eval": (96, 32, 32, 32),
}
VIDEO_FRAMES = 8
VIDEO_FRAME_STRIDE = 8  # TUM records at 30 fps: every 8th frame = 3.75 fps
TUM_FPS = 30.0

IMAGE_INSTRUCTIONS = [
    "Describe this image in detail.",
    "Write a one-sentence caption for this image.",
    "List the objects you can see and where they are in the image.",
    "What is the closest object to the camera, and what is the farthest?",
    "How many chairs are visible in this image? Explain briefly.",
    "If a robot arm had to tidy up this scene, what steps should it take?",
    "Is there anything in this scene that could be a safety hazard? Explain.",
    "Describe the layout of this space as if guiding a blind person through it.",
    "What could a person do in this place? Give three activities.",
    "What materials are the main surfaces made of?",
    "Estimate the distance between the camera and the main object in the scene.",
    "What is the lighting like in this image, and where does the light come from?",
    "Which object would be easiest for a robot gripper to pick up? Why?",
    "Describe the colors and textures in this image.",
    "What is the main object in the image, and what is it used for?",
    "Is the camera looking up, down, or straight ahead? How can you tell?",
    "Point out any text, labels or screens visible in the image.",
    "What time of day does it appear to be, and why?",
    "Plan a path for a mobile robot to cross this space without collisions.",
    "Give a short inventory of the items on the main surface.",
]
VIDEO_INSTRUCTIONS = [
    "Describe what happens in this video.",
    "How does the camera move in this video?",
    "List the objects that appear in the video in the order they appear.",
    "Summarize the scene shown in this video in two sentences.",
    "Is the camera operator walking, turning, or standing still? Explain.",
    "What would a robot need to know to navigate the space shown in this video?",
]
TEXT_PROMPTS = [
    "Explain why a glass of ice water gets wet on the outside.",
    "A robot must stack three boxes of different sizes. In what order should it stack them and why?",
    "What is the difference between a two-finger gripper and a suction cup gripper?",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a car travels at 72 km/h, how many meters does it cover in 5 seconds?",
    "List five safety rules for working near an industrial robot.",
    "Explain the concept of center of mass to a ten-year-old.",
    "Why do objects fall at the same rate in a vacuum?",
    "Describe how to make a cup of tea step by step.",
    "What is inverse kinematics?",
    "A ball rolls off a table 1 m high. Roughly how long does it take to hit the floor?",
    "Summarize, in three sentences, a story about a lost robot finding its way home.",
    "What are the pros and cons of lidar compared to cameras for robots?",
    "Translate 'The box is on the left side of the shelf' into French and Japanese.",
    "Explain what a PID controller does.",
    "How should a warehouse robot handle a person suddenly stepping into its path?",
    "What happens to water when it freezes, and why can it break pipes?",
    "Give three examples of levers in everyday life.",
    "Write a haiku about a quiet factory at night.",
    "What is the difference between precision and accuracy?",
    "A shelf holds 12 boxes. A robot removes a third of them and then adds 5. How many boxes are on the shelf?",
    "Explain friction and give an example where it is useful.",
    "Describe the steps to replace a flat bicycle tire.",
    "Why is it harder to balance on a bicycle when it is stopped?",
    "What is a digital twin in manufacturing?",
    "Explain how a camera lens focuses light.",
    "Write a SQL query that counts orders per customer from a table orders(customer_id, order_id).",
    "What should a cleaning robot do when its battery is low?",
    "Compare walking robots and wheeled robots for use in homes.",
    "Explain the difference between weight and mass.",
    "How does a refrigerator keep food cold?",
    "Give a short checklist for safely lifting a heavy box.",
    "What is torque? Give an everyday example.",
    "Describe how bees communicate the location of food.",
    "If you pour water into a tilted glass, what shape does the surface take and why?",
    "Write a short email asking a colleague to review a report by Friday.",
    "What is the role of a gyroscope in a smartphone?",
    "Explain why ships made of steel float.",
    "A robot arm has two links of 30 cm and 20 cm. What is the farthest distance it can reach from its base?",
    "List the main parts of a car's braking system.",
    "What is the difference between supervised learning and reinforcement learning?",
    "How would you teach a robot to fold a towel?",
    "Explain how traffic lights at an intersection are typically coordinated.",
    "Why do we see lightning before we hear thunder?",
    "Give three tips for taking a sharp photo in low light.",
    "What does a thermostat do?",
    "Explain the water cycle briefly.",
    "Write a JavaScript function that checks whether a string is a palindrome.",
    "How can a drone keep a stable position in windy conditions?",
    "What are the main causes of falls of elderly people at home, and how can sensors help?",
    "Explain static and dynamic friction with the example of pushing a box.",
    "Plan a simple morning routine for a household robot.",
    "What is the boiling point of water at high altitude, and why does it change?",
    "Describe how a conveyor belt sorting system might detect defective items.",
    "If three gears are meshed in a row and the first turns clockwise, which way does the third turn?",
    "Explain what a convolutional neural network is in simple terms.",
    "What precautions should be taken when charging lithium-ion batteries?",
    "How do noise-cancelling headphones work?",
    "Give an example of a situation where a robot should ask a human for help.",
    "What is the difference between a map and a floor plan?",
    "Why does a spinning top stay upright?",
    "Write the opening paragraph of a short story set in a smart city.",
    "How does GPS determine a position?",
    "What is the purpose of a fuse in an electrical circuit?",
]
CV_BENCH_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def image_size_kwargs() -> dict:
    return {"size": {"shortest_edge": IMAGE_MIN_PIXELS, "longest_edge": DEVICE_IMAGE_MAX_PIXELS}}


def video_size_kwargs() -> dict:  # the processor's video budget covers all frames: one frame's budget times frames
    return {"size": {"shortest_edge": 4096, "longest_edge": DEVICE_IMAGE_MAX_PIXELS * VIDEO_FRAMES}}


def encode(processor, content: list[dict], thinking: bool, images=None, videos=None, video_fps=None) -> dict:
    """The chat-templated prompt of one user turn as processor outputs (pixel values as uint8 codes)."""
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking)
    kwargs = {}
    if images:
        kwargs["images"] = images
        kwargs["images_kwargs"] = image_size_kwargs()
    if videos:
        metadata = [{"fps": video_fps, "total_num_frames": len(v), "frames_indices": list(range(len(v)))} for v in videos]
        kwargs["videos"] = videos
        kwargs["videos_kwargs"] = {**video_size_kwargs(), "do_sample_frames": False, "video_metadata": metadata}
    inputs = dict(processor(text=[text], return_tensors="pt", **kwargs))
    for key in PIXEL_KEYS:
        if key in inputs:
            codes = (inputs[key].float() + 1.0) * 127.5
            if not torch.allclose(codes, codes.round(), atol=1e-3):
                raise RuntimeError(f"{key}: pixel values are not on the uint8 grid")
            inputs[key] = codes.round().to(torch.uint8)
    for key in list(inputs):
        if key.endswith("_metadata") or not isinstance(inputs[key], torch.Tensor):
            inputs.pop(key)
    return {"inputs": inputs, "prompt_text": text}


def tum_frames(scene: str) -> list[str]:
    return sorted(glob.glob(str(TUM_RGBD / f"rgbd_dataset_{scene}" / "rgb" / "*.png")))


def mipnerf_views(scene: str) -> list[str]:
    return sorted(glob.glob(str(MIPNERF360 / scene / "images_8" / "*")))


def image_pool(split: str) -> list[tuple[str, str]]:
    """(source name, path) of the split's scenes, every 30th TUM frame and every 10th Mip-NeRF view."""
    pool = []
    for scene in TUM_SCENES[split]:
        pool += [(f"tum_{scene}", p) for p in tum_frames(scene)[::30]]
    for scene in MIPNERF_SCENES[split]:
        pool += [(f"mipnerf_{scene}", p) for p in mipnerf_views(scene)[::10]]
    return pool


def text_prompts(split: str) -> list[str]:
    return TEXT_PROMPTS[:32] if split == "calib" else TEXT_PROMPTS[32:]


def benchmark_items(name: str) -> list[dict]:
    """Questions of a benchmark as {prompt, image bytes, answer, answer kind, task}."""
    items = []
    if name == "cv_bench":
        for part in ("test_2d", "test_3d"):
            frame = pd.read_parquet(CV_BENCH / f"{part}.parquet")
            for row in frame.itertuples():
                items.append({"id": f"cv_bench_{int(row.idx):04d}", "prompt": f"{row.prompt}\n{CV_BENCH_INSTRUCTION}",
                              "image": row.image["bytes"], "answer": row.answer.strip("()"), "answer_kind": "letter",
                              "task": f"{row.type} {row.task}"})
    elif name == "realworldqa":
        frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(REALWORLDQA / "data" / "*.parquet")))]
        for i, row in enumerate(pd.concat(frames).itertuples()):
            answer = str(row.answer).strip()
            kind = "letter" if len(answer) == 1 and answer in "ABCDEF" else "short"
            items.append({"id": f"realworldqa_{i:04d}", "prompt": row.question, "image": row.image["bytes"],
                          "answer": answer, "answer_kind": kind, "task": "RealWorldQA"})
    else:
        raise ValueError(name)
    return items


def write(out_dir: Path, name: str, record: dict) -> None:
    path = out_dir / f"{name}.pt"
    temporary = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(record, temporary)
    os.replace(temporary, path)


def prepare_split(split: str, processor) -> None:
    out_dir = DATA_DIR / "cache" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"cosmos3-edge-{split}")
    image_count, video_count, text_count, benchmark_count = COUNTS[split]
    index = 0

    def thinking_for(i: int) -> bool:  # a quarter of each kind
        return i % 4 == 3

    pool = image_pool(split)
    for i, (source, path) in enumerate(rng.sample(pool, image_count)):
        instruction = rng.choice(IMAGE_INSTRUCTIONS)
        record = encode(processor, [{"type": "image"}, {"type": "text", "text": instruction}], thinking_for(i),
                        images=[Image.open(path).convert("RGB")])
        record.update(kind="image", thinking=thinking_for(i), source=source, files=[path], instruction=instruction)
        write(out_dir, f"{index:03d}_image_{source}", record)
        index += 1
    clip_frames = VIDEO_FRAMES * VIDEO_FRAME_STRIDE
    for i in range(video_count):
        scene = TUM_SCENES[split][i % len(TUM_SCENES[split])]
        frames = tum_frames(scene)
        start = rng.randrange(0, len(frames) - clip_frames)
        files = frames[start : start + clip_frames : VIDEO_FRAME_STRIDE]
        video = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
        instruction = rng.choice(VIDEO_INSTRUCTIONS)
        record = encode(processor, [{"type": "video"}, {"type": "text", "text": instruction}], thinking_for(i),
                        videos=[video], video_fps=TUM_FPS / VIDEO_FRAME_STRIDE)
        record.update(kind="video", thinking=thinking_for(i), source=f"tum_{scene}", files=files,
                      instruction=instruction)
        write(out_dir, f"{index:03d}_video_tum_{scene}", record)
        index += 1
    for i, prompt in enumerate(text_prompts(split)[:text_count]):
        record = encode(processor, [{"type": "text", "text": prompt}], thinking_for(i))
        record.update(kind="text", thinking=thinking_for(i), source="text", files=[], instruction=prompt)
        write(out_dir, f"{index:03d}_text", record)
        index += 1
    for benchmark in ("cv_bench", "realworldqa") if benchmark_count else ():
        for item in rng.sample(benchmark_items(benchmark), benchmark_count):
            write(out_dir, f"{index:03d}_{item['id']}", benchmark_record(processor, item))
            index += 1
    print(f"{split}: {index} samples -> {out_dir}", flush=True)


def benchmark_record(processor, item: dict) -> dict:
    image = Image.open(io.BytesIO(item["image"])).convert("RGB")
    record = encode(processor, [{"type": "image"}, {"type": "text", "text": item["prompt"]}], False, images=[image])
    record.update(kind="benchmark", thinking=False, source=item["id"].rsplit("_", 1)[0], files=[],
                  instruction=item["prompt"],
                  benchmark={k: item[k] for k in ("id", "answer", "answer_kind", "task") if k in item})
    return record


def prepare_bench(processor) -> None:
    out_dir = DATA_DIR / "cache" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for benchmark in ("cv_bench", "realworldqa"):
        for item in benchmark_items(benchmark):
            if not (out_dir / f"{item['id']}.pt").exists():
                write(out_dir, item["id"], benchmark_record(processor, item))
            count += 1
            if count % 500 == 0:
                print(f"bench: {count} questions", flush=True)
    print(f"bench: {count} questions -> {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["calib", "eval", "bench"], required=True)
    args = parser.parse_args()
    processor = load_processor()
    if args.split == "bench":
        prepare_bench(processor)
    else:
        prepare_split(args.split, processor)


if __name__ == "__main__":
    main()
