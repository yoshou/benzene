"""Calibration statistics: BF16 answers of the calibration prompts, per-Linear Hessians (X^T X) and input absmax.

The model runs on --device; Hessians accumulate in fp32 on --hessian-device in passes (module groups) that fit its
memory, then go to OUT_DIR/hessians/<module>.pt. Every calibration prompt runs teacher-forced with its BF16 answer, so
the statistics cover prompt (prefill) and generated (decode) positions. q/k/v projections share their input, so only
q_proj's Hessian is stored.

  python -m benzene.models.cosmos3_edge.calibrate answers [--shard 0/2 --device cuda:0]
      # BF16 greedy answers of the calibration prompts
  python -m benzene.models.cosmos3_edge.calibrate hessian --group vision
  python -m benzene.models.cosmos3_edge.calibrate hessian --group llm_low
  python -m benzene.models.cosmos3_edge.calibrate hessian --group llm_high
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import torch

from .paths import DATA_DIR, OUT_DIR
from .runner import answer_logits, generate, load_model, load_sample

HESSIAN_DIR = OUT_DIR / "hessians"
ANSWER_DIR = OUT_DIR / "calibration_answers"
MAX_NEW_TOKENS = {False: 256, True: 512}  # by thinking

GROUPS = {
    "vision": r"^model\.(visual\.encoder|projector)\.",
    "llm_low": r"^model\.language_model\.layers\.([0-9]|1[0-3])\.",
    "llm_high": r"^(model\.language_model\.layers\.(1[4-9]|2[0-7])\.|lm_head$)",
}


def hessian_key(name: str) -> str:
    """k/v projections share q_proj's input."""
    return re.sub(r"\.(k_proj|v_proj)$", ".q_proj", name)


def calibration_files(count: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache/calib/*.pt")))
    return files[:count] if count else files


def in_shard(index: int, shard: str | None) -> bool:
    if not shard:
        return True
    part, parts = (int(x) for x in shard.split("/"))
    return index % parts == part


def load_answer(name: str) -> torch.Tensor:
    return torch.load(ANSWER_DIR / f"{name}.pt")["answer_ids"]


def command_answers(args) -> None:
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.device, torch.bfloat16)
    for i, path in enumerate(calibration_files(args.count)):
        name = Path(path).stem
        out = ANSWER_DIR / f"{name}.pt"
        if out.exists() or not in_shard(i, args.shard):
            continue
        sample = load_sample(path, args.device, torch.bfloat16)
        answer = generate(model, sample["inputs"], MAX_NEW_TOKENS[sample["thinking"]])
        temporary = out.with_suffix(f".tmp{os.getpid()}")
        torch.save({"answer_ids": answer}, temporary)
        os.replace(temporary, out)
        print(f"{i} {name}: {len(answer)} tokens", flush=True)


def command_hessian(args) -> None:
    model = load_model(args.device, torch.bfloat16)
    pattern = re.compile(GROUPS[args.group])
    HESSIAN_DIR.mkdir(parents=True, exist_ok=True)
    accumulators: dict[str, dict] = {}

    def hook(key):
        def accumulate(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(args.hessian_device, torch.float32)
            accumulator = accumulators[key]
            accumulator["H"].addmm_(x.T, x)
            accumulator["rows"] += x.shape[0]
            accumulator["amax"] = torch.maximum(accumulator["amax"], x.abs().amax(0))

        return accumulate

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and pattern.search(name):
            key = hessian_key(name)
            if key != name:
                continue
            k = module.in_features
            accumulators[key] = {"H": torch.zeros(k, k, device=args.hessian_device), "rows": 0,
                                 "amax": torch.zeros(k, device=args.hessian_device)}
            module.register_forward_hook(hook(key))
    gib = sum(a["H"].numel() * 4 for a in accumulators.values()) / 2**30
    print(f"{args.group}: {len(accumulators)} Hessians, {gib:.1f} GiB on {args.hessian_device}", flush=True)
    start = time.time()
    for i, path in enumerate(calibration_files(args.count)):
        sample = load_sample(path, args.device, torch.bfloat16)
        answer_logits(model, sample["inputs"], load_answer(sample["name"]))
        if i % 16 == 0:
            print(f"  {i} samples {time.time() - start:.0f} s", flush=True)
    for key, accumulator in accumulators.items():
        torch.save({"H": (accumulator["H"] / accumulator["rows"]).cpu(), "n": accumulator["rows"],
                    "amax": accumulator["amax"].cpu()}, HESSIAN_DIR / f"{key}.pt")
    (HESSIAN_DIR / f".done_{args.group}").write_text(str(len(calibration_files(args.count))))
    print(f"saved {len(accumulators)} to {HESSIAN_DIR} ({time.time() - start:.0f} s)", flush=True)


def command_activation_range(args) -> None:
    """Per-channel min/max of every Linear input (static activation ranges for the HMX int8 static recipes)."""
    model = load_model(args.device, torch.bfloat16)
    ranges: dict[str, list] = {}

    def hook(name):
        def record(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            low, high = x.amin(0), x.amax(0)
            if name in ranges:
                ranges[name][0] = torch.minimum(ranges[name][0], low)
                ranges[name][1] = torch.maximum(ranges[name][1], high)
            else:
                ranges[name] = [low, high]

        return record

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook(name))
    for path in calibration_files(args.count):
        sample = load_sample(path, args.device, torch.bfloat16)
        answer_logits(model, sample["inputs"], load_answer(sample["name"]))
    torch.save({k: (v[0].cpu(), v[1].cpu()) for k, v in ranges.items()}, OUT_DIR / "activation_range.pt")
    print(f"saved ranges of {len(ranges)} Linear inputs", flush=True)


_ACTIVATION_RANGE: dict | None = None


def load_actrange(name: str) -> tuple:
    global _ACTIVATION_RANGE
    if _ACTIVATION_RANGE is None:
        _ACTIVATION_RANGE = torch.load(OUT_DIR / "activation_range.pt")
    return _ACTIVATION_RANGE[name]


def load_hessian(name: str, device: str) -> dict:
    return torch.load(HESSIAN_DIR / f"{hessian_key(name)}.pt", map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["answers", "hessian", "activation_range"])
    parser.add_argument("--group", choices=list(GROUPS))
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--shard", default=None, help="k/n: answers only for samples with index %% n == k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hessian-device", default="cuda:1")
    args = parser.parse_args()
    {"answers": command_answers, "hessian": command_hessian, "activation_range": command_activation_range}[
        args.command](args)
    sys.stdout.flush()
    os._exit(0)  # skip ROCm teardown (it can segfault after all work is saved)


if __name__ == "__main__":
    main()
