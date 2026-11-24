"""Fidelity of quantization recipes against the BF16 reference, and their benchmark accuracy.

  python -m benzene.models.cosmos3_edge.evaluate ref [--shard 0/2 --device cuda:0]
      # BF16 greedy answers of the eval prompts + top-k log-probs
  python -m benzene.models.cosmos3_edge.evaluate run fp16 w8a8_sq05s_gptq ...
      # recipes, teacher-forced on the reference answers
  python -m benzene.models.cosmos3_edge.evaluate bench bf16 w8a8_sq05s_gptq ...
      # accuracy on CV-Bench / RealWorldQA (greedy, 8 new tokens)
  python -m benzene.models.cosmos3_edge.evaluate bench ... --per-task 200
      # the first 200 questions of every task (id order)
  python -m benzene.models.cosmos3_edge.evaluate table

Fidelity (eval split; every answer position, teacher-forced on the BF16 answer):
  kl      mean KL(BF16 || recipe) per answer token. The reference keeps its 4096 most likely tokens and the rest of
          the vocabulary as one bucket, which makes this a lower bound of the full KL (the bucket's probability: see
          `evaluate.py table`).
  top1    agreement of the recipe's most likely token with the reference's
  prefix  mean fraction of an answer that greedy decoding with the recipe reproduces (up to the first disagreement)
  same    answers reproduced entirely
Rows per sample kind: image, video, text, thinking (any kind with thinking enabled), benchmark (short answers).
Benchmark accuracy: the first option letter of the answer (multiple choice) or the normalized answer (RealWorldQA
short answers) against the ground truth; next to the bf16 run, the questions that flip either way.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .paths import DATA_DIR, OUT_DIR
from .runner import answer_logits, generate, load_model, load_processor, load_sample, prompt_length

REFERENCE_DIR = OUT_DIR / "reference"
RESULT_DIR = OUT_DIR / "results"
TOP_TOKENS = 4096
MAX_NEW_TOKENS = {False: 256, True: 512}  # by thinking
BENCH_NEW_TOKENS = 8
KINDS = ("image", "video", "text", "thinking", "benchmark")


def sample_files(split: str, count: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache" / split / "*.pt")))
    return files[:count] if count else files


def in_shard(index: int, shard: str | None) -> bool:
    if not shard:
        return True
    part, parts = (int(x) for x in shard.split("/"))
    return index % parts == part


def model_dtype(recipe: str) -> torch.dtype:
    return torch.bfloat16 if recipe == "bf16" else torch.float16


def load_recipe_model(recipe: str, device: str):
    from .recipes import apply_recipe

    model = load_model(device, model_dtype(recipe))
    info = apply_recipe(model, recipe)
    return model, info


# ---------------------------------------------------------------------------------------------- reference


def command_reference(args) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.device, torch.bfloat16)
    processor = load_processor()
    for i, path in enumerate(sample_files("eval", args.count)):
        out = REFERENCE_DIR / f"{Path(path).stem}.pt"
        if not in_shard(i, args.shard):
            continue
        existing = torch.load(out) if out.exists() else None
        if existing is not None and existing["top_ids"].shape[1] >= TOP_TOKENS:
            continue
        sample = load_sample(path, args.device, torch.bfloat16)
        if existing is not None:  # an answer kept with fewer top tokens: only the teacher-forced pass again
            answer = existing["answer_ids"]
        else:
            answer = generate(model, sample["inputs"], MAX_NEW_TOKENS[sample["thinking"]])
        log_probs = torch.log_softmax(answer_logits(model, sample["inputs"], answer), -1)
        top = log_probs.topk(TOP_TOKENS, -1)
        text = processor.tokenizer.decode(answer)
        temporary = out.with_suffix(f".tmp{os.getpid()}")  # shards / restarts never see a partial file
        torch.save({"answer_ids": answer, "top_ids": top.indices.int().cpu(), "top_log_probs": top.values.cpu(),
                    "text": text}, temporary)
        os.replace(temporary, out)
        print(f"{i} {Path(path).stem}: {len(answer)} tokens | {text[:100]!r}", flush=True)


# ---------------------------------------------------------------------------------------------- fidelity


def bucketed_kl(reference_top_ids: torch.Tensor, reference_top_log_probs: torch.Tensor,
                log_probs: torch.Tensor) -> torch.Tensor:
    """KL(reference || recipe) per position over the reference's top tokens plus one bucket for the rest."""
    p_top = reference_top_log_probs.float().exp()
    q_top_log = log_probs.gather(-1, reference_top_ids.long())
    kl = (p_top * (reference_top_log_probs.float() - q_top_log)).sum(-1)
    p_rest = (1.0 - p_top.sum(-1)).clamp_min(0.0)
    q_rest = (1.0 - q_top_log.exp().sum(-1)).clamp_min(1e-12)
    rest = torch.where(p_rest > 0, p_rest * (p_rest.clamp_min(1e-12).log() - q_rest.log()), torch.zeros_like(p_rest))
    return kl + rest


def sample_kinds(sample: dict) -> list[str]:
    kinds = [sample["kind"]]
    if sample["thinking"]:
        kinds.append("thinking")
    return kinds


def command_run(args) -> None:
    from benzene.quantization import set_mixed_split

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for recipe in args.recipes:
        model, info = load_recipe_model(recipe, args.device)
        dtype = model_dtype(recipe)
        rows = []
        start = time.time()
        files = sample_files("eval", args.count)
        for i, path in enumerate(files):
            if i % 32 == 0:
                print(f"[{recipe}] {time.strftime('%H:%M:%S')} sample {i}/{len(files)}", flush=True)
            reference = torch.load(REFERENCE_DIR / f"{Path(path).stem}.pt")
            sample = load_sample(path, args.device, dtype)
            set_mixed_split(prompt_length(sample))  # prompt positions run the prefill weights of mixed recipes
            logits = answer_logits(model, sample["inputs"], reference["answer_ids"])
            set_mixed_split(None)
            log_probs = torch.log_softmax(logits, -1)
            top_ids = reference["top_ids"].to(log_probs.device)
            kl = bucketed_kl(top_ids, reference["top_log_probs"].to(log_probs.device), log_probs)
            agree = (log_probs.argmax(-1) == top_ids[:, 0].long()).float()
            disagreements = (agree == 0).nonzero()
            first_disagreement = int(disagreements[0]) if len(disagreements) else len(agree)
            rows.append({"sample": Path(path).stem, "kinds": sample_kinds(sample), "tokens": len(agree),
                         "kl": kl.mean().item(), "kl_max": kl.max().item(), "top1": agree.mean().item(),
                         "prefix": first_disagreement / len(agree), "same": first_disagreement == len(agree)})
        summary = summarize(rows)
        summary.update(recipe=recipe, eval_seconds=time.time() - start, **info)
        (RESULT_DIR / f"{recipe}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print(HEADER)
        print(format_row(summary), flush=True)
        del model
        torch.cuda.empty_cache()


def summarize_rows(rows: list[dict]) -> dict:
    tokens = sum(r["tokens"] for r in rows)
    return {
        "n": len(rows),
        "tokens": tokens,
        "kl": sum(r["kl"] * r["tokens"] for r in rows) / tokens,
        "top1": sum(r["top1"] * r["tokens"] for r in rows) / tokens,
        "prefix": float(np.mean([r["prefix"] for r in rows])),
        "same": int(sum(r["same"] for r in rows)),
    }


def summarize(rows: list[dict]) -> dict:
    summary = summarize_rows(rows)
    summary["by_kind"] = {kind: summarize_rows([r for r in rows if kind in r["kinds"]])
                          for kind in KINDS if any(kind in r["kinds"] for r in rows)}
    return summary


HEADER = (f"{'recipe':22s} {'n':>4s} {'KL':>8s} {'top1':>6s} {'prefix':>6s} {'same':>5s} | "
          + " ".join(f"{k + ' KL/top1':>17s}" for k in KINDS))


def format_row(summary: dict) -> str:
    kinds = " ".join(f"{summary['by_kind'][k]['kl']:8.4f}/{summary['by_kind'][k]['top1']:.3f}"
                     if k in summary["by_kind"] else f"{'-':>17s}" for k in KINDS)
    return (f"{summary['recipe']:22s} {summary['n']:4d} {summary['kl']:8.4f} {summary['top1']:6.3f} "
            f"{summary['prefix']:6.3f} {summary['same']:5d} | {kinds}")


# ---------------------------------------------------------------------------------------------- benchmark accuracy


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def is_correct(prediction: str, benchmark: dict) -> bool:
    if benchmark["answer_kind"] == "letter":
        match = re.search(r"\b([A-F])\b", prediction)
        return match is not None and match.group(1) == benchmark["answer"]
    words = normalize(prediction).split()
    expected = normalize(benchmark["answer"]).split()
    return words[: len(expected)] == expected


def bench_files(per_task: int | None) -> list[str]:
    """The bench split's questions in id order; with per_task, the first per_task of every task."""
    files = sample_files("bench")
    if not per_task:
        return files
    chosen, counts = [], {}
    for path in files:
        task = torch.load(path, weights_only=False)["benchmark"]["task"]
        if counts.get(task, 0) < per_task:
            counts[task] = counts.get(task, 0) + 1
            chosen.append(path)
    return chosen


def command_bench(args) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    processor = load_processor()
    files = bench_files(args.per_task)
    for recipe in args.recipes:
        model, info = load_recipe_model(recipe, args.device)
        dtype = model_dtype(recipe)
        rows = []
        start = time.time()
        for i, path in enumerate(files):
            if not in_shard(i, args.shard):
                continue
            if len(rows) % 200 == 0:
                print(f"[{recipe} bench] {time.strftime('%H:%M:%S')} question {i}/{len(files)}", flush=True)
            sample = load_sample(path, args.device, dtype)
            answer = generate(model, sample["inputs"], BENCH_NEW_TOKENS)
            prediction = processor.tokenizer.decode(answer, skip_special_tokens=True)
            benchmark = sample["benchmark"]
            rows.append({"id": benchmark["id"], "task": benchmark["task"], "prediction": prediction,
                         "correct": is_correct(prediction, benchmark)})
        shard = (args.shard or "0/1").replace("/", "of")
        out = RESULT_DIR / f"bench_{recipe}_{shard}.json"
        out.write_text(json.dumps({"recipe": recipe, "seconds": time.time() - start, "rows": rows, **info}, indent=1))
        print(f"[{recipe} bench] {sum(r['correct'] for r in rows)}/{len(rows)} correct -> {out}", flush=True)
        del model
        torch.cuda.empty_cache()


def bench_rows(recipe: str) -> dict[str, dict]:
    rows = {}
    for path in sorted(RESULT_DIR.glob(f"bench_{recipe}_*.json")):
        for row in json.loads(path.read_text())["rows"]:
            rows[row["id"]] = row
    return rows


def print_bench_table() -> None:
    recipes = sorted({re.sub(r"^bench_(.*)_\d+of\d+\.json$", r"\1", p.name) for p in RESULT_DIR.glob("bench_*.json")})
    if not recipes:
        return
    reference = bench_rows("bf16")
    print(f"\n{'recipe':22s} {'n':>5s} {'CV-Bench':>9s} {'2D':>6s} {'3D':>6s} {'RWQA':>6s} {'all':>6s}  vs bf16")
    for recipe in recipes:
        rows = bench_rows(recipe)

        def accuracy(select) -> str:
            chosen = [r["correct"] for r in rows.values() if select(r)]
            return f"{100 * np.mean(chosen):6.2f}" if chosen else f"{'-':>6s}"

        flips = ""
        if recipe != "bf16" and reference:
            common = [i for i in rows if i in reference]
            gained = sum(rows[i]["correct"] and not reference[i]["correct"] for i in common)
            lost = sum(reference[i]["correct"] and not rows[i]["correct"] for i in common)
            flips = f"  +{gained} -{lost} of {len(common)}"
        print(f"{recipe:22s} {len(rows):5d} {accuracy(lambda r: r['id'].startswith('cv_bench')):>9s} "
              f"{accuracy(lambda r: r['task'].startswith('2D')):>6s} {accuracy(lambda r: r['task'].startswith('3D'))} "
              f"{accuracy(lambda r: r['id'].startswith('realworldqa'))} {accuracy(lambda r: True)}{flips}")


def print_reference_tail() -> None:
    files = sorted(REFERENCE_DIR.glob("*.pt"))
    if not files:
        return
    rest = torch.cat([1 - torch.load(f)["top_log_probs"].float().exp().sum(-1) for f in files])
    print(f"reference: {len(files)} answers, {len(rest)} positions; probability outside the top {TOP_TOKENS} tokens: "
          f"< 1e-3 at {100 * (rest < 1e-3).float().mean():.1f}% of positions, < 1e-2 at "
          f"{100 * (rest < 1e-2).float().mean():.1f}%, max {rest.max():.4f}")


def command_table(args) -> None:
    print_reference_tail()
    print(HEADER)
    for path in sorted(RESULT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if path.name.startswith(("bench_", "device_")):  # benchmark accuracy / checks/generate.py results
            continue
        print(format_row(json.loads(path.read_text())["summary"]))
    print_bench_table()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ref", "run", "bench", "table"])
    parser.add_argument("recipes", nargs="*")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", default=None, help="k/n: only samples with index %% n == k (ref, bench)")
    parser.add_argument("--per-task", type=int, default=None, help="bench: the first N questions of every task")
    args = parser.parse_args()
    {"ref": command_reference, "run": command_run, "bench": command_bench, "table": command_table}[args.command](args)
    sys.stdout.flush()
    os._exit(0)  # skip ROCm teardown (it can segfault after all work is saved)


if __name__ == "__main__":
    main()
