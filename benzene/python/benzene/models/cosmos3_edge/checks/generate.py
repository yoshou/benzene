"""End-to-end generation on the board (benzene cosmos3_edge_generate) against PyTorch for eval samples.

  python -m benzene.models.cosmos3_edge.checks.generate bundles --samples 0-7
      # device input bundles (prepare_bundle.py)
  python -m benzene.models.cosmos3_edge.checks.generate fake-quant w8a8_sq05s_gptq --samples 0-7
      # the recipe's greedy answers in PyTorch (GPU)
  python -m benzene.models.cosmos3_edge.checks.generate device w8a8_sq05s_gptq --samples 0-7 [--dec4 dev_w8a8_q4hs]

device pushes the bundles, runs one generate per sample (greedy, at most as many tokens as the BF16 reference answer)
and compares the device's tokens with the BF16 reference answer and with the fake-quant answer of the same recipe
(for --dec4 the mixed recipe): identical answers, mean common-prefix fraction, the first generated token's KL against
the fake-quant first-token distribution, and the stage times the device printed. The model file, its table and the
embedding table must be on the board (paths.BOARD_DIR, from export_hexmodel.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board
from benzene.command_line import Indices

from ..paths import BOARD_DIR, DATA_DIR, OUT_DIR, WORK
from ..recipes import DEVICE_RECIPE

BUNDLE_DIR = WORK / "bundle"
FAKE_QUANT_DIR = OUT_DIR / "fake_quant_answers"
BOARD_BUNDLE_DIR = f"{BOARD_DIR}/bundle"



def eval_files() -> list[str]:
    return sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))


def reference(name: str) -> dict:
    return torch.load(OUT_DIR / "reference" / f"{name}.pt")


def command_bundles(args) -> None:
    from ..prepare_bundle import write_bundle
    from ..runner import load_model, load_sample

    model = load_model("cpu", torch.float32)
    files = eval_files()
    for index in args.samples:
        sample = load_sample(files[index], "cpu", torch.float32)
        sizes = write_bundle(model, sample, BUNDLE_DIR / sample["name"])
        print(f"{index} {sample['name']}: {sizes}", flush=True)


def command_fake_quant(args) -> None:
    from ..evaluate import load_recipe_model, model_dtype
    from ..runner import generate, load_sample

    model, _ = load_recipe_model(args.recipe, args.device)
    out = FAKE_QUANT_DIR / args.recipe
    out.mkdir(parents=True, exist_ok=True)
    files = eval_files()
    for index in args.samples:
        sample = load_sample(files[index], args.device, model_dtype(args.recipe))
        answer_tokens = len(reference(sample["name"])["answer_ids"])
        logits = first_token_logits(model, sample["inputs"])
        answer = generate(model, sample["inputs"], answer_tokens)
        torch.save({"answer_ids": answer, "first_logits": logits.cpu()}, out / f"{sample['name']}.pt")
        print(f"{index} {sample['name']}: {len(answer)} tokens", flush=True)


@torch.inference_mode()
def first_token_logits(model, inputs: dict) -> torch.Tensor:
    model.model.rope_deltas = None
    return model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()


def run_on_board(model: str, bundle: str, out: str, max_new_tokens: int, decode_4bit: bool) -> str:
    command = board.benzene_command(f"cosmos3_edge_generate {model}.hexmodel {bundle} {out} --max-new-tokens "
                                    f"{max_new_tokens}" + (" --dec4" if decode_4bit else ""))
    result = board.ssh(f"rm -rf {out} && mkdir -p {out} && {command}")
    if result.returncode != 0:
        raise RuntimeError(f"board run failed: {result.stdout} {result.stderr}")
    return result.stdout


def common_prefix(a: list[int], b: list[int]) -> int:
    count = 0
    while count < min(len(a), len(b)) and a[count] == b[count]:
        count += 1
    return count


def command_device(args) -> None:
    from ..runner import load_processor

    tokenizer = load_processor().tokenizer
    model = f"{BOARD_DIR}/cosmos3_edge-{args.recipe}"
    fake_quant_recipe = args.dec4 or args.recipe
    files = eval_files()
    rows = []
    for index in args.samples:
        name = Path(files[index]).stem
        reference_ids = reference(name)["answer_ids"].tolist()
        fake_quant = torch.load(FAKE_QUANT_DIR / fake_quant_recipe / f"{name}.pt")
        fake_quant_ids = fake_quant["answer_ids"].tolist()
        board.upload(f"{BUNDLE_DIR / name}/", f"{BOARD_BUNDLE_DIR}/{name}/")
        with tempfile.TemporaryDirectory() as directory:
            board_out = f"{BOARD_DIR}/out/{name}"
            output = run_on_board(model, f"{BOARD_BUNDLE_DIR}/{name}", board_out, len(reference_ids), bool(args.dec4))
            board.download(f"{board_out}/", f"{directory}/")
            device_ids = np.fromfile(Path(directory) / "generated_ids.i32", np.int32).tolist()
            device_logits = torch.from_numpy(np.fromfile(Path(directory) / "first_logits.f32", np.float32))
        p = torch.log_softmax(fake_quant["first_logits"].double(), -1)
        q = torch.log_softmax(device_logits.double(), -1)
        first_kl = float((p.exp() * (p - q)).sum())
        timing = re.search(r"vision ([\d.]+) s.*prefill ([\d.]+) s \((\d+) tokens\).*decode (\d+) tokens ([\d.]+) s "
                           r"\(([\d.]+) ms/token.*total ([\d.]+) s", output)
        row = {"sample": name, "tokens": len(device_ids), "same_as_fake_quant": device_ids == fake_quant_ids,
               "same_as_bf16": device_ids == reference_ids,
               "prefix_fake_quant": common_prefix(device_ids, fake_quant_ids) / max(len(fake_quant_ids), 1),
               "prefix_bf16": common_prefix(device_ids, reference_ids) / max(len(reference_ids), 1),
               "first_token_kl": first_kl}
        if timing:
            row.update(vision=float(timing.group(1)), prefill=float(timing.group(2)), prompt=int(timing.group(3)),
                       decode_ms_per_token=float(timing.group(6)), total=float(timing.group(7)))
        rows.append(row)
        print(f"{index} {name}: {row['tokens']} tokens, same as fake-quant {row['same_as_fake_quant']} (prefix "
              f"{row['prefix_fake_quant']:.2f}), as BF16 {row['same_as_bf16']} (prefix {row['prefix_bf16']:.2f}), "
              f"first-token KL {first_kl:.2e} | {output.strip().splitlines()[-1][:160]}", flush=True)
        if args.show:
            print("  device:", repr(tokenizer.decode(device_ids)[:300]))
            print("  BF16:  ", repr(tokenizer.decode(reference_ids)[:300]))
    summary = {"samples": len(rows), "same_as_fake_quant": sum(r["same_as_fake_quant"] for r in rows),
               "same_as_bf16": sum(r["same_as_bf16"] for r in rows),
               "prefix_fake_quant": float(np.mean([r["prefix_fake_quant"] for r in rows])),
               "prefix_bf16": float(np.mean([r["prefix_bf16"] for r in rows])),
               "first_token_kl_mean": float(np.mean([r["first_token_kl"] for r in rows]))}
    tag = args.recipe + (f"_dec4_{args.dec4}" if args.dec4 else "")
    out = OUT_DIR / "results" / f"device_generate_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary), "->", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["bundles", "fake-quant", "device"])
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--samples", nargs="+", action=Indices, default=list(range(8)), help="eval sample indices")
    parser.add_argument("--dec4", metavar="RECIPE", help="decode on the 4-bit weights (the mixed recipe's answers)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show", action="store_true", help="print the device and BF16 answers")
    args = parser.parse_args()
    {"bundles": command_bundles, "fake-quant": command_fake_quant, "device": command_device}[args.command](args)


if __name__ == "__main__":
    main()
