"""Language model prefill + teacher-forced decode on the DSP (benzene cosmos3_edge_llm_decode) against the BF16
reference answer.

The layer-0 input rows of the capture (capture_activations.py: the prompt and the BF16 reference answer of one eval
sample, embeddings with the vision tokens merged) and their M-RoPE tables drive the device: the prompt is prefilled,
then each answer token is decoded on the session KV cache. The device logits are compared with the reference's top-k
log-probabilities (KL with the rest of the vocabulary as one bucket, top-1 agreement; evaluate.bucketed_kl), next to the
recipe's fake-quant numbers for the same sample (results/<recipe>.json of evaluate.py run).

  python -m benzene.models.cosmos3_edge.checks.llm_decode w8a8_sq05s_gptq
  python -m benzene.models.cosmos3_edge.checks.llm_decode w8a8_sq05s_gptq --dec4 dev_w8a8_q4hs
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board

from ..evaluate import bucketed_kl
from ..paths import BOARD_DIR, OUT_DIR
from ..recipes import DEVICE_RECIPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--dec4", metavar="RECIPE", help="decode on the model file's 4-bit weights (benzene --dec4); "
                                                         "RECIPE = the mixed recipe they were exported from")
    args = parser.parse_args()
    capture = torch.load(OUT_DIR / "acts" / args.recipe / "model.language_model.layers.0.layer.pt")
    sample = capture["sample"]
    reference = torch.load(OUT_DIR / "reference" / f"{sample}.pt")
    answer_tokens = len(reference["answer_ids"])
    rows = capture["x"].shape[0]
    prompt_tokens = rows - answer_tokens
    x = capture["x"].numpy().astype(np.float32)
    rope = np.concatenate([capture["cos"][:, :64].numpy(), capture["sin"][:, :64].numpy()]).astype(np.float32)
    model = f"{BOARD_DIR}/cosmos3_edge-{args.recipe}"
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory)
        (local / "bundle").mkdir()
        (local / "bundle" / "x.f32").write_bytes(x.tobytes())
        (local / "bundle" / "rope.f32").write_bytes(rope.tobytes())
        (local / "bundle" / "meta.txt").write_text(f"{prompt_tokens} {answer_tokens}\n")
        work = board.empty_directory("pr")
        board.upload(f"{directory}/bundle/", f"{work}/bundle/")
        result = board.run_benzene(f"cosmos3_edge_llm_decode {model}.hexmodel pr/bundle pr/out" +
                                   (" --dec4" if args.dec4 else ""))
        print(result.stdout.strip(), result.stderr.strip())
        if result.returncode:
            sys.exit(f"board run failed ({result.returncode})")
        board.download(f"{work}/out/logits.f32", f"{directory}/")
        logits = np.frombuffer((local / "logits.f32").read_bytes(), np.float32).reshape(answer_tokens, -1)
    log_probs = torch.log_softmax(torch.from_numpy(logits.astype(np.float64)), -1)
    kl = bucketed_kl(reference["top_ids"], reference["top_log_probs"].double(), log_probs)
    top1 = (log_probs.argmax(-1) == reference["top_ids"][:, 0].long()).double().mean().item()
    evaluated = {row["sample"]: row
             for row in json.loads((OUT_DIR / "results" / f"{args.dec4 or args.recipe}.json").read_text())["rows"]}
    fake_quant = evaluated.get(sample, {})
    print(f"sample {sample}: prompt {prompt_tokens} tokens, answer {answer_tokens} tokens; device vs BF16 reference: "
          f"KL {kl.mean().item():.4f} (first token {kl[0].item():.4f}), top1 {top1:.3f}; PyTorch fake-quant: KL "
          f"{fake_quant.get('kl', float('nan')):.4f}, top1 {fake_quant.get('top1', float('nan')):.3f}")


if __name__ == "__main__":
    main()
