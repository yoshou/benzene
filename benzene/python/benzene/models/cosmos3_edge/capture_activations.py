"""Capture the real inputs / fake-quant outputs the on-device checks (checks/) compare with.

Runs one evaluation sample teacher-forced on its BF16 reference answer through the model with a recipe applied and
saves, per whole decoder / vision layer of LAYERS, its input, output and (language model) M-RoPE cos / sin or (vision)
image boundaries, and the vision post_layernorm input and the projector output (vision_tail).

  python -m benzene.models.cosmos3_edge.capture_activations w8a8_sq05s_gptq --device cpu --sample 0
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from .paths import DATA_DIR, OUT_DIR
from .recipes import DEVICE_RECIPE
from .runner import answer_logits, load_model, load_sample

# the language model's first and last layers (checks/llm_prefill.py, llm_decode.py), the first vision layer
# (checks/vision.py)
LAYERS = ["model.language_model.layers.0", "model.language_model.layers.27", "model.visual.encoder.layers.0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", type=int, default=0, help="index in the eval split")
    parser.add_argument("--out", default=str(OUT_DIR / "acts"))
    args = parser.parse_args()
    from .recipes import apply_recipe

    torch.set_num_threads(32)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = load_model(args.device, dtype)
    apply_recipe(model, args.recipe)
    path = sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))[args.sample]
    reference = torch.load(OUT_DIR / "reference" / (Path(path).stem + ".pt"))
    sample = load_sample(path, args.device, dtype)

    layer_captures: dict[str, dict] = {}

    def layer_hook(name):
        def capture(module, positional, keywords, output):
            if name in layer_captures:
                return
            hidden = positional[0] if positional else keywords["hidden_states"]
            result = output[0] if isinstance(output, tuple) else output
            record = {"x": hidden.detach().float().cpu(), "y": result.detach().float().cpu()}
            if hidden.dim() == 3:  # language model layers: [1, tokens, hidden] and the M-RoPE cos / sin
                cos, sin = (t.detach().float().cpu() for t in keywords["position_embeddings"])
                record = {k: v[0] for k, v in record.items()}
                record.update(cos=cos[0], sin=sin[0])
            if "cu_seqlens" in keywords or len(positional) > 1:  # vision layers: packed images
                record["cu_seqlens"] = (keywords["cu_seqlens"] if "cu_seqlens" in keywords else positional[1]).cpu()
            layer_captures[name] = record

        return capture

    hooks = [model.get_submodule(name).register_forward_hook(layer_hook(name), with_kwargs=True) for name in LAYERS]

    def tail_input(module, inputs, output):
        layer_captures.setdefault("vision_tail", {})["x"] = inputs[0].detach().float().cpu()

    def tail_output(module, inputs, output):
        layer_captures.setdefault("vision_tail", {})["y"] = output.detach().float().cpu()

    hooks.append(model.get_submodule("model.visual.post_layernorm").register_forward_hook(tail_input))
    hooks.append(model.get_submodule("model.projector").register_forward_hook(tail_output))
    answer_logits(model, sample["inputs"], reference["answer_ids"])
    for hook in hooks:
        hook.remove()
    out = Path(args.out) / args.recipe
    out.mkdir(parents=True, exist_ok=True)
    for name, record in layer_captures.items():
        record["sample"] = sample["name"]
        torch.save(record, out / f"{name}.layer.pt")
        print(name, "layer", {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in record.items()}, flush=True)


if __name__ == "__main__":
    main()
