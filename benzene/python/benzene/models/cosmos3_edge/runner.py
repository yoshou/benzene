"""Model loading, cached samples as model inputs, greedy generation and teacher-forced logits of the Cosmos 3 Edge
reasoner (transformers Cosmos3EdgeForConditionalGeneration)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import MODEL_DIR

PIXEL_KEYS = ("pixel_values", "pixel_values_videos")


def load_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(MODEL_DIR))


def load_model(device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    from transformers import Cosmos3EdgeForConditionalGeneration

    model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), dtype=dtype, device_map=device, attn_implementation="sdpa"
    )
    model.eval()
    return model


def load_sample(path: str | Path, device: str, dtype: torch.dtype) -> dict[str, Any]:
    """A cached sample (prepare_data.py) with its model inputs on `device`; pixel values come back from their uint8
    codes (the processor's normalization x / 127.5 - 1 is exact on them)."""
    record = torch.load(path, map_location="cpu", weights_only=False)
    inputs = dict(record["inputs"])
    for key in PIXEL_KEYS:
        if key in inputs:
            inputs[key] = (inputs[key].float() / 127.5 - 1.0).to(dtype)
    record["inputs"] = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    record["name"] = Path(path).stem
    return record


def prompt_length(sample: dict[str, Any]) -> int:
    return int(sample["inputs"]["input_ids"].shape[1])


def _new_sequence(model) -> None:
    model.model.rope_deltas = None  # positions are recomputed for every prompt


@torch.inference_mode()
def generate(model, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> torch.Tensor:
    """Greedy continuation: the new token ids (with the end-of-turn id when the model ended its turn)."""
    _new_sequence(model)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
    )
    return output[0, inputs["input_ids"].shape[1] :].cpu()


@torch.inference_mode()
def answer_logits(model, inputs: dict[str, torch.Tensor], answer_ids: torch.Tensor) -> torch.Tensor:
    """One forward over prompt + answer: the fp32 logits of the positions that predict the answer tokens (the last
    prompt position .. the second-to-last answer token), [answer length, vocabulary]."""
    _new_sequence(model)
    answer = answer_ids.to(inputs["input_ids"].device)[None]
    full = dict(inputs)
    full["input_ids"] = torch.cat([inputs["input_ids"], answer], 1)
    full["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(answer)], 1)
    full["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(answer)], 1)
    output = model(**full, use_cache=False, logits_to_keep=answer.shape[1] + 1)
    return output.logits[0, :-1].float()
