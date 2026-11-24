"""Evaluation of LoMa recipes on the evaluation pairs (benzene.models.lightglue.prepare_data --split eval, with their
colour frames: runner.load_pair); the metrics and commands: benzene.models.lightglue.evaluation.

A recipe is "<name>[:<settings>]": name fp32 (the reference network), bf16 (it in autocast bf16, as LoMa runs by
default) or a quantization recipe of recipes.py ("<cnn>+<dino>+<matcher>", "fp16"); settings a runner.Settings name
(det<size>-desc<size>-k<keypoints>-<color>, e.g. det1024-desc784x784-k2048-rgb), by default SETTINGS - the
reference's (the fp32 outputs every recipe's deviation columns compare with).

  python -m benzene.models.loma.evaluate ref
  python -m benzene.models.loma.evaluate run fp32:detnative-descnative14-k2048-rgb bf16 fq+w8a8+fp16
  python -m benzene.models.loma.evaluate rescore <name> --seeds 5
  python -m benzene.models.loma.evaluate table
"""

from __future__ import annotations

import re
import sys

import torch

from ..lightglue.evaluation import AccuracyEvaluation
from .paths import OUT_DIR
from .runner import Settings, load_model, load_pair, run_pair

# the device's settings: DaD at LoMa's 1024 long side, DeDoDe-G at the image size rounded to multiples of 14, gray
# images, 2048 keypoints
SETTINGS = Settings(1024, "native14", 2048, "gray")


def parse_settings(name: str) -> Settings:
    m = re.fullmatch(r"det(native|\d+)-desc(native14|native56|\d+x\d+)-k(\d+)-(rgb|gray)", name)
    if m is None:
        raise ValueError(f"settings {name!r}: det<native|size>-desc<native14|native56|WxH>-k<n>-<rgb|gray>")
    detector = m.group(1) if m.group(1) == "native" else int(m.group(1))
    descriptor = m.group(2) if m.group(2).startswith("native") else tuple(int(v) for v in m.group(2).split("x"))
    return Settings(detector, descriptor, int(m.group(3)), m.group(4))


def parse_recipe(recipe: str | None) -> tuple[str, Settings]:
    if recipe is None:
        return "fp32", SETTINGS
    name, _, settings = recipe.partition(":")
    return name, parse_settings(settings) if settings else SETTINGS


def runner(recipe: str | None, device: str):
    """The function running one pair with a recipe (None: the fp32 reference)."""
    name, settings = parse_recipe(recipe)
    net = load_model(device=device)
    if name in ("fp32", "bf16"):
        ac = None if name == "fp32" else torch.bfloat16
        return lambda rec: run_pair(net, rec, settings, device, ac, ac)
    from .recipes import apply_recipe

    model, extract_ac, match_ac = apply_recipe(net, name, settings, device)
    return lambda rec: run_pair(model, rec, settings, device, extract_ac, match_ac)


accuracy_evaluation = AccuracyEvaluation(OUT_DIR, load_pair, runner)

if __name__ == "__main__":
    sys.exit(accuracy_evaluation.main())
