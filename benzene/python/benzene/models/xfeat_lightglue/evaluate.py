"""Evaluation of XFeat + LightGlue recipes on the evaluation pairs (benzene.models.lightglue.prepare_data --split eval,
cropped by runner.load_pair); the metrics and commands: benzene.models.lightglue.evaluation.

  python -m benzene.models.xfeat_lightglue.evaluate ref
      # fp32 outputs of every pair -> OUT_DIR/ref
  python -m benzene.models.xfeat_lightglue.evaluate run fp16 dev+fp16
      # recipes -> OUT_DIR/results/<recipe>.json, matches -> OUT_DIR/outs
  python -m benzene.models.xfeat_lightglue.evaluate rescore <name> --seeds 5
  python -m benzene.models.xfeat_lightglue.evaluate table
"""

from __future__ import annotations

import sys

from ..lightglue.evaluation import AccuracyEvaluation, recipe_runner
from .paths import OUT_DIR
from .recipes import apply_recipe, autocast_dtypes
from .runner import load_models, load_pair, run_pair

accuracy_evaluation = AccuracyEvaluation(OUT_DIR, load_pair,
                                         recipe_runner(load_models, run_pair, apply_recipe, autocast_dtypes))

if __name__ == "__main__":
    sys.exit(accuracy_evaluation.main())
