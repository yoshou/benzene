"""Paths of the XFeat + LightGlue port: its work directory and its directory on the board (the LightGlue checkout and
the evaluation pairs are shared: benzene.models.lightglue.paths)."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("XFEAT_LIGHTGLUE_WORK", workspace.WORK / "xfeat_lightglue"))
SRC_DIR = WORK / "src"  # the verlab/accelerated_features checkout (commit e92685f): XFeat's code and weights
WEIGHTS = SRC_DIR / "weights"  # xfeat.pt, xfeat-lighterglue.pt
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("xfeat_lightglue")  # BOARD_DIR: model files on the board
