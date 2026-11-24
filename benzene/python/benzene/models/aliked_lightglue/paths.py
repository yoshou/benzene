"""Paths of the ALIKED + LightGlue port: its work directory and its directory on the board (the LightGlue checkout -
whose lightglue/aliked.py is ALIKED's code - and the evaluation pairs are shared: benzene.models.lightglue.paths)."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("ALIKED_LIGHTGLUE_WORK", workspace.WORK / "aliked_lightglue"))
WEIGHTS = WORK / "weights"  # aliked-n16.pth (Shiaoming/ALIKED models/), aliked_lightglue.pth (LightGlue v0.1_arxiv)
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("aliked_lightglue")  # BOARD_DIR: model files on the board
