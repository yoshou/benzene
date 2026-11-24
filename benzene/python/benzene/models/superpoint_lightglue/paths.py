"""Paths of the SuperPoint + LightGlue port: its work directory and its directory on the board (the LightGlue checkout
and the evaluation pairs are shared: benzene.models.lightglue.paths)."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("SUPERPOINT_LIGHTGLUE_WORK", workspace.WORK / "superpoint_lightglue"))
WEIGHTS = WORK / "weights"  # the LightGlue v0.1_arxiv release files
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("superpoint_lightglue")  # BOARD_DIR: model files on the board
