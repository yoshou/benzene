"""Paths of the LoMa port: its work directory (the davnords/LoMa checkout, the weights, evaluation outputs) and its
directory on the board (the evaluation pairs are shared: benzene.models.lightglue.paths)."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("LOMA_WORK", workspace.WORK / "loma"))
SRC_DIR = WORK / "src"  # the davnords/LoMa checkout (commit ecc947d, lomatch 1.2.0): what network.py follows
WEIGHTS = WORK / "weights"  # loma_B.pt (LoMa-B with its DaD detector and DeDoDe-G descriptor), dad.pth
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("loma")  # BOARD_DIR: model files on the board
