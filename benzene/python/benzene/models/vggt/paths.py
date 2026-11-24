"""Paths of the VGGT port (facebook/VGGT-1B): its work directory and its directory on the board.

The model code is the facebookresearch/vggt checkout in WORK/src, put on sys.path here (nothing is installed into the
venv), so `import vggt` finds it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("VGGT_WORK", workspace.WORK / "vggt"))
MODEL_DIR = Path(os.environ.get("VGGT_MODEL", WORK / "hf/VGGT-1B"))
SRC_DIR = WORK / "src"
DATA_DIR = WORK / "data"
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("vggt")  # BOARD_DIR: model files on the board

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
