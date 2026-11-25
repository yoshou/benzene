"""Paths of the AnySplat port (lhjiang/anysplat): its work directory and its directory on the board.

The network is VGGT's: the facebookresearch/vggt checkout of the VGGT port (benzene.models.vggt.paths puts it on
sys.path, so `import vggt` finds it) with AnySplat's weights, plus AnySplat's Gaussian head and Gaussians (network.py).
The InternRobotics/AnySplat checkout in WORK/src is what network.py follows; it is not imported (its modules need
CUDA-only packages).
"""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

from ..vggt import paths as _vggt_paths  # noqa: F401 - the vggt checkout on sys.path

WORK = Path(os.environ.get("ANYSPLAT_WORK", workspace.WORK / "anysplat"))
MODEL_DIR = Path(os.environ.get("ANYSPLAT_MODEL", WORK / "hf/anysplat"))
SRC_DIR = WORK / "src"
DATA_DIR = WORK / "data"
OUT_DIR = WORK / "out"
__getattr__ = workspace.board_directory("anysplat")  # BOARD_DIR: model files on the board
