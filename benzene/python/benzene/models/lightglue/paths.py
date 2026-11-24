"""Paths of what the LightGlue matchers share: the cvg/LightGlue checkout and the evaluation image pairs."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("LIGHTGLUE_WORK", workspace.WORK / "lightglue"))
SRC_DIR = WORK / "src"  # the cvg/LightGlue checkout (commit 5a650c4), source.modules() / aliked_module()
DATA_DIR = WORK / "data"  # pairs/{calib, eval} (prepare_data.py)
