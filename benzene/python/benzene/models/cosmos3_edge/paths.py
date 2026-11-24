"""Paths of the Cosmos 3 Edge port (the reasoner of nvidia/Cosmos3-Edge: vision encoder, projector and language
model; the generator tower is not used): its work directory and its directory on the board."""

from __future__ import annotations

import os
from pathlib import Path

from benzene import workspace

WORK = Path(os.environ.get("COSMOS3_EDGE_WORK", workspace.WORK / "cosmos3_edge"))
MODEL_DIR = Path(os.environ.get("COSMOS3_EDGE_MODEL", WORK / "hf/Cosmos3-Edge"))
DATA_DIR = WORK / "data"
OUT_DIR = WORK / "out"
# BOARD_DIR: model files, embedding table, bundles and outputs on the board
__getattr__ = workspace.board_directory("cosmos3_edge")
