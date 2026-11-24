"""The cvg/LightGlue checkout in paths.SRC_DIR (commit 5a650c4) as Python modules.

Imported without its package __init__ (which pulls in the kornia / OpenCV based extractors; neither is in the venv and
nothing is installed into it): only lightglue/superpoint.py and lightglue/lightglue.py are loaded (modules()), and
lightglue/aliked.py on request (aliked_module()), with stub kornia / cv2 modules for the imports of utils.py and the
extractors' colour conversions that benzene never calls.
"""

from __future__ import annotations

import importlib
import sys
import types

from .paths import SRC_DIR


def _unused(*args, **kwargs):
    raise RuntimeError("kornia stub: not available in benzene")


def modules():
    """(lightglue.superpoint, lightglue.lightglue) of the checkout."""
    if "lightglue.lightglue" in sys.modules:
        return sys.modules["lightglue.superpoint"], sys.modules["lightglue.lightglue"]
    for name in ("kornia", "kornia.color", "cv2"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["kornia"].color = sys.modules["kornia.color"]
    sys.modules["kornia.color"].rgb_to_grayscale = _unused
    sys.modules["kornia.color"].grayscale_to_rgb = _unused  # (aliked.py: benzene passes it RGB images)
    pkg = types.ModuleType("lightglue")
    pkg.__path__ = [str(SRC_DIR / "lightglue")]
    sys.modules["lightglue"] = pkg
    return importlib.import_module("lightglue.superpoint"), importlib.import_module("lightglue.lightglue")


def aliked_module():
    """lightglue.aliked of the checkout (ALIKED and its parts: DKD, SDDH, DeformableConv2d)."""
    modules()
    return importlib.import_module("lightglue.aliked")
