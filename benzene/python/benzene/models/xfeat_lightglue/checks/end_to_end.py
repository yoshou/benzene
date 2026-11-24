"""XFeat + LightGlue on the board (benzene xfeat_lightglue) against PyTorch for evaluation pairs
(benzene.models.lightglue.device_check). PyTorch runs exported_model.records in device_model.DeviceXFeat (the device's
codes, constants and arithmetic; the post-processing of detectAndCompute in PyTorch fp32), LightGlue in its fp16 mixed
precision; also the image normalization (the device's fp32 mean / 1 / std bits).

  python -m benzene.models.xfeat_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2
  python -m benzene.models.xfeat_lightglue.checks.end_to_end dev+fp16 --all --save device_dev+fp16
  python -m benzene.models.xfeat_lightglue.checks.end_to_end dev+fp16 --all --adaptive --save device_dev+fp16a
      (LighterGlue's point pruning on both sides: benzene xfeat_lightglue --adaptive, the recipe fp16a in PyTorch)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ...lightglue.device_check import DeviceCheck
from ..device_model import DeviceXFeat, image_statistics
from ..evaluate import accuracy_evaluation
from ..exported_model import records
from ..paths import BOARD_DIR, WORK
from ..recipes import DEVICE_RECIPE
from ..runner import ADAPTIVE, DESCRIPTOR_DIMENSIONS, load_models, run_pair


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class XFeatCheck(DeviceCheck):
    command = "xfeat_lightglue"
    board_work = "xf"
    features = "xf"
    descriptor_dimensions = DESCRIPTOR_DIMENSIONS
    adaptive_help = "LighterGlue's adaptive inference (point pruning)"
    normalizations: list[tuple[float, float]]
    normalization_mismatches = 0

    def pytorch_model(self, recipe: str, device: str):
        net, lg = load_models(device)
        pytorch_net = DeviceXFeat(records(net, self.model_file(recipe)), device)
        return (lambda rec, device: run_pair(pytorch_net, lg, rec, device)), lg

    def read_pair(self, out: Path, index: int) -> None:
        if index == 0:
            self.normalizations = []
        for image in range(2):
            mean, inverse_std = np.fromfile(out / f"{index}.xfn{image}", np.float32).tolist()
            self.normalizations.append((mean, inverse_std))

    def check_images(self, rec: dict, index: int) -> None:
        for i in (0, 1):  # the normalization: PyTorch's fp32 values bit for bit
            expected = image_statistics(rec[f"image{i}"])
            n = 2 * index + i
            got = self.normalizations[n] if n < len(self.normalizations) else (float("nan"), float("nan"))
            if any(f32_bits(a) != f32_bits(b) for a, b in zip(expected, got)):
                self.normalization_mismatches += 1
                print(f"{rec['name']} image {i}: normalization {got} vs PyTorch {expected}")

    def report(self, files: list[str], with_pytorch: bool) -> None:
        print(f"normalization: {2 * len(files) - self.normalization_mismatches} of {2 * len(files)} images bit for "
              f"bit")


if __name__ == "__main__":
    XFeatCheck(accuracy_evaluation, WORK, BOARD_DIR, DEVICE_RECIPE, ADAPTIVE).main()
