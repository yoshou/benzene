"""ALIKED + LightGlue on the board (benzene aliked_lightglue) against PyTorch for evaluation pairs
(benzene.models.lightglue.device_check). PyTorch runs exported_model.records in device_model.DeviceAliked (the device's
codes, constants and arithmetic), LightGlue in its fp16 mixed precision; keypoints, scores and fp16 descriptors bit for
bit (images identical).

  python -m benzene.models.aliked_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2
  python -m benzene.models.aliked_lightglue.checks.end_to_end dev+fp16 --all --save device_dev+fp16
  python -m benzene.models.aliked_lightglue.checks.end_to_end dev+fp16 --all --adaptive --save device_dev+fp16a
      (LightGlue's adaptive inference: benzene aliked_lightglue --adaptive, the recipe fp16a in PyTorch)
"""

from __future__ import annotations

from ...lightglue.device_check import DeviceCheck, identical
from ..device_model import DeviceAliked
from ..evaluate import accuracy_evaluation
from ..exported_model import records
from ..paths import BOARD_DIR, WORK
from ..recipes import DEVICE_RECIPE
from ..runner import ADAPTIVE, DESCRIPTOR_DIMENSIONS, load_models, run_pair


class AlikedCheck(DeviceCheck):
    command = "aliked_lightglue"
    board_work = "al"
    features = "al"
    descriptor_dimensions = DESCRIPTOR_DIMENSIONS
    adaptive_help = "LightGlue's adaptive inference"

    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.same = {"keypoints": 0, "scores": 0, "descriptors": 0}
        self.images = 0

    def pytorch_model(self, recipe: str, device: str):
        net, lg = load_models(device)
        pytorch_net = DeviceAliked(records(net, self.model_file(recipe)), net.dkd, device)
        return (lambda rec, device: run_pair(pytorch_net, lg, rec, device)), lg

    def compare(self, device: dict, pytorch: dict) -> None:
        for i in (0, 1):
            self.images += 1
            self.same["keypoints"] += identical(device[f"kpts{i}"], pytorch[f"kpts{i}"])
            self.same["scores"] += identical(device[f"scores{i}"], pytorch[f"scores{i}"])
            self.same["descriptors"] += identical(device[f"desc{i}"], pytorch[f"desc{i}"])

    def describe_matches(self, device: dict, pytorch: dict) -> str:
        keypoints = f"{len(device['kpts0'])} / {len(pytorch['kpts0'])} keypoints"
        return f"{super().describe_matches(device, pytorch)}; {keypoints}"

    def report(self, files: list[str], with_pytorch: bool) -> None:
        if with_pytorch:
            print(f"bit for bit of {self.images} images: keypoints {self.same['keypoints']}, scores "
                  f"{self.same['scores']}, descriptors {self.same['descriptors']}")


if __name__ == "__main__":
    AlikedCheck(accuracy_evaluation, WORK, BOARD_DIR, DEVICE_RECIPE, ADAPTIVE).main()
