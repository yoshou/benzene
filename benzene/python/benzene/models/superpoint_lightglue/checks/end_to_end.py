"""SuperPoint + LightGlue on the board (benzene superpoint_lightglue) against PyTorch for evaluation pairs
(benzene.models.lightglue.device_check). PyTorch runs SuperPoint as exported_model.exported_superpoint (the device's
weights and integer arithmetic of the convolutions, the post-processing in PyTorch fp32), LightGlue in its fp16 mixed
precision.

  python -m benzene.models.superpoint_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2
  python -m benzene.models.superpoint_lightglue.checks.end_to_end dev+fp16 --all --save device_dev+fp16 --device-only
  python -m benzene.models.superpoint_lightglue.checks.end_to_end dev+fp16 --all --adaptive --save device_dev+fp16a
      (LightGlue's adaptive inference on both sides: benzene superpoint_lightglue --adaptive, the recipe fp16a in
      PyTorch)
"""

from __future__ import annotations

from ...lightglue.device_check import DeviceCheck
from ..evaluate import accuracy_evaluation
from ..exported_model import exported_superpoint
from ..paths import BOARD_DIR, WORK
from ..recipes import DEVICE_RECIPE
from ..runner import ADAPTIVE, DESCRIPTOR_DIMENSIONS, load_models, run_pair


class SuperPointCheck(DeviceCheck):
    command = "superpoint_lightglue"
    board_work = "sg"
    features = "sp"
    descriptor_dimensions = DESCRIPTOR_DIMENSIONS
    adaptive_help = "LightGlue's adaptive inference (early stop, point pruning)"

    def pytorch_model(self, recipe: str, device: str):
        sp, lg = load_models(device)
        exported_superpoint(sp, self.model_file(recipe))
        return (lambda rec, device: run_pair(sp, lg, rec, device)), lg


if __name__ == "__main__":
    SuperPointCheck(accuracy_evaluation, WORK, BOARD_DIR, DEVICE_RECIPE, ADAPTIVE).main()
