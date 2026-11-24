"""Device input bundle of image pairs for the benzene XFeat + LightGlue runtime (benzene xfeat_lightglue model
bundle out; format: benzene.models.lightglue.bundle): the evaluation pairs cropped to multiples of 32
(runner.load_pair), the accuracy evaluation's exact input.

  python -m benzene.models.xfeat_lightglue.prepare_bundle --pairs 0 1 2 --out work/xfeat_lightglue/bundle/test
  python -m benzene.models.xfeat_lightglue.prepare_bundle --sequence rgbd_dataset_freiburg1_desk --frames 62 \
      --out work/xfeat_lightglue/bundle/seq
      (consecutive camera frames as pairs (0, 1), (2, 3), ..; TUM's 640 x 480 needs no crop)
"""

from __future__ import annotations

from ..lightglue import bundle
from .runner import load_pair

if __name__ == "__main__":
    bundle.main(load_pair)
