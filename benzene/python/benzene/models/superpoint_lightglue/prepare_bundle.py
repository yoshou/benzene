"""Device input bundle of image pairs for the benzene SuperPoint + LightGlue runtime (benzene superpoint_lightglue
model bundle out; format: benzene.models.lightglue.bundle; the exact input of the accuracy evaluation: gray / 255).

  python -m benzene.models.superpoint_lightglue.prepare_bundle --pairs 0 1 2 \
      --out work/superpoint_lightglue/bundle/test
  python -m benzene.models.superpoint_lightglue.prepare_bundle --sequence rgbd_dataset_freiburg1_desk --frames 62 \
      --out work/superpoint_lightglue/bundle/seq
      (consecutive camera frames as pairs (0, 1), (2, 3), ..)
"""

from __future__ import annotations

from ..lightglue import bundle
from .runner import load_pair

if __name__ == "__main__":
    bundle.main(load_pair)
