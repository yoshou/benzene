"""Device input bundle of a cached sample for the benzene AnySplat command (benzene anysplat model.hexmodel bundle_dir
out_dir): the VGGT port's bundle of the sample's contexts, meta.txt "S H W" and images.u8 [S][3][H][W] (exactly the
PyTorch model's input times 255).

  python -m benzene.models.anysplat.prepare_bundle --sample 0 --out work/anysplat/bundle/000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from ..vggt.prepare_bundle import write_bundle  # noqa: F401 - re-exported for the checks
from .paths import DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    f = sorted(glob.glob(str(DATA_DIR / f"cache/{args.split}/*.pt")))[args.sample]
    print(Path(f).name, write_bundle(f, Path(args.out)))


if __name__ == "__main__":
    main()
