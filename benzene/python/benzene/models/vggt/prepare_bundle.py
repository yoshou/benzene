"""Device input bundle of a cached sample for the benzene VGGT runtime (benzene vggt model.hexmodel bundle_dir
out_dir): meta.txt "S H W" and images.u8 [S][3][H][W] (the preprocessed frames, exactly the PyTorch model's input
times 255).

  python -m benzene.models.vggt.prepare_bundle --sample 0 --out work/vggt/bundle/000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from .paths import DATA_DIR


def write_bundle(sample: str | Path, out: Path, frames: int | None = None) -> dict:
    """frames: only the sample's first `frames` frames"""
    rec = torch.load(sample, map_location="cpu", weights_only=False)
    img = rec["images"][:frames] if frames else rec["images"]
    S, _, H, W = img.shape
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.txt").write_text(f"{S} {H} {W}\n")
    (out / "images.u8").write_bytes(img.contiguous().numpy().tobytes())
    return {"S": S, "H": H, "W": W}


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
