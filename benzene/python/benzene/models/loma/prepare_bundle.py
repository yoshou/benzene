"""Device input bundle of image pairs for the benzene LoMa runtime (benzene loma model bundle out): per image its gray
codes at the detector's and at the descriptor's size of the settings (runner.network_codes: PIL's resize of the gray
image, the accuracy evaluation's exact input).

meta.txt: the pair count, then per image "detector_height detector_width descriptor_height descriptor_width";
images.u8: per image the detector codes, then the descriptor codes.

  python -m benzene.models.loma.prepare_bundle --pairs 0 1 2 --out work/loma/bundle/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benzene.command_line import Indices

from ..lightglue.evaluation import eval_files
from .runner import descriptor_shape, detector_shape, image_rgb, load_pair, network_codes


def image_codes(rec: dict, i: int, settings) -> tuple[np.ndarray, np.ndarray]:
    """(detector codes [h, w], descriptor codes [h', w']) of image i of a pair."""
    rgb = image_rgb(rec, i, settings)
    height, width = rgb.shape[:2]
    detector = network_codes(rgb, detector_shape(width, height, settings.detector_size), "cpu").numpy()
    descriptor = network_codes(rgb, descriptor_shape(width, height, settings.descriptor_size), "cpu").numpy()
    return detector, descriptor


def write_bundle(files: list[str], settings, out: Path) -> list[tuple[int, ...]]:
    out.mkdir(parents=True, exist_ok=True)
    sizes, blobs = [], []
    for f in files:
        rec = load_pair(f)
        for i in (0, 1):
            detector, descriptor = image_codes(rec, i, settings)
            sizes.append((*detector.shape, *descriptor.shape))
            blobs += [detector.tobytes(), descriptor.tobytes()]
    (out / "meta.txt").write_text(f"{len(files)}\n" + "".join(" ".join(map(str, s)) + "\n" for s in sizes))
    (out / "images.u8").write_bytes(b"".join(blobs))
    return sizes


DESCRIPTOR_DIMENSIONS = 256


def read_features(path: Path) -> dict:
    """A device output file (<pair>.lm<image>): positions [k, 2], scores [k], descriptors [k, 256]."""
    raw = path.read_bytes()
    k = int(np.frombuffer(raw[:4], np.int32)[0])
    positions = np.frombuffer(raw[4 : 4 + 8 * k], np.float32).reshape(k, 2)
    scores = np.frombuffer(raw[4 + 8 * k : 4 + 12 * k], np.float32)
    descriptors = np.frombuffer(raw[4 + 12 * k :], np.float32).reshape(k, DESCRIPTOR_DIMENSIONS)
    return {"positions": positions, "scores": scores, "descriptors": descriptors}


def main() -> None:
    from .evaluate import SETTINGS, parse_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", action=Indices, default=[0])
    parser.add_argument("--settings", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    settings = parse_settings(args.settings) if args.settings else SETTINGS
    files = eval_files()
    print(write_bundle([files[i] for i in args.pairs], settings, Path(args.out)))


if __name__ == "__main__":
    main()
