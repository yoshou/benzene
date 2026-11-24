"""Device input bundles of the feature matchers (benzene superpoint_lightglue / xfeat_lightglue / aliked_lightglue
model bundle out) and their output files.

Bundle: meta.txt "n" then n lines "H0 W0 H1 W1", images.u8 = per pair the two grayscale images [H][W] (uint8, exactly
the accuracy evaluation's input). Outputs per pair i: <i>.<ext><s> (the extractor's features of image s: int32 k,
keypoints [k][2] f32, scores [k] f32, descriptors [k][D] fp16), <i>.lg (int32 n0, matches0 [n0] int32, mscores0 [n0]
f32).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from benzene.command_line import Indices

from .evaluation import eval_files
from .prepare_data import TUM_RGBD, _tum_list, gray


def write_bundle(records: list[dict], out: Path) -> list[tuple[int, int, int, int]]:
    """A bundle of pairs (records with uint8 images image0 / image1)."""
    out.mkdir(parents=True, exist_ok=True)
    sizes, data = [], []
    for rec in records:
        a, b = rec["image0"].numpy(), rec["image1"].numpy()
        sizes.append((*a.shape, *b.shape))
        data += [np.ascontiguousarray(a, np.uint8).tobytes(), np.ascontiguousarray(b, np.uint8).tobytes()]
    (out / "meta.txt").write_text(f"{len(records)}\n" + "".join(f"{h0} {w0} {h1} {w1}\n" for h0, w0, h1, w1 in sizes))
    (out / "images.u8").write_bytes(b"".join(data))
    return sizes


def write_sequence(scene: str, start: int, frames: int, out: Path) -> tuple[int, int, int]:
    """frames consecutive TUM frames (30 Hz) of a scene from `start`, grayscale as prepare_data.gray, as frame pairs
    (0, 1), (2, 3), ..."""
    _, rgb = _tum_list(TUM_RGBD / scene / "rgb.txt")
    imgs = [gray(TUM_RGBD / scene / rgb[start + i][0])[0] for i in range(frames - frames % 2)]
    out.mkdir(parents=True, exist_ok=True)
    H, W = imgs[0].shape
    (out / "meta.txt").write_text(f"{len(imgs) // 2}\n" + f"{H} {W} {H} {W}\n" * (len(imgs) // 2))
    (out / "images.u8").write_bytes(b"".join(np.ascontiguousarray(a).tobytes() for a in imgs))
    return len(imgs), H, W


def read_features(path: Path, dimensions: int) -> dict:
    """A device features file: keypoints [k][2], keypoint_scores [k], descriptors fp16 [k][dimensions] (as f32)."""
    raw = path.read_bytes()
    k = int(np.frombuffer(raw[:4], np.int32)[0])
    o = 4
    kp = np.frombuffer(raw[o : o + 8 * k], np.float32).reshape(k, 2)
    o += 8 * k
    sc = np.frombuffer(raw[o : o + 4 * k], np.float32)
    o += 4 * k
    ds = np.frombuffer(raw[o : o + 2 * dimensions * k], np.float16).reshape(k, dimensions).astype(np.float32)
    return {"keypoints": kp, "keypoint_scores": sc, "descriptors": ds}


def read_matches(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A device matches file: matches0 [n0] int32, mscores0 [n0] f32."""
    raw = path.read_bytes()
    n0 = int(np.frombuffer(raw[:4], np.int32)[0])
    return (np.frombuffer(raw[4 : 4 + 4 * n0], np.int32).copy(),
            np.frombuffer(raw[4 + 4 * n0 : 4 + 8 * n0], np.float32).copy())


def main(load_pair: Callable[[str], dict]) -> None:
    """A matcher's prepare_bundle: evaluation pairs as load_pair reads them, or consecutive frames of a TUM scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", action=Indices, default=[0])
    parser.add_argument("--sequence", help="a TUM scene: consecutive frames instead of evaluation pairs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=62)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.sequence:
        print(write_sequence(args.sequence, args.start, args.frames, Path(args.out)))
        return
    files = eval_files()
    print(write_bundle([load_pair(files[i]) for i in args.pairs], Path(args.out)))
