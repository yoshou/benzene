"""Build cached multi-view samples for the accuracy evaluation of the quantized VGGT.

A sample is S frames of one scene, preprocessed exactly as vggt.utils.load_fn.load_and_preprocess_images (mode
"crop": width 518, height rounded to a multiple of 14) and stored as the exact uint8 values (the loader's tensor is
uint8 / 255), plus ground truth where the dataset has it:
  gt_extri  [S, 3, 4] world-to-camera (OpenCV axes), gt_intri [S, 3, 3] for the preprocessed image size,
  gt_depth  [S, H, W] fp16 metres (0 = no measurement; TUM only, nearest-resized to the preprocessed grid).

Sources (the shared raw datasets, benzene.workspace; scripts/download-inputs.sh):
  tum     TUM RGB-D sequences (Kinect 640x480, depth registered to RGB, motion-capture poses). A sample takes S frames
          from a random start with a random stride of 8..32 frames (0.27..1.07 s at 30 Hz).
  mip360  Mip-NeRF 360 scenes (images_8, COLMAP poses from poses_bounds.npy). Even samples: S of the 24 views nearest
          to a random anchor; odd samples: S random views of the scene (wide baseline).
Scenes are split between calibration and evaluation (no scene in both).

  python -m benzene.models.vggt.prepare_data --split calib --count 64
  python -m benzene.models.vggt.prepare_data --split eval --count 96
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benzene.workspace import MIPNERF360, TUM_RGBD

from .paths import DATA_DIR

SPLITS = {
    "calib": {"tum": ["rgbd_dataset_freiburg1_desk2", "rgbd_dataset_freiburg1_room",
                      "rgbd_dataset_freiburg3_structure_texture_far", "rgbd_dataset_freiburg2_desk"],
              "mip360": ["bicycle", "garden", "treehill", "counter", "room"]},
    "eval": {"tum": ["rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg1_plant", "rgbd_dataset_freiburg1_teddy",
                     "rgbd_dataset_freiburg1_360", "rgbd_dataset_freiburg3_long_office_household"],
             "mip360": ["flowers", "stump", "bonsai", "kitchen"]},
}
TUM_K = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
         "freiburg3": (535.4, 539.2, 320.1, 247.6)}
TARGET = 518


def preprocess(path: Path) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """load_and_preprocess_images (crop) for one image -> (uint8 [3, H, W], (sx, sy) resize factors, (left, top) crop
    offsets)."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    w, h = img.size
    nw, nh = TARGET, round(h * (TARGET / w) / 14) * 14
    a = np.asarray(img.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.uint8).transpose(2, 0, 1)
    y0 = (nh - TARGET) // 2 if nh > TARGET else 0
    if nh > TARGET:
        a = a[:, y0 : y0 + TARGET]
    return np.ascontiguousarray(a), (nw / w, nh / h), (0, y0)


def scale_intri(fx, fy, cx, cy, s, offset) -> np.ndarray:
    """Pinhole intrinsics after a resize by s = (sx, sy) (pixel centres at +0.5) and a crop at offset = (left, top)."""
    return np.array([[fx * s[0], 0, (cx + 0.5) * s[0] - 0.5 - offset[0]],
                     [0, fy * s[1], (cy + 0.5) * s[1] - 0.5 - offset[1]], [0, 0, 1]], np.float32)


def quat_to_rot(q) -> np.ndarray:  # (qx, qy, qz, qw)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def w2c_from_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([R.T, (-R.T @ t)[:, None]], 1).astype(np.float32)


# ---------------------------------------------------------------------------- TUM RGB-D
def _tum_list(p: Path) -> tuple[np.ndarray, list[str]]:
    ts, rest = [], []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        ts.append(float(f[0]))
        rest.append(f[1:])
    return np.array(ts), rest


class Tum:
    def __init__(self, name: str):
        self.dir = TUM_RGBD / name
        self.rgb_t, rgb = _tum_list(self.dir / "rgb.txt")
        self.rgb = [r[0] for r in rgb]
        self.dep_t, dep = _tum_list(self.dir / "depth.txt")
        self.dep = [r[0] for r in dep]
        self.gt_t, gt = _tum_list(self.dir / "groundtruth.txt")
        self.gt = np.array(gt, np.float64)  # tx ty tz qx qy qz qw (camera-to-world)
        self.K = TUM_K[next(k for k in TUM_K if k in name)]
        # frames with a depth image and a pose within 20 ms
        di = np.clip(np.searchsorted(self.dep_t, self.rgb_t), 1, len(self.dep_t) - 1)
        di = np.where(np.abs(self.dep_t[di - 1] - self.rgb_t) < np.abs(self.dep_t[di] - self.rgb_t), di - 1, di)
        gi = np.clip(np.searchsorted(self.gt_t, self.rgb_t), 1, len(self.gt_t) - 1)
        gi = np.where(np.abs(self.gt_t[gi - 1] - self.rgb_t) < np.abs(self.gt_t[gi] - self.rgb_t), gi - 1, gi)
        ok = (np.abs(self.dep_t[di] - self.rgb_t) < 0.02) & (np.abs(self.gt_t[gi] - self.rgb_t) < 0.02)
        self.valid = np.nonzero(ok)[0]
        self.di, self.gi = di, gi

    def pick(self, rng: np.random.Generator, S: int) -> list[int]:
        stride = int(rng.choice([8, 12, 16, 24, 32]))
        n = len(self.valid)
        stride = min(stride, (n - 1) // (S - 1))
        start = int(rng.integers(0, n - (S - 1) * stride))
        return [int(self.valid[start + k * stride]) for k in range(S)]

    def frame(self, i: int, preprocess=preprocess) -> dict:
        """Frame i through `preprocess` (this module's, or another port's of the same signature)."""
        img, s, (left, top) = preprocess(self.dir / self.rgb[i])
        H, W = img.shape[1:]
        d = np.asarray(Image.open(self.dir / self.dep[self.di[i]]), np.float32) / 5000.0
        h0, w0 = d.shape
        ys = np.clip(((np.arange(H) + top + 0.5) / s[1]).astype(int), 0, h0 - 1)  # nearest source pixel
        xs = np.clip(((np.arange(W) + left + 0.5) / s[0]).astype(int), 0, w0 - 1)
        g = self.gt[self.gi[i]]
        return {"name": self.rgb[i], "img": img, "intri": scale_intri(*self.K, s, (left, top)),
                "extri": w2c_from_c2w(quat_to_rot(g[3:7]), g[:3]), "depth": d[ys][:, xs].astype(np.float16)}


# ---------------------------------------------------------------------------- Mip-NeRF 360
class Mip360:
    def __init__(self, name: str):
        self.dir = MIPNERF360 / name
        pb = np.load(self.dir / "poses_bounds.npy")
        p = pb[:, :15].reshape(-1, 3, 5)
        self.c2w = np.concatenate([p[:, :, 1:2], p[:, :, 0:1], -p[:, :, 2:3]], 2)  # LLFF [down, right, back] -> OpenCV
        self.t = p[:, :, 3]
        self.hwf = p[0, :, 4]  # full-resolution H, W, focal
        self.files = sorted((self.dir / "images_8").iterdir())
        assert len(self.files) == len(pb), f"{name}: {len(self.files)} images vs {len(pb)} poses"

    def pick(self, rng: np.random.Generator, S: int, local: bool) -> list[int]:
        n = len(self.files)
        if not local:
            return sorted(rng.choice(n, S, replace=False).tolist())
        a = int(rng.integers(n))
        near = np.argsort(np.linalg.norm(self.t - self.t[a], axis=1))[:24]
        return sorted([a] + rng.choice(near[1:], S - 1, replace=False).tolist())

    def frame(self, i: int, preprocess=preprocess) -> dict:
        """Frame i through `preprocess` (this module's, or another port's of the same signature)."""
        img, s, offset = preprocess(self.files[i])
        w8 = Image.open(self.files[i]).size[0]
        H, W, f = self.hwf
        r = w8 / W  # images_8 relative to the full resolution
        K = scale_intri(f * r, f * r, W * r / 2 - 0.5, H * r / 2 - 0.5, s, offset)
        return {"name": self.files[i].name, "img": img, "intri": K, "extri": w2c_from_c2w(self.c2w[i], self.t[i]),
                "depth": None}


def build(split: str, count: int, S: int, seed: int) -> None:
    out_dir = DATA_DIR / "cache" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sources = [("tum", n) for n in SPLITS[split]["tum"]] + [("mip360", n) for n in SPLITS[split]["mip360"]]
    loaded: dict = {}
    for k in range(count):
        ds, scene = sources[k % len(sources)]  # round-robin over the scenes
        out = out_dir / f"{k:03d}_{ds}_{scene.replace('rgbd_dataset_', '')}.pt"
        sub = np.random.default_rng([seed, k])  # per sample: a larger --count keeps the existing samples
        if out.exists():
            continue
        if scene not in loaded:
            loaded[scene] = Tum(scene) if ds == "tum" else Mip360(scene)
        src = loaded[scene]
        idx = src.pick(sub, S) if ds == "tum" else src.pick(sub, S, local=(k // len(sources)) % 2 == 0)
        fr = [src.frame(i) for i in idx]
        shapes = {f["img"].shape for f in fr}
        assert len(shapes) == 1, f"{scene}: mixed shapes {shapes}"
        rec = {"dataset": ds, "scene": scene, "frames": [f["name"] for f in fr],
               "images": torch.from_numpy(np.stack([f["img"] for f in fr])),
               "gt_extri": torch.from_numpy(np.stack([f["extri"] for f in fr])),
               "gt_intri": torch.from_numpy(np.stack([f["intri"] for f in fr])),
               "gt_depth": None if fr[0]["depth"] is None else torch.from_numpy(np.stack([f["depth"] for f in fr]))}
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        torch.save(rec, tmp)
        os.replace(tmp, out)
        print(f"{out.name} {tuple(rec['images'].shape)} frames {rec['frames'][0]} .. {rec['frames'][-1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else {"calib": 1, "eval": 2}[args.split]
    build(args.split, args.count, args.frames, seed)


if __name__ == "__main__":
    main()
