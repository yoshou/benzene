"""Image pairs with ground-truth relative poses for the accuracy evaluation of the LightGlue matchers
(superpoint_lightglue, xfeat_lightglue).

A pair is two grayscale uint8 frames of one scene at the camera resolution (the device input: SuperPoint reads
gray / 255, XFeat the gray values; gray = round(0.299 R + 0.587 G + 0.114 B), the weights of kornia's rgb_to_grayscale
SuperPoint's official extractor applies), their pinhole intrinsics and the relative pose T_0to1 (camera 0 -> camera 1,
OpenCV axes).

Sources (the shared raw datasets of benzene.workspace), same scene split as VGGT's (no scene in both):
  tum     TUM RGB-D sequences, 640 x 480: frame i and i + gap, gap in {5, 10, 15, 20, 30} frames (0.17 .. 1 s at 30 Hz,
          a camera stream with dropped frames), baseline >= 3 cm (a relative pose with a direction to score)
  mip360  Mip-NeRF 360 scenes, images_8 centre-cropped to multiples of 8 (616 x 408 .. 648 x 416 outdoor, 384 x 256
          indoor), COLMAP poses: a random view and one of its 6 nearest views
Scenes are taken round-robin; pair k of a split depends only on (seed, k).

  python -m benzene.models.lightglue.prepare_data --split calib --count 64
  python -m benzene.models.lightglue.prepare_data --split eval --count 180
"""

from __future__ import annotations

import argparse
import os

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
GAPS = (5, 10, 15, 20, 30)


def gray(path, crop: tuple[int, int] | None = None) -> tuple[np.ndarray, int, int]:
    """uint8 [H, W] grayscale (centre crop to `crop` = (W, H) if given) and the crop offsets (x0, y0)."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float64)
    g = np.clip(np.round(a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114), 0, 255).astype(np.uint8)
    if crop is None:
        return g, 0, 0
    x0, y0 = (g.shape[1] - crop[0]) // 2, (g.shape[0] - crop[1]) // 2
    return np.ascontiguousarray(g[y0 : y0 + crop[1], x0 : x0 + crop[0]]), x0, y0


def quat_to_rot(q) -> np.ndarray:  # (qx, qy, qz, qw)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _tum_list(p) -> tuple[np.ndarray, list[list[str]]]:
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
        rgb_t, rgb = _tum_list(self.dir / "rgb.txt")
        gt_t, gt = _tum_list(self.dir / "groundtruth.txt")
        gt = np.array(gt, np.float64)  # tx ty tz qx qy qz qw (camera-to-world)
        gi = np.clip(np.searchsorted(gt_t, rgb_t), 1, len(gt_t) - 1)
        gi = np.where(np.abs(gt_t[gi - 1] - rgb_t) < np.abs(gt_t[gi] - rgb_t), gi - 1, gi)
        ok = np.abs(gt_t[gi] - rgb_t) < 0.02  # a pose within 20 ms
        self.files = [rgb[i][0] for i in range(len(rgb))]
        self.pose = [c2w(quat_to_rot(gt[gi[i]][3:7]), gt[gi[i]][:3]) for i in range(len(rgb))]
        self.ok = ok
        fx, fy, cx, cy = TUM_K[next(k for k in TUM_K if k in name)]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def pair(self, rng: np.random.Generator) -> tuple[int, int]:
        n = len(self.files)
        for _ in range(1000):
            gap = int(rng.choice(GAPS))
            i = int(rng.integers(0, n - gap))
            j = i + gap
            if not (self.ok[i] and self.ok[j]):
                continue
            if np.linalg.norm(self.pose[i][:3, 3] - self.pose[j][:3, 3]) >= 0.03:
                return i, j
        raise RuntimeError(f"{self.dir.name}: no pair with a 3 cm baseline")

    def frame(self, i: int) -> dict:
        g, _, _ = gray(self.dir / self.files[i])
        return {"name": self.files[i], "image": g, "K": self.K.copy(), "c2w": self.pose[i]}


class Mip360:
    def __init__(self, name: str):
        self.dir = MIPNERF360 / name
        pb = np.load(self.dir / "poses_bounds.npy")
        p = pb[:, :15].reshape(-1, 3, 5)
        R = np.concatenate([p[:, :, 1:2], p[:, :, 0:1], -p[:, :, 2:3]], 2)  # LLFF [down, right, back] -> OpenCV
        self.pose = [c2w(R[i], p[i, :, 3]) for i in range(len(pb))]
        self.hwf = p[0, :, 4]  # full-resolution H, W, focal
        self.files = sorted((self.dir / "images_8").iterdir())
        assert len(self.files) == len(pb), f"{name}: {len(self.files)} images vs {len(pb)} poses"
        w8, h8 = Image.open(self.files[0]).size
        self.crop = (w8 // 8 * 8, h8 // 8 * 8)
        H, W, f = self.hwf
        r = w8 / W
        self.K = np.array([[f * r, 0, W * r / 2 - 0.5 - (w8 - self.crop[0]) // 2],
                           [0, f * r, H * r / 2 - 0.5 - (h8 - self.crop[1]) // 2], [0, 0, 1]])

    def pair(self, rng: np.random.Generator) -> tuple[int, int]:
        n = len(self.files)
        i = int(rng.integers(n))
        t = np.array([T[:3, 3] for T in self.pose])
        near = np.argsort(np.linalg.norm(t - t[i], axis=1))[1:7]
        return i, int(rng.choice(near))

    def frame(self, i: int) -> dict:
        g, _, _ = gray(self.files[i], self.crop)
        return {"name": self.files[i].name, "image": g, "K": self.K.copy(), "c2w": self.pose[i]}


def build(split: str, count: int, seed: int) -> None:
    out_dir = DATA_DIR / "pairs" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [("tum", n) for n in SPLITS[split]["tum"]] + [("mip360", n) for n in SPLITS[split]["mip360"]]
    loaded: dict = {}
    for k in range(count):
        ds, scene = sources[k % len(sources)]
        out = out_dir / f"{k:03d}_{ds}_{scene.replace('rgbd_dataset_', '')}.pt"
        if out.exists():
            continue
        if scene not in loaded:
            loaded[scene] = Tum(scene) if ds == "tum" else Mip360(scene)
        src = loaded[scene]
        i, j = src.pair(np.random.default_rng([seed, k]))
        f0, f1 = src.frame(i), src.frame(j)
        T = np.linalg.inv(f1["c2w"]) @ f0["c2w"]  # camera 0 -> camera 1
        rec = {"dataset": ds, "scene": scene, "names": (f0["name"], f1["name"]),
               "image0": torch.from_numpy(f0["image"]), "image1": torch.from_numpy(f1["image"]),
               "K0": torch.from_numpy(f0["K"]), "K1": torch.from_numpy(f1["K"]), "T_0to1": torch.from_numpy(T)}
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        torch.save(rec, tmp)
        os.replace(tmp, out)
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"{out.name} {tuple(f0['image'].shape)} {f0['name']} / {f1['name']}: baseline "
              f"{np.linalg.norm(T[:3, 3]):.3f}, rotation {ang:.1f} deg", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    build(args.split, args.count, args.seed if args.seed is not None else {"calib": 1, "eval": 2}[args.split])


if __name__ == "__main__":
    main()
