"""AnySplat on the board (benzene anysplat) against PyTorch for eval samples.

Pushes the model file (export_hexmodel.py) and each sample's bundle (its contexts), runs the device and scores what
it makes, its Gaussians (gaussians.ply), depth and cameras, against the fp32 reference of the accuracy evaluation
(evaluate.py ref) with the evaluation's metrics (evaluate.score: the targets rendered from the board's Gaussians against
the fp32 renders and the held-out images, depth, cameras), so that the device rows compare directly with the recipe rows
of evaluate.py table; then the medians of the board's stage times over the samples. With --raw also the raw Gaussian
head outputs (gs) and the board's Gaussians against Python's Gaussians of the same outputs (the board's voxelization and
adapter, per voxel); with --heads the board's depth and Gaussian heads against PyTorch's fp32 heads on the board's own
transformer outputs (the DPT heads alone). The renders and comparisons run on --device (a GPU).

  python -m benzene.models.anysplat.checks.end_to_end w8a8_tfc --samples 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board
from benzene.command_line import Indices

from ..device_layout import OUTPUT_CHANNELS
from ..evaluate import REF_DIR, SUBSAMPLE, load_ref, score, summarize, target_cameras
from ..gaussians import VOXEL_SIZE, Gaussians, make_gaussians, read_ply
from ..network import RAW_CHANNELS, RAW_CONFIDENCE, RAW_HARMONICS, RAW_OPACITY, RAW_ROTATION, RAW_SCALES
from ..paths import BOARD_DIR, DATA_DIR, OUT_DIR, WORK
from ..prepare_bundle import write_bundle
from ..rasterizer import render
from ..recipes import DEVICE_RECIPE

BOARD_OUT = "/dev/shm/benzene-anysplat"  # the outputs in RAM: the PLY (~0.5 GB) not through the eMMC's writeback
# The board's milliseconds per stage, from the command's lines.
TIME_LINES = {"transformer": r"transformer ([\d.]+) ms", "camera head": r"camera head ([\d.]+) ms",
              "depth head": r"depth head ([\d.]+) ms", "gaussian head": r"gaussian head ([\d.]+) ms",
              "gaussians (CPU)": r"gaussians ([\d.]+) ms \|"}


def push_model(recipe: str) -> str:
    src = WORK / "export" / f"anysplat-{recipe}.hexmodel"
    if not src.exists():
        sys.exit(f"{src} missing: run export_hexmodel {recipe}")
    board.ssh(f"mkdir -p {BOARD_DIR}", check=True)
    board.upload([src, src.with_suffix(".tab")], f"{BOARD_DIR}/", inplace=True)
    return f"{BOARD_DIR}/{src.name}"


def run_device(model: str, sample: str, td: Path, raw: bool, tokens: bool) -> tuple[dict, dict, str]:
    """The board's outputs of a sample: pose_enc (the camera head's last iteration), depth, depth_conf, the Gaussians of
    gaussians.ply, with raw the head's raw outputs [S, H, W, RAW_CHANNELS], with tokens the kept transformer outputs
    [4, S, tokens per frame, 2048]; plus the bundle's sizes and the run's log."""
    meta = write_bundle(sample, td / "bundle")
    S, H, W = meta["S"], meta["H"], meta["W"]
    work = board.empty_directory("as")
    board.upload(f"{td}/bundle/", f"{work}/bundle/")
    board.empty_directory(BOARD_OUT)
    try:
        flags = (" --raw" if raw else "") + (" --tokens" if tokens else "")
        r = board.run_benzene(f"anysplat {model} as/bundle {BOARD_OUT}{flags}")
        if r.returncode:
            sys.exit(f"board run failed ({r.returncode}): {r.stdout} {r.stderr}")
        board.download(f"{BOARD_OUT}/", f"{td}/out/")
    finally:
        board.ssh(f"rm -rf {BOARD_OUT}", check=True)
    rd = lambda n, dt: np.frombuffer((td / "out" / n).read_bytes(), dt).copy()  # noqa: E731
    d = torch.from_numpy(rd("depth.f32", np.float32).reshape(S, H, W, 2))
    out = {"pose_enc": torch.from_numpy(rd("pose.f32", np.float32).reshape(4, S, 9)[-1]), "depth": d[..., 0],
           "depth_conf": d[..., 1], "gaussians": read_ply(td / "out" / "gaussians.ply")}
    if raw:
        head = rd("gaussian_head.f16", np.float16).reshape(S, H, W, OUTPUT_CHANNELS)[..., :RAW_CHANNELS]
        out["raw"] = torch.from_numpy(head.astype(np.float32))
    if tokens:
        out["tokens"] = torch.from_numpy(rd("inter.f16", np.float16).reshape(4, S, -1, 2048).astype(np.float32))
    return out, meta, (r.stdout + r.stderr).strip()


def device_times(log: str) -> dict:
    """The stage times of a run's log (TIME_LINES), in milliseconds."""
    times = {}
    for name, pattern in TIME_LINES.items():
        m = re.search(pattern, log)
        if m:
            times[name] = float(m.group(1))
    return times


def voxel_keys(g: Gaussians) -> np.ndarray:
    """The voxels of the Gaussians' means (a fused mean stays in its voxel), packed 21 bits per axis."""
    k = (g.means.double() / VOXEL_SIZE).round().long() + (1 << 20)
    return ((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]).cpu().numpy()


def compare_gaussians(board_g: Gaussians, python_g: Gaussians) -> str:
    """The board's Gaussians against Python's of the same head outputs, voxel by voxel."""
    _, ia, ib = np.intersect1d(voxel_keys(board_g), voxel_keys(python_g), return_indices=True)
    ia, ib = torch.from_numpy(ia), torch.from_numpy(ib)
    parts = [f"{len(board_g)} on the board, {len(python_g)} in Python, {len(ia)} in the same voxels:"]
    for name in ("means", "scales", "rotations", "harmonics", "opacities"):
        a, b = getattr(board_g, name)[ia].double(), getattr(python_g, name)[ib].double()
        parts.append(f"{name} max {(a - b).abs().max().item():.2e} rel {((a - b).norm() / b.norm()).item():.1e}")
    return " ".join(parts)


@torch.no_grad()
def compare_heads(dev: dict, s: dict, device: str) -> str:
    """The board's depth and Gaussian heads against PyTorch's fp32 heads on the board's kept transformer outputs."""
    from ..runner import load_model

    model = load_model(device)
    tokens = [None] * model.aggregator.depth
    for k, layer in enumerate(model.depth_head.intermediate_layer_idx):
        tokens[layer] = dev["tokens"][k][None].to(device)
    images, start = s["x"][None], model.aggregator.patch_start_idx
    d, dc = model.depth_head(tokens, images=images, patch_start_idx=start, frames_chunk_size=2)
    g = model.gaussian_param_head(tokens, images=images, patch_start_idx=start, frames_chunk_size=2)
    d, dc, g = d[0, ..., 0].cpu(), dc[0].cpu(), g[0].permute(0, 2, 3, 1).cpu()
    del model
    torch.cuda.empty_cache()
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    groups = (("opacity", RAW_OPACITY), ("scales", RAW_SCALES), ("rotation", RAW_ROTATION), ("sh", RAW_HARMONICS),
              ("confidence", RAW_CONFIDENCE))
    channels = " ".join(f"{n} {rel(dev['raw'][..., i], g[..., i]):.4f}" for n, i in groups)
    return (f"depth {rel(dev['depth'], d):.4f} confidence {rel(dev['depth_conf'], dc):.4f} | Gaussian head "
            f"{rel(dev['raw'], g):.4f}: {channels}")


def save_images(stem: Path, renders: torch.Tensor, ref_renders: torch.Tensor, targets: torch.Tensor) -> None:
    """Per target a PNG of the board's render, fp32's render and the held-out image side by side."""
    from PIL import Image

    stem.parent.mkdir(parents=True, exist_ok=True)
    for t in range(renders.shape[0]):
        row = torch.cat([renders[t].cpu(), ref_renders[t].float().cpu(), targets[t].cpu()], dim=2)
        Image.fromarray((row.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
            f"{stem}_target{t}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--samples", nargs="+", action=Indices, default=[0])
    parser.add_argument("--raw", action="store_true", help="also the raw head outputs and the board's Gaussians "
                                                           "against Python's")
    parser.add_argument("--heads", action="store_true",
                        help="the board's heads against PyTorch's on the board's transformer outputs (with --raw)")
    parser.add_argument("--device", default="cuda:0", help="where the renders and the comparisons run")
    parser.add_argument("--save", help="write the per-sample rows and the summary to OUT_DIR/results/<name>.json")
    parser.add_argument("--images", help="a directory for PNGs of the targets: the board's render, fp32's, the image")
    args = parser.parse_args()
    from ..runner import load_sample

    model = push_model(args.recipe)
    files = sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))
    rows = []
    for i in args.samples:
        f = files[i]
        with tempfile.TemporaryDirectory() as td:
            dev, meta, log = run_device(model, f, Path(td), args.raw or args.heads, args.heads)
        print(log)
        s = load_sample(f, args.device)
        r = load_ref(REF_DIR / (Path(f).stem + ".pt"))
        q = {k: dev[k] for k in ("pose_enc", "depth", "depth_conf")}
        if "raw" in dev:
            q["raw_sub"] = dev["raw"][:, ::SUBSAMPLE, ::SUBSAMPLE]
        g = dev["gaussians"].to(args.device)
        extri, intri = target_cameras(r["joint_pose_enc"].to(args.device), dev["pose_enc"].to(args.device),
                                      (meta["H"], meta["W"]))
        renders = render(g, extri, intri, meta["H"], meta["W"])
        row = score(q, renders, len(g), s, r) | {"ms": device_times(log)}
        rows.append(row)
        if args.images:
            save_images(Path(args.images) / Path(f).stem, renders, r["renders"], s["xt"])
        print(f"{Path(f).stem} ({meta['S']} frames {meta['H']}x{meta['W']}, {len(g)} Gaussians, fp32 "
              f"{row['gaussians_ref']}): device vs fp32 reference: target renders PSNR {row['render']:.2f} dB, "
              + (f"Gaussian head {100 * row['gs']:.3f}%, " if "raw" in dev else "")
              + f"depth {100 * row['dep']:.3f}% (all {100 * row['dep_all']:.3f}%), rot {row['rot']:.3f} deg, trans "
              f"{100 * row['trans']:.3f}%, FoV {row['fov']:.3f} deg; targets vs held-out images PSNR {row['psnr']:.2f} "
              f"dB SSIM {row['ssim']:.4f} (fp32 {row['psnr_ref']:.2f}, {row['ssim_ref']:.4f}); GT pose AUC@30 "
              f"{row['auc30']:.4f} (fp32 {row['auc30_ref']:.4f})"
              + (f", depth AbsRel {row['absrel']:.4f} (fp32 {row['absrel_ref']:.4f})" if "absrel" in row else ""),
              flush=True)
        if "raw" in dev:
            python_g = make_gaussians(dev["pose_enc"].to(args.device), dev["depth"].to(args.device),
                                      dev["raw"].to(args.device))
            print("  Gaussians:", compare_gaussians(dev["gaussians"], python_g.to("cpu")), flush=True)
        if args.heads:
            print("  heads on the board's transformer outputs vs PyTorch fp32:", compare_heads(dev, s, args.device),
                  flush=True)
    summ = summarize(rows)
    print("mean over", len(rows), "samples:", json.dumps({k: round(v, 5) for k, v in summ.items() if k != "n"}))
    print(f"median board times over {len(rows)} samples [ms]:",
          ", ".join(f"{name} {np.median([r['ms'][name] for r in rows if name in r['ms']]):.0f}" for name in TIME_LINES))
    if args.save:
        out = OUT_DIR / "results" / f"{args.save}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": {**summ, "recipe": args.save}, "rows": rows}, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
