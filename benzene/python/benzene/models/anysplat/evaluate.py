"""Fidelity of AnySplat quantization recipes against the fp32 reference, and novel-view synthesis of the held-out
target views of the eval samples (prepare_data).

  python -m benzene.models.anysplat.evaluate ref
      # fp32 reference outputs of the eval samples (OUT_DIR/ref)
  python -m benzene.models.anysplat.evaluate run fp16 w8a8_tfc ...
      # recipes -> OUT_DIR/results/<recipe>.json
  python -m benzene.models.anysplat.evaluate table

The targets' cameras, as AnySplat's eval_nvs makes them: the fp32 model predicts the cameras of the contexts and the
targets together (frame 0 the world origin, as in a contexts-only run; the reference keeps them), and the targets'
positions are scaled to a model's own contexts-only run by the ratio of the contexts' mean camera distance from the
origin (eval_nvs takes the ratio of the mean camera coordinates instead, which is unstable near the origin).

Deviation from the reference (per sample, then averaged):
  render   PSNR [dB] of the recipe's renders of the targets against the fp32 renders
  gs       relative RMS error of the raw Gaussian head outputs against fp32's (every 4th row and column: the reference
           keeps that subsample)
  dep, dep_all, rot, trans, fov, conf   depth, cameras, depth confidence as the VGGT port's (vggt.evaluate)
  (summaries also give p5 of render, p50 and p95 of dep and rot)
Ground truth:
  psnr, ssim   of the target renders against the held-out target images (SSIM as AnySplat's compute_ssim, skimage's
               structural_similarity: Gaussian window 11, sigma 1.5, sample covariance, data range 1)
  auc30        pose AUC@30 of the contexts' cameras against the GT poses
  absrel, d1   depth against the TUM depth maps after one median scale (as the VGGT port's)
LPIPS (AnySplat's third metric) needs the VGG weights of the lpips package and is left out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..vggt import evaluate as vggt_evaluate
from ..vggt.evaluate import gt_metrics
from .gaussians import cameras, make_gaussians
from .paths import DATA_DIR, OUT_DIR
from .rasterizer import render

REF_DIR = OUT_DIR / "ref"
RES_DIR = OUT_DIR / "results"
SUBSAMPLE = 4  # the reference's raw Gaussian outputs: every 4th row and column


def eval_files(count: int | None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))
    return files[:count] if count else files


def camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """-R^T t of world-to-camera extrinsics [S, 3, 4]."""
    R, t = extrinsics[:, :, :3], extrinsics[:, :, 3]
    return -(R.transpose(1, 2) * t[:, None, :]).sum(-1)


def target_cameras(joint_pose_enc: torch.Tensor, pose_enc: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
    """The targets' cameras (extrinsics [T, 3, 4], intrinsics [T, 3, 3]) for a contexts-only run pose_enc [S, 9]: the
    joint run's (pose_enc of S contexts, then T targets), positions scaled to the run."""
    S = pose_enc.shape[0]
    joint_extri, joint_intri = cameras(joint_pose_enc.float(), hw)
    own_extri, _ = cameras(pose_enc.float(), hw)
    scale = camera_centers(own_extri).norm(dim=-1).mean() / camera_centers(joint_extri[:S]).norm(dim=-1).mean()
    extri = joint_extri[S:].clone()
    extri[:, :, 3] *= scale
    return extri, joint_intri[S:]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """AnySplat's compute_psnr of one image pair [3, H, W] (both clipped to [0, 1])."""
    return (-10 * ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean().log10()).item()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """skimage structural_similarity(win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0) of two
    [3, H, W] images."""
    g = torch.exp(-0.5 * (torch.arange(11, dtype=torch.float64, device=a.device) - 5) ** 2 / 1.5**2)
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]

    def blur(z):
        return F.conv2d(z[:, None], kernel)[:, 0]  # the valid region: skimage crops 5 pixels per side

    x, y = a.double().clamp(0, 1), b.double().clamp(0, 1)
    ux, uy = blur(x), blur(y)
    norm = 121 / 120  # sample covariance
    vx, vy, vxy = norm * (blur(x * x) - ux * ux), norm * (blur(y * y) - uy * uy), norm * (blur(x * y) - ux * uy)
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.mean().item()


def gaussian_renders(p: dict, joint_pose_enc: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The targets rendered from a run's Gaussians (p: predict()'s outputs on the GPU) -> ([T, 3, H, W], the Gaussian
    count)."""
    hw = p["depth"].shape[-2:]
    g = make_gaussians(p["pose_enc"], p["depth"], p["raw"])
    extri, intri = target_cameras(joint_pose_enc.to(p["depth"].device), p["pose_enc"], hw)
    return render(g, extri, intri, *hw), len(g)


def nvs_metrics(renders: torch.Tensor, targets: torch.Tensor) -> dict:
    return {"psnr": float(np.mean([psnr(r, t) for r, t in zip(renders, targets)])),
            "ssim": float(np.mean([ssim(r, t) for r, t in zip(renders, targets)]))}


def deviation(q: dict, r: dict) -> dict:
    """q: a run's outputs (CPU fp32; raw subsampled, gs NaN without it), r: the reference's: gs, then the VGGT port's
    deviation."""
    gs = ((q["raw_sub"] - r["raw_sub"]).norm() / r["raw_sub"].norm()).item() if "raw_sub" in q else float("nan")
    return {"gs": gs, **vggt_evaluate.deviation(q, r)}


def stored(p: dict) -> dict:
    """What a run keeps of predict()'s outputs: CPU fp32, the raw outputs subsampled."""
    return {"pose_enc": p["pose_enc"].float().cpu(), "depth": p["depth"].float().cpu(),
            "depth_conf": p["depth_conf"].float().cpu(),
            "raw_sub": p["raw"][:, ::SUBSAMPLE, ::SUBSAMPLE].float().cpu()}


def command_reference(args) -> None:
    from .runner import camera_poses, load_model, load_sample, predict

    REF_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.device)
    for i, f in enumerate(eval_files(args.count)):
        out = REF_DIR / (Path(f).stem + ".pt")
        if out.exists() or (args.shard and i % int(args.shard.split("/")[1]) != int(args.shard.split("/")[0])):
            continue
        s = load_sample(f, args.device)
        t0 = time.time()
        p = predict(model, s["x"])
        joint = camera_poses(model, torch.cat([s["x"], s["xt"]]))
        renders, count = gaussian_renders(p, joint)
        rec = stored(p) | {"joint_pose_enc": joint.cpu(), "renders": renders.half().cpu(), "gaussians": count}
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        torch.save({k: (v.half() if k in ("depth", "depth_conf", "raw_sub") else v) for k, v in rec.items()}, tmp)
        os.replace(tmp, out)
        print(f"{i} {Path(f).stem} {time.time() - t0:.1f}s {count} gaussians {nvs_metrics(renders, s['xt'])} "
              f"{gt_metrics(stored(p), s)}", flush=True)


def load_ref(path: Path) -> dict:
    r = torch.load(path)
    return {k: v.float() if torch.is_tensor(v) else v for k, v in r.items()}


def score(q: dict, renders: torch.Tensor, gaussians: int, s: dict, r: dict) -> dict:
    """One run's row: q stored() outputs, its target renders and Gaussian count, the sample s, the reference r."""
    ref_renders = r["renders"].to(renders.device)
    return {"sample": s["name"], "dataset": s["dataset"], "gaussians": gaussians, "gaussians_ref": r["gaussians"],
            "render": float(np.mean([psnr(a, b) for a, b in zip(renders, ref_renders)])), **deviation(q, r),
            **nvs_metrics(renders, s["xt"]), **{k + "_ref": v for k, v in nvs_metrics(ref_renders, s["xt"]).items()},
            **gt_metrics(q, s), **{k + "_ref": v for k, v in gt_metrics(r, s).items()}}


def command_run(args) -> None:
    from .recipes import RECIPES, apply_recipe
    from .runner import load_model, load_sample, predict

    RES_DIR.mkdir(parents=True, exist_ok=True)
    for recipe in args.recipes:
        prec = RECIPES[recipe][0]
        model = load_model(args.device)
        info = apply_recipe(model, recipe)
        rows, t0 = [], time.time()
        files = eval_files(args.count)
        for i, f in enumerate(files):
            if i % 10 == 0:
                print(f"[{recipe}] {time.strftime('%H:%M:%S')} sample {i}/{len(files)}", flush=True)
            s = load_sample(f, args.device)
            r = load_ref(REF_DIR / (Path(f).stem + ".pt"))
            p = predict(model, s["x"], agg=prec["agg"], head=prec["head"])
            renders, count = gaussian_renders(p, r["joint_pose_enc"])
            rows.append(score(stored(p), renders, count, s, r))
        summ = summarize(rows)
        summ.update(recipe=recipe, eval_seconds=time.time() - t0, **info)
        (RES_DIR / f"{recipe}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
        print(HEADER)
        print(format_row(summ), flush=True)
        del model
        torch.cuda.empty_cache()


KEYS = ["render", "gs", "dep", "dep_all", "rot", "trans", "fov", "conf", "psnr", "psnr_ref", "ssim", "ssim_ref",
        "auc30", "auc30_ref", "absrel", "absrel_ref", "d1", "d1_ref", "gaussians", "gaussians_ref"]


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in KEYS:
        v = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["render_p5"] = float(np.percentile([r["render"] for r in rows], 5))
    for k in ("dep", "rot"):
        out[k + "_p50"] = float(np.percentile([r[k] for r in rows], 50))
        out[k + "_p95"] = float(np.percentile([r[k] for r in rows], 95))
    return out


HEADER = (f"{'recipe':16s} {'n':>3s} {'render':>6s} {'p5':>6s} {'gs%':>6s} {'dep%':>6s} {'p95':>6s} {'rot':>6s} "
          f"{'trans%':>6s} {'fov':>6s} | {'psnr':>6s} {'(ref)':>6s} {'ssim':>6s} {'(ref)':>6s} {'auc30':>6s} "
          f"{'(ref)':>6s} {'absrel':>6s} {'(ref)':>6s} {'d1':>6s}")


def format_row(s: dict) -> str:
    return (f"{s['recipe']:16s} {s['n']:3d} {s['render']:6.2f} {s['render_p5']:6.2f} {100 * s['gs']:6.3f} "
            f"{100 * s['dep']:6.3f} {100 * s['dep_p95']:6.3f} {s['rot']:6.3f} {100 * s['trans']:6.3f} "
            f"{s['fov']:6.3f} | {s['psnr']:6.2f} {s['psnr_ref']:6.2f} {s['ssim']:6.4f} {s['ssim_ref']:6.4f} "
            f"{s['auc30']:6.4f} {s['auc30_ref']:6.4f} {s['absrel']:6.4f} {s['absrel_ref']:6.4f} {s['d1']:6.4f}")


def command_table(args) -> None:
    print(HEADER)
    for f in sorted(RES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        print(format_row(json.loads(f.read_text())["summary"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ref", "run", "table"])
    parser.add_argument("recipes", nargs="*")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", default=None, help="k/n: ref only for samples with index %% n == k")
    args = parser.parse_args()
    {"ref": command_reference, "run": command_run, "table": command_table}[args.command](args)
    sys.stdout.flush()
    os._exit(0)  # skip ROCm teardown (it can segfault after all work is saved)


if __name__ == "__main__":
    main()
