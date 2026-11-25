"""Fidelity of VGGT quantization recipes against the fp32 reference, plus ground-truth metrics.

  python -m benzene.models.vggt.evaluate ref
      # fp32 reference outputs of the eval samples (OUT_DIR/ref)
  python -m benzene.models.vggt.evaluate run fp16 w8a8_tf ...
      # recipes -> OUT_DIR/results/<recipe>.json
  python -m benzene.models.vggt.evaluate table

Deviation from the reference (per sample, then averaged):
  dep      mean |d / d_ref - 1| over the pixels whose reference depth confidence is in the sample's upper half
  dep_all  the same over all pixels
  pts      mean |P - P_ref| over the upper half by reference point confidence, divided by the mean distance of those
           reference points from their median (scene size)
  rot      mean rotation angle between the predicted and the reference camera of each frame [deg]
  trans    mean |t - t_ref| / mean |t_ref| of the cameras (frame 0 is the origin in both)
  fov      mean |FoV - FoV_ref| over both axes [deg]
  conf     mean |log c - log c_ref| of the depth confidence
Ground truth (where the dataset has it):
  auc30    pose AUC@30: over all frame pairs, the relative rotation / translation-direction errors against the GT poses
           (translation sign ambiguity as in VGGSfM), accuracy of max(err_R, err_t) < t averaged for t = 1..30 deg
  absrel / d1  depth against the TUM depth maps after one median scale per sample: mean |d s - g| / g, share of
           max(d s / g, g / (d s)) < 1.25
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

from .paths import DATA_DIR, OUT_DIR

REF_DIR = OUT_DIR / "ref"
RES_DIR = OUT_DIR / "results"


def eval_files(count: int | None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))
    return files[:count] if count else files


def cameras(pose_enc: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    e, i = pose_encoding_to_extri_intri(pose_enc[None].double(), hw)
    return e[0], i[0]


def rot_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def upper_half(conf: torch.Tensor) -> torch.Tensor:
    return conf >= conf.flatten().median()


def deviation(q: dict, r: dict) -> dict:
    """q's deviation from the reference r; pts only when r has world points (ports of VGGT-architecture models
    without the point head leave it out)."""
    hw = r["depth"].shape[-2:]
    m = upper_half(r["depth_conf"])
    rel = (q["depth"] / r["depth"] - 1).abs()
    out = {"dep": rel[m].mean().item(), "dep_all": rel.mean().item()}
    if "world_points" in r:
        mp = upper_half(r["world_points_conf"])
        pr = r["world_points"][mp]
        scale = (pr - pr.median(0).values).norm(dim=-1).mean()
        out["pts"] = ((q["world_points"][mp] - pr).norm(dim=-1).mean() / scale).item()
    eq, iq = cameras(q["pose_enc"], hw)
    er, ir = cameras(r["pose_enc"], hw)
    tn = er[:, :, 3].norm(dim=-1)
    fq = torch.rad2deg(q["pose_enc"][:, 7:].double())
    fr = torch.rad2deg(r["pose_enc"][:, 7:].double())
    return out | {
        "rot": rot_angle(eq[:, :, :3], er[:, :, :3]).mean().item(),
        "trans": ((eq[:, :, 3] - er[:, :, 3]).norm(dim=-1).mean() / tn[1:].mean()).item(),
        "fov": (fq - fr).abs().mean().item(),
        "conf": (q["depth_conf"].log() - r["depth_conf"].log()).abs().mean().item(),
    }


def pose_auc(extri: torch.Tensor, gt: torch.Tensor, max_t: int = 30) -> float:
    """AUC of max(relative rotation error, relative translation direction error) over all frame pairs."""
    S = extri.shape[0]
    i, j = torch.triu_indices(S, S, 1)

    def rel(e):
        R, t = e[:, :, :3].double(), e[:, :, 3].double()
        Rr = R[j] @ R[i].transpose(-1, -2)
        return Rr, t[j] - (Rr @ t[i][..., None])[..., 0]

    Rp, tp = rel(extri)
    Rg, tg = rel(gt)
    er = rot_angle(Rp, Rg)
    c = (tp * tg).sum(-1) / (tp.norm(dim=-1) * tg.norm(dim=-1)).clamp_min(1e-12)
    et = torch.rad2deg(torch.arccos(c.clamp(-1, 1)))
    et = torch.minimum(et, (180 - et).abs())
    e = torch.maximum(er, et).numpy()
    return float(np.mean([(e < t).mean() for t in range(1, max_t + 1)]))


def gt_metrics(p: dict, s: dict) -> dict:
    out = {"auc30": pose_auc(cameras(p["pose_enc"], p["depth"].shape[-2:])[0], s["gt_extri"])}
    if s["gt_depth"] is not None:
        g = s["gt_depth"].float()
        v = (g > 0.1) & (g < 10)
        d = p["depth"][v]
        sc = g[v].median() / d.median()
        ds = d * sc
        out["absrel"] = ((ds - g[v]).abs() / g[v]).mean().item()
        out["d1"] = (torch.maximum(ds / g[v], g[v] / ds) < 1.25).float().mean().item()
    return out


def command_reference(args) -> None:
    from .runner import load_model, load_sample, predict

    REF_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.device)
    for i, f in enumerate(eval_files(args.count)):
        out = REF_DIR / (Path(f).stem + ".pt")
        if out.exists() or (args.shard and i % int(args.shard.split("/")[1]) != int(args.shard.split("/")[0])):
            continue
        s = load_sample(f, args.device)
        t0 = time.time()
        p = predict(model, s["x"], agg=None, head=None)
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        torch.save({k: (v.half() if k in ("world_points", "depth", "depth_conf", "world_points_conf") else v)
                    for k, v in p.items()} | {"fp32": True}, tmp)
        os.replace(tmp, out)
        print(f"{i} {Path(f).stem} {time.time() - t0:.1f}s {gt_metrics(p, s)}", flush=True)


def load_ref(path: Path) -> dict:
    r = torch.load(path)
    return {k: v.float() if torch.is_tensor(v) else v for k, v in r.items()}


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
            p = predict(model, s["x"], agg=prec["agg"], head=prec["head"])
            r = load_ref(REF_DIR / (Path(f).stem + ".pt"))
            rows.append({"sample": Path(f).stem, "dataset": s["dataset"], **deviation(p, r),
                         **gt_metrics(p, s), **{k + "_ref": v for k, v in gt_metrics(r, s).items()}})
        summ = summarize(rows)
        summ.update(recipe=recipe, eval_seconds=time.time() - t0, **info)
        (RES_DIR / f"{recipe}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
        print(HEADER)
        print(format_row(summ), flush=True)
        del model
        torch.cuda.empty_cache()


KEYS = ["dep", "dep_all", "pts", "rot", "trans", "fov", "conf", "auc30", "auc30_ref", "absrel", "absrel_ref", "d1",
        "d1_ref"]


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in KEYS:
        v = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(v)) if v else float("nan")
    for k in ("dep", "pts", "rot"):
        out[k + "_p95"] = float(np.percentile([r[k] for r in rows], 95))
    return out


HEADER = (f"{'recipe':16s} {'n':>3s} {'dep%':>6s} {'p95':>6s} {'all%':>6s} {'pts%':>6s} {'p95':>6s} {'rot':>6s} "
          f"{'p95':>6s} {'trans%':>6s} {'fov':>6s} {'conf':>6s} | {'auc30':>6s} {'(ref)':>6s} {'absrel':>6s} "
          f"{'(ref)':>6s} {'d1':>6s}")


def format_row(s: dict) -> str:
    return (f"{s['recipe']:16s} {s['n']:3d} {100 * s['dep']:6.3f} {100 * s['dep_p95']:6.3f} {100 * s['dep_all']:6.3f} "
            f"{100 * s['pts']:6.3f} {100 * s['pts_p95']:6.3f} {s['rot']:6.3f} {s['rot_p95']:6.3f} "
            f"{100 * s['trans']:6.3f} {s['fov']:6.3f} {s['conf']:6.3f} | {s['auc30']:6.4f} {s['auc30_ref']:6.4f} "
            f"{s['absrel']:6.4f} {s['absrel_ref']:6.4f} {s['d1']:6.4f}")


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
