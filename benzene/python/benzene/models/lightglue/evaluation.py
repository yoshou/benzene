"""Evaluation of a feature matcher (keypoints + descriptors + LightGlue) on the image pairs of prepare_data.py: metrics
against a reference run and the ground truth, and an accuracy evaluation's result files and command line.

Deviation from the fp32 reference (per pair, averaged):
  kp     share of the reference keypoints found at the same pixel (both images)
  desc   1 - cosine of the descriptors of those keypoints (x 1e3)
  score  mean |score - reference score| / mean reference score of those keypoints (%)
  mP/mR  precision / recall of the matches against the reference matches (a match = its two keypoint pixels)
Against the ground truth: prec = share of matches with a symmetric epipolar distance < 5e-4, AUC@5/10/20 of the
relative pose from the matches (geometry.estimate_pose, same seed per pair for every recipe; rescore --seeds: the
median error over several seeds), nm = matches per pair.

An accuracy evaluation's outputs (AccuracyEvaluation, under its out_dir): ref/<pair>.pt (fp32 outputs) +
ref/summary.json, results/<name>.json (per-pair rows and the summary), outs/<name>.pt ({pair: kpts0, kpts1, matches0},
enough for gt_metrics). Its command line (AccuracyEvaluation.main):
  ref                      fp32 outputs of every pair
  rescore-ref [--seeds S]  the reference's ground-truth metrics again
  run <recipes>            recipes -> results/<recipe>.json, outs/<recipe>.pt
  rescore <names> --seeds 5
  table
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .geometry import EPI_TH, epipolar_error, estimate_pose, pose_auc, pose_error
from .paths import DATA_DIR


def pair_files(split: str, count: int | None = None) -> list[str]:
    files = sorted(glob.glob(str(DATA_DIR / f"pairs/{split}/*.pt")))
    return files[:count] if count else files


def eval_files(count: int | None = None) -> list[str]:
    return pair_files("eval", count)


def calibration_files(count: int | None = None) -> list[str]:
    return pair_files("calib", count)


def load_pair(path: str | Path) -> dict:
    """A pair file of prepare_data.py (images, intrinsics, relative pose) with its name (the file stem)."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["name"] = Path(path).stem
    return rec


def _key(k: torch.Tensor) -> torch.Tensor:
    k = k.round().long()
    return k[:, 1] * 65536 + k[:, 0]


def _common(kq: torch.Tensor, kr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (iq, ir) of the keypoints at the same pixel."""
    a, b = _key(kq), _key(kr)
    pos = {int(v): i for i, v in enumerate(b.tolist())}
    iq = [i for i, v in enumerate(a.tolist()) if v in pos]
    ir = [pos[int(a[i])] for i in iq]
    return torch.tensor(iq, dtype=torch.long), torch.tensor(ir, dtype=torch.long)


def _matches(o: dict) -> set[tuple[int, int]]:
    m = o["matches0"]
    v = torch.nonzero(m >= 0)[:, 0]
    k0, k1 = _key(o["kpts0"][v]), _key(o["kpts1"][m[v]])
    return set(zip(k0.tolist(), k1.tolist()))


def deviation(q: dict, r: dict) -> dict:
    kp, desc, score = [], [], []
    for i in (0, 1):
        iq, ir = _common(q[f"kpts{i}"], r[f"kpts{i}"])
        kp.append(len(ir) / max(1, len(r[f"kpts{i}"])))
        if len(ir):
            cos = (q[f"desc{i}"][iq] * r[f"desc{i}"][ir]).sum(-1) / (
                q[f"desc{i}"][iq].norm(dim=-1) * r[f"desc{i}"][ir].norm(dim=-1))
            desc.append(float((1 - cos).mean()))
            sr = r[f"scores{i}"][ir]
            score.append(float((q[f"scores{i}"][iq] - sr).abs().mean() / sr.mean()))
    mq, mr = _matches(q), _matches(r)
    both = len(mq & mr)
    return {"kp": float(np.mean(kp)), "desc": float(np.mean(desc)) if desc else float("nan"),
            "score": float(np.mean(score)) if score else float("nan"),
            "mP": both / max(1, len(mq)), "mR": both / max(1, len(mr))}


def gt_metrics(o: dict, rec: dict, seed: int, nseeds: int = 1) -> dict:
    """Matches, epipolar precision, pose error (with nseeds > 1 the median over RANSAC seeds seed + 1000 s: one draw's
    error flips between a few and tens of degrees on the pairs with few inliers)."""
    m = o["matches0"]
    v = torch.nonzero(m >= 0)[:, 0]
    k0, k1 = o["kpts0"][v], o["kpts1"][m[v]]
    if len(v) == 0:
        return {"nm": 0, "prec": 0.0, "err": 180.0}
    e = epipolar_error(k0, k1, rec["K0"], rec["K1"], rec["T_0to1"])
    errs = []
    for sd in range(nseeds):
        pose = estimate_pose(k0, k1, rec["K0"], rec["K1"], seed=seed + 1000 * sd)
        errs.append(180.0 if pose is None else pose_error(*pose, rec["T_0to1"]))
    out = {"nm": len(v), "prec": float((e < EPI_TH).float().mean()), "err": float(np.median(errs))}
    if nseeds > 1:
        out["errs"] = errs
    return out


def summarize(rows: list[dict]) -> dict:
    s = {k: float(np.nanmean([r[k] for r in rows])) for k in ("kp", "desc", "score", "mP", "mR", "nm", "prec")
         if k in rows[0]}
    s.update(zip(("auc5", "auc10", "auc20"), pose_auc([r["err"] for r in rows])))
    s["n"] = len(rows)
    return s


def format_row(name: str, s: dict) -> str:
    dv = (f"{100 * s['kp']:6.2f} {1e3 * s['desc']:7.3f} {100 * s['score']:6.2f} {100 * s['mP']:6.2f} "
          f"{100 * s['mR']:6.2f}") if "kp" in s else " " * 34
    return (f"{name:24s} {s['n']:4d} {dv} | {s['nm']:6.1f} {100 * s['prec']:6.2f} {100 * s['auc5']:6.2f} "
            f"{100 * s['auc10']:6.2f} {100 * s['auc20']:6.2f}")


HEADER = (f"{'recipe':24s} {'n':>4s} {'kp%':>6s} {'desc':>7s} {'scr%':>6s} {'mP%':>6s} {'mR%':>6s} | {'nm':>6s} "
          f"{'prec%':>6s} {'auc5':>6s} {'auc10':>6s} {'auc20':>6s}")


def recipe_runner(load_models: Callable, run_pair: Callable, apply_recipe: Callable, autocast_dtypes: Callable):
    """A matcher's AccuracyEvaluation runner: the models on the device with the recipe applied (None: the fp32
    reference), running one pair."""

    def runner(recipe: str | None, device: str) -> Callable[[dict], dict]:
        extractor, lg = load_models(device)
        extractor_ac, lg_ac = None, None
        if recipe is not None:
            extractor, _ = apply_recipe(extractor, lg, recipe, device)
            extractor_ac, lg_ac = autocast_dtypes(recipe)
        return lambda rec: run_pair(extractor, lg, rec, device, extractor_ac, lg_ac)

    return runner


class AccuracyEvaluation:
    """A matcher's accuracy evaluation: its result files under out_dir and its command line. load_pair reads a pair file
    (the input images); runner(recipe, device) returns the function that runs one pair (recipe None: the fp32 reference)
    to an output dict kpts0, kpts1, scores0, scores1, desc0, desc1, matches0, mscores0 (CPU tensors)."""

    def __init__(self, out_dir: Path, load_pair: Callable[[str], dict],
                 runner: Callable[[str | None, str], Callable[[dict], dict]]):
        self.ref_dir = out_dir / "ref"
        self.results_dir = out_dir / "results"
        self.outs_dir = out_dir / "outs"
        self.load_pair = load_pair
        self.runner = runner

    def load_ref(self, name: str) -> dict:
        return torch.load(self.ref_dir / f"{name}.pt", map_location="cpu")

    def save_results(self, name: str, rows: list[dict], outs: dict | None) -> dict:
        s = summarize(rows)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / f"{name}.json").write_text(json.dumps({"summary": {**s, "recipe": name}, "rows": rows},
                                                                  indent=1))
        if outs is not None:
            self.outs_dir.mkdir(parents=True, exist_ok=True)
            torch.save(outs, self.outs_dir / f"{name}.pt")
        return s

    def ref(self, count: int | None, device: str) -> None:
        run = self.runner(None, device)
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for k, f in enumerate(eval_files(count)):
            rec = self.load_pair(f)
            o = run(rec)
            torch.save(o, self.ref_dir / f"{rec['name']}.pt")
            rows.append({"pair": rec["name"], **gt_metrics(o, rec, k)})
        (self.ref_dir / "summary.json").write_text(json.dumps({"summary": summarize(rows), "rows": rows}, indent=1))

    def rescore_ref(self, count: int | None, seeds: int) -> None:
        rows = []
        for k, f in enumerate(eval_files(count)):
            rec = self.load_pair(f)
            rows.append({"pair": rec["name"], **gt_metrics(self.load_ref(rec["name"]), rec, k, seeds)})
        s = summarize(rows)
        (self.ref_dir / "summary.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
        print(HEADER)
        print(format_row("fp32 (reference)", s), flush=True)

    def run(self, recipe: str, count: int | None, device: str) -> dict:
        t0 = time.time()
        run = self.runner(recipe, device)
        rows, outs = [], {}
        for k, f in enumerate(eval_files(count)):
            rec = self.load_pair(f)
            o = run(rec)
            rows.append({"pair": rec["name"], **deviation(o, self.load_ref(rec["name"])), **gt_metrics(o, rec, k)})
            outs[rec["name"]] = {key: o[key] for key in ("kpts0", "kpts1", "matches0")}
        s = self.save_results(recipe, rows, outs)
        print(format_row(recipe, s), f"({time.time() - t0:.0f}s)", flush=True)
        return s

    def rescore(self, names: list[str], seeds: int) -> None:
        """Ground-truth metrics of saved outputs again (the deviation columns are kept)."""
        files = {Path(f).stem: (k, f) for k, f in enumerate(eval_files())}
        for name in names:
            outs = torch.load(self.outs_dir / f"{name}.pt")
            res = self.results_dir / f"{name}.json"
            old = {r["pair"]: r for r in json.loads(res.read_text())["rows"]} if res.exists() else {}
            rows = []
            for pair, o in outs.items():
                k, f = files[pair]
                rows.append({**old.get(pair, {"pair": pair}), **gt_metrics(o, self.load_pair(f), k, seeds)})
            print(format_row(name, self.save_results(name, rows, None)), flush=True)

    def table(self) -> None:
        print(HEADER)
        ref = self.ref_dir / "summary.json"
        if ref.exists():
            print(format_row("fp32 (reference)", json.loads(ref.read_text())["summary"]))
        for p in sorted(self.results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            print(format_row(p.stem, json.loads(p.read_text())["summary"]))

    def main(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["ref", "rescore-ref", "run", "rescore", "table"])
        parser.add_argument("recipes", nargs="*")
        parser.add_argument("--count", type=int, default=None)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--seeds", type=int, default=1,
                            help="rescore: median pose error over this many RANSAC seeds")
        args = parser.parse_args()
        if args.command == "ref":
            self.ref(args.count, args.device)
        elif args.command == "rescore-ref":
            self.rescore_ref(args.count, args.seeds)
        elif args.command == "rescore":
            self.rescore(args.recipes, args.seeds)
        elif args.command == "run":
            print(HEADER, flush=True)
            for r in args.recipes:
                self.run(r, args.count, args.device)
        else:
            self.table()
