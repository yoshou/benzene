"""VGGT on the board (benzene vggt) against PyTorch for eval samples.

Pushes the model file (export_hexmodel.py) and each sample's bundle, runs the device and scores its depth, world
points and cameras against the fp32 reference outputs of the accuracy evaluation (evaluate.py ref) with its metrics
(evaluate.deviation, ground truth by evaluate.gt_metrics), so that the device rows compare directly with the recipe
rows of evaluate.py table. With --tokens also the kept transformer outputs (frame | global blocks 4, 11, 17,
23) against the fp32 aggregator and the recipe's fake-quant aggregator (needs a GPU).

  python -m benzene.models.vggt.checks.end_to_end w8a8_all --samples 0 1 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board
from benzene.command_line import Indices

from ..evaluate import REF_DIR, deviation, gt_metrics, load_ref, summarize
from ..paths import BOARD_DIR, DATA_DIR, OUT_DIR, WORK
from ..prepare_bundle import write_bundle
from ..recipes import DEVICE_RECIPE

KEEP = (4, 11, 17, 23)


def push_model(recipe: str) -> str:
    src = WORK / "export" / f"vggt-{recipe}.hexmodel"
    if not src.exists():
        sys.exit(f"{src} missing: run export_hexmodel {recipe}")
    board.ssh(f"mkdir -p {BOARD_DIR}", check=True)
    board.upload([src, src.with_suffix(".tab")], f"{BOARD_DIR}/", inplace=True)
    return f"{BOARD_DIR}/{src.name}"


def run_device(model: str, sample: str, td: Path, tokens: bool, frames: int | None = None) -> tuple[dict, dict, str]:
    """Device outputs of a sample (its first `frames` frames) as the PyTorch predict() dict (and the kept tokens with
    tokens), plus the run's log."""
    meta = write_bundle(sample, td / "bundle", frames)
    S, H, W = meta["S"], meta["H"], meta["W"]
    work = board.empty_directory("vg")
    board.ssh(f"mkdir -p {work}/out", check=True)
    board.upload(f"{td}/bundle/", f"{work}/bundle/")
    r = board.run_benzene(f"vggt {model} vg/bundle vg/out" + (" --tokens" if tokens else ""))
    if r.returncode:
        sys.exit(f"board run failed ({r.returncode}): {r.stdout} {r.stderr}")
    board.download(f"{work}/out/", f"{td}/out/")
    rd = lambda n, dt: np.frombuffer((td / "out" / n).read_bytes(), dt).copy()  # noqa: E731
    d = torch.from_numpy(rd("depth.f32", np.float32).reshape(S, H, W, 2))
    p = torch.from_numpy(rd("points.f32", np.float32).reshape(S, H, W, 4))
    out = {"pose_enc": torch.from_numpy(rd("pose.f32", np.float32).reshape(4, S, 9)[-1]),
           "depth": d[..., 0], "depth_conf": d[..., 1], "world_points": p[..., :3], "world_points_conf": p[..., 3]}
    tok = rd("inter.f16", np.float16).reshape(4, -1, 2048).astype(np.float32) if tokens else None
    return out, {"tokens": tok, **meta}, (r.stdout + r.stderr).strip()


@torch.no_grad()
def torch_tokens(sample: str, recipe: str | None) -> np.ndarray:
    """Kept aggregator outputs [4][S * P][2048] of the fp32 model (recipe None) or the recipe's fake-quant model."""
    from ..recipes import RECIPES, apply_recipe
    from ..runner import load_model, load_sample

    model = load_model("cuda:0")
    agg = None
    if recipe:
        apply_recipe(model, recipe, verbose=False)
        agg = RECIPES[recipe][0]["agg"]
    s = load_sample(sample, "cuda:0")
    with torch.autocast("cuda", dtype=agg or torch.float32, enabled=agg is not None):
        tok, _ = model.aggregator(s["x"][None])
    out = np.stack([tok[k][0].float().reshape(-1, 2048).cpu().numpy() for k in KEEP])
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE)
    parser.add_argument("--samples", nargs="+", action=Indices, default=[0])
    parser.add_argument("--tokens", action="store_true", help="also compare the kept transformer outputs (GPU)")
    parser.add_argument("--save", help="write the per-sample rows and the summary to OUT_DIR/results/<name>.json")
    parser.add_argument("--frames", type=int,
                        help="only the first N frames of each sample (fp32 reference computed here on "
                             "the GPU; ground truth of those frames)")
    args = parser.parse_args()
    model = push_model(args.recipe)
    files = sorted(glob.glob(str(DATA_DIR / "cache/eval/*.pt")))
    rows = []
    for i in args.samples:
        f = files[i]
        s = torch.load(f, map_location="cpu", weights_only=False)
        with tempfile.TemporaryDirectory() as td:
            dev, extra, log = run_device(model, f, Path(td), args.tokens, args.frames)
        print(log)
        if args.frames:
            from ..runner import load_model, predict

            n = args.frames
            s = {**s, "images": s["images"][:n], "gt_extri": s["gt_extri"][:n],
                 "gt_depth": None if s["gt_depth"] is None else s["gt_depth"][:n]}
            ref = predict(load_model("cuda:0"), s["images"].cuda().float() / 255.0)
            torch.cuda.empty_cache()
        else:
            ref = load_ref(REF_DIR / (Path(f).stem + ".pt"))
        row = {"sample": Path(f).stem, "dataset": s["dataset"], **deviation(dev, ref), **gt_metrics(dev, s),
               **{k + "_ref": v for k, v in gt_metrics(ref, s).items()}}
        rows.append(row)
        print(f"{Path(f).stem} ({extra['S']} frames {extra['H']}x{extra['W']}): device vs fp32 reference: depth "
              f"{100 * row['dep']:.3f}% (all {100 * row['dep_all']:.3f}%), points {100 * row['pts']:.3f}%, rot "
              f"{row['rot']:.3f} deg, trans {100 * row['trans']:.3f}%, FoV {row['fov']:.3f} deg, conf {row['conf']:.3f}; "
              f"GT pose AUC@30 {row['auc30']:.4f} (fp32 {row['auc30_ref']:.4f})"
              + (f", depth AbsRel {row['absrel']:.4f} (fp32 {row['absrel_ref']:.4f})" if "absrel" in row else ""), flush=True)
        if args.tokens:
            r32, fq = torch_tokens(f, None), torch_tokens(f, args.recipe)
            rel = lambda a, b: np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean())  # noqa: E731
            t = extra["tokens"]
            for k, l in enumerate(KEEP):
                print(f"  tokens block {l:2d} rel. rms vs fp32: device {rel(t[k], r32[k]):.4f}, fake-quant "
                      f"{rel(fq[k], r32[k]):.4f}, device vs fake-quant {rel(t[k], fq[k]):.4f}")
    summ = summarize(rows)
    print("mean over", len(rows), "samples:", json.dumps({k: round(v, 5) for k, v in summ.items() if k != "n"}))
    if args.save:
        out = OUT_DIR / "results" / f"{args.save}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": {**summ, "recipe": args.save}, "rows": rows}, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
