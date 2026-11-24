"""The feature matchers' end-to-end checks (models/<matcher>/checks/end_to_end.py): benzene's <matcher> command runs the
exported model file on evaluation pairs (format: bundle), compared with

  PyTorch  the exported model run in PyTorch (DeviceCheck.pytorch_model): keypoints at the same pixel, |score| and
           descriptor differences, matches in both (match precision / recall)
  fp32     the accuracy evaluation's metrics (deviation against the fp32 reference outputs, gt_metrics against the
           ground truth), so the device rows compare directly with the recipe rows of evaluate.py table; --save writes
           them
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from benzene import board
from benzene.command_line import Indices

from . import bundle
from .evaluation import HEADER, AccuracyEvaluation, deviation, eval_files, format_row, gt_metrics, summarize
from .matcher import set_adaptive


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The same shape and the same fp32 bits."""
    return a.shape == b.shape and bool(torch.equal(a.float().view(torch.int32), b.float().view(torch.int32)))


class DeviceCheck:
    """A matcher's check: its benzene command (also its model files' name: WORK/export/<command>-<recipe>.hexmodel),
    directory under the board's benzene directory, feature file extension and descriptor dimensions; the hooks below
    add what a matcher checks beyond the shared comparison."""

    command: str
    board_work: str
    features: str
    descriptor_dimensions: int
    adaptive_help: str

    def __init__(self, evaluation: AccuracyEvaluation, work: Path, board_dir: str, device_recipe: str,
                 adaptive: tuple[float, float]):
        self.evaluation = evaluation
        self.work = work
        self.board_dir = board_dir
        self.device_recipe = device_recipe
        self.adaptive = adaptive

    def model_file(self, recipe: str) -> Path:
        return self.work / "export" / f"{self.command}-{recipe}.hexmodel"

    def pytorch_model(self, recipe: str, device: str) -> tuple[Callable[[dict, str], dict], torch.nn.Module]:
        """(run one pair with the exported model on device, its LightGlue)."""
        raise NotImplementedError

    def read_pair(self, out: Path, index: int) -> None:
        """Reads the device's further outputs of pair `index`."""

    def check_images(self, rec: dict, index: int) -> None:
        """Compares the device's further outputs of the pair with PyTorch's (with or without the PyTorch model)."""

    def compare(self, device: dict, pytorch: dict) -> None:
        """Compares one pair's outputs beyond the deviation."""

    def describe_matches(self, device: dict, pytorch: dict) -> str:
        return f"{int((device['matches0'] >= 0).sum())} vs {int((pytorch['matches0'] >= 0).sum())}"

    def report(self, files: list[str], with_pytorch: bool) -> None:
        """Prints the totals of check_images and compare."""

    def push_model(self, recipe: str) -> str:
        source = self.model_file(recipe)
        if not source.exists():
            sys.exit(f"{source} missing: run export_hexmodel {recipe}")
        board.ssh(f"mkdir -p {self.board_dir}", check=True)
        board.upload([source, source.with_suffix(".tab")], f"{self.board_dir}/", inplace=True)
        return f"{self.board_dir}/{source.name}"

    def run_device(self, model: str, files: list[str], td: Path, extra: str = "") -> tuple[list, str]:
        bundle.write_bundle([self.evaluation.load_pair(f) for f in files], td / "bundle")
        work = board.empty_directory(self.board_work)
        board.ssh(f"mkdir -p {work}/out", check=True)
        board.upload(f"{td}/bundle/", f"{work}/bundle/")
        r = board.run_benzene(f"{self.command} {model} {self.board_work}/bundle {self.board_work}/out {extra}")
        log = (r.stdout + r.stderr).strip()
        if r.returncode:
            sys.exit(f"board run failed ({r.returncode}): {log}")
        out = td / "out"
        out.mkdir(exist_ok=True)
        board.download(f"{work}/out/", f"{out}/")
        outs = []
        for i in range(len(files)):
            self.read_pair(out, i)
            a, b = (bundle.read_features(out / f"{i}.{self.features}{s}", self.descriptor_dimensions) for s in (0, 1))
            m0, ms = bundle.read_matches(out / f"{i}.lg")
            t = lambda x: torch.from_numpy(np.array(x))  # noqa: E731
            outs.append({"kpts0": t(a["keypoints"]), "kpts1": t(b["keypoints"]), "scores0": t(a["keypoint_scores"]),
                         "scores1": t(b["keypoint_scores"]), "desc0": t(a["descriptors"]),
                         "desc1": t(b["descriptors"]), "matches0": t(m0).long(), "mscores0": t(ms)})
        return outs, log

    def main(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("recipe", nargs="?", default=self.device_recipe)
        parser.add_argument("--pairs", nargs="+", action=Indices, default=[0])
        parser.add_argument("--all", action="store_true", help="every evaluation pair")
        parser.add_argument("--device-only", action="store_true", help="skip the comparison with PyTorch (no GPU)")
        parser.add_argument("--save", help="write the per-pair rows and the summary to OUT_DIR/results/<name>.json")
        parser.add_argument("--adaptive", action="store_true", help=self.adaptive_help)
        parser.add_argument("--device", default="cuda:0")
        args = parser.parse_args()
        model = self.push_model(args.recipe)
        files = eval_files() if args.all else [eval_files()[i] for i in args.pairs]
        with tempfile.TemporaryDirectory() as td:
            outs, log = self.run_device(model, files, Path(td), "--adaptive" if args.adaptive else "")
        print(log)
        if not args.device_only:
            run_pytorch, lg = self.pytorch_model(args.recipe, args.device)
            lg.conf.mp = True  # (the device recipe's LightGlue: fp16)
            set_adaptive(lg, self.adaptive if args.adaptive else None)
        rows, keep = [], {}
        for k, (f, o) in enumerate(zip(files, outs)):
            rec = self.evaluation.load_pair(f)
            self.check_images(rec, k)
            if not args.device_only:
                pytorch = run_pytorch(rec, args.device)
                self.compare(o, pytorch)
                d = deviation(o, pytorch)
                print(f"{rec['name']}: device vs PyTorch: keypoints same pixel {100 * d['kp']:.2f}%, score "
                      f"{100 * d['score']:.4f}%, descriptor 1 - cos {d['desc']:.2e}, matches precision "
                      f"{100 * d['mP']:.2f}% recall {100 * d['mR']:.2f}% ({self.describe_matches(o, pytorch)})",
                      flush=True)
            rows.append({"pair": rec["name"], **deviation(o, self.evaluation.load_ref(rec["name"])),
                         **gt_metrics(o, rec, eval_files().index(f))})
            keep[rec["name"]] = {key: o[key] for key in ("kpts0", "kpts1", "matches0")}
        self.report(files, not args.device_only)
        s = self.evaluation.save_results(args.save, rows, keep) if args.save else summarize(rows)
        print(HEADER)
        print(format_row("device " + args.recipe + ("a" if args.adaptive else ""), s))
