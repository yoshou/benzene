"""LoMa on the board (benzene loma) against PyTorch for evaluation pairs.

Pushes the model file (export_hexmodel; its records: export_hexmodel.RECORDS) and bundles of the pairs, runs the
device in batches and per pair compares:
  PyTorch  the device model run in PyTorch (device_model.DeviceLoMa with the exported records, DINOv2 W8A8, the
           matcher in fp16 - recipe dev+w8a8+fp16): keypoints and probabilities bit for bit, the matches (precision /
           recall; DINOv2 and the matcher are not bit-exact on the device)
  fp32     the accuracy evaluation's metrics (deviation against the fp32 reference outputs, gt_metrics against the
           ground truth), so the device row compares directly with the recipe rows of evaluate.py table; --save writes
           them (evaluate rescore <name> --seeds 5 for the 5-seed AUC)
and prints the device's mean times.

  python -m benzene.models.loma.checks.end_to_end --pairs 0 1 2
  python -m benzene.models.loma.checks.end_to_end --all --save device_dev+w8a8+fp16
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from benzene import board
from benzene.command_line import Indices

from ...lightglue.bundle import read_matches
from ...lightglue.device_check import identical
from ...lightglue.evaluation import HEADER, deviation, eval_files, format_row, gt_metrics, summarize
from ..device_model import DeviceLoMa
from ..dinov2_quantization import quantized_dinov2
from ..evaluate import SETTINGS, accuracy_evaluation
from ..export_hexmodel import MODEL, RECORDS
from ..paths import BOARD_DIR
from ..prepare_bundle import read_features, write_bundle
from ..runner import load_model, load_pair, run_pair

BATCH = 20  # pairs per bundle (the board's output files: ~4 MB per pair)


def upload_model() -> str:
    src = MODEL
    board.ssh(f"mkdir -p {BOARD_DIR}", check=True)
    board.upload([src, src.with_suffix(".tab")], f"{BOARD_DIR}/", inplace=True)
    return f"{BOARD_DIR}/{src.name}"


def pixel_keypoints(positions: np.ndarray, rec: dict, i: int) -> torch.Tensor:
    """DeviceLoMa.extract's keypoints of normalized positions: size * (positions + 1) / 2 - 0.5 in fp32."""
    height, width = rec[f"image{i}"].shape[:2]
    size = torch.tensor([width, height], dtype=torch.float32)
    return size * (torch.from_numpy(positions.copy()) + 1) / 2 - 0.5


def run_device(model: str, files: list[str], td: Path) -> tuple[list[dict], list[str]]:
    outs, logs = [], []
    for start in range(0, len(files), BATCH):
        chunk = files[start : start + BATCH]
        write_bundle(chunk, SETTINGS, td / "bundle")
        work = board.empty_directory("lm")
        board.ssh(f"mkdir -p {work}/out", check=True)
        board.upload(f"{td}/bundle/", f"{work}/bundle/")
        r = board.run_benzene(f"loma {model} lm/bundle lm/out")
        log = (r.stdout + r.stderr).strip()
        if r.returncode:
            raise SystemExit(f"board run failed ({r.returncode}): {log}")
        logs.append(r.stdout)
        out = td / f"out{start}"
        board.download(f"{work}/out/", f"{out}/")
        for p, f in enumerate(chunk):
            rec = load_pair(f)
            feats = [read_features(out / f"{p}.lm{i}") for i in (0, 1)]
            matches, match_scores = read_matches(out / f"{p}.lg")
            o = {"matches0": torch.from_numpy(matches).long(), "mscores0": torch.from_numpy(match_scores)}
            for i in (0, 1):
                o[f"kpts{i}"] = pixel_keypoints(feats[i]["positions"], rec, i)
                o[f"kpts_normalized{i}"] = torch.from_numpy(feats[i]["positions"].copy())
                o[f"scores{i}"] = torch.from_numpy(feats[i]["scores"].copy())
                o[f"desc{i}"] = torch.from_numpy(feats[i]["descriptors"].copy())
            outs.append(o)
        print(f"device: {start + len(chunk)} / {len(files)} pairs", flush=True)
    return outs, logs


IMAGE_LINE = re.compile(
    r"pair \d+ image \d+ (?P<height>\d+)x(?P<width>\d+): loma (?P<image>[\d.]+) ms .*?"
    r"detector: encoder (?P<detector_encoder>[\d.]+) block1 (?P<detector_block1>[\d.]+) hidden "
    r"(?P<detector_hidden>[\d.]+) \(depthwise (?P<detector_depthwise>[\d.]+)\) out (?P<detector_out>[\d.]+) "
    r"contexts \+ logits (?P<detector_contexts>[\d.]+) \| keypoints: softmax \+ NMS (?P<softmax_nms>[\d.]+) selection "
    r"(?P<selection>[\d.]+) positions (?P<positions>[\d.]+)\n  descriptor (?P<descriptor_height>\d+)x"
    r"(?P<descriptor_width>\d+): encoder (?P<descriptor_encoder>[\d.]+) DINOv2 (?P<dinov2>[\d.]+) \(norm1 "
    r"(?P<dinov2_norm1>[\d.]+) qkv (?P<dinov2_qkv>[\d.]+) attention (?P<dinov2_attention>[\d.]+) output "
    r"(?P<dinov2_output>[\d.]+) norm2 (?P<dinov2_norm2>[\d.]+) fc1 (?P<dinov2_fc1>[\d.]+) fc2 (?P<dinov2_fc2>[\d.]+)\) "
    r"block1 "
    r"(?P<descriptor_block1>[\d.]+) hidden (?P<descriptor_hidden>[\d.]+) \(depthwise "
    r"(?P<descriptor_depthwise>[\d.]+)\) out (?P<descriptor_out>[\d.]+) contexts \+ sums "
    r"(?P<descriptor_contexts>[\d.]+) sparse (?P<sparse>[\d.]+)")
MATCHER_LINE = re.compile(
    r"pair \d+: lightglue (?P<matcher>[\d.]+) ms .*?setup (?P<setup>[\d.]+) layers (?P<layers>[\d.]+) \(\d+ run, "
    r"[\d.]+ per layer: qkv (?P<qkv>[\d.]+) attention (?P<attention>[\d.]+) out\+FFN (?P<ffn>[\d.]+) HVX "
    r"(?P<hvx>[\d.]+)\) final (?P<final>[\d.]+) assignment (?P<assignment>[\d.]+)")


def mean_times(logs: list[str]) -> str:
    """The device's mean milliseconds per stage from the command's lines; each run's first image and first pair are
    left out (they grow the workspaces)."""
    images, matchers, sizes = [], [], {}
    for log in logs:
        image_rows = [m.groupdict() for m in IMAGE_LINE.finditer(log)]
        matcher_rows = [m.groupdict() for m in MATCHER_LINE.finditer(log)]
        for row in image_rows:
            key = f"{row['width']}x{row['height']} / {row['descriptor_width']}x{row['descriptor_height']}"
            sizes[key] = sizes.get(key, 0) + 1
        images += image_rows[1:]
        matchers += matcher_rows[1:]
    if not images or not matchers:
        return "device mean: no timing lines"
    m = {k: float(np.mean([float(r[k]) for r in images])) for k in images[0] if "width" not in k and "height" not in k}
    q = {k: float(np.mean([float(r[k]) for r in matchers])) for k in matchers[0]}
    detector = sum(m[k] for k in ("detector_encoder", "detector_block1", "detector_hidden", "detector_out",
                                  "detector_contexts"))  # (depthwise: part of hidden)
    keypoints = m["softmax_nms"] + m["selection"] + m["positions"]
    descriptor = sum(m[k] for k in ("descriptor_encoder", "dinov2", "descriptor_block1", "descriptor_hidden",
                                    "descriptor_out", "descriptor_contexts", "sparse"))
    return (f"device means over {len(images)} images / {len(matchers)} pairs (each run's first left out); sizes "
            f"(detector / descriptor): {', '.join(f'{k} x{n}' for k, n in sizes.items())}\n"
            f"  image {m['image']:.1f} ms = detector {detector:.1f} (encoder {m['detector_encoder']:.1f}, block1 "
            f"{m['detector_block1']:.1f}, hidden {m['detector_hidden']:.1f} (depthwise {m['detector_depthwise']:.1f}), "
            f"out {m['detector_out']:.1f}, contexts + "
            f"logits {m['detector_contexts']:.1f}) + keypoints {keypoints:.1f} (softmax + NMS {m['softmax_nms']:.1f}, "
            f"selection {m['selection']:.1f}, positions {m['positions']:.1f}) + descriptor {descriptor:.1f} (encoder "
            f"{m['descriptor_encoder']:.1f}, DINOv2 {m['dinov2']:.1f} (norm1 {m['dinov2_norm1']:.1f}, qkv "
            f"{m['dinov2_qkv']:.1f}, attention {m['dinov2_attention']:.1f}, output {m['dinov2_output']:.1f}, norm2 "
            f"{m['dinov2_norm2']:.1f}, fc1 {m['dinov2_fc1']:.1f}, fc2 {m['dinov2_fc2']:.1f}), block1 "
            f"{m['descriptor_block1']:.1f}, hidden "
            f"{m['descriptor_hidden']:.1f} (depthwise {m['descriptor_depthwise']:.1f}), out {m['descriptor_out']:.1f}, "
            f"contexts + sums "
            f"{m['descriptor_contexts']:.1f}, sparse {m['sparse']:.1f})\n"
            f"  matcher {q['matcher']:.1f} ms = setup {q['setup']:.2f} + layers {q['layers']:.1f} (per layer: qkv "
            f"{q['qkv']:.2f}, attention {q['attention']:.2f}, out + FFN {q['ffn']:.2f}, HVX {q['hvx']:.2f}) + final "
            f"{q['final']:.2f} + assignment {q['assignment']:.2f}\n"
            f"  pair (two images + matcher) {2 * m['image'] + q['matcher']:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", action=Indices, default=[0])
    parser.add_argument("--all", action="store_true", help="every evaluation pair")
    parser.add_argument("--device-only", action="store_true", help="skip the comparison with PyTorch (no GPU)")
    parser.add_argument("--save", help="write the per-pair rows and the summary to OUT_DIR/results/<name>.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model = upload_model()
    files = eval_files() if args.all else [eval_files()[i] for i in args.pairs]
    with tempfile.TemporaryDirectory() as td:
        outs, logs = run_device(model, files, Path(td))
    print(mean_times(logs))
    if not args.device_only:
        net = load_model(device=args.device)
        records = torch.load(RECORDS, map_location="cpu")
        assert records["settings"] == SETTINGS.name, (records["settings"], SETTINGS.name)
        dinov2 = quantized_dinov2(net._descriptor.encoder.frozen_dinov2.dinov2_vitl14, "w8a8", SETTINGS, args.device,
                                  False)
        pytorch_model = DeviceLoMa(records, net, SETTINGS, args.device, dinov2)
    rows, keep = [], {}
    same = {"keypoints": 0, "probabilities": 0}
    agreement = []
    for f, o in zip(files, outs):
        rec = load_pair(f)
        if not args.device_only:
            pytorch = run_pair(pytorch_model, rec, SETTINGS, args.device, None, torch.float16)
            for i in (0, 1):
                same["keypoints"] += identical(o[f"kpts_normalized{i}"], pytorch[f"kpts_normalized{i}"])
                same["probabilities"] += identical(o[f"scores{i}"], pytorch[f"scores{i}"])
            d = deviation(o, pytorch)
            agreement.append((d["mP"], d["mR"]))
            print(f"{rec['name']}: device vs PyTorch: keypoints same pixel {100 * d['kp']:.2f}%, descriptor 1 - cos "
                  f"{d['desc']:.2e}, matches precision {100 * d['mP']:.2f}% recall {100 * d['mR']:.2f}% "
                  f"({int((o['matches0'] >= 0).sum())} vs {int((pytorch['matches0'] >= 0).sum())})", flush=True)
        rows.append({"pair": rec["name"], **deviation(o, accuracy_evaluation.load_ref(rec["name"])),
                     **gt_metrics(o, rec, eval_files().index(f))})
        keep[rec["name"]] = {key: o[key] for key in ("kpts0", "kpts1", "matches0")}
    if not args.device_only:
        print(f"bit for bit of {2 * len(files)} images: keypoints {same['keypoints']}, probabilities "
              f"{same['probabilities']}; matches vs PyTorch: precision {100 * np.mean([a[0] for a in agreement]):.2f}% "
              f"recall {100 * np.mean([a[1] for a in agreement]):.2f}%")
    s = accuracy_evaluation.save_results(args.save, rows, keep) if args.save else summarize(rows)
    print(HEADER)
    print(format_row("device dev+w8a8+fp16", s))


if __name__ == "__main__":
    main()
