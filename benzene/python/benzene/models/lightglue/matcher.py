"""Matching two feature sets with a LightGlue model of the checkout (source.modules), and its adaptive inference."""

from __future__ import annotations

from collections.abc import Callable

import torch


def autocast(dtype):
    """CUDA autocast to dtype (None: off)."""
    return torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None)


@torch.inference_mode()
def match(lg, f0: dict, f1: dict, size0, size1, ac=None) -> dict:
    """Matches of features f0 -> f1 (keypoints [k, 2] pixel centres, descriptors [k, D]) of images of size0 / size1
    (W, H): matches0 [k0] (index into image 1 or -1), mscores0 [k0]."""
    dev = f0["keypoints"].device
    d0 = {"keypoints": f0["keypoints"][None], "descriptors": f0["descriptors"][None],
          "image_size": torch.tensor([size0], device=dev, dtype=torch.float32)}
    d1 = {"keypoints": f1["keypoints"][None], "descriptors": f1["descriptors"][None],
          "image_size": torch.tensor([size1], device=dev, dtype=torch.float32)}
    with autocast(ac):
        out = lg({"image0": d0, "image1": d1})
    return {"matches0": out["matches0"][0], "mscores0": out["matching_scores0"][0].float()}


def run_pair(extract: Callable, extractor, lg, rec: dict, device: str, extractor_ac=None, lg_ac=None) -> dict:
    """The features of both images of a pair (extract(extractor, image, device, ac): keypoints, keypoint_scores,
    descriptors) matched by lg; on the CPU: kpts{0,1}, scores{0,1}, desc{0,1}, matches0, mscores0."""
    f0 = extract(extractor, rec["image0"], device, extractor_ac)
    f1 = extract(extractor, rec["image1"], device, extractor_ac)
    s0 = (rec["image0"].shape[1], rec["image0"].shape[0])
    s1 = (rec["image1"].shape[1], rec["image1"].shape[0])
    m = match(lg, f0, f1, s0, s1, lg_ac)
    out = {"kpts0": f0["keypoints"], "kpts1": f1["keypoints"], "scores0": f0["keypoint_scores"],
           "scores1": f1["keypoint_scores"], "desc0": f0["descriptors"], "desc1": f1["descriptors"], **m}
    return {k: v.cpu() for k, v in out.items()}


def set_adaptive(lg, confidences: tuple[float, float] | None) -> None:
    """LightGlue's adaptive inference with (depth_confidence, width_confidence) - early stop, point pruning; -1 turns
    one off - or all layers on every keypoint (None). Points are pruned at every keypoint count, as LightGlue does on
    the CPU and the device does (on CUDA it prunes only above 1024 keypoints: a GPU gather-overhead threshold)."""
    lg.conf.depth_confidence, lg.conf.width_confidence = confidences if confidences is not None else (-1, -1)
    lg.pruning_keypoint_thresholds = {k: -1 for k in type(lg).pruning_keypoint_thresholds}
