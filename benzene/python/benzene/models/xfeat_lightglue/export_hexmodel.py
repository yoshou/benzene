"""Device model file for XFeat + LightGlue (benzene xfeat_lightglue):
WORK/export/xfeat_lightglue-<recipe>.hexmodel (+ .tab).

  python -m benzene.models.xfeat_lightglue.export_hexmodel dev+fp16

XFeat, the device recipe (recipes.quantize_dev, device_model.DeviceXFeat), tensors in protocol::xfeat::tensor order:
  xfeat.block1.0 .. 3  the HVX layers (benzene/dsp/models/xfeat/xfeat.cpp Block1Arguments): int8 weight words
                       (4 input channels or a row's 3 taps per word), then 32-lane fp32 / int32 vectors of constants
                       (padded to 128 bytes)
  xfeat.block2.0 ..    benzene.kernels.convolution_int8.convolution_int8_stream (K tap-major; block2's and block3.0's
                       24 input channels padded to 32, block2's 24 output channels to 32 columns, heatmap.2's one to 32,
                       keypoint.3's 65 to 96); the signed inputs (block2.0, heatmap.0) with zero point 128
  xfeat.keypoint.0     its stream (the uint8 constants unused), then fp32 k [64], beta [64], int32 tap sums [64]: the
                       device folds the image normalization in per image
  xfeat.fusion_scales  fp32 [4][64]: the fusion's k3, k4, k5, then the heatmap head's inverse input scale
LightGlue (XFeat's LighterGlue), fp16: benzene.models.lightglue.export (96 dimensions, 1 head, 6 layers, input
projection 64 -> 96).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benzene.kernels.convolution_int8 import convolution_int8_stream

from ..lightglue.export import write_model_file
from .device_model import SIGNED_INPUTS
from .paths import WORK
from .recipes import DEVICE_RECIPE, XFEAT_RECIPES, apply_recipe, split
from .runner import DETECTION_THRESHOLD, MAX_KPTS, load_models

HVX_LAYERS = ("block1.0", "block1.1", "block1.2", "block1.3")
HMX_LAYERS = ("block2.0", "block2.1", "block3.0", "block3.1", "block3.2", "block4.0", "block4.1", "block4.2",
              "block5.0", "block5.1", "block5.2", "block5.3", "fusion.0", "fusion.1", "fusion.2", "heatmap.0",
              "heatmap.1", "heatmap.2", "keypoint.0", "keypoint.1", "keypoint.2", "keypoint.3")


def lanes(values, dtype=np.float32) -> np.ndarray:
    """A 32-lane vector (128 bytes) of up to 32 values, zero beyond."""
    v = np.zeros(32, dtype)
    a = np.asarray(values, dtype)
    v[: len(a)] = a
    return v


def padded(data: bytes) -> bytes:
    return data + bytes((-len(data)) % 128)


def block1_tensor(name: str, r: dict) -> bytes:
    """An HVX layer's tensor (xfeat.cpp Block1Arguments)."""
    codes = r["codes"].numpy().astype(np.int8)  # [N, C, 3, 3]
    N, C = codes.shape[:2]
    if name == "block1.0":  # words [channel][row]: the row's taps, then 0
        words = np.zeros((N, 3, 4), np.int8)
        words[:, :, :3] = codes[:, 0]
        constants = [lanes(r["k"]), lanes(r["beta"])] + [lanes(r["tap_sums"][:, c], np.int32) for c in range(9)]
    else:  # words [channel][tap][C / 4]: 4 input channels each
        words = codes.transpose(0, 2, 3, 1).reshape(N, 9, C // 4, 4)
        if name == "block1.3":
            constants = [lanes(r[k]) for k in ("scale", "bias", "skip_weight16", "skip_weight", "skip_bias",
                                                "inverse_scale")]
        else:
            constants = [lanes(r["m"]), lanes(r["b"])]
    return padded(np.ascontiguousarray(words).tobytes()) + b"".join(c.tobytes() for c in constants)


def device_codes(codes: np.ndarray, channels: int, columns: int) -> np.ndarray:
    """codes [N, C, kh, kw] -> [columns, K] in the device's tap-major K order with C padded to `channels`."""
    N, C, kh, kw = codes.shape
    c = np.zeros((columns, channels, kh, kw), np.int8)
    c[:N, :C] = codes
    return c.transpose(0, 2, 3, 1).reshape(columns, -1)


def hmx_tensor(name: str, r: dict) -> bytes:
    """An HMX layer's convolution_int8 stream (keypoint.0: plus its normalization constants)."""
    codes = r["codes"].numpy().astype(np.int8)
    N, C = codes.shape[:2]
    channels = 32 if C == 24 else C
    columns = -(-N // 32) * 32
    dc = device_codes(codes, channels, columns)
    zero_point = 128 if name in SIGNED_INPUTS else 0
    if "m" in r:  # uint8 output for the next layer (extra columns: codes 0 -> q 0)
        m, b = np.zeros(columns, np.float32), np.full(columns, 0.5, np.float32)
        m[:N], b[:N] = r["m"], r["b"]
        return convolution_int8_stream(dc, u8=(m, b), zero_point=zero_point)
    if "scale" in r:  # fp32 rows
        sc, bi = np.zeros(columns, np.float32), np.zeros(columns, np.float32)
        sc[:N], bi[:N] = r["scale"], r["bias"]
        return convolution_int8_stream(dc, f32=(sc, bi), zero_point=zero_point)
    # keypoint.0: the stream with placeholder constants, then k, beta, tap sums (per image on the device)
    placeholder = np.zeros(columns, np.float32)
    stream = convolution_int8_stream(dc, u8=(placeholder, placeholder), zero_point=zero_point)
    tail = (np.asarray(r["k"], np.float32).tobytes() + np.asarray(r["beta"], np.float32).tobytes() +
            np.asarray(r["tap_sums"][:, 0], np.int32).tobytes())
    return stream + tail


def xfeat_tensors(records: dict) -> dict[str, bytes]:
    out = {f"xfeat.{name}": block1_tensor(name, records[name]) for name in HVX_LAYERS}
    out.update({f"xfeat.{name}": hmx_tensor(name, records[name]) for name in HMX_LAYERS})
    f = records["fusion"]
    out["xfeat.fusion_scales"] = np.concatenate(
        [f["k3"], f["k4"], f["k5"], records["fusion.2"]["inverse_scale"]]).astype(np.float32).tobytes()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE,
                        help="<xfeat>+<lg>: XFeat with a 'dev' recipe, LightGlue fp16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    xfeat_name, lg_name = split(args.recipe)
    if not isinstance(XFEAT_RECIPES[xfeat_name], dict) or lg_name != "fp16":
        raise SystemExit(f"{args.recipe}: the device runs XFeat with a 'dev' recipe and LightGlue in fp16")
    t0 = time.time()
    net, lg = load_models(args.device)
    _, records = apply_recipe(net, lg, args.recipe, args.device, verbose=False)
    write_model_file(WORK / "export" / f"xfeat_lightglue-{args.recipe}.hexmodel",
                     {"model": "xfeat_lightglue", "recipe": args.recipe, "max_kpts": MAX_KPTS,
                      "detection_threshold": DETECTION_THRESHOLD}, xfeat_tensors(records), lg, t0)


if __name__ == "__main__":
    main()
