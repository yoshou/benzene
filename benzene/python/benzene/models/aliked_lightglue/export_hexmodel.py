"""Device model file for ALIKED + LightGlue (benzene aliked_lightglue):
WORK/export/aliked_lightglue-<recipe>.hexmodel (+ .tab).

  python -m benzene.models.aliked_lightglue.export_hexmodel dev+fp16

ALIKED, the device recipe (recipes.quantize_dev, device_model.DeviceAliked), tensors in protocol::aliked::tensor
order (names in TENSORS; layouts benzene/dsp/models/aliked/aliked.cpp):
  convolution_int8 layers  the weight stream (benzene.kernels.convolution_int8.convolution_int8_stream: K tap-major,
                           k = tap C + c with C the device's channels (16 padded to 32), the per-k zero points in its
                           store bias;
                           output columns padded to 32 with zero weights) and an auxiliary tensor: a SELU layer's
                           constants [10][N] (Sp, Bp, St, Bt, Dn, Q0 .. Q4; padding columns give code 0), then (3x3
                           layers) the input's zero points [C] as bytes, the padding codes
    block2.conv2           3x3 over the 64 channels [block2.mid | x1.pool | 0]: the downsample at the centre tap
    block3/4.conv1, .conv2 1x1 over the deformable columns [tap][C] (+ the pooled input for conv2), K padded to 128
    desc.offset0           1x1 over the 3x3 patches [tap][128]
    desc.aggregation       1x1 over the 16 samples' features [p][128]
  block1.conv1             int16 weight words [16][9] (both halfwords), SELU constants [16][10]
  score_head.2 / 4 / 6     int8 weight words [N][9][groups of 4 inputs], corrections sum(w z) [N], SELU constants
                           [N][10] (.6: S, B)
  stage constants          aliked.cpp StageConstants: pools, the aggregations' dequantization, score1's SELU
                           constants and the score head's zero points, desc.patch / desc.samples quantizers
LightGlue, fp16: benzene.models.lightglue.export (256 dimensions, 4 heads, 9 layers, input projection 128 -> 256).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from benzene.kernels.convolution_int8 import convolution_int8_stream

from ..lightglue.export import write_model_file
from .device_model import EXP2_F32
from .paths import WORK
from .recipes import ALIKED_RECIPES, DEVICE_LAYERS, DEVICE_RECIPE, apply_recipe, split
from .runner import DETECTION_THRESHOLD, MAX_KPTS, load_models

SELU_KEYS = ("Sp", "Bp", "St", "Bt", "Dn") + tuple(f"Q{i}" for i in range(len(EXP2_F32)))
# the protocol::aliked::tensor order: (name, kind) with kind "stream", "aux", "hvx" or "stages"
TENSORS = [
    ("block1.conv1", "hvx"),
    ("block1.conv2", "stream"), ("block1.conv2", "aux"),
    ("block2.conv1", "stream"), ("block2.conv1", "aux"),
    ("block2.conv2", "stream"), ("block2.conv2", "aux"),
    ("block3.conv1.offset", "stream"), ("block3.conv1.offset", "aux"),
    ("block3.conv1", "stream"), ("block3.conv1", "aux"),
    ("block3.conv2.offset", "stream"), ("block3.conv2.offset", "aux"),
    ("block3.conv2", "stream"), ("block3.conv2", "aux"),
    ("block4.conv1.offset", "stream"), ("block4.conv1.offset", "aux"),
    ("block4.conv1", "stream"), ("block4.conv1", "aux"),
    ("block4.conv2.offset", "stream"), ("block4.conv2.offset", "aux"),
    ("block4.conv2", "stream"), ("block4.conv2", "aux"),
    ("conv1", "stream"), ("conv1", "aux"),
    ("conv2", "stream"), ("conv2", "aux"),
    ("conv3", "stream"), ("conv3", "aux"),
    ("conv4", "stream"), ("conv4", "aux"),
    ("score_head.0.1", "stream"), ("score_head.0.2", "stream"), ("score_head.0.3", "stream"),
    ("score_head.0.4", "stream"),
    ("score_head.2", "hvx"), ("score_head.4", "hvx"), ("score_head.6", "hvx"),
    ("desc.offset0", "stream"), ("desc.offset0", "aux"),
    ("desc.offset2", "stream"),
    ("desc.sf", "stream"), ("desc.sf", "aux"),
    ("desc.aggregation", "stream"),
    ("stages", "stages"),
]
DEVICE_CHANNELS = {"image": 1, "block1.mid": 32, "x1": 32, "x1.pool": 32}  # inputs padded on the device


def tensor_name(name: str, kind: str) -> str:
    return "aliked.stages" if kind == "stages" else f"aliked.{name}" + (".aux" if kind == "aux" else "")


def padded(data: bytes) -> bytes:
    return data + bytes((-len(data)) % 128)


def vector_bytes(values, length: int = 32, dtype=np.float32) -> bytes:
    """Values in vectors of 32 lanes (128 bytes each), zero beyond."""
    v = np.zeros(-(-max(len(values), 1) // length) * length, dtype)
    a = np.asarray(values, dtype)
    v[: len(a)] = a
    return v.tobytes()


def numpy(x) -> np.ndarray:
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ---------------------------------------------------------------------------------------------- convolution_int8
def device_matrix(records: dict, name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(codes [N, K], per-k zero points [K], device input channels C, taps) of an HMX layer in the device's K order."""
    layer = records["layers"][name]
    source, kind, _ = DEVICE_LAYERS[name]
    zero = numpy(records["activations"][source]["zero"]).astype(np.int64)
    codes = numpy(layer["codes"]).astype(np.int64)  # [N, C, kh, kw]
    N, C = codes.shape[:2]
    if name == "block2.conv2":  # 3x3 over [block2.mid (32) | x1.pool (16) | 0 (16)], the downsample at the centre tap
        residual = numpy(layer["residual_codes"]).astype(np.int64)[:, :, 0, 0]  # [N, 16]
        full = np.zeros((N, 64, 3, 3), np.int64)
        full[:, :C] = codes
        full[:, C : C + residual.shape[1], 1, 1] = residual
        z = np.zeros(64, np.int64)
        z[:C] = zero
        z[C : C + residual.shape[1]] = numpy(records["activations"][layer["residual_input"]]["zero"])
        return full.transpose(0, 2, 3, 1).reshape(N, -1), np.tile(z, 9), 64, 9
    if kind == "3x3":
        channels = DEVICE_CHANNELS.get(source, C)
        full = np.zeros((N, channels, 3, 3), np.int64)
        full[:, :C] = codes
        z = np.zeros(channels, np.int64)
        z[:C] = zero
        return full.transpose(0, 2, 3, 1).reshape(N, -1), np.tile(z, 9), channels, 9
    if kind == "1x1":
        channels = DEVICE_CHANNELS.get(source, C)
        full = np.zeros((N, channels), np.int64)
        full[:, :C] = codes[:, :, 0, 0]
        z = np.zeros(channels, np.int64)
        z[:C] = zero
        return full, z, channels, 1
    if kind in ("deformable", "patch"):  # columns [tap][C] (+ the pooled input), K padded to 128
        matrix = codes.reshape(N, C, 9).transpose(0, 2, 1).reshape(N, -1)
        z = np.tile(zero, 9)
        if "residual_codes" in layer:
            residual = numpy(layer["residual_codes"]).astype(np.int64)[:, :, 0, 0]
            matrix = np.concatenate([matrix, residual], 1)
            z = np.concatenate([z, numpy(records["activations"][layer["residual_input"]]["zero"]).astype(np.int64)])
        K = -(-matrix.shape[1] // 128) * 128
        matrix = np.pad(matrix, ((0, 0), (0, K - matrix.shape[1])))
        z = np.pad(z, (0, K - len(z)))
        return matrix, z, K, 1
    assert kind == "aggregation"
    return codes.reshape(N, -1), np.tile(zero, 16), codes.shape[1], 1


def hmx_tensors(records: dict, name: str) -> tuple[bytes, bytes | None]:
    """(weight stream, auxiliary tensor or None) of an HMX layer."""
    layer = records["layers"][name]
    source, kind, output = DEVICE_LAYERS[name]
    matrix, zero, channels, taps = device_matrix(records, name)
    N = matrix.shape[0]
    columns = -(-N // 32) * 32
    codes = np.zeros((columns, matrix.shape[1]), np.int8)
    codes[:N] = matrix
    if output is None:  # fp32 rows
        scale, bias = np.zeros(columns, np.float32), np.zeros(columns, np.float32)
        scale[:N], bias[:N] = numpy(layer["S"]), numpy(layer["B"])
        stream = convolution_int8_stream(codes, f32=(scale, bias), zero_point=zero)
        aux = b""
    else:  # SELU: the stream's uint8 constants unused
        stream = convolution_int8_stream(codes, u8=(np.zeros(columns, np.float32), np.zeros(columns, np.float32)),
                              zero_point=zero)
        constants = np.zeros((len(SELU_KEYS), columns), np.float32)
        constants[SELU_KEYS.index("Bp")] = 0.5  # (padding columns: code 0)
        constants[SELU_KEYS.index("Dn")] = 0.5
        for i, key in enumerate(SELU_KEYS):
            constants[i, :N] = numpy(layer[key])
        aux = constants.tobytes()
    if taps == 9:  # the padding codes: the input's zero points
        aux = padded(aux) + vector_bytes(zero[:channels], 128, np.uint8)
    return stream, (aux if aux else None)


# ---------------------------------------------------------------------------------------------- HVX layers
def block1_conv1_tensor(records: dict) -> bytes:
    layer = records["layers"]["block1.conv1"]
    codes = numpy(layer["codes"]).astype(np.int64)[:, 0].reshape(16, 9)  # [channel][tap]
    assert np.abs(codes).max() <= 32767
    words = (codes & 0xFFFF) | ((codes & 0xFFFF) << 16)
    constants = np.stack([numpy(layer[k]) for k in SELU_KEYS], 1).astype(np.float32)  # [channel][10]
    return padded(words.astype(np.uint32).tobytes()) + padded(constants.tobytes())


def score_head_tensor(records: dict, name: str) -> bytes:
    layer = records["layers"][name]
    source = DEVICE_LAYERS[name][0]
    codes = numpy(layer["codes"]).astype(np.int64)  # [N, C, 3, 3]
    N, C = codes.shape[:2]
    groups = C // 4
    zero = numpy(records["activations"][source]["zero"]).astype(np.int64)
    w = codes.reshape(N, groups, 4, 9).transpose(0, 3, 1, 2)  # [N][tap][group][4]
    words = (w[..., 0] & 0xFF) | ((w[..., 1] & 0xFF) << 8) | ((w[..., 2] & 0xFF) << 16) | ((w[..., 3] & 0xFF) << 24)
    corrections = (codes * zero[None, :, None, None]).sum((1, 2, 3)).astype(np.int32)
    if DEVICE_LAYERS[name][2] is not None:
        constants = np.stack([numpy(layer[k]) for k in SELU_KEYS], 1).astype(np.float32)
    else:
        constants = np.array([numpy(layer["S"])[0], numpy(layer["B"])[0]], np.float32)
    return (padded(words.astype(np.uint32).tobytes()) + padded(corrections.tobytes()) +
            padded(constants.tobytes()))


def stage_constants(records: dict) -> bytes:
    """aliked.cpp StageConstants."""
    pools = records["pools"]
    activations = records["activations"]
    out = []
    for pooled, channels in (("x1.pool", 32), ("x2.pool", 32), ("x3.pool", 64)):
        K = np.zeros(channels, np.float32)
        Z = np.full(channels, 0.5, np.float32)
        k, z = numpy(pools[pooled]["K"]), numpy(pools[pooled]["Z"])
        K[: len(k)], Z[: len(z)] = k, z
        out += [vector_bytes(K), vector_bytes(Z)]
    for part in ("scale", "zero"):
        for k in range(1, 5):
            out.append(vector_bytes(numpy(activations[f"agg{k}"][part])))
    score1 = records["score1"]
    for key in SELU_KEYS:
        out.append(vector_bytes(np.tile(numpy(score1[key]), 4)))
    out.append(vector_bytes(np.tile(numpy(activations["score1"]["zero"]), 4)))
    out.append(vector_bytes(np.tile(numpy(activations["score2"]["zero"]), 8)))
    out.append(vector_bytes(np.tile(numpy(activations["score3"]["zero"]), 8)))
    for name in ("desc.patch", "desc.samples"):
        out += [vector_bytes(numpy(records[name]["R"])), vector_bytes(numpy(records[name]["Z"]))]
    data = b"".join(out)
    assert len(data) == 1440 * 4
    return data


def aliked_tensors(records: dict) -> dict[str, bytes]:
    out = {}
    for name, kind in TENSORS:
        if kind == "stream":
            stream, aux = hmx_tensors(records, name)
            out[tensor_name(name, "stream")] = stream
            if aux is not None:
                out[tensor_name(name, "aux")] = aux
        elif kind == "hvx":
            out[tensor_name(name, kind)] = (block1_conv1_tensor(records) if name == "block1.conv1"
                                            else score_head_tensor(records, name))
        elif kind == "stages":
            out[tensor_name(name, kind)] = stage_constants(records)
    expected = [tensor_name(n, k) for n, k in TENSORS]
    assert list(out) == expected, set(expected) ^ set(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE,
                        help="<aliked>+<lg>: ALIKED with the 'dev' recipe, LightGlue fp16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    aliked_name, lg_name = split(args.recipe)
    spec = ALIKED_RECIPES[aliked_name]
    if not (isinstance(spec, dict) and spec.get("dev")) or lg_name != "fp16":
        raise SystemExit(f"{args.recipe}: the device runs ALIKED with the 'dev' recipe and LightGlue in fp16")
    t0 = time.time()
    net, lg = load_models(args.device)
    _, records = apply_recipe(net, lg, args.recipe, args.device, verbose=False)
    write_model_file(WORK / "export" / f"aliked_lightglue-{args.recipe}.hexmodel",
                     {"model": "aliked_lightglue", "recipe": args.recipe, "max_kpts": MAX_KPTS,
                      "detection_threshold": DETECTION_THRESHOLD}, aliked_tensors(records), lg, t0)


if __name__ == "__main__":
    main()
