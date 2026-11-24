"""The exported ALIKED in PyTorch: the device records (device_model.DeviceAliked) read back from the device model file -
the codes and constants the device runs (sums over the raw codes plus the streams' zero-point biases), the reference
of the device checks (checks/end_to_end.py), which so do not depend on quantizing again (GPTQ on a GPU)."""

from __future__ import annotations

import numpy as np
import torch

from benzene.kernels.convolution_int8 import TILE_BYTES, read_convolution_int8_stream
from benzene.model_file import load_export

from .export_hexmodel import DEVICE_CHANNELS, SELU_KEYS, TENSORS, tensor_name
from .hooked_network import CONVOLUTIONS, HookedAliked
from .recipes import DEVICE_LAYERS, DEVICE_RESIDUALS


def stream_bias(stream: bytes, K: int, N: int) -> np.ndarray:
    """The store bias words (-sum(w z)) of a convolution_int8 stream, int64 [N]."""
    KT = -(-K // 32)
    out = np.zeros(N, np.int64)
    for cb in range(N // 32):
        o = cb * (KT + 1) * TILE_BYTES + KT * TILE_BYTES
        blocks = np.frombuffer(stream[o : o + 1024], np.uint32).reshape(4, 64)
        out[cb * 32 : (cb + 1) * 32] = blocks[0, 32:].astype(np.int64)
    return np.where(out >= 2**31, out - 2**32, out)


def torch_int(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x).astype(np.int64)).to(torch.int32)


def records(net, path) -> dict:
    """The device records of the model file at path (the fp32 net gives the layers' shapes)."""
    ex = load_export(path)
    shapes = {name: tuple(w.shape) for name, (w, _) in HookedAliked(net).weights.items()}
    data = {tensor_name(n, k): ex[tensor_name(n, k)][0] for n, k in TENSORS}
    out = {"layers": {}, "pools": {}}
    stages = np.frombuffer(data["aliked.stages"], np.float32)
    for name, (source, kind, output) in DEVICE_LAYERS.items():
        record = {"kind": kind, "output": output, "input": source}
        N, C, kh, kw = shapes[name]
        if name in ("block1.conv1", "score_head.2", "score_head.4", "score_head.6"):
            raw = data[tensor_name(name, "hvx")]
            if name == "block1.conv1":
                words = np.frombuffer(raw[: 16 * 9 * 4], np.int32).reshape(16, 9)
                codes = (words & 0xFFFF).astype(np.int64)
                codes = np.where(codes >= 32768, codes - 65536, codes).reshape(16, 1, 3, 3)
                constants = np.frombuffer(raw[640 : 640 + 16 * 10 * 4], np.float32).reshape(16, 10)
                record.update(codes=torch_int(codes), bias2=torch.zeros(16, dtype=torch.int64),
                              padding=torch.zeros(1, dtype=torch.float64))
            else:
                groups = C // 4
                weight_bytes = -(-N * 9 * groups * 4 // 128) * 128
                words = np.frombuffer(raw[: N * 9 * groups * 4], np.int8).reshape(N, 9, groups, 4)
                codes = words.transpose(0, 2, 3, 1).reshape(N, C, 3, 3)
                corrections = np.frombuffer(raw[weight_bytes : weight_bytes + 4 * N], np.int32)
                constants_offset = weight_bytes + -(-4 * N // 128) * 128
                zero = stages[{"score_head.2": 832, "score_head.4": 864, "score_head.6": 896}[name]:][:C]
                record.update(codes=torch_int(codes), bias2=-torch.from_numpy(corrections.astype(np.int64)),
                              padding=torch.from_numpy(zero.astype(np.float64)))
                if output is not None:
                    constants = np.frombuffer(raw[constants_offset : constants_offset + 40 * N], np.float32).reshape(N, 10)
                else:
                    s_b = np.frombuffer(raw[constants_offset : constants_offset + 8], np.float32)
                    record.update(S=torch.tensor(s_b[:1]), B=torch.tensor(s_b[1:2]))
            if output is not None:
                for i, key in enumerate(SELU_KEYS):
                    record[key] = torch.from_numpy(constants[:, i].copy())
            out["layers"][name] = record
            continue
        stream = data[tensor_name(name, "stream")]
        columns = -(-N // 32) * 32
        residual = DEVICE_RESIDUALS.get(name)
        if name == "block2.conv2":
            channels, taps = 64, 9
        elif kind == "3x3":
            channels, taps = DEVICE_CHANNELS.get(source, C), 9
        elif kind == "1x1":
            channels, taps = DEVICE_CHANNELS.get(source, C), 1
        elif kind in ("deformable", "patch"):
            extra = shapes[residual][1] if residual else 0
            channels, taps = -(-(9 * C + extra) // 128) * 128, 1
        else:
            channels, taps = C, 1
        K = channels * taps
        matrix, p, q = read_convolution_int8_stream(stream, K, columns, output is not None)
        matrix = matrix.astype(np.int64)[:N]
        if name == "block2.conv2":
            full = matrix.reshape(N, 3, 3, 64).transpose(0, 3, 1, 2)
            codes = full[:, :C]
            record["residual_codes"] = torch_int(full[:, C : C + shapes[residual][1], 1:2, 1:2])
        elif kind == "3x3":
            codes = matrix.reshape(N, 3, 3, channels).transpose(0, 3, 1, 2)[:, :C]
        elif kind == "1x1":
            codes = matrix[:, :C, None, None]
        elif kind in ("deformable", "patch"):
            codes = matrix[:, : 9 * C].reshape(N, 9, C).transpose(0, 2, 1).reshape(N, C, 3, 3)
            if residual:
                record["residual_codes"] = torch_int(matrix[:, 9 * C : 9 * C + shapes[residual][1], None, None])
        else:
            codes = matrix[:, :, None, None]
        if residual:
            record["residual_input"] = CONVOLUTIONS[residual][0]
        record["codes"] = torch_int(codes)
        record["bias2"] = torch.from_numpy(stream_bias(stream, K, columns)[:N])
        aux = data.get(tensor_name(name, "aux"), b"")
        constants_bytes = 0
        if output is not None:
            constants = np.frombuffer(aux[: len(SELU_KEYS) * columns * 4], np.float32).reshape(len(SELU_KEYS), columns)
            for i, key in enumerate(SELU_KEYS):
                record[key] = torch.from_numpy(constants[i, :N].copy())
            constants_bytes = -(-len(SELU_KEYS) * columns * 4 // 128) * 128
        else:
            record.update(S=torch.from_numpy(p[:N].copy()), B=torch.from_numpy(q[:N].copy()))
        if taps == 9:
            padding = np.frombuffer(aux[constants_bytes : constants_bytes + channels], np.uint8)
            record["padding"] = torch.from_numpy(padding[:C].astype(np.float64))
        out["layers"][name] = record
    for name in ("block3.conv1", "block3.conv2", "block4.conv1", "block4.conv2"):  # (the offset convolution's input)
        out["layers"][name]["input_zero"] = out["layers"][name + ".offset"]["padding"].clone()
    # stage constants (benzene/dsp/models/aliked/aliked.cpp StageConstants)
    for pooled, (k_at, z_at, channels, window) in {"x1.pool": (0, 32, 16, 2), "x2.pool": (64, 96, 32, 4),
                                                    "x3.pool": (128, 192, 64, 4)}.items():
        out["pools"][pooled] = {"window": window, "K": torch.from_numpy(stages[k_at : k_at + channels].copy()),
                                "Z": torch.from_numpy(stages[z_at : z_at + channels].copy())}
    out["dequantize"] = {f"agg{k + 1}": {"scale": torch.from_numpy(stages[256 + 32 * k : 288 + 32 * k].copy()),
                                         "zero": torch.from_numpy(stages[384 + 32 * k : 416 + 32 * k].copy())}
                         for k in range(4)}
    out["score1"] = {key: torch.from_numpy(stages[512 + 32 * i : 512 + 32 * i + 8].copy())
                     for i, key in enumerate(SELU_KEYS)}
    out["desc.patch"] = {"R": torch.from_numpy(stages[928:1056].copy()), "Z": torch.from_numpy(stages[1056:1184].copy())}
    out["desc.samples"] = {"R": torch.from_numpy(stages[1184:1312].copy()),
                           "Z": torch.from_numpy(stages[1312:1440].copy())}
    return out


__all__ = ["records"]
