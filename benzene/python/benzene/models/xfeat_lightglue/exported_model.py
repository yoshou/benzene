"""The exported XFeat in PyTorch: the device records (device_model.DeviceXFeat) read back from the device model file -
the codes and constants the device runs, the reference of the device checks (checks/end_to_end.py), which so do not
depend on quantizing again (GPTQ on a GPU)."""

from __future__ import annotations

import numpy as np
import torch

from benzene.kernels.convolution_int8 import convolution_int8_stream_bytes, read_convolution_int8_stream
from benzene.model_file import load_export

from .device_model import SIGNED_INPUTS
from .export_hexmodel import HMX_LAYERS, HVX_LAYERS
from .recipes import folded


def _vectors(data: bytes, offset: int, count: int, dtype=np.float32) -> np.ndarray:
    return np.frombuffer(data[offset : offset + 128 * count], dtype).reshape(count, 32)


def records(net, path) -> dict:
    """The device records of the model file at path (the fp32 net gives the layers' shapes)."""
    ex = load_export(path)
    out = {}
    for name in HVX_LAYERS:
        data, _ = ex[f"xfeat.{name}"]
        N, C = folded(net, name)[0].shape[:2]
        if name == "block1.0":
            words = np.frombuffer(data[:48], np.int8).reshape(N, 3, 4)
            codes = words[:, :, :3][:, None]  # [N, 1, 3, 3]
            v = _vectors(data, 128, 2)
            sums = _vectors(data, 128 + 256, 9, np.int32)[:, :N].T.copy()  # [N, 9]
            out[name] = {"codes": torch.from_numpy(codes.copy()), "zero_point": 0, "k": v[0, :N].copy(),
                         "beta": v[1, :N].copy(), "tap_sums": sums}
            continue
        weight_bytes = N * 9 * C
        words = np.frombuffer(data[:weight_bytes], np.int8).reshape(N, 3, 3, C)
        codes = torch.from_numpy(words.transpose(0, 3, 1, 2).copy())
        constants = -(-weight_bytes // 128) * 128
        if name == "block1.3":
            v = _vectors(data, constants, 6)
            keys = ("scale", "bias", "skip_weight16", "skip_weight", "skip_bias", "inverse_scale")
            out[name] = {"codes": codes, "zero_point": 0, **{k: v[i, :N].copy() for i, k in enumerate(keys)}}
        else:
            v = _vectors(data, constants, 2)
            out[name] = {"codes": codes, "zero_point": 0, "m": v[0, :N].copy(), "b": v[1, :N].copy()}
    for name in HMX_LAYERS:
        data, _ = ex[f"xfeat.{name}"]
        N, C, kh, kw = folded(net, name)[0].shape
        channels = 32 if C == 24 else C
        columns = -(-N // 32) * 32
        K = kh * kw * channels
        f32_output = name in ("fusion.2", "heatmap.2", "keypoint.3")
        codes, p, q = read_convolution_int8_stream(data, K, columns, not f32_output)
        codes = codes.reshape(columns, kh, kw, channels).transpose(0, 3, 1, 2)[:N, :C]
        record = {"codes": torch.from_numpy(np.ascontiguousarray(codes)),
                  "zero_point": 128 if name in SIGNED_INPUTS else 0}
        if name == "keypoint.0":
            tail = convolution_int8_stream_bytes(K, columns)
            record.update(k=np.frombuffer(data[tail : tail + 4 * N], np.float32).copy(),
                          beta=np.frombuffer(data[tail + 256 : tail + 256 + 4 * N], np.float32).copy(),
                          tap_sums=np.frombuffer(data[tail + 512 : tail + 512 + 4 * N], np.int32).copy()[:, None])
        elif f32_output:
            record.update(scale=p[:N].copy(), bias=q[:N].copy())
        else:
            record.update(m=p[:N].copy(), b=q[:N].copy())
        out[name] = record
    scales = np.frombuffer(ex["xfeat.fusion_scales"][0], np.float32).reshape(4, 64)
    out["fusion"] = {"k3": scales[0].copy(), "k4": scales[1].copy(), "k5": scales[2].copy()}
    out["fusion.2"]["inverse_scale"] = scales[3].copy()
    return out


__all__ = ["records"]
