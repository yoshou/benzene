"""The LoMa device model file (benzene dsp/models/loma): the convolutional networks of the device recipe
(device_records), DINOv2 W8A8 (dinov2_quantization's cached weights) and the matcher in fp16 (the LightGlue export),
for the settings of evaluate.SETTINGS; the device records it was made from go to RECORDS (the checks' device model).

Tensors (layer blobs: 4096-aligned parts, benzene.model_file.pack_parts):
  <conv>            an int8 HMX convolution (kernels::convolution_int8 stream, K tap-major k = tap * C + c, the input
                    zero points' -sum(w z) in its store biases): uint8 outputs (m, b) or fp32 rows (S, B; N padded to
                    32); the full-resolution refiners' block1 reads 128 channels (64 + 32, zero weights for the last
                    32), an out convolution has N = split + context padded to 32
  <conv>.dw         a 5x5 depthwise layer (kernels::depthwise_int8): parts weights int8 [25][C], initial int32 [C],
                    multipliers fp32 [C], biases fp32 [C], zero points uint8 [C]
  <point>.quantizer the codes of an fp32 activation (a refiner's upsampled context, DINOv2's output): R = f32(1 / s),
                    Zq = f32(z + 0.5) fp32 [C] each
  desc.dinov2.<i>   the 24 DINOv2 blocks (protocol vision_transformer::block_part, the VGGT export's layout)
  desc.dinov2.tail  parts: final norm weight, bias [1024]; the class token plus its position [1024]; the patch
                    embedding of the gray image as an fp16 GEMM stream (K = 14 x 14 = 196 padded to 256; the RGB weights
                    summed)
  desc.dinov2.positions  f32 [1369][1024]: the 37 x 37 position grid (the host resizes it, bicubic, per patch grid)
  lg.*              the matcher (benzene.models.lightglue.export: LoMa's layers have LightGlue's names; no token
                    confidences)
  loma.config       f32 [3]: the detector image's zero point, the descriptor image's, the keypoint count (what the
                    host passes to the DSP)
Header: the settings, the image zero points of the two gray first convolutions, the matcher's shape.

  python -m benzene.models.loma.export_hexmodel
      # -> WORK/export/loma-dev.hexmodel (+ .tab)
"""

from __future__ import annotations

import argparse
import time
import types

import numpy as np
import torch

from benzene.kernels.convolution_int8 import convolution_int8_stream
from benzene.kernels.gemm_fp16 import gemm_fp16_stream
from benzene.kernels.gemm_w8a8 import int8_codes, gemm_w8a8_stream
from benzene.model_file import Writer, pack_parts

from ..lightglue.export import lightglue_tensors
from .hooked_network import HookedDescriptor, HookedDetector
from .paths import WORK

PATCH_INPUTS = 256  # 14 x 14 gray pixels per patch, zero-padded to a multiple of 64 (gemm_fp16's K)
MODEL = WORK / "export" / "loma-dev.hexmodel"
RECORDS = WORK / "export" / "loma-dev.records.pt"  # the device records the model file was made from


def numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def records_on_cpu(records: dict) -> dict:
    groups = {group: {name: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in record.items()}
                      for name, record in records[group].items()} for group in ("layers", "activations")}
    return {"settings": records["settings"], **groups}


def device_codes(record: dict, input_channels: int | None = None) -> np.ndarray:
    """int8 [N, K] in the device's K order (tap-major); a 1x1 layer's K padded to input_channels, a gray layer's 9
    taps to 32; N padded to 32."""
    codes = numpy(record["codes"]).astype(np.int8)
    N, C, k, _ = codes.shape
    matrix = codes.transpose(0, 2, 3, 1).reshape(N, k * k * C)
    K = 32 if record["kind"] == "gray" else (input_channels or matrix.shape[1])
    matrix = np.pad(matrix, ((0, (-N) % 32), (0, K - matrix.shape[1])))
    return matrix


def padded(values: np.ndarray, N: int) -> np.ndarray:
    return np.pad(np.asarray(values, np.float32), (0, N - len(values)))


def convolution_tensor(record: dict, input_channels: int | None = None) -> bytes:
    codes = device_codes(record, input_channels)
    N, K = codes.shape
    if record["kind"] == "gray":
        zero_point = int(numpy(record["padding"])[0])
    else:
        z = numpy(record["padding"]).astype(np.int64)
        taps = numpy(record["codes"]).shape[-1] ** 2
        zero_point = np.pad(np.tile(z, taps) if taps == 9 else z, (0, K - len(z) * taps))  # (K order: tap-major)
    if record["output"] == "u8":
        stream = convolution_int8_stream(codes, u8=(padded(numpy(record["m"]), N), padded(numpy(record["b"]), N)),
                              zero_point=zero_point)
    else:
        stream = convolution_int8_stream(codes, f32=(padded(numpy(record["S"]), N), padded(numpy(record["B"]), N)),
                              zero_point=zero_point)
    return pack_parts([stream])[0]


def depthwise_tensor(record: dict) -> tuple[bytes, np.ndarray]:
    codes = numpy(record["codes"]).astype(np.int8).reshape(-1, 25)
    return pack_parts([np.ascontiguousarray(codes.T).tobytes(), numpy(record["bias2"]).astype(np.int32).tobytes(),
                       numpy(record["m"]).astype(np.float32), numpy(record["b"]).astype(np.float32),
                       numpy(record["padding"]).astype(np.uint8).tobytes()])


def quantizer_tensor(activation: dict) -> tuple[bytes, np.ndarray]:
    scale, zero = activation["scale"].double().cpu(), activation["zero"].double().cpu()
    return pack_parts([(1.0 / scale).float().numpy(), (zero + 0.5).float().numpy()])


def block1_input_channels(name: str) -> int | None:
    """The device's input channels of a 1x1 layer whose K is padded: the full-resolution refiners' block1 (64 feature
    + 32 context channels in a map of 128)."""
    return 128 if name in ("det.dec1.block1.conv1", "desc.dec1.block1.conv1") else None


def output_part(record: dict, columns: slice) -> dict:
    """An out convolution's record restricted to output channels `columns` (fp32 rows)."""
    return {**record, "codes": record["codes"][columns], "S": record["S"][columns], "B": record["B"][columns]}


def convolution_tensors(records: dict, plans) -> dict:
    """The networks' tensors by name (the host's reference order: protocol.h loma): an out convolution as two, its
    context (the channels after `split`; not at the last scale) and its deltas (the first `split`: .logit / .deltas)."""
    out = {}
    for plan in plans:
        for name in plan.convolutions:
            record = records["layers"][name]
            if record["kind"] == "dw":
                out[name] = depthwise_tensor(record)
            elif name.endswith(".out"):
                last = name.startswith(f"{plan.prefix}.dec{plan.scales[-1]}.")
                if not last:
                    context = output_part(record, slice(plan.split, None))
                    out[name + ".context"] = (convolution_tensor(context), np.array([0], np.int32))
                deltas = output_part(record, slice(0, plan.split))
                out[name + (".logit" if plan.prefix == "det" else ".deltas")] = (convolution_tensor(deltas),
                                                                                 np.array([0], np.int32))
            else:
                out[name] = (convolution_tensor(record, block1_input_channels(name)), np.array([0], np.int32))
        for scale in plan.scales[1:]:
            point = f"{plan.prefix}.dec{scale}.context"
            out[point + ".quantizer"] = quantizer_tensor(records["activations"][point])
    out["desc.dino.quantizer"] = quantizer_tensor(records["activations"]["desc.dino"])
    return out


def dinov2_block(dinov2, i: int, settings, device: str) -> tuple[bytes, np.ndarray]:
    from .dinov2_quantization import weight_record

    block = dinov2.blocks[i]
    record = {part: weight_record(dinov2, f"blocks.{i}.{part}", settings, device)
              for part in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")}

    def linear(part: str, column_multiplier=None):
        codes, scales = int8_codes(record[part])
        scales = scales.astype(np.float32)
        if column_multiplier is not None:
            scales = (scales * column_multiplier).astype(np.float32)
        return gemm_w8a8_stream(codes), scales

    def inverse_smooth(part: str) -> np.ndarray:
        return (1.0 / record[part]["smooth"].float()).numpy().astype(np.float32)

    ls1, ls2 = numpy(block.ls1.gamma).astype(np.float32), numpy(block.ls2.gamma).astype(np.float32)
    qkv, qkv_scales = linear("attn.qkv")
    proj, proj_scales = linear("attn.proj", ls1)
    fc1, fc1_scales = linear("mlp.fc1")
    fc2, fc2_scales = linear("mlp.fc2", ls2)
    i_qkv, i_fc1 = inverse_smooth("attn.qkv"), inverse_smooth("mlp.fc1")
    return pack_parts([qkv, qkv_scales, numpy(block.attn.qkv.bias),
                       proj, proj_scales, numpy(block.attn.proj.bias) * ls1,
                       fc1, fc1_scales, numpy(block.mlp.fc1.bias),
                       fc2, fc2_scales, numpy(block.mlp.fc2.bias) * ls2,
                       numpy(block.norm1.weight) * i_qkv, numpy(block.norm1.bias) * i_qkv, inverse_smooth("attn.proj"),
                       numpy(block.norm2.weight) * i_fc1, numpy(block.norm2.bias) * i_fc1, inverse_smooth("mlp.fc2"),
                       np.zeros(256, np.float32)])


def dinov2_tail(dinov2) -> tuple[bytes, np.ndarray]:
    weight = numpy(dinov2.patch_embed.proj.weight).astype(np.float32).sum(1).reshape(1024, 196)  # gray: RGB summed
    position = numpy(dinov2.pos_embed)[0].astype(np.float32)
    return pack_parts([numpy(dinov2.norm.weight), numpy(dinov2.norm.bias),
                       numpy(dinov2.cls_token)[0, 0] + position[0],
                       gemm_fp16_stream(np.pad(weight, ((0, 0), (0, PATCH_INPUTS - 196))),
                                    numpy(dinov2.patch_embed.proj.bias))])


def matcher_model(net) -> types.SimpleNamespace:
    """LoMa's matcher with the interface of the LightGlue export (conf, state_dict)."""
    names = ("input_proj", "posenc", "transformers", "log_assignment")
    state = {k: v for k, v in net.state_dict().items() if k.split(".")[0] in names}
    layers = len(net.transformers)
    dim = net.transformers[0].self_attn.Wqkv.in_features
    conf = types.SimpleNamespace(n_layers=layers, num_heads=net.transformers[0].self_attn.heads, descriptor_dim=dim,
                                 input_dim=net.input_proj.in_features if hasattr(net.input_proj, "in_features")
                                 else dim)
    return types.SimpleNamespace(conf=conf, state_dict=lambda: state)


def main() -> None:
    from .device_records import quantize_dev
    from .evaluate import SETTINGS, parse_settings
    from .runner import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings = parse_settings(args.settings) if args.settings else SETTINGS
    t0 = time.time()
    net = load_model(device=args.device)
    records = quantize_dev(net, settings, args.device, verbose=False)
    plans = (HookedDetector(net._detector), HookedDescriptor(net._descriptor))
    dinov2 = net._descriptor.encoder.frozen_dinov2.dinov2_vitl14
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    path = MODEL
    torch.save(records_on_cpu(records), RECORDS)  # (the checks' device model: GPTQ may differ between runs)
    writer = Writer(path)
    for name, blob in convolution_tensors(records, plans).items():
        writer.blob(name, blob)
    for i in range(len(dinov2.blocks)):
        writer.blob(f"desc.dinov2.{i}", dinov2_block(dinov2, i, settings, args.device))
    writer.blob("desc.dinov2.tail", dinov2_tail(dinov2))
    writer.f32("desc.dinov2.positions", numpy(dinov2.pos_embed)[0, 1:].astype(np.float32))
    detector_zero = int(numpy(records["layers"]["det.enc.0"]["padding"])[0])
    descriptor_zero = int(numpy(records["layers"]["desc.enc.0"]["padding"])[0])
    writer.f32("loma.config", np.array([detector_zero, descriptor_zero, settings.keypoints], np.float32))
    for name, tensor in lightglue_tensors(matcher_model(net), token_confidences=False).items():
        if isinstance(tensor, bytes):
            writer.raw(name, tensor)
        else:
            writer.f32(name, tensor)
    header = {"model": "loma_b", "settings": settings.name, "keypoints": settings.keypoints,
              "detector_image_zero": detector_zero, "descriptor_image_zero": descriptor_zero,
              "n_layers": len(net.transformers), "heads": net.transformers[0].self_attn.heads, "dim": 256}
    size = writer.finish(header)
    print(f"{path}: {size / 2**20:.1f} MiB, {len(writer.tensors)} tensors ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
