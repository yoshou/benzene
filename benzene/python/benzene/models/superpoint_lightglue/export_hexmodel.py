"""Device model file for SuperPoint + LightGlue (benzene superpoint_lightglue):
WORK/export/superpoint_lightglue-<recipe>.hexmodel (+ .tab).

  python -m benzene.models.superpoint_lightglue.export_hexmodel dev+fp16

SuperPoint, int8 (recipe "dev", recipes.quantize_sp_dev): one weight stream per convolution
(benzene.kernels.convolution_int8.convolution_int8_stream, device K order tap-major), convPa and convDa as one GEMM
(N = 512: two output maps of 256 channels), convPb padded to 96 columns. A uint8 output carries its consumer's
calibrated per-channel input scale (m = s_acc / s_y, b = bias / s_y + 0.5); the 1x1 heads put out fp32 (s_acc, bias).
LightGlue, fp16: benzene.models.lightglue.export (256 dimensions, 4 heads of 64, 9 layers, no input projection).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benzene.kernels.convolution_int8 import convolution_int8_stream

from ..lightglue.export import write_model_file
from .paths import WORK
from .recipes import DEVICE_RECIPE, SP_RECIPES, apply_recipe, split
from .runner import MAX_KPTS, load_models

SP_LAYERS = ("conv1a", "conv1b", "conv2a", "conv2b", "conv3a", "conv3b", "conv4a", "conv4b")


def dev_codes(rec: dict) -> np.ndarray:
    c = rec["codes"]  # [Cout, Cin, kh, kw]
    return c.permute(0, 2, 3, 1).reshape(c.shape[0], -1).numpy()  # k = (ky * kw + kx) * Cin + ci


def sp_streams(records: dict, sp) -> dict[str, tuple[bytes, dict]]:
    out = {}
    for name in SP_LAYERS:
        r = records[name]
        out[f"sp.{name}"] = (convolution_int8_stream(dev_codes(r), u8=(r["m"].numpy(), r["b"].numpy())),
                             dict(K=int(dev_codes(r).shape[1]), N=int(r["codes"].shape[0])))
    pa, da = records["convPa"], records["convDa"]
    codes = np.concatenate([dev_codes(pa), dev_codes(da)])
    m = np.concatenate([pa["m"].numpy(), da["m"].numpy()])
    b = np.concatenate([pa["b"].numpy(), da["b"].numpy()])
    out["sp.convPaDa"] = (convolution_int8_stream(codes, u8=(m, b)), dict(K=int(codes.shape[1]), N=512))
    pb = records["convPb"]
    cpb = np.zeros((96, 256), np.int8)
    cpb[:65] = dev_codes(pb)
    sc, bi = np.zeros(96, np.float32), np.zeros(96, np.float32)
    sc[:65], bi[:65] = pb["s_acc"].numpy(), sp.convPb.conv.bias.detach().float().cpu().numpy()
    out["sp.convPb"] = (convolution_int8_stream(cpb, f32=(sc, bi)), dict(K=256, N=96))
    db = records["convDb"]
    convDb_bias = sp.convDb.conv.bias.detach().float().cpu().numpy()
    out["sp.convDb"] = (convolution_int8_stream(dev_codes(db), f32=(db["s_acc"].numpy(), convDb_bias)),
                        dict(K=256, N=256))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", default=DEVICE_RECIPE,
                        help="<sp>+<lg>: the SuperPoint recipe must be the device one (dev), LightGlue fp16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sp_name, lg_name = split(args.recipe)
    if not (isinstance(SP_RECIPES[sp_name], dict) and SP_RECIPES[sp_name].get("dev")) or lg_name != "fp16":
        raise SystemExit(f"{args.recipe}: the device runs SuperPoint with a 'dev' recipe and LightGlue in fp16")
    t0 = time.time()
    sp, lg = load_models(args.device)
    _, records = apply_recipe(sp, lg, args.recipe, args.device, verbose=False)
    write_model_file(WORK / "export" / f"superpoint_lightglue-{args.recipe}.hexmodel",
                     {"model": "superpoint_lightglue", "recipe": args.recipe, "max_kpts": MAX_KPTS},
                     sp_streams(records, sp), lg, t0)


if __name__ == "__main__":
    main()
