# Arduino VENTUNO Q NPU development (benzene)

benzene runs models on the Arduino VENTUNO Q's Hexagon NPU (Qualcomm QCS8275, Hexagon v75 cDSP) with its own HVX /
HMX / DMA kernels through FastRPC (no QAIRT / QNN). This repository holds benzene and the devcontainer it is built
and checked in.

## Repository layout

```
.devcontainer/      the container (post-create pulls the toolchain image and installs the models' venv)
config/             board connection settings (board.env)
scripts/            the toolchain image and venv settings, the models' venv install, the download of the models'
                    external inputs into work/ and a GPU check
benzene/            own HVX / HMX / DMA runtime on the Hexagon NPU, and the models it runs
  include/ dsp/ host/           C++: the DSP library (runtime, kernels, models) and the host runner
  python/                       the models' venv packages, pinned (requirements*.txt)
  python/benzene/               Python: the kernels' weight streams, the model file, board access, quantization,
                                and per model (models/<model>) its quantization recipes and their accuracy
                                evaluation, export, device inputs, checks
build/              build outputs (not tracked)
work/               model data: datasets, checkpoints, evaluation outputs, exported models (not tracked)
```

## Getting started

1. Open this folder in VS Code and run **Dev Containers: Reopen in Container**.
2. On first creation, post-create does the following (takes a few minutes):
   - pulls the Snapdragon toolchain image (Hexagon SDK + LLVM, ~1.7 GB; benzene builds in it) into the host's Docker
   - installs the models' venv (`scripts/install-venv.sh`, several GB) into the volume `ventunoq-torch`
     (`/opt/torch/venv-models`): Python 3.12.16 and the packages pinned in
     `benzene/python/requirements-rocm.txt` (PyTorch for ROCm) and `benzene/python/requirements.txt`
3. Set `BOARD_HOST` in `config/board.env`. It currently uses the `Host ventunoq` alias from the host's `~/.ssh/config`.
   - The IP, user, and key come from `~/.ssh/config`. post-create copies the host's `~/.ssh` into the container.
   - Without an alias, set `BOARD_HOST` to `<name>.local` or an IP address, and register your key with `ssh-copy-id arduino@<host>`.
4. Fetch the models' external inputs into `work/` (about 23 GB) with `scripts/download-inputs.sh`: the source
   checkouts at pinned commits, the weights checked by SHA-256, the Hugging Face models and datasets at pinned
   revisions, and the TUM RGB-D and Mip-NeRF 360 datasets. What is already there is checked and kept.

### Key settings (`containerEnv` in `.devcontainer/devcontainer.json`)

| Variable | Default | Description |
|---|---|---|
| `TOOLCHAIN_AUTO_PULL` | `1` | Set to `0` to skip pulling the Snapdragon toolchain image in post-create (`benzene/build.sh` pulls it on its first run) |
| `VENV_AUTO_INSTALL` | `1` | Set to `0` to skip installing the models' venv in post-create (`scripts/install-venv.sh` installs it later) |

## benzene

One DSP library (`benzene/dsp`: runtime, kernels, models) and the host runner (`benzene/host`), cross-compiled in the
upstream Snapdragon toolchain image (`ghcr.io/snapdragon-toolchain/arm64-linux`, Hexagon SDK + LLVM) through the
host's Docker.

```bash
benzene/build.sh                  # -> build/benzene/pkg/{bin/benzene, lib/libbenzene_skel.so}
```

The models: `cosmos3_edge` (the reasoner of nvidia/Cosmos3-Edge), `vggt` (facebook/VGGT-1B), the feature matchers
`superpoint_lightglue` (SuperPoint + LightGlue), `xfeat_lightglue` (XFeat + its LighterGlue) and `aliked_lightglue`
(ALIKED + LightGlue), and `loma` (davnords/LoMa). The matchers share `benzene.models.lightglue` (the LightGlue checkout,
matching, the evaluation pairs and metrics, the device input bundles, the model file's LightGlue tensors and the device
checks) and on the DSP `dsp/models/lightglue`. The host-side Python is the package `benzene` in `benzene/python`: per
model (`benzene.models.<model>`) the quantization recipes and their accuracy evaluation (`calibrate`, `evaluate`;
`prepare_data` in `vggt`, `cosmos3_edge` and, for the matchers, `lightglue`), the export to the device model file
(`export_hexmodel`), device input bundles (`prepare_bundle`) and checks of the device against PyTorch (`checks`; the
Cosmos3-Edge stage checks compare with `cosmos3_edge.capture_activations`, LoMa's DINOv2 recipes come from
`loma.dinov2_quantization`); shared by all of them the kernels' weight streams (`benzene.kernels`), the model file, the
board access (`benzene.board`, settings from `config/board.env`) and the quantization methods. Run the modules with
`benzene/python` on the module path, in the models' venv:

```bash
export PYTHONPATH=$PWD/benzene/python                 # from the repository root
PY=/opt/torch/venv-models/bin/python
$PY -c "from benzene import board; board.push_build()"  # push the build to $BOARD_BENZENE_DIR on the board
$PY -m benzene.models.vggt.evaluate table
$PY -m benzene.models.superpoint_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2
$PY -m benzene.models.xfeat_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2
```

Each model keeps its data in `work/<model>`; the raw datasets the models share are in `work/datasets`.
`scripts/download-inputs.sh` fetches the external inputs; the modules make the rest (prepared data,
calibration, exported models, device input bundles).

## Measurements on the board

Measured 2026-11-24 with the checks below, on the board's Hexagon v75 cDSP (1498 MHz, 4 HVX contexts,
1 HMX, 8 MiB VTCM). Times are on the DSP. Per image: the median, leaving out the first image of a run (warm-up); per
pair: the range over the 3 evaluation pairs (TUM RGB-D); LoMa: means, leaving out each run's first image and pair. The
device's outputs are compared with the same recipe run in PyTorch, unless the entry names another reference.

| Model (recipe) | Input | Time on the DSP | Device vs PyTorch |
|---|---|---|---|
| SuperPoint + LightGlue (`dev+fp16`) | 480×640, up to 1024 keypoints | SuperPoint 8.3 ms; LightGlue 22–27 ms, adaptive 14–15 ms | keypoints at the same pixels, scores and descriptors equal to rounding; matches: precision 98.4%, recall 98.7% (adaptive 99.5%, 98.8%) |
| XFeat + LighterGlue (`dev+fp16`) | 480×640, 1024 keypoints | XFeat 3.9 ms; LighterGlue 7.0–9.8 ms, adaptive 5.3–8.3 ms | same as SuperPoint's; matches 98.2%, 98.3% (adaptive 97.6%, 98.2%) |
| ALIKED + LightGlue (`dev+fp16`) | 480×640, up to 1024 keypoints | ALIKED 27.7 ms; LightGlue 12–20 ms, adaptive 7–12 ms | keypoints, scores and descriptors bit for bit; matches 99.1%, 99.3% (adaptive 99.6%, 99.5%) |
| LoMa (`dev+w8a8+fp16`) | detector 1024×768, descriptor 644×476 | 441 ms per image, matcher 151 ms, 1.03 s per pair | keypoints and probabilities bit for bit; matches 98.4%, 98.3% |
| VGGT-1B (`w8a8_tfc`) | 8 frames 392×518 (8328 tokens) | transformer 4.43 s, heads 1.52 s (camera 0.08, depth 0.74, points 0.70) | vs fp32: depth 0.35%, points 0.78%, rotation 0.11°; pose AUC@30 0.917 (fp32 0.917) |
| Cosmos3-Edge vision (`w8a8_sq05s_gptq`) | 8 images, 9600 patches | 1.33 s | relative RMS error 6.6%; vs fp32 6.7% (PyTorch 6.8%) |
| Cosmos3-Edge LLM prefill | 1024 tokens | 288 ms | relative RMS error 6.7%; vs fp32 7.3% (PyTorch 6.9%) |
| Cosmos3-Edge LLM decode | 5 tokens after a 2490-token video prompt | 48.1 ms per token; 4-bit decode weights 31.6 ms | vs BF16: KL 0.0034, top-1 100% (PyTorch 0.012, 83%); 4-bit 0.037, 83% |
| Cosmos3-Edge answers (`generate`) | 8 samples, one image each, 293–337 prompt tokens | vision 0.22 s, prefill 0.11 s, 44.9 ms per token (4-bit decode weights 27.9 ms) | 1 of 8 answers identical, 31% common prefix, first-token KL 0.036 (4-bit: 1 of 8, 35%, 0.038) |

The checks, with `PYTHONPATH` and `PY` as above (Cosmos3-Edge's need its exported model files in `~/cosmos3_edge` on
the board):

```bash
$PY -m benzene.models.superpoint_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2      # and with --adaptive
$PY -m benzene.models.xfeat_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2           # and with --adaptive
$PY -m benzene.models.aliked_lightglue.checks.end_to_end dev+fp16 --pairs 0 1 2          # and with --adaptive
$PY -m benzene.models.loma.checks.end_to_end --pairs 0 1 2
$PY -m benzene.models.vggt.checks.end_to_end w8a8_tfc --samples 0 1 2
$PY -m benzene.models.cosmos3_edge.checks.vision w8a8_sq05s_gptq
$PY -m benzene.models.cosmos3_edge.checks.llm_prefill w8a8_sq05s_gptq --tokens 1024
$PY -m benzene.models.cosmos3_edge.checks.llm_decode w8a8_sq05s_gptq                     # and with --dec4 dev_w8a8_q4hs
$PY -m benzene.models.cosmos3_edge.checks.generate device w8a8_sq05s_gptq --samples 0-7  # and with --dec4 dev_w8a8_q4hs
```

## Host GPU (AMD / ROCm)

The development PC's AMD GPUs are available in the container through PyTorch for ROCm, for the model work that needs
a GPU (running the float model, calibration, quantization, the checks' references).

```bash
scripts/gpu-check.sh   # device access, then a matmul on every GPU compared with the CPU
torch-env              # activate the models' venv
python -c 'import torch; print(torch.cuda.device_count())'   # ROCm uses the torch.cuda API ("cuda:0", ...)
```

- The host needs the amdgpu kernel driver (verified: ROCm 7.2.2, 2× RX 9060 XT = gfx1200, 16 GB each).
  The PyTorch wheels bundle the ROCm user-space libraries, so ROCm is not installed in the container.
- PyTorch comes from the `rocm7.2` wheel index (`download.pytorch.org/whl/rocm7.2`, in `scripts/install-venv.sh`);
  keep it close to the host's ROCm. The venv has no pip: change packages in `benzene/python/requirements*.txt` and
  run `scripts/install-venv.sh`.
- The GPU device nodes (`/dev/kfd`, `/dev/dri/renderD*`) come from the `/dev` bind and keep the host's group IDs
  (`render` is 110 on the host and does not exist in the container). The local feature `.devcontainer/host-gpu`
  adds `vscode` to those groups by GID at every container start. Without it, `vscode` cannot open `/dev/kfd`.
- CUDA-only packages (e.g. `flash-attn` CUDA builds, NVIDIA-only kernels) do not work; use ROCm builds or PyTorch's SDPA attention.

## Container notes

- `--network=host` plus the avahi socket mount lets the container resolve `*.local` names. If the host has no avahi-daemon, remove that mount line.
- `--privileged` plus the `/dev` bind gives access to the host's GPUs.
- The host's `~/.ssh` is mounted read-only and copied to `~/.ssh` by post-create.
- Docker is the host's daemon (docker-outside-of-docker). Bind-mount sources in `docker run -v` are resolved on the host, so use host paths: `LOCAL_WORKSPACE_FOLDER` (`remoteEnv`) holds the workspace's host path.
