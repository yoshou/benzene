# shellcheck shell=bash disable=SC2034
# Shared by the scripts: the repository root, the Snapdragon toolchain image (Hexagon SDK + LLVM) that
# benzene/build.sh builds in, the models' venv
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPDRAGON_TOOLCHAIN_IMAGE="${SNAPDRAGON_TOOLCHAIN_IMAGE:-ghcr.io/snapdragon-toolchain/arm64-linux:v0.8}"
VENV="${VENV:-/opt/torch/venv-models}"  # the models' venv (scripts/install-venv.sh)
