#!/usr/bin/env bash
# The models' venv ($VENV of _common.sh: /opt/torch/venv-models, in the ventunoq-torch volume): Python 3.12.16 managed
# by uv and exactly the packages of benzene/python/requirements-rocm.txt (PyTorch for ROCm) and requirements.txt (the
# rest).
# Run it again after changing the requirement files; it does not uninstall packages taken out of them.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
PYTHON_VERSION=3.12.16
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/torch/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/torch/uv-cache}"

uv python install "$PYTHON_VERSION"
[[ -x "$VENV/bin/python" ]] || uv venv --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"
uv pip install --python "$VENV/bin/python" --no-deps --index-url https://download.pytorch.org/whl/rocm7.2 \
  -r "$WS/benzene/python/requirements-rocm.txt"
uv pip install --python "$VENV/bin/python" --no-deps --index-url https://pypi.org/simple \
  -r "$WS/benzene/python/requirements.txt"
uv pip check --python "$VENV/bin/python"
"$VENV/bin/python" -c 'import torch; print("torch", torch.__version__, "/ HIP", torch.version.hip)'
