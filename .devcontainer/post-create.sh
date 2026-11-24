#!/usr/bin/env bash
# One-time initialization, run after the devcontainer is created
set -uo pipefail
cd "$(dirname "$0")/.."
WS="$(pwd)"

log()  { printf '\033[1;36m[post-create]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[post-create] WARN:\033[0m %s\n' "$*"; }

# --- Copy SSH config from the host --------------------------------------------
if [[ -d "$HOME/.ssh-host" ]]; then
  log "Copying host ~/.ssh"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  cp -a "$HOME/.ssh-host/." "$HOME/.ssh/" 2>/dev/null || true
  chmod -R go-rwx "$HOME/.ssh" 2>/dev/null || true
fi

# --- Board connection settings ------------------------------------------------
[[ -f config/board.env ]] || cp config/board.env.example config/board.env

# --- Snapdragon toolchain image (Hexagon SDK + LLVM) for benzene/build.sh ----
# Kept by the host's Docker
if [[ "${TOOLCHAIN_AUTO_PULL:-1}" == "1" ]]; then
  IMAGE="$(source scripts/_common.sh && echo "$SNAPDRAGON_TOOLCHAIN_IMAGE")"
  log "Pulling the Snapdragon toolchain image $IMAGE (~1.7 GB on first run)"
  docker pull "$IMAGE" || warn "Pulling $IMAGE failed. benzene/build.sh pulls it on its first run"
fi

# --- The models' venv (PyTorch for the host's AMD GPUs and the rest) ----------
if [[ "${VENV_AUTO_INSTALL:-1}" == "1" ]]; then
  log "Installing the models' venv (downloads several GB on first run only)"
  bash scripts/install-venv.sh || warn "Installing the models' venv failed. Re-run 'bash scripts/install-venv.sh'"
fi

# --- Shell initialization -----------------------------------------------------
MARK="# >>> ventuno-q devcontainer >>>"
VENV="$(source scripts/_common.sh && echo "$VENV")"
if ! grep -qF "$MARK" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<RC

$MARK
export VENTUNO_WS="$WS"
export PATH="\$VENTUNO_WS/scripts:\$PATH"
# Switch to the models' venv (PyTorch for the host's AMD GPUs)
torch-env() { deactivate 2>/dev/null || true; source $VENV/bin/activate; }
# <<< ventuno-q devcontainer <<<
RC
fi

chmod +x scripts/*.sh 2>/dev/null || true
log "Done."
