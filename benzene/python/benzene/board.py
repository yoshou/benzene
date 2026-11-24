"""The board (the VENTUNO Q's Linux side) as the checks use it: running benzene there and copying files.

The connection settings come from config/board.env (config/board.env.example when it does not exist; another file
with BOARD_ENV). benzene runs from BOARD_BENZENE_DIR, the directory push_build copies the build to (default
~/benzene); every model keeps its model files in its own directory under the board user's home
(models/<model>/paths.py, BOARD_DIR).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from benzene.workspace import REPOSITORY


def _read_settings() -> dict[str, str]:
    path = Path(os.environ.get("BOARD_ENV", REPOSITORY / "config/board.env"))
    if not path.exists():
        path = REPOSITORY / "config/board.env.example"
    settings = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip().strip("\"'")
    return settings


_SETTINGS = _read_settings()


def _setting(name: str, default: str | None = None) -> str:
    """A setting from the settings file, else the environment, else the default."""
    value = _SETTINGS.get(name) or os.environ.get(name) or default
    if value is None:
        raise KeyError(f"{name} is not set (config/board.env)")
    return value


USER = _setting("BOARD_USER")
HOST = f"{USER}@{_setting('BOARD_HOST')}"
HOME = f"/home/{USER}"
BENZENE_DIR = _setting("BOARD_BENZENE_DIR", f"{HOME}/benzene")
_SSH_OPTIONS = ["-p", _setting("BOARD_SSH_PORT", "22"), "-o", "ConnectTimeout=10"]
# where the DSP loads the skel from: the benzene directory, then the system's DSP libraries
_DSP_LIBRARY_PATH = f"{BENZENE_DIR};/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp"


def ssh(command: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command on the board; stdout and stderr are captured as text."""
    return subprocess.run(["ssh", *_SSH_OPTIONS, HOST, command], capture_output=True, text=True, check=check)


def upload(local: str | Path | list, remote: str, inplace: bool = False) -> None:
    """rsync -a of local files or directories (a trailing / copies a directory's contents) to a path on the board;
    inplace: update changed files in place (large model files)."""
    sources = [str(path) for path in (local if isinstance(local, list) else [local])]
    options = ["--inplace"] if inplace else []
    subprocess.run(["rsync", "-a", *options, "-e", shlex.join(["ssh", *_SSH_OPTIONS]), *sources, f"{HOST}:{remote}"],
                   check=True)


def download(remote: str, local: str | Path) -> None:
    """rsync -a of a file or directory on the board to a local path."""
    subprocess.run(["rsync", "-a", "-e", shlex.join(["ssh", *_SSH_OPTIONS]), f"{HOST}:{remote}", str(local)], check=True)


def empty_directory(path: str) -> str:
    """Create the board directory (relative paths: under BENZENE_DIR) or empty it; returns its absolute path."""
    absolute = path if path.startswith("/") else f"{BENZENE_DIR}/{path}"
    ssh(f"rm -rf {absolute} && mkdir -p {absolute}", check=True)
    return absolute


def benzene_command(arguments: str) -> str:
    """The board shell command that runs ./benzene <arguments> in BENZENE_DIR (relative paths in the arguments are
    relative to it)."""
    return f"cd {BENZENE_DIR} && ADSP_LIBRARY_PATH='{_DSP_LIBRARY_PATH}' ./benzene {arguments}"


def run_benzene(arguments: str) -> subprocess.CompletedProcess:
    """Run ./benzene <arguments> on the board (see benzene_command); output captured, exit status not checked."""
    return ssh(benzene_command(arguments))


def push_build() -> None:
    """Copy the local build (build/benzene/pkg, or BENZENE_OUT/pkg like benzene/build.sh) to BENZENE_DIR."""
    package = Path(os.environ.get("BENZENE_OUT", REPOSITORY / "build/benzene")) / "pkg"
    ssh(f"mkdir -p {BENZENE_DIR}", check=True)
    upload([package / "bin/benzene", package / "lib/libbenzene_skel.so"], f"{BENZENE_DIR}/")
