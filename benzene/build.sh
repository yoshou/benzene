#!/usr/bin/env bash
# Build the benzene host runner (aarch64) and DSP skel (v75) in the Snapdragon toolchain image.
# Output: build/benzene/pkg/{bin/benzene, lib/libbenzene_skel.so}
set -euo pipefail
source "$(dirname "$0")/../scripts/_common.sh"
SRC="${BENZENE_SRC:-$WS/benzene}"  # BENZENE_SRC / BENZENE_OUT: build another tree (A/B checks)
OUT="${BENZENE_OUT:-$WS/build/benzene}"
IMAGE="$SNAPDRAGON_TOOLCHAIN_IMAGE"
HOST_WS="${LOCAL_WORKSPACE_FOLDER:-$WS}"
[[ "${1:-}" == --clean ]] && rm -rf "$OUT"
mkdir -p "$OUT"
docker run --rm --platform linux/amd64 -u "$(id -u):$(id -g)" -e HOME=/tmp \
  --mount "type=bind,source=$HOST_WS,target=$WS" -w "$SRC" "$IMAGE" bash -c "
    set -euo pipefail
    cmake -S . -B '$OUT/build' -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE='$SRC/host/cmake-toolchain.cmake' \
      -DHEXAGON_SDK_ROOT=\$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=\$HEXAGON_TOOLS_ROOT \
      -DCMAKE_INSTALL_PREFIX='$OUT/pkg' -DPREBUILT_LIB_DIR=linux_aarch64 -DDSP_VERSION=${DSP_VERSION:-v75} >/dev/null
    cmake --build '$OUT/build' -j \$(nproc)
    cmake --install '$OUT/build' >/dev/null
  "
find "$OUT/pkg" -type f -printf '%P\n'
