# Development container for benzene on the Arduino VENTUNO Q's NPU (for x86_64 hosts). benzene builds in the
# Snapdragon toolchain image through the host's Docker; the models' venv (PyTorch for the host's AMD GPUs and the
# rest) lives in a named volume.
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      # General build tools
      build-essential cmake ninja-build pkg-config git curl wget ca-certificates \
      unzip zip xz-utils rsync jq file less vim \
      # Python
      python3 python3-dev python3-pip python3-venv python3-setuptools \
      # Board connectivity (network / mDNS)
      openssh-client iputils-ping libnss-mdns avahi-utils \
    && rm -rf /var/lib/apt/lists/*

# Resolve mDNS names (<hostname>.local) through the host's avahi-daemon
RUN sed -i 's/^hosts:.*/hosts:          files mdns4_minimal [NOTFOUND=return] dns mdns4/' /etc/nsswitch.conf

# uv installs the models' venv (scripts/install-venv.sh)
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /usr/local/bin/

# Mount point for the models' venv volume. Ownership set in the image is inherited by a newly created volume
RUN mkdir -p /opt/torch && chown -R vscode:vscode /opt/torch
