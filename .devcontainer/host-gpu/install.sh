#!/usr/bin/env bash
# Installs the entrypoint that grants the remote user access to the host's GPU device nodes.
set -euo pipefail
USERNAME="${_REMOTE_USER:-vscode}"

cat > /usr/local/share/host-gpu-init.sh <<SCRIPT
#!/bin/sh
# Runs at every container start (feature entrypoint).
# /dev is bind-mounted from the host, so the GPU nodes keep the host's group IDs (e.g. render = 110),
# which may not exist, or mean another group, in the container. Add the user to each owning group by GID.
USERNAME="${USERNAME}"
SUDO=""; [ "\$(id -u)" -eq 0 ] || SUDO="sudo -n"
for dev in /dev/kfd /dev/dri/card* /dev/dri/renderD*; do
  [ -c "\$dev" ] || continue
  gid="\$(stat -c %g "\$dev")"
  [ "\$gid" -eq 0 ] && continue
  grp="\$(getent group "\$gid" | cut -d: -f1)"
  if [ -z "\$grp" ]; then
    grp="hostgpu\$gid"
    \$SUDO groupadd -g "\$gid" "\$grp"
  fi
  id -nG "\$USERNAME" | tr ' ' '\n' | grep -qx "\$grp" || \$SUDO usermod -aG "\$grp" "\$USERNAME"
done
exec "\$@"
SCRIPT
chmod +x /usr/local/share/host-gpu-init.sh
