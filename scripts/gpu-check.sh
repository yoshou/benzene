#!/usr/bin/env bash
# Check that the host's AMD GPUs are usable from the container: device access, then PyTorch (ROCm) on each GPU
set -uo pipefail
source "$(dirname "$0")/_common.sh"
PY=$VENV/bin/python
ng=0

echo "== Device nodes"
ls -l /dev/kfd /dev/dri/renderD* 2>&1
echo "== Groups of $(id -un): $(id -nG)"
for d in /dev/kfd /dev/dri/renderD*; do
  if [[ -r $d && -w $d ]]; then echo "OK: $d"; else echo "NG: no read/write access to $d (rebuild the container)"; ng=1; fi
done

[[ -x $PY ]] || { echo "The models' venv is not installed: bash scripts/install-venv.sh" >&2; exit 1; }
echo "== PyTorch"
"$PY" - <<'PY' || ng=1
import sys, torch
print(f"torch {torch.__version__} / HIP {torch.version.hip}")
if not torch.cuda.is_available():
    sys.exit("NG: torch.cuda.is_available() is False")
a = torch.randn(1024, 1024)
ref = a @ a
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    d = f"cuda:{i}"
    err = ((a.to(d) @ a.to(d)).cpu() - ref).abs().max().item()
    print(f"{d}: {p.name} ({p.gcnArchName}, {p.total_memory / 2**30:.1f} GiB)  matmul max err vs CPU = {err:.2e}")
    if err > 1e-2:
        sys.exit(f"NG: {d} result differs from the CPU")
PY

(( ng )) && { echo "== NG"; exit 1; }
echo "== OK"
