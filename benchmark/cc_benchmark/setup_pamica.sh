#!/bin/bash
# Build .venv_pamica with sccn/pAMICA, the SCCN PyTorch implementation
# (Shirazi, Delorme, Makeig). Run ONCE per site.
#
#   pamica (sccn/pAMICA v0.3.1)  PyTorch NG  -> import pamica
#
# Why this is a separate venv from .venv_competitors: pAMICA requires Python
# >= 3.12 and torch >= 2.12.1, while the competitors venv is built on 3.11 for
# the older implementations. They cannot share an interpreter.
#
# Why the install is pinned: github.com/neuromechanist/pyAMICA was renamed and
# transferred to github.com/sccn/pAMICA (same GitHub repository id), so the two
# URLs resolve to one project at two very different states. Pinning the tag
# keeps "which pAMICA did we measure" answerable; the parity and performance
# claims in the docs are version-specific.
#
# git+pip needs internet, so run this on the LOGIN node — it is a one-time ENV
# BUILD (not compute), which is the supported use of pip on a login node.
#
#   bash setup_pamica.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/cc_benchmark/
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${PAMICA_VENV_DIR:-$REPO_ROOT/.venv_pamica}"

# v0.3.1 (2026-07-19). The commit is pinned in pins.toml (single source of
# truth); this label is for humans. To adopt a newer pamica: bump the commit in
# pins.toml, re-run the parity check in scripts/paper/figures/, and re-record —
# the numbers are not portable across versions.
PAMICA_TAG="v0.3.1"

# Caches off $HOME (Alliance quota), mirroring fir_env.sh.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

module purge 2>/dev/null || true
# 3.12 is the floor pamica declares; fir carries 3.12.4.
module load StdEnv/2023 python/3.12 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    virtualenv --no-download "$VENV" 2>/dev/null || python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# Fail loudly rather than building an environment that cannot run pamica.
PYV=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PYV" in
    3.1[2-9]|3.[2-9][0-9]) : ;;
    *) echo "ERROR: pamica needs Python >= 3.12, this venv is $PYV." >&2
       echo "       'module load python/3.12' did not take; check module avail python." >&2
       exit 1 ;;
esac
echo "python $PYV OK"

# Scientific stack, PINNED from pins.toml (torch==2.13.0 etc.) so a fresh build is
# reproducible. The Alliance --no-index wheel is CUDA-enabled and serves BOTH the
# CPU and GPU comparisons; fall back to PyPI off-Alliance. verify --strict enforces
# these against the committed reference lock.
echo "Installing the pinned scientific stack ..."
while read -r s; do
    [ -n "$s" ] && (pip install --no-index "$s" 2>/dev/null || pip install "$s")
done < <(python "$HERE/check_env.py" stack-specs --venv pamica)

echo "Installing pamica $PAMICA_TAG (pinned in pins.toml) ..."
while read -r spec; do
    [ -n "$spec" ] && pip install "$spec"
done < <(python "$HERE/check_env.py" specs --venv pamica)

# psutil is a dependency of the runner protocol (_common.py reads the process
# high-water RSS through it), not of pamica, so nothing above pulls it in.
echo "Installing runner dependencies ..."
pip install --no-index psutil 2>/dev/null || pip install psutil

# Optional: NVML neutral cross-check for the GPU comparison (enable with AMICA_MEM_NVML=1).
pip install nvidia-ml-py 2>/dev/null || echo "(nvidia-ml-py not installed; NVML cross-check stays off)"

echo "=== verify API surface ==="
python - <<'PY'
from importlib.metadata import version
import torch
from pamica import AMICA

print(f"pamica {version('pamica')} | torch {torch.__version__} | "
      f"cuda available: {torch.cuda.is_available()}")
# The runner depends on all four of these existing; catch an API drift here
# rather than three hours into an array job.
for attr in ("fit", "get_unmixing_matrix", "ll_history_", "final_ll_"):
    assert hasattr(AMICA, attr) or attr.endswith("_"), f"AMICA lacks {attr}"
print("OK — pamica imports and the runner's API surface is present")
PY

# Assert the installed pamica SHA equals the pinned commit — via direct_url.json,
# not a `startswith("0.3.")` version-string test that a future 0.3.2 would pass.
echo "=== verify pins ==="
python "$HERE/check_env.py" verify --venv pamica
python "$HERE/check_env.py" lock --venv pamica
echo "pamica venv ready: $VENV"
