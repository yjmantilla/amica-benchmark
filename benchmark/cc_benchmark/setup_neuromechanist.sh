#!/bin/bash
# Build .venv_neuromechanist with the March-2025 sccn/pAMICA pure-NumPy snapshot
# (formerly neuromechanist/pyAMICA). Run ONCE per site.
#
#   pyAMICA  (sccn/pAMICA @526aa32, pre-rename snapshot)  pure NumPy  -> import pyAMICA
#
# Why a SEPARATE venv from .venv_competitors: this snapshot's distribution name
# is "pyAMICA", which canonicalizes to "pyamica" (PEP 503) — the same project
# name as DerAndereJohannes/pyamica. Installed into the competitors venv it
# UNINSTALLS pyamica (the default-compared implementation). Isolating it here
# keeps both usable. It is only needed for the optional
# `--include-neuromechanist-snapshot` comparison.
#
# Pure NumPy: no torch, so this venv is small. Commit pinned in pins.toml.
#
# git+pip needs internet, so run this on the LOGIN node — it is a one-time ENV
# BUILD (not compute), the supported use of pip on a login node.
#
#   bash setup_neuromechanist.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/cc_benchmark/
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${NEUROMECHANIST_VENV_DIR:-$REPO_ROOT/.venv_neuromechanist}"

# Caches off $HOME (Alliance quota), mirroring the other setup scripts.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    virtualenv --no-download "$VENV" 2>/dev/null || python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# numpy (the implementation is pure NumPy) + psutil (the runner protocol reads
# the process high-water RSS through it; see _common.py).
echo "Installing runner dependencies (pinned stack from pins.toml + psutil) ..."
while read -r s; do
    [ -n "$s" ] && (pip install --no-index "$s" 2>/dev/null || pip install "$s")
done < <(python "$HERE/check_env.py" stack-specs --venv neuromechanist)
pip install --no-index psutil 2>/dev/null || pip install psutil

echo "Installing the pinned pyAMICA snapshot (see pins.toml) ..."
while read -r spec; do
    [ -n "$spec" ] && pip install "$spec"
done < <(python "$HERE/check_env.py" specs --venv neuromechanist)

echo "=== verify imports ==="
python -c "import numpy, pyAMICA; print('OK — numpy', numpy.__version__, '| pyAMICA imports')"

echo "=== verify pins ==="
python "$HERE/check_env.py" verify --venv neuromechanist
python "$HERE/check_env.py" lock --venv neuromechanist
echo "neuromechanist venv ready: $VENV"
