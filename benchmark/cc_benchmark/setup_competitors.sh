#!/bin/bash
# Build .venv_competitors with the 3 external AMICA reimplementations used by the
# cross-implementation MEMORY comparison (benchmark/comparator). Run ONCE per site.
#
#   pyamica  (DerAndereJohannes)  PyTorch     -> import pyamica
#   amica    (scott-huberty)      PyTorch     -> import amica
#
# The March-2025 sccn/pAMICA snapshot (pure NumPy, `import pyAMICA`) is NO LONGER
# installed here: its distribution name "pyAMICA" canonicalizes to "pyamica"
# (PEP 503), so pip treats it as the same project as DerAndereJohannes/pyamica
# and installing it UNINSTALLS pyamica. It now gets its own venv -> see
# setup_neuromechanist.sh (used only by --include-neuromechanist-snapshot).
#
# Current pAMICA (v0.3.1, package `pamica`) is NOT here either: it needs Python
# >= 3.12. It gets its own venv -> see setup_pamica.sh.
#
# All commits are pinned in pins.toml (single source of truth); this script
# installs from it and then asserts installed == intended via check_env.py.
#
# git+pip needs internet, so run this on the LOGIN node — it is a one-time ENV BUILD
# (not compute), which is the supported use of pip on a login node. On Alliance the
# `--no-index torch` wheel is CUDA-capable and serves both the CPU and GPU comparisons.
#
#   bash setup_competitors.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/cc_benchmark/
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${COMPETITORS_VENV_DIR:-$REPO_ROOT/.venv_competitors}"

# Caches off $HOME (Alliance quota), mirroring fir_env.sh.
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

# torch: Alliance wheelhouse wheel (--no-index) is CUDA-enabled and works for BOTH the CPU
# and GPU comparisons; fall back to PyPI off-Alliance.
echo "Installing torch ..."
pip install --no-index torch 2>/dev/null || pip install torch

# The competitor AMICA implementations, installed from pins.toml. These are the
# PUBLISHED PyPI releases (pyamica==0.3.0, amica-python==0.1.1), recovered from
# the surviving publication venv; a version pin is the stricter identity for a
# released wheel, and check_env.py asserts the installed version below.
echo "Installing the pinned competitor implementations (see pins.toml) ..."
while read -r spec; do
    [ -n "$spec" ] && pip install "$spec"
done < <(python "$HERE/check_env.py" specs --venv competitors)

# Optional: NVML neutral cross-check for the GPU comparison (enable with AMICA_MEM_NVML=1).
pip install nvidia-ml-py 2>/dev/null || echo "(nvidia-ml-py not installed; NVML cross-check stays off)"

echo "=== verify imports ==="
python -c "import torch, pyamica, amica; print('OK — torch', torch.__version__, '| pyamica + amica(scott) import')"

# Assert installed == intended (fails loudly on any drift) and record the
# as-built lock (resolved torch/numpy/... alongside the pinned commits).
echo "=== verify pins ==="
python "$HERE/check_env.py" verify --venv competitors
python "$HERE/check_env.py" lock --venv competitors
echo "competitors venv ready: $VENV"
