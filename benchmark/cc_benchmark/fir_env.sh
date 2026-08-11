#!/bin/bash
# FIR cluster environment setup for AMICA benchmarking
# Ref: https://github.com/BabaSanfour/crash-course/tree/main/module_05_advanced_alliance_ai_workflows

# `set -u` must be OFF from here to the end of the module block. Callers that
# use `set -euo pipefail` otherwise die before any compute runs: the Compute
# Canada profile reads SKIP_CC_CVMFS while unset, and the GPU/partition tests
# below read variables that simply do not exist in a CPU job. Restored at the
# bottom to whatever the caller had.
_amica_had_u=0
case $- in *u*) _amica_had_u=1 ;; esac
set +u

# sbatch and srun give a NON-login shell, in which the `module` function is not
# defined at all. Without this the module loads below silently no-op and the
# job runs against the wrong interpreter.
[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ] && \
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

# ── Module loads ──
# Force all caches to scratch to avoid /home quota Input/output errors
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export XDG_DATA_HOME="/scratch/$USER/.local/share"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$PIP_CACHE_DIR"

# ── Per-user/site config: copy env.template -> env.local and edit (env.local is gitignored).
# Sourced here if present so accounts/paths can be overridden without editing this file.
_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_ENV_DIR/env.local" ] && source "$_ENV_DIR/env.local"

module purge
module load StdEnv/2023 || true
module load python/3.11
module load scipy-stack

# Load CUDA and cuDNN for JAX GPU support
module load cuda/12.6
module load cudnn

# Export path for XLA to find CUDA
if [ -n "${CUDA_HOME:-}" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi

# ── Dataset paths (crash-course rule: reuse shared /project datasets) ──
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"

# ── NumPy/MKL thread tuning ──
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ── Results directory ──
# Temporarily pointing to scratch because the team /project quota is full!
RESULTS_DIR="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_python_validation_v3}"
export AMICA_RESULTS_DIR="$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ── Virtual environment ──
# Use a persistent venv in the repo root (created on the login node with internet)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv_fir"

# Is this a GPU job? Both variables are absent in a CPU job, hence the guards.
if [[ "${SLURM_JOB_PARTITION:-}" == *"gpu"* ]] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    AMICA_GPU_JOB=true
else
    AMICA_GPU_JOB=false
fi

# Completeness test, not a directory test. `[ -d "$VENV_PATH" ]` calls a venv
# usable the instant virtualenv creates it, so a build that dies part-way
# leaves a stub that every later job then "reuses" -- which is how job 53097938
# failed on `import jax` twelve minutes after 53097937 died mid-bootstrap and
# left 13 packages behind. Probe the imports every job actually needs.
# `amica` is in the probe: the [jax-cpu]/[jax-gpu] extras install JAX but NOT the
# algorithm under test, so a venv that imports jax but not amica is incomplete.
# (Only presence is probed here, not the commit: AMICA_SRC legitimately overrides
# the baseline amica at run time for OUR runner, so a commit test here would
# trigger spurious mid-job rebuilds. The exact baseline commit is asserted at
# setup time by check_env.py verify, below.)
AMICA_VENV_PROBE="import numpy, scipy, mne, amica"
if [ "$AMICA_GPU_JOB" = true ]; then
    AMICA_VENV_PROBE="$AMICA_VENV_PROBE, jax"
fi

if [ -x "$VENV_PATH/bin/python" ] && \
   "$VENV_PATH/bin/python" -c "$AMICA_VENV_PROBE" >/dev/null 2>&1; then
    REINSTALL=false
else
    REINSTALL=true
    # Guard the rm: REPO_ROOT comes from a `cd`, and an empty one would make
    # VENV_PATH "/.venv_fir". Only ever delete a path we actually constructed.
    if [ -d "$VENV_PATH" ] && [ -n "$REPO_ROOT" ] && [ "$VENV_PATH" = "$REPO_ROOT/.venv_fir" ]; then
        echo "fir_env: venv at $VENV_PATH is incomplete ($AMICA_VENV_PROBE failed); rebuilding."
        rm -rf "$VENV_PATH"
    elif [ -d "$VENV_PATH" ]; then
        echo "fir_env: venv at $VENV_PATH looks incomplete but the path is unexpected; not deleting." >&2
        echo "fir_env: remove it by hand and re-run." >&2
        [ "$_amica_had_u" = 1 ] && set -u
        return 1 2>/dev/null || exit 1
    fi
fi



if [ "$REINSTALL" = true ]; then
    echo "Setting up virtual environment at $VENV_PATH..."
    virtualenv --no-download "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --no-index --upgrade pip

    # Extras must match this repository's pyproject.toml, which defines
    # jax-cpu, jax-gpu and amica -- NOT the all/gpu extras that amica-python
    # defines. This file was copied across from that repo, so it asked pip for
    # extras that do not exist here; pip warns and installs the base
    # dependencies only, leaving the venv without jax and without the
    # algorithm. Every job sourcing this file then failed on its first import.
    if [ "$AMICA_GPU_JOB" = true ]; then
        echo "GPU job detected, installing with [jax-gpu] extra..."
        pip install -e "$REPO_ROOT[jax-gpu]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "CPU job detected, installing with [jax-cpu] extra..."
        pip install -e "$REPO_ROOT[jax-cpu]"
    fi

    # Install additional benchmarking dependencies. Compute nodes on fir do
    # reach PyPI (verified HTTP/2 200 to pypi.org from fc30669), so this does
    # not need to run on a login node.
    #
    # openneuro-py is deliberately NOT installed. It only *downloads* datasets,
    # and the cluster reads pre-staged BIDS trees from /project. It pulls qh3,
    # which has no wheel for this platform and builds through maturin/bindgen,
    # failing on a missing libclang -- which aborted the whole environment
    # build under `set -e` after everything else had installed correctly.
    pip install mne-bids

    # Install the reference `amica` package itself. The [jax-cpu]/[jax-gpu]
    # extras above install JAX but NOT amica (this repo's `amica` extra is
    # separate), which previously left a "complete" venv that could import jax
    # but not the algorithm under test — every job then failed on its first
    # `import amica`, or silently used an unrelated pre-existing install.
    # Pinned in pins.toml; AMICA_SRC still overrides it at run time for OUR runner.
    while read -r spec; do
        [ -n "$spec" ] && pip install "$spec"
    done < <(python "$SCRIPT_DIR/check_env.py" specs --venv fir)

    echo "Environment installed:"
    python -c "import numpy, scipy, mne, amica; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| mne', mne.__version__, '| amica', getattr(amica, '__version__', '?'))"
    if [ "$AMICA_GPU_JOB" = true ]; then
        python -c "import jax; print('jax', jax.__version__, '| devices', jax.devices())"
    fi

    # Record the as-built lock right after a fresh install.
    python "$SCRIPT_DIR/check_env.py" lock --venv fir
else
    echo "Activating existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Assert the baseline amica build == pins.toml on EVERY job — fresh OR reused
# venv — so any fir-sourcing paper run (v3 / comparators / heldout / scaling /
# reval) fails fast instead of silently measuring an off-pin amica. This checks
# the INSTALLED baseline; AMICA_SRC's per-runner override is asserted separately
# by the submit scripts that use it. Opt out with AMICA_SKIP_PIN_CHECK=1 for
# ad-hoc/dev use.
if [ "${AMICA_SKIP_PIN_CHECK:-0}" != "1" ]; then
    if ! python "$SCRIPT_DIR/check_env.py" verify --venv fir; then
        echo "fir_env: FATAL — installed amica != pins.toml (set AMICA_SKIP_PIN_CHECK=1 to bypass)." >&2
        [ "$_amica_had_u" = 1 ] && set -u
        return 1 2>/dev/null || exit 1
    fi
fi

# Restore the caller's `set -u` (see the top of this file).
[ "$_amica_had_u" = 1 ] && set -u
unset _amica_had_u AMICA_VENV_PROBE
