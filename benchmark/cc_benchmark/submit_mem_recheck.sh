#!/bin/bash
# Peak host RSS across the six paired ds004505 recordings, full batch vs blocked.
#
# Re-measures the campaign behind the manuscript's memory paragraph, which
# predates the E-step blocking change and now understates it badly: on sub-01
# the same comparison is 11.39 GiB full batch against 2.43 GiB blocked, where
# the archived campaign reported 11.32 against 6.63 and the paragraph quotes a
# median reduction of 54% (range 42-63%).
#
# One recording is not six. The paragraph states a range and a median across
# recordings, so rewriting it from a single re-measured point would be inventing
# the other five.
#
# Memory is iteration-independent -- the peak is set by the array sizes, not the
# loop count -- so 60 iterations suffices and matches what five of the six
# archived runs used. Only the two amica arms run: the paragraph compares this
# package's full-batch and automatic-blocking paths with each other, not with
# other implementations.
#
#SBATCH --job-name=amica_mem_recheck
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=1-6
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env.sh || exit 1

# --- point the amica runner at the current package -------------------------
# The cluster's checkout predates both the package rename (amica_python ->
# amica) and the E-step blocking this job exists to measure: /scratch/$USER/
# amica-python is the old repo on an old branch, and its venv has amica_python
# installed editable from a third checkout entirely. Reinstalling would mean
# pip on a login node, or rebuilding a working venv to run one benchmark.
# implementation_perf.run_subprocess copies os.environ into every runner, so a
# fresh clone on PYTHONPATH reaches them without touching the venv.
#
#   git clone -b perf/cpu-profiling git@github.com:snesmaeili/amica.git /scratch/$USER/amica-blocked
# AMICA_SRC is read by implementation_perf.py and applied to OUR runner only.
# It must not go on PYTHONPATH globally: scott-huberty's package is imported as
# `amica` too, so a global PYTHONPATH shadows it with ours and its runner dies
# with "cannot import name 'AMICA' from amica".
export AMICA_SRC="${AMICA_SRC:-/scratch/$USER/amica-blocked}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-/scratch/$USER/amica-python/.venv_fir/bin/python}"

# The orchestrator defaults the competitors venv to <benchmark repo>/.venv_competitors,
# which does not exist on fir -- it lives under the amica-python tree. Left
# unset, every competitor run dies instantly with "venv python missing" and the
# task still exits 0, so the array looks like it succeeded while producing
# nothing. The pAMICA venv, by contrast, IS where the default expects it.
export COMPETITORS_VENV="${COMPETITORS_VENV:-/scratch/$USER/amica-python/.venv_competitors/bin/python}"
export PAMICA_VENV="${PAMICA_VENV:-/scratch/$USER/amica-benchmark/.venv_pamica/bin/python}"
for _v in "$AMICA_PYTHON_VENV" "$COMPETITORS_VENV" "$PAMICA_VENV"; do
    [ -x "$_v" ] || { echo "FATAL: no interpreter at $_v" >&2; exit 1; }
done

# installed == intended: assert each competitor venv holds the pinned commit
# from pins.toml before the first fit. Catches silent upstream HEAD drift and a
# clobbered install (e.g. the pyamica/pyAMICA name collision). Cheap; the amica
# build itself is asserted by the AMICA_SRC check just below.
"$COMPETITORS_VENV" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv competitors || exit 1
"$PAMICA_VENV"      "$SLURM_SUBMIT_DIR/check_env.py" verify --venv pamica      || exit 1

# Fail fast rather than benchmark the wrong code. The old checkout imports and
# runs perfectly well; it would just quietly produce a curve for a different
# implementation, which is the one failure mode this whole campaign cannot
# survive. (Runs on the compute node -- importing jax is compute.)
AMICA_SRC="$AMICA_SRC" PYTHONPATH="$AMICA_SRC" "$AMICA_PYTHON_VENV" - <<'PYCHECK' || exit 1
import os, sys
import amica
from amica import AmicaConfig
src = os.path.realpath(amica.__file__)
want = os.path.realpath(os.environ["AMICA_SRC"])
if not src.startswith(want):
    sys.exit(f"FATAL: imported amica from {src}, expected under {want}")
if AmicaConfig().chunk_size != "auto":
    sys.exit("FATAL: this build predates E-step blocking (chunk_size default is not 'auto')")
print(f"amica OK: {src} | default chunk_size={AmicaConfig().chunk_size!r}")
PYCHECK

# installed == intended for the MEASURED amica: assert AMICA_SRC's HEAD is the
# commit pinned in pins.toml, not merely "some E-step-blocked build" — a git pull
# in that checkout otherwise silently changes what this campaign measures. Opt out
# for dev iteration with AMICA_ALLOW_SRC_DRIFT=1.
if [ "${AMICA_ALLOW_SRC_DRIFT:-0}" != "1" ]; then
    _want_amica=$("$AMICA_PYTHON_VENV" "$SLURM_SUBMIT_DIR/check_env.py" pin --venv fir --name amica)
    _got_amica=$(git -C "$AMICA_SRC" rev-parse HEAD 2>/dev/null)
    if [ "$_got_amica" != "$_want_amica" ]; then
        echo "FATAL: AMICA_SRC HEAD ${_got_amica:-<none>} != pinned amica ${_want_amica}" >&2
        echo "       (set AMICA_ALLOW_SRC_DRIFT=1 to run a non-pinned checkout on purpose)" >&2
        exit 1
    fi
    if ! git -C "$AMICA_SRC" diff --quiet HEAD 2>/dev/null; then
        echo "FATAL: AMICA_SRC has uncommitted changes (dirty worktree at pinned HEAD)" >&2
        echo "       (set AMICA_ALLOW_SRC_DRIFT=1 to measure a dirty checkout on purpose)" >&2
        exit 1
    fi
    echo "amica pin OK: AMICA_SRC HEAD == ${_want_amica} (clean)"
fi

# Record which commit produced these numbers. The package is reached through a
# source checkout, so a `git pull` in that directory silently changes what a
# later job measures; a SHA in the log makes that auditable after the fact
# instead of reconstructable only from memory.
echo "amica-blocked commit: $(git -C "$AMICA_SRC" rev-parse --short HEAD 2>/dev/null) $(git -C "$AMICA_SRC" log -1 --format=%s 2>/dev/null)"
# ---------------------------------------------------------------------------

SUBJECT="$SLURM_ARRAY_TASK_ID"
export AMICA_COMPARATOR_RESULTS="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/mem_recheck"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

# Walltime: the archived runs were 60 iterations at 64 components on recordings
# of 0.8-1.4 M samples. At the measured per-iteration cost (1.45 s blocked,
# 1.99 s full batch at 785k samples, scaling with sample count) the pair comes
# to roughly 12 minutes plus preprocessing. 90 minutes is generous.
echo "=== memory re-check: ds004505 sub-0${SUBJECT}, 64 components, 60 iterations ==="
python ../comparator/implementation_perf.py     --dataset ds004505     --subject "$SUBJECT"     --input-level "${AMICA_INPUT_LEVEL:-bids}"     --n-components 64     --max-iter 60     --amica-device cpu --competitor-device cpu     --amica-chunk-size auto     --out-tag "mem_recheck/sub-0${SUBJECT}"     --skip amica_python_numpy pyamica_torch scott_huberty_torch pamica_torch

echo "=== DONE sub-0${SUBJECT}. Results under $AMICA_COMPARATOR_RESULTS/ ==="
