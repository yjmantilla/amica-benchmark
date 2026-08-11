#!/bin/bash
# Cluster CPU: runtime as a function of iteration count, every implementation.
#
# Produces the CPU panel of the runtime-vs-iterations figure. No implementation
# records per-iteration times, and hooking four third-party training loops to add
# them would make each line a different measurement, so the curve is built the
# uniform way: the same fit run to four iteration caps, each timed end to end.
# Every plotted point is a measured fit.
#
# Why all four points are re-run rather than appended to the archived 100- and
# 600-iteration results: amica now blocks the E-step by default (chunk_size=
# "auto"), so its archived points came from a different implementation. Mixing
# them into one line would draw a curve no single version of the code produces.
# The competitors are unchanged, but they are re-run alongside so that every
# point in the figure comes from one campaign on one node.
#
# This also re-measures peak RSS at 64 components x 785,328 samples, where the
# archived numbers (amica full batch 11.28 GiB, blocked 6.63 GiB) predate the
# blocking change.
#
# One array task per implementation: a failure costs one line, not the figure,
# and the short implementations do not wait behind the long ones.
#
#SBATCH --job-name=amica_iter_curve_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1               # modules + .venv_fir + env.local (BIDS_ROOT, AMICA_RESULTS_DIR, ...)

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

# Walltime justification, from the archived 100- and 600-iteration runs on this
# exact problem (per-iteration cost = (t600 - t100) / 500):
#   amica_python_jax          2.73 s/iter -> ~1.7 h for 100+400+700+1000
#   amica_python_jax_chunked  2.92 s/iter -> ~1.8 h   (expected lower now)
#   pamica_torch              3.33 s/iter -> ~2.0 h
#   scott_huberty_torch       3.84 s/iter -> ~2.3 h
#   fortran_amica17           5.96 s/iter -> ~3.6 h
#   pyamica_torch             7.47 s/iter -> ~4.6 h   <- sets the 6 h request
ALL_IMPLS=(amica_python_jax amica_python_jax_chunked pamica_torch \
           scott_huberty_torch pyamica_torch fortran_amica17)
KEEP="${ALL_IMPLS[$SLURM_ARRAY_TASK_ID]}"

# The orchestrator runs every implementation per invocation, so isolating one
# per array task means skipping the complement.
SKIP="amica_python_numpy"
for impl in "${ALL_IMPLS[@]}"; do
    [ "$impl" = "$KEEP" ] && continue
    [ "$impl" = "fortran_amica17" ] && continue     # gated by --include-fortran, not --skip
    SKIP="$SKIP $impl"
done

FORTRAN_OPT=""
if [ "$KEEP" = "fortran_amica17" ]; then
    # Default to the archived reference build, not whatever amica17 is nearest.
    # /scratch/$USER/fortran_parity/amica17_build/amica17 is an ablation build;
    # using it once produced a worst matched |r| of 0.2765 against an archived
    # 0.9391, which reads as a parity failure rather than the wrong binary.
    # Hence the checksum: this row is only meaningful for the reference build.
    # Default to the group-readable staged copy; the expected sha is the SINGLE
    # source of truth from pins.toml (no hard-coded second copy that can drift).
    export AMICA17_BIN="${AMICA17_BIN:-/project/rrg-kjerbi/sesma-shared/amica-repro/amica17}"
    AMICA17_SHA_EXPECTED="${AMICA17_SHA_EXPECTED:-$("$AMICA_PYTHON_VENV" "$SLURM_SUBMIT_DIR/check_env.py" fortran-sha)}"
    export AMICA17_SHA_EXPECTED          # run_fortran.py also asserts it per fit
    if [ -x "${AMICA17_BIN}" ]; then
        _sha=$(sha256sum "$AMICA17_BIN" | awk '{print $1}')
        if [ "$_sha" != "$AMICA17_SHA_EXPECTED" ]; then
            echo "FATAL: $AMICA17_BIN has sha $_sha, expected $AMICA17_SHA_EXPECTED (pins.toml)" >&2
            exit 1
        fi
        echo "Fortran reference binary verified: $AMICA17_BIN (sha ${_sha:0:12}…)"
        module load openmpi/4.1.5 flexiblas 2>/dev/null || true
        export GNU_TIME_BIN="${GNU_TIME_BIN:-/usr/bin/time}"
        FORTRAN_OPT="--include-fortran"
    else
        echo "FATAL: no Fortran binary at $AMICA17_BIN" >&2
        exit 1
    fi
fi

# A distinct tag per component count, so a 30-component campaign cannot land on
# top of a 64-component one. The two are not interchangeable: per-iteration cost
# is not linear in the component count (measured 3.92/6.83/17.45/19.87 ms/iter at
# C=16/32/48/64 at fixed sample count, where a linear fit misses by up to 46%), so
# mixing them in one directory would silently blend two scalings into one curve.
# Running the laptop fixture (mne_sample) here makes the cluster and local
# panels differ in hardware alone, instead of in dataset, rank and recording
# length at once. MNE_DATASETS_SAMPLE_PATH is set explicitly rather than read
# from MNE's stored config, which goes stale whenever data moves and then
# fails in a way that reads as a benchmark bug rather than a path problem.
export MNE_DATASETS_SAMPLE_PATH="${MNE_DATASETS_SAMPLE_PATH:-$HOME/mne_data}"
export AMICA_ITER_TAG="${AMICA_ITER_TAG:-itercurve_cpu}"
export AMICA_COMPARATOR_RESULTS="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/itercurve/${AMICA_ITER_TAG}"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== task $SLURM_ARRAY_TASK_ID: $KEEP (skipping: $SKIP) ==="
python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"

# Overridable so a single missing point can be refilled without re-running the
# whole campaign, which at 2200 iterations per implementation is hours.
ITERS="${AMICA_ITERS:-100 400 700 1000}"
for IT in $ITERS; do
    echo "--- $KEEP @ max_iter=$IT ---"
    python ../comparator/implementation_perf.py \
        --dataset "${AMICA_MEM_DATASET:-ds004505}" \
        --subject "${AMICA_MEM_SUBJECT:-1}" \
        --input-level "${AMICA_INPUT_LEVEL:-bids}" \
        --n-components "${AMICA_MEM_NCOMP:-64}" \
        --max-iter "$IT" \
        --amica-device cpu --competitor-device cpu \
        --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
        $FORTRAN_OPT \
        --out-tag "${AMICA_ITER_TAG}/iter${IT}" \
        --skip $SKIP
done

echo "=== DONE ($KEEP, C=${AMICA_MEM_NCOMP:-64}). Results under $AMICA_COMPARATOR_RESULTS/ ==="
