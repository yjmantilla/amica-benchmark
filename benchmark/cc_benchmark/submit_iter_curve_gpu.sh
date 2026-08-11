#!/bin/bash
# Cluster GPU: runtime as a function of iteration count, every GPU-capable
# implementation.
#
# Produces the GPU panel of the runtime-vs-iterations figure, and replaces the
# archived GPU runtime points, which cannot support a curve. On the archived
# runs amica_python_jax_chunked takes 39.9 s at 100 iterations and 14.4 s at 600
# -- a negative per-iteration cost -- while scott_huberty and pyamica both land
# on exactly 48.6 s at 600. At those magnitudes the measurement is dominated by
# compilation and warm-up rather than by the fit, so the archived numbers cannot
# be extended to 1000 iterations; they have to be replaced.
#
# Two things are done differently here for that reason:
#   * four iteration caps, so the per-iteration cost comes from a slope over a
#     range rather than from a difference between two warm-up-dominated points;
#   * a discarded warm-up fit before each timed one, so JIT compilation and
#     cuBLAS/cuDNN autotuning are not charged to the first timed iteration.
#
# One array task per iteration cap, all implementations inside it, so each task
# is short and a failure costs one point rather than the whole panel.
#
#SBATCH --job-name=amica_iter_curve_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1               # modules (incl. cuda/cudnn) + .venv_fir + env.local

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

ITERS=(100 400 700 1000)
IT="${ITERS[$SLURM_ARRAY_TASK_ID]}"

# Walltime justification. The only archived GPU per-iteration cost that is not
# self-contradictory is pamica_torch at 1.88 s/iter, which sets the ceiling:
# ~31 min at 1000 iterations. The other three are well under a minute per run at
# any of these caps. 3 h is deliberate headroom, because the archived GPU
# timings are exactly what this job exists to distrust.
export AMICA_COMPARATOR_RESULTS="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/itercurve/gpu"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== GPU iteration curve: max_iter=$IT ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,compute_mode --format=csv,noheader || true

# Check the interpreters that actually run the fits, not the one fir_env.sh
# happens to activate. That venv carries a CPU-only jaxlib, so probing it prints
# "[CpuDevice(id=0)]" on a perfectly good GPU node -- which is exactly what made
# a working allocation look broken once already. A GPU panel built from runners
# that silently fell back to CPU would be worse than no panel, so this is fatal.
gpu_check() {
    "$1" - "$2" <<'PYCHK'
import importlib.util as u, sys
label = sys.argv[1]
ok = True
if u.find_spec("jax") is not None:
    import jax
    devs = jax.devices()
    ok = any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in devs)
    print(f"  {label} jax devices: {devs}")
if u.find_spec("torch") is not None:
    import torch
    ok = torch.cuda.is_available()
    print(f"  {label} torch {torch.__version__} cuda: {ok}")
    if ok:
        torch.zeros(1024, 1024, device="cuda"); torch.cuda.synchronize()
sys.exit(0 if ok else 1)
PYCHK
}
gpu_check "$AMICA_PYTHON_VENV" amica       || { echo "FATAL: amica venv cannot see the GPU" >&2; exit 1; }
gpu_check "$COMPETITORS_VENV"  competitors || { echo "FATAL: competitors venv cannot see the GPU" >&2; exit 1; }
gpu_check "$PAMICA_VENV"       pamica      || { echo "FATAL: pamica venv cannot see the GPU" >&2; exit 1; }

# Warm-up: one short fit whose timing is thrown away, so compilation and
# autotuning do not land inside the measured run.
echo "--- warm-up (discarded) ---"
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter 10 \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    --nvml-crosscheck \
    --out-tag "itercurve_gpu/warmup_iter${IT}" \
    --skip amica_python_numpy amica_python_jax > /dev/null 2>&1 || true

echo "--- timed run: max_iter=$IT ---"
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter "$IT" \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    --nvml-crosscheck \
    --out-tag "itercurve_gpu/iter${IT}" \
    --skip amica_python_numpy amica_python_jax

echo "=== DONE (max_iter=$IT). Results under $AMICA_COMPARATOR_RESULTS/itercurve_gpu/ ==="
