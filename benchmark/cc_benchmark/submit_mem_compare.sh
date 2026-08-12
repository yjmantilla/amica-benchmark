#!/bin/bash
# Cross-implementation MEMORY comparison — CPU peak RSS (portable capsule version).
#
# Reproducible on any Alliance/Compute-Canada cluster:
#   1) cp env.template env.local   &&  edit (account / BIDS_ROOT / results / dataset)
#   2) bash setup_competitors.sh    (build .venv_competitors: pyamica + scott + neuromechanist)
#   3) sbatch submit_mem_compare.sh        # or: bash submit_all.sh  (passes --account/--partition)
#
# Measures absolute + delta peak RSS (resource.getrusage high-water mark) for AMICA-Python
# in BOTH configs (full-batch and chunked = the memory/speed dial), pyamica, scott-huberty,
# neuromechanist — and Fortran AMICA 1.7 ONLY if you built amica17 and exported AMICA17_BIN.
# Memory is iteration-independent, so 1 subject / ~100 iter captures the peak. Aggregate +
# figure run locally afterwards (see README): aggregate_pilot.py -> plot_pilot.py (fig12).
#
# The #SBATCH lines are fir defaults; override per-site via submit_all.sh or `sbatch --account ...`.
#SBATCH --job-name=amica_mem_compare
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1              # modules + .venv_fir + env.local (BIDS_ROOT, AMICA_RESULTS_DIR, ...)
REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"

# installed == intended: assert the pinned competitor + pamica commits before the
# orchestrator's first fit (these venvs are what implementation_perf.py launches).
# Skip a venv that is absent (matches this job's amica-only fallback posture);
# fail fast only when it exists but drifts from pins.toml.
for _pair in "competitors:${COMPETITORS_VENV:-$REPO_ROOT/.venv_competitors/bin/python}" \
             "pamica:${PAMICA_VENV:-$REPO_ROOT/.venv_pamica/bin/python}"; do
    _venv="${_pair%%:*}"; _py="${_pair#*:}"
    if [ -x "$_py" ]; then
        "$_py" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv "$_venv" \
            || { echo "FATAL: $_venv venv drifted from pins.toml" >&2; exit 1; }
    fi
done

[ -x "$REPO_ROOT/.venv_competitors/bin/python" ] || \
  echo "WARN: $REPO_ROOT/.venv_competitors missing — run setup_competitors.sh; the competitor impls will be skipped/error."

# CPU results in their own subdir (GPU job uses .../comparator/gpu) so the shared
# amica_python_jax_chunked JSON never collides between the two jobs.
export AMICA_COMPARATOR_RESULTS="${AMICA_COMPARATOR_RESULTS:-${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/comparator/cpu}"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

# Fortran AMICA 1.7 is OPTIONAL — included only if you built amica17 and exported
# AMICA17_BIN. Default to the group-readable staged copy; the expected sha is the
# single source of truth from pins.toml, asserted here AND per-fit by run_fortran.
FORTRAN_OPT=""
export AMICA17_BIN="${AMICA17_BIN:-$SLURM_SUBMIT_DIR/../../fortran/amica17}"
if [ -x "${AMICA17_BIN}" ]; then
    AMICA17_SHA_EXPECTED="${AMICA17_SHA_EXPECTED:-$(python "$SLURM_SUBMIT_DIR/check_env.py" fortran-sha)}"
    export AMICA17_SHA_EXPECTED
    _sha=$(sha256sum "$AMICA17_BIN" | awk '{print $1}')
    if [ "$_sha" != "$AMICA17_SHA_EXPECTED" ]; then
        echo "FATAL: $AMICA17_BIN has sha $_sha, expected $AMICA17_SHA_EXPECTED (pins.toml)" >&2
        exit 1
    fi
    module load openmpi/4.1.5 flexiblas 2>/dev/null || true
    export GNU_TIME_BIN="${GNU_TIME_BIN:-/usr/bin/time}"
    FORTRAN_OPT="--include-fortran"
    echo "Fortran amica17 verified ($AMICA17_BIN, sha ${_sha:0:12}…) -> including it."
else
    echo "Fortran amica17 not built / AMICA17_BIN unset -> skipping it (optional)."
fi

# Optional custom run tag; default is the orchestrator's '<dataset>_sub-NN', which
# aggregate_pilot.py recognizes.
TAG_OPT=""; [ -n "${AMICA_MEM_TAG:-}" ] && TAG_OPT="--out-tag ${AMICA_MEM_TAG}"

echo "=== CPU memory comparison: dataset=${AMICA_MEM_DATASET:-ds004505} subject=${AMICA_MEM_SUBJECT:-1} ==="
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter "${AMICA_MEM_ITER:-100}" \
    --amica-device cpu --competitor-device cpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    $FORTRAN_OPT $TAG_OPT \
    --skip amica_python_numpy

echo "=== DONE. Results under $AMICA_COMPARATOR_RESULTS/ . Aggregate + figure locally: ==="
echo "  python ../comparator/aggregate_pilot.py --root '$AMICA_COMPARATOR_RESULTS' --impls all"
echo "  python ../comparator/plot_pilot.py      --root '$AMICA_COMPARATOR_RESULTS'"
