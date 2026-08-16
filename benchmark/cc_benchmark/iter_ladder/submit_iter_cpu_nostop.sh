#!/bin/bash
# CPU stops-off campaign (fir bycore): iteration-matched (AMICA_DISABLE_EARLYSTOP=1) chunk-sweep and
# iteration-ladder, 250 iters (memory iter-independent; s/iter stable), all 25 subjects, 10 reps/cell,
# with the node-load monitor. High parallelism on purpose (we are memory-contended regardless). 12h wall
# so the slow small-chunk cells (esp. pyamica@1024) fit. Writes to iter_ladder/cpu_nostop (SEPARATE from
# the old @1000 CPU baseline). Manifest line: "SUBJECT IMPL CHUNK MAXITER REP".
#   python iter_ladder/build_manifest_cpu_nostop.py 25 10
#   sbatch --array=1-6250%50 iter_ladder/submit_iter_cpu_nostop.sh iter_ladder/manifest_cpu_sweep_nostop.txt
#   sbatch --array=1-3750%50 iter_ladder/submit_iter_cpu_nostop.sh iter_ladder/manifest_cpu_ladder_nostop.txt
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yorguin/iter_ladder_cells/cpunostop-%A_%a.out
#SBATCH --error=/scratch/yorguin/iter_ladder_cells/cpunostop-%A_%a.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source fir_env.sh || exit 1
export AMICA_SKIP_PIN_CHECK=1
export AMICA_DISABLE_EARLYSTOP=1       # iteration-matched: run the full max_iter, stops off
export BIDS_ROOT_DS4505=/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids
MANIFEST="${1:-${MANIFEST:?set manifest as arg1}}"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST"; exit 1; }
mkdir -p /scratch/yorguin/iter_ladder_cells

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$line" ] && { echo "no manifest line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
read -r S IMPL C IT REP <<< "$line"
REP=${REP:-1}
st=$(printf "sub-%02d" "$S")
R=/scratch/yorguin/amica-benchmark-repro
CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz

ALL="amica_python_jax amica_python_jax_chunked amica_python_numpy neuromechanist_numpy pyamica_torch scott_huberty_torch pamica_torch fortran_amica17"
SKIP=""; for k in $ALL; do [ "$k" = "$IMPL" ] || SKIP="$SKIP $k"; done

export AMICA_PYTHON_VENV=$R/.venv_fir_gpu/bin/python
export COMPETITORS_VENV=$R/.venv_competitors_main/bin/python
export PAMICA_VENV=$R/.venv_pamica_main/bin/python
export AMICA_SRC=/scratch/yorguin/amica_main_src
export AMICA_SCOTT_BATCH=$C AMICA_PYAMICA_CHUNK=$C AMICA_PAMICA_BLOCK_SIZE=$C AMICA_FORTRAN_BLOCK=$C
export AMICA_COMPARATOR_RESULTS=/scratch/yorguin/iter_ladder/cpu_nostop   # SEPARATE from the @1000 baseline
mkdir -p "$AMICA_COMPARATOR_RESULTS"

FOPT=""
if [ "$IMPL" = fortran_amica17 ]; then
  FOPT="--include-fortran"
  export AMICA17_BIN=/project/rrg-kjerbi/yorguin/amica_fortran_reference/amica17
fi
CIN=""; [ -f "$CACHE" ] && CIN="--cached-input $CACHE"

MONDIR="$AMICA_COMPARATOR_RESULTS/nodemon"; mkdir -p "$MONDIR"
MON="$MONDIR/${st}_${IMPL}_c${C}_i${IT}_r${REP}_${SLURM_JOB_ID:-0}.csv"
bash iter_ladder/node_monitor.sh "$MON" "${MON_INTERVAL:-10}" &
MONPID=$!
trap 'kill $MONPID 2>/dev/null' EXIT

echo "=== CPU stops-off cell: sub-$S impl=$IMPL chunk=$C max_iter=$IT rep=$REP ==="
python ../comparator/implementation_perf.py \
    --dataset ds004505 --subject "$S" --input-level bids \
    --n-components 64 --max-iter "$IT" --seeds 0 \
    --amica-device cpu --competitor-device cpu \
    --amica-chunk-size "$C" \
    $CIN --out-tag "c${C}_i${IT}_r${REP}" $FOPT \
    --skip $SKIP
kill $MONPID 2>/dev/null
echo "CPU_NOSTOP_CELL_DONE sub-$S $IMPL c$C i$IT r$REP"
