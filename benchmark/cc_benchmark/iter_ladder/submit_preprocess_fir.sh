#!/bin/bash
# One-time PCA-cache build on FIR for the CPU stops-off campaign (array: one subject per task).
# fir has only 5 caches; this builds all 25 so the sweep cells reuse them via --cached-input instead
# of re-preprocessing per cell. Cheap (~1-2 min/subject).
#   python iter_ladder/build_manifest_cpu_nostop.py   # (not needed for prep)
#   sbatch --array=1-25 iter_ladder/submit_preprocess_fir.sh
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yorguin/iter_ladder_cells/prep-%A_%a.out
#SBATCH --error=/scratch/yorguin/iter_ladder_cells/prep-%A_%a.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source fir_env.sh || exit 1
export AMICA_SKIP_PIN_CHECK=1
export BIDS_ROOT_DS4505=/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids
R=/scratch/yorguin/amica-benchmark-repro
source "$R/.venv_fir_gpu/bin/activate" 2>/dev/null || true
mkdir -p /scratch/yorguin/realchunk_cache /scratch/yorguin/iter_ladder_cells
S=$SLURM_ARRAY_TASK_ID
st=$(printf "sub-%02d" "$S")
CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz
if [ -f "$CACHE" ]; then echo "cache exists, skip $CACHE"; exit 0; fi
echo "=== preprocess-only (fir): ds004505 $st -> $CACHE ==="
python ../comparator/implementation_perf.py \
    --dataset ds004505 --subject "$S" --input-level bids \
    --n-components 64 --resample-sfreq 250 --seeds 0 \
    --cached-input "$CACHE" --preprocess-only || echo "PREPROCESS_FAIL $st"
echo "PREP_DONE $st"
