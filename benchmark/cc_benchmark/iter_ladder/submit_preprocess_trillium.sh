#!/bin/bash
# One-time PCA-cache build on TRILLIUM for the GPU iteration ladder. Trillium is whole-node
# scheduling (no --mem/--cpus/--ntasks; a 25-task array would grab 25 whole nodes), so this is a
# SINGLE job that loops all 25 subjects on one node (no fit, just PCA projection -> ~1-2 min each).
# The ladder cells then reuse the caches via --cached-input.
#   sbatch iter_ladder/submit_preprocess_trillium.sh
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yorguin/iter_ladder_cells/pre-%j.out
#SBATCH --error=/scratch/yorguin/iter_ladder_cells/pre-%j.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 scipy-stack/2026a >/dev/null 2>&1
export AMICA_SKIP_PIN_CHECK=1
export BIDS_ROOT_DS4505=/scratch/yorguin/ds004505
mkdir -p /scratch/yorguin/iter_ladder_cells /scratch/yorguin/realchunk_cache

R=/scratch/yorguin/amica-benchmark-repro
source $R/.venv_fir_gpu/bin/activate 2>/dev/null || true

for S in $(seq 1 25); do
  st=$(printf "sub-%02d" "$S")
  CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz
  if [ -f "$CACHE" ]; then echo "cache exists, skip: $CACHE"; continue; fi
  echo "=== preprocess-only (Trillium): ds004505 $st -> $CACHE ==="
  python ../comparator/implementation_perf.py \
      --dataset ds004505 --subject "$S" --input-level bids \
      --n-components 64 --resample-sfreq 250 --seeds 0 \
      --cached-input "$CACHE" --preprocess-only || echo "PREPROCESS_FAIL $st"
done
echo "PREPROCESS_ALL_DONE"
