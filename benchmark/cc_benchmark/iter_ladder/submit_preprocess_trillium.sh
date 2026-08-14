#!/bin/bash
# One-time PCA-cache build on TRILLIUM for the GPU iteration ladder. The fir caches don't exist
# on Trillium's filesystem, so build the 25 per-subject projected inputs once here; the ladder
# cells then reuse them via --cached-input (no re-preprocessing per cell). No fit, so this is a
# small CPU-only preprocess. One array task per subject: sbatch --array=1-25.
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yorguin/iter_ladder_cells/pre-%A_%a.out
#SBATCH --error=/scratch/yorguin/iter_ladder_cells/pre-%A_%a.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 scipy-stack/2026a >/dev/null 2>&1
export AMICA_SKIP_PIN_CHECK=1
export BIDS_ROOT_DS4505=/scratch/yorguin/ds004505
mkdir -p /scratch/yorguin/iter_ladder_cells /scratch/yorguin/realchunk_cache

R=/scratch/yorguin/amica-benchmark-repro
export AMICA_PYTHON_VENV=$R/.venv_fir_gpu/bin/python
export AMICA_SRC=/scratch/yorguin/amica_main_src
# activate a venv with mne/scipy for preprocessing (JAX venv carries scipy-stack)
source $R/.venv_fir_gpu/bin/activate 2>/dev/null || true

S=${SLURM_ARRAY_TASK_ID}
st=$(printf "sub-%02d" "$S")
CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz
[ -f "$CACHE" ] && { echo "cache exists, skip: $CACHE"; exit 0; }

echo "=== preprocess-only (Trillium): ds004505 $st -> $CACHE ==="
python ../comparator/implementation_perf.py \
    --dataset ds004505 --subject "$S" --input-level bids \
    --n-components 64 --resample-sfreq 250 --seeds 0 \
    --cached-input "$CACHE" --preprocess-only
echo "PREPROCESS_DONE $st"
