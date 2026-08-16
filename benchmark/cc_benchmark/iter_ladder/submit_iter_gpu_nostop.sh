#!/bin/bash
# ITERATION-MATCHED GPU re-run (Trillium H100): every implementation runs the FULL max_iter with
# all early-stops DISABLED (AMICA_DISABLE_EARLYSTOP=1), so "wall time to N iterations" is a clean,
# convergence-heterogeneity-free throughput comparison. Also captures the NVML post-init memory floor
# (nvml_post_init_gb) and full ll_history. Writes to a SEPARATE dir so the existing @3000 baseline
# (iter_ladder/gpu/c*_i3000, the post-hoc-convergence source) is NOT overwritten.
#
# Manifest lines: "SUBJECT IMPL CHUNK MAXITER" (one array task per line). Build with
# build_manifest_nostop.py. Launch (on Trillium, from benchmark/cc_benchmark/):
#   python iter_ladder/build_manifest_nostop.py <n_subjects>
#   sbatch --array=1-<N>%8 iter_ladder/submit_iter_gpu_nostop.sh iter_ladder/manifest_gpu_nostop.txt
#SBATCH --account=rrg-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yorguin/iter_ladder_cells/gpunostop-%A_%a.out
#SBATCH --error=/scratch/yorguin/iter_ladder_cells/gpunostop-%A_%a.out
# NOTE: Trillium is whole-node scheduling -- do NOT add --mem/--cpus-per-task/--ntasks (rejected).
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 scipy-stack/2026a cuda/12.6 cudnn >/dev/null 2>&1
[ -n "$CUDA_HOME" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export AMICA_NVML_CROSSCHECK=1          # framework-neutral whole-GPU VRAM + post-init floor
export AMICA_DISABLE_EARLYSTOP=1        # iteration-matched: run the full max_iter, stops off
export AMICA_SKIP_PIN_CHECK=1
export BIDS_ROOT_DS4505=/scratch/yorguin/ds004505
export XDG_CACHE_HOME=/scratch/yorguin/.cache XDG_DATA_HOME=/scratch/yorguin/.local/share
export JAX_COMPILATION_CACHE_DIR=/scratch/yorguin/.cache/amica_jax MPLCONFIGDIR=/scratch/yorguin/.cache/mpl
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$MPLCONFIGDIR"
MANIFEST="${1:-${MANIFEST:-iter_ladder/manifest_gpu_nostop.txt}}"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST (pass as arg 1 on Trillium)"; exit 1; }
mkdir -p /scratch/yorguin/iter_ladder_cells

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$line" ] && { echo "no manifest line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
read -r S IMPL C IT <<< "$line"
st=$(printf "sub-%02d" "$S")
R=/scratch/yorguin/amica-benchmark-repro
CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz

ALL="amica_python_jax amica_python_jax_chunked amica_python_numpy neuromechanist_numpy pyamica_torch scott_huberty_torch pamica_torch fortran_amica17"
SKIP=""; for k in $ALL; do [ "$k" = "$IMPL" ] || SKIP="$SKIP $k"; done

export AMICA_PYTHON_VENV=$R/.venv_fir_gpu/bin/python
export COMPETITORS_VENV=$R/.venv_competitors_main/bin/python
export PAMICA_VENV=$R/.venv_pamica_main/bin/python
export AMICA_SRC=/scratch/yorguin/amica_main_src
source "$R/.venv_fir_gpu/bin/activate" 2>/dev/null || true
export AMICA_SCOTT_BATCH=$C AMICA_PYAMICA_CHUNK=$C AMICA_PAMICA_BLOCK_SIZE=$C
export AMICA_COMPARATOR_RESULTS=/scratch/yorguin/iter_ladder/gpu_nostop   # SEPARATE from the baseline
mkdir -p "$AMICA_COMPARATOR_RESULTS"

CIN=""; [ -f "$CACHE" ] && CIN="--cached-input $CACHE"
echo "=== iter-matched GPU cell (Trillium, stops OFF): sub-$S impl=$IMPL chunk=$C max_iter=$IT ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
python ../comparator/implementation_perf.py \
    --dataset ds004505 --subject "$S" --input-level bids \
    --n-components 64 --max-iter "$IT" --seeds 0 \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size "$C" \
    $CIN --out-tag "c${C}_i${IT}" \
    --skip $SKIP
echo "ITER_MATCHED_CELL_DONE sub-$S $IMPL c$C i$IT"
