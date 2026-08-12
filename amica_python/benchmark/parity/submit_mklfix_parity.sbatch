#!/bin/bash
#SBATCH --job-name=amica17_mklfix
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Fortran <-> amica-python parity on synth6 (6ch x 50k, m=3, Newton, 1000 iter,
# shared sphere+mean+fix_init -> identical start state isolates pure algorithmic
# agreement). Rebuilds the vendored amica17 from source (fortran/, incl. fix#3)
# and compares its W / LL to amica-python. Reference numbers (final ΔLL ~7e-8,
# W matched |r| 0.9999999999) are in fortran/README.md; re-validated 2026-08-12
# on the fix#3 build (sha 665b5771): final ΔLL 7.49e-8, W matched |r| 0.99999999993.
#
# Submit FROM THIS DIRECTORY (amica_python/benchmark/parity/):
#   sbatch submit_mklfix_parity.sbatch
# Override the scratch work dir with AMICA_PARITY_WORK if desired.
set -euo pipefail

# Repo root: this script lives at amica_python/benchmark/parity/ — three levels
# below the repo root. SLURM_SUBMIT_DIR is where `sbatch` was invoked (falls back
# to the script's own dir for a bare `bash` run).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$SUBMIT_DIR/../../.." && pwd)"
BUILD="$REPO/fortran"                                   # vendored source + build.sh
WORK="${AMICA_PARITY_WORK:-/scratch/$USER/fortran_parity/mklfix50k}"
ENVSH="$REPO/benchmark/cc_benchmark/fir_env.sh"
mkdir -p "$(dirname "$WORK")"

echo "############ PHASE 0: prep (numpy / amica-python env) ############"
module purge
source "$ENVSH"
# amica_python is a vendored dir at the repo root, NOT a pip-installed package in
# .venv_fir (this is amica-benchmark, not Sina's amica-python repo whose editable
# install put it on the path), so add the repo root for the `python -m` calls.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python -c "import sys, numpy; print('py', sys.executable, 'numpy', numpy.__version__)"
mkdir -p "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity prep \
    --workdir "$WORK" --dataset synth6 --n-channels 6 --n-samples 50000 \
    --m 3 --do-newton 1 --max-iter 1000 --seed 0 \
    --container-data "$WORK/data" --container-out "$WORK/out"

echo "############ PHASE 1: build amica17 (-DMKL, patched, fix#3) + run (gcc/openmpi env) ############"
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
export OMP_NUM_THREADS=1
cd "$BUILD"
bash build.sh
file "$BUILD/amica17"; sha256sum "$BUILD/amica17"
cd "$WORK"
rm -rf "$WORK/out"
mpirun -np 1 "$BUILD/amica17" "$WORK/amica.param"
echo "----- Fortran out.txt (last 3 lines) -----"
tail -3 "$WORK/out/out.txt" || true

echo "############ PHASE 2: compare to amica-python (amica-python env) ############"
module purge
source "$ENVSH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity compare --workdir "$WORK"
echo "############ DONE ############"
