# Cross-implementation performance benchmark — ds004505

> **⚠️ Superseded.** The cross-implementation conclusions here (a single 100-iteration,
> default-batching run) were replaced by the chunk-size study in
> [`../xperf_chunksize/`](../xperf_chunksize/README.md), which shows the gaps are dominated
> by each library's batching knob (`chunk_size`/`block_size`/`chunk_t`/`batch_size`) and by
> how many iterations each impl actually runs under its own early-stop — not by release↔main.
> **Read that study for any speed/memory numbers**; it uses a realistic iteration budget
> (GPU 3000 / CPU 1000), NVML memory, and reports convergence (n_iter + ll_final) alongside
> time. The per-second rankings quoted in earlier versions of this file (from the 100-iter
> draft) are obsolete and have been removed to avoid confusion.

Timing + peak-memory measures for six AMICA implementations (including the new
SCCN **pAMICA**) fitting the same EEG data on CPU and GPU. This directory holds
the **measures only** — the raw ICA outputs (unmixing matrices, sources, fitted
models) are deliberately not committed.

Run: 2026-08-12 · Compute Canada **fir** · dataset **ds004505**, 25 subjects,
64 components, 100 iterations, 1 seed. 250/250 cells, 0 failures.

**Hardware.** CPU — AMD EPYC **Turin** (Zen 5) node, 192 cores / 768 GB, of which
**8 cores + 40 GB** were allocated per job (`def-kjerbi_cpu`). GPU — 1× **NVIDIA
H100 80GB HBM3** (nodes carry 4× H100), with **4 CPU cores + 32 GB** per job
(`def-kjerbi_gpu`). Software base: StdEnv/2023 · gcc 12.3 · cuda 12.6 · cudnn ·
Python 3.11.

## Files

| file | what |
|---|---|
| `xperf_measures.csv` | ground truth: one row per `device × subject × implementation` — `fit_time_s`, `peak_mem_gb`, `iters` |
| `xperf_summary.json` | derived aggregates (mean/std/min/max, peak mem) + paired significance tests |
| `make_summary.py` | regenerates `xperf_summary.json` from the CSV (`python make_summary.py`; needs scipy). Auditable — the stats reproduce from the committed measures alone. |
| `xperf_report.html` | standalone visual report (open in a browser) |

## Headline — removed (obsolete 100-iteration, default-batching run)

The per-second rankings this section used to show (a single 100-iteration run at each
library's *default* batching) are **obsolete and misleading** — the follow-up chunk-size
study showed those gaps were dominated by the batching knob and by how many iterations
each impl actually runs, not by the implementations themselves. The numbers have been
removed to avoid them being quoted. **For any speed/memory/convergence numbers, use
[`../xperf_chunksize/`](../xperf_chunksize/README.md)** (realistic iteration budget, NVML
memory, n_iter + ll_final reported). The only claim that carries over unchanged: all
implementations agreed numerically here (|W corr| 0.997–1.000 vs the Fortran reference),
i.e. they compute the same answer.

## How to reproduce the measures

These numbers were produced by the benchmark harness **with PRs #2–#6 applied**
(central pins, provenance stamping, the Fortran `fix_init` fix, aggregator, docs).
Reproduce from the repo alone:

1. **Build the environments** (login node) — one venv per implementation, all
   version-pinned in `benchmark/cc_benchmark/pins.toml`:
   ```bash
   bash benchmark/cc_benchmark/setup_competitors.sh     # pyamica + scott-huberty
   bash benchmark/cc_benchmark/setup_pamica.sh          # pAMICA v0.3.1
   bash benchmark/cc_benchmark/setup_neuromechanist.sh  # (optional snapshot)
   bash fortran/build.sh                                # amica17 from vendored source
   ```
   For **amica-JAX on GPU** you need the CUDA JAX backend, which the default
   `.venv_fir` (jax-cpu) lacks. Build a `.venv_fir_gpu`:
   `jax==0.10.2 jaxlib==0.10.2 jax_cuda12_plugin jax_cuda12_pjrt` (Alliance
   wheelhouse) + `amica@92003b4`, under `module load cuda/12.6 cudnn`. See
   `todo.md` / the `setup_fir_gpu.sh` follow-up.

2. **Run per subject** (SLURM) — the canonical comparator scripts, unchanged
   except for env-var overrides. Two things that are NOT the script defaults and
   matter for a faithful reproduction:
   - `AMICA17_BIN` → the repo-built `fortran/amica17` (the stale default points at
     a pre-built binary with a different sha; the sha gate rejects it).
   - `JAX_COMPILATION_CACHE_DIR` → a **per-subject** scratch dir. amica caches its
     JAX compilation under `~/.cache/amica/jax_cache` by default; a shared dir
     **races** under concurrent jobs (intermittent `JaxRuntimeError`) and warms the
     cache inconsistently across subjects, which skews `fit_time`. A per-subject
     dir makes every run **cold-consistent** (each pays its own compile, like the
     torch impls pay first-iteration warmup).
   ```bash
   cd benchmark/cc_benchmark
   for s in $(seq 1 25); do st=$(printf sub-%02d $s)
     # CPU: all six impls incl Fortran
     sbatch --export=ALL,AMICA_MEM_SUBJECT=$s,AMICA_MEM_DATASET=ds004505,\
   AMICA_MEM_NCOMP=64,AMICA_MEM_ITER=100,AMICA_MEM_TAG=$st,\
   AMICA17_BIN=$PWD/../../fortran/amica17,\
   JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache/cpu_$st \
       submit_mem_compare.sh
     # GPU: amica (via .venv_fir_gpu) + torch impls
     sbatch --export=ALL,AMICA_MEM_SUBJECT=$s,AMICA_MEM_DATASET=ds004505,\
   AMICA_MEM_NCOMP=64,AMICA_MEM_ITER=100,AMICA_MEM_GPU_TAG=$st,\
   AMICA_PYTHON_VENV=$PWD/../../.venv_fir_gpu/bin/python,\
   JAX_COMPILATION_CACHE_DIR=/scratch/$USER/jax_cache/gpu_$st \
       submit_mem_gpu.sh
   done
   ```

3. **Collect** — each job writes `implementation_perf_<subject>.json` under its
   results dir; flatten `fit_time_s` + `peak_gb` per `device × subject ×
   implementation` into `xperf_measures.csv`, then `python make_summary.py`.

## Provenance & caveats

- Implementations: `amica @92003b4` (snesmaeili/amica, branch `perf/cpu-profiling`) ·
  `pAMICA v0.3.1 (0e6b7f5)` · `scott amica-python 0.1.1 (cad98a6c)` ·
  `pyamica (a8a4d7e0)` · `Fortran amica17` built from vendored source
  (sha `665b5771…`, parity-validated ΔLL 7e-8). Torch 2.12/2.13, numpy 2.4.2,
  jax 0.10.2 — all pinned in `pins.toml`.
- **Note on `pAMICA` / `neuromechanist`:** the current SCCN implementation is
  **`sccn/pAMICA`** (package `pamica`, the `pamica_torch` cell). The benchmark's
  separate `neuromechanist_numpy` cell (opt-in, not in this run) is **not a
  distinct project** — it is a *historical NumPy snapshot of the same
  `sccn/pAMICA` repo* at commit `526aa32` (formerly hosted at
  `neuromechanist/pyAMICA` before the transfer to SCCN). So "neuromechanist" and
  "pAMICA" are two points in one SCCN codebase's history, not two competitors.
- **Single seed, 100 iterations, 25 subjects.** Enough to separate pAMICA from the
  field by orders of magnitude; the amica-vs-scott GPU tie needs multiple seeds to
  resolve either way.
- `fit_time_s` is the wall time of `fit()` **including** first-iteration/JIT
  compilation, measured cold and consistently for every implementation.
- Fortran is **CPU-only** here (numerical reference, not a GPU contender).
- The pAMICA-is-slow finding (barely accelerates CPU→GPU, ~2.5× vs amica's ~23×) is
  inferred from the CPU↔GPU scaling, **not** a kernel-level profile.
