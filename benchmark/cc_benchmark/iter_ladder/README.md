# Iteration ladder — time to reach N iterations

Total fit time as a function of iteration count, per implementation, at a **fixed** chunk
(65536). One complete fit run to each `max_iter` cap, timed end to end (Sina's ladder method).
Slope of the resulting line ≈ steady per-iteration cost; intercept ≈ compile/setup.

Not the chunk sweep — here chunk is held constant and **iterations** are the swept variable.

## Design (locked)
| device | cluster | subjects | impls | chunk | caps (max_iter) |
|---|---|---|---|---|---|
| GPU (H100) | **SciNet Trillium** (`def-kjerbi`, compute) | 25 | jamica, scott-huberty, pyamica, pAMICA | 65536 | 100,250,500,1000,2000,3000 |
| CPU (`bycore`) | **fir** (`rrg-kjerbi_cpu`) | 5 | + Fortran amica17 | 65536 | 100,250,500,1000 |

GPU runs on Trillium (matches the published chunk-campaign provenance); CPU on fir. Venvs +
`amica_main_src` are already on both filesystems; the PCA caches are **not** on Trillium, so build
them there once (`submit_preprocess_trillium.sh`).

### CPU repetitions & contention monitoring
CPU cells are memory-bandwidth-bound, so a shared `bycore` node's *other tenants* distort timing.
Two mitigations, both on by default:
- **Reps as separate array tasks** (`build_manifests.py --cpu-reps 10`): each rep lands on a
  different node/time, so we can keep the least-contended reps and the spread quantifies contention.
  Separate tasks (not 10 sequential fits in one job) keep every rep under the 3 h wall.
  CPU cell count = 5 subj × 5 impl × 4 caps × **reps** (reps=10 -> 1000 cells).
- **Per-rep node sampler** (`node_monitor.sh`, backgrounded by `submit_iter_cpu.sh`): every 10 s it
  logs node loadavg, node-wide CPU-busy% (`/proc/stat`), memory used, and **co-tenant reserved
  cores** (`squeue -w <node>`, other users) to a sidecar under `.../cpu/nodemon/`. `aggregate_ladder.py`
  folds each trace into the tidy CSV (`cotenant_cores_mean/max`, `node_busy_mean`, `load1_mean`)
  next to `fit_time_s`, so timing can be regressed on contention. `aggregate_ladder.py` also emits
  `iter_ladder_summary.csv` with BOTH modes per (device,impl,cap): **use-all** (median/mean/IQR) and
  **quiet-only** (reps with co-tenant cores <= `--quiet-cotenant-max`, then min/median). The plotter
  takes `--stat {median,mean,min}` and `--filter-quiet` to draw either view. (`perf_event_paranoid=2` on fir
  blocks system-wide/uncore DRAM-bandwidth counters, so co-tenant cores + node busy% are the proxy.)

CPU is trimmed (5 subj, cap 1000) because 25-subj × 3000 on CPU is ~55 CPU-days and blows the
wall clock. Strategy is **shared nodes, high throttle (`%32`), 10 reps, sampler on** — a "statistical
exclusive": we can't reserve a quiet node fair-share-cheaply, so we oversample and keep the reps the
monitor certifies ran quiet (`--filter-quiet`). `--exclusive` was rejected as fair-share-hostile.

## Run (from `benchmark/cc_benchmark/`)
```bash
python iter_ladder/build_manifests.py --cpu-reps 10   # writes manifest_{gpu,cpu}.txt (cpu=1000 cells)

# --- GPU, on Trillium ---
sbatch --array=1-25 iter_ladder/submit_preprocess_trillium.sh            # build 25 caches (once)
MANIFEST=iter_ladder/manifest_gpu.txt sbatch --array=1-600%8 iter_ladder/submit_iter_gpu.sh

# --- CPU, on fir --- (caches from the chunk campaign are reused; add --exclusive + EXCLUSIVE=1
#     for the cleanest absolute times; raise %N to finish sooner)
MANIFEST=iter_ladder/manifest_cpu.txt sbatch --array=1-1000%32 iter_ladder/submit_iter_cpu.sh
```
Each cell runs `implementation_perf.py` at one `(subject, impl, chunk, max_iter)`, reusing the
per-subject PCA caches (`/scratch/yorguin/realchunk_cache/`), and writes a result JSON under
`/scratch/yorguin/iter_ladder/{gpu,cpu}/c65536_i<cap>/`.

## Aggregate + plot (locally, after pulling results)
```bash
python iter_ladder/aggregate_ladder.py \
    --gpu-root /scratch/yorguin/iter_ladder/gpu \
    --cpu-root /scratch/yorguin/iter_ladder/cpu \
    --out iter_ladder/iter_ladder_data.csv        # tidy: device,impl,subject,chunk,max_iter,fit_time_s
python iter_ladder/plot_time_vs_iter.py --csv iter_ladder/iter_ladder_data.csv \
    --out iter_ladder/time_vs_iter.png            # + .svg
```
If a jamica build exposes native `iteration_times`, the runner now persists it and the aggregator
also emits `*_periter.csv` (per-iteration cumulative time) for a fine-grained jamica curve.

## Notes / caveats
- **Provenance:** GPU runs on SciNet Trillium (matches the published chunk-campaign numbers); CPU on
  fir. Venvs + `amica_main_src` are replicated on both; PCA caches are built on Trillium once via
  `submit_preprocess_trillium.sh`.
- **CPU absolutes** still carry the node-contention caveat — trust the slopes/ordering, not the
  last-decimal seconds. Fortran is single-threaded (`OMP_NUM_THREADS=1`), so its CPU line is not
  per-core comparable to the 8-thread Python impls.
- Re-run ladder ≈ Σ(caps) iters/cell: GPU Σ=6850, CPU Σ=1850 per (impl, subject).
