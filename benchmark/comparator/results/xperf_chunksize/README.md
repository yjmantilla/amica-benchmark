# Chunk-size cross-implementation study (ds004505)

Fit time, **convergence**, and peak memory for each Python AMICA implementation as a function of its
batch/chunk-size setting, on GPU and CPU, on a real EEG dataset (ds004505). The rendered report is the
deliverable; everything else here regenerates or backs it.

## Files
- `xperf_chunk_report.html` — the report (content fragment; the standalone version opens in a browser).
- `xperf_chunk_report_standalone.html` — full HTML document.
- `gen_report.py` — generator + the single source of truth for the plotted medians (cross-checked
  against `raw/`). Re-run to regenerate the HTML + `chunk_sweep_data.csv`.
- `chunk_sweep_data.csv` — tidy medians/IQR/convergence, regenerated from `gen_report.py`.
- `raw/chunk_{gpu3000,gpumem,cpu1000}_{summary,percell}.csv` — **authoritative raw aggregate**, computed
  straight from the per-cell result JSONs (one row per cell in `*_percell.csv`; per-(impl,chunk)
  median/IQR/ll_final/n_iter/n_samples in `*_summary.csv`).
- `NOTES_measurement.md` — caveats: iteration-budget ≠ convergence, CPU contention, NVML vs allocator,
  the two jamica keys.

## What this answers
1. **Each implementation's batch/chunk knob is a real dial for fit *time*** — up to ~25× within one
   implementation (amica-python, GPU; others 8–17×). For the torch implementations it also moves peak
   VRAM (~2.7–3.6× NVML); jamica's GPU memory is flat in the median (~5.4 GiB) on its chunked path
   (per-subject 3.4–5.4 GiB, rising to 5.4–7.4 at 262K for the longest recordings).
2. **Large chunks are fastest on both devices** (contention-free) — on the GPU and on whole CPU nodes the
   largest chunk wins for the Python/JAX impls; the earlier "small/mid wins on CPU" flip was a contention
   artifact. (The single-threaded Fortran reference is roughly flat.)
3. **The GPU fits are iteration-matched (early-stops disabled → all run the full 3000 iterations),** so
   GPU wall time is directly per-iteration-comparable (s/iter × 3000 = wall). Final log-likelihoods sit in
   a tight but non-identical band (three within ~0.001 nats; pAMICA ~0.011 nats lower — a genuine
   convergence-quality gap at matched iterations, still slowly improving at 3000, beyond which was not
   tested). This is not proof of numerically equivalent decompositions (no component matching this pass).
   **The CPU section is the whole-node exclusive, iteration-matched (250-iter) Narval run** — clean
   absolutes, 25 subjects, all 5 impls incl Fortran. It differs from the GPU only in iteration budget, so
   don't compare GPU and CPU seconds.

## Measurement corrections baked into this version
- **NVML is the headline VRAM** (framework-neutral whole-GPU peak). Per-framework allocator counters
  (JAX `peak_bytes_in_use`, torch `max_memory_allocated`) understate the footprint ~1.2–3.3× and are
  not comparable across frameworks.
- **jamica is measured on its chunked path** (`amica_python_jax_chunked`). Its full-batch key
  (`amica_python_jax`, `chunk_size=None`) is a separate program: ~13.4 GiB NVML median (per-subject up
  to ~21) at the same GPU speed, much slower + ~19.8 GiB on CPU — discussed only in the memory note.
- **`262144` is the largest tested chunk, not "full-batch"** — recordings are 785k–1.36M samples, so
  262144 is ~19–33% of the data.

## Provenance
- GPU fit @3000: `/scratch/yorguin/iter_ladder/gpu/c<chunk>_i3000/` (Trillium H100, 20–25 subj/cell).
- GPU memory (NVML+alloc): `raw/chunk_gpumem_*.csv` — iteration-independent; jamica-chunked from the
  i3000 run (logged NVML for jamica only), torch impls + jamica-fullbatch from the i1000 run.
- CPU @250 (matched): `raw/narval_nostop_i250_summary.csv` (Narval 64-core Zen2, whole-node exclusive;
  iteration-matched; per-subject median over 25 subj; all 5 impls incl Fortran).
- Builds (main): jamica `df18b5e` · amica-python `e15e158` · pyamica `a8a4d7e` · pAMICA `0c4da39` ·
  Fortran ref `665b577`. 64 PCA components.
