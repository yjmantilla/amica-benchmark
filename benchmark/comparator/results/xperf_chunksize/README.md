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
- `raw/nostop_{gpu3000,gpumem,gpumem_decomp,ladder_i*}_summary.csv` (GPU, iteration-matched),
  `raw/nostop_gpu_ext_summary.csv` (GPU large-chunk / full-batch extension: 512K/1M/full, fit + all three
  memory counters incl. reserved, 1M restricted to the 20 subjects >1,048,576), `raw/nostop_gpu_pool_summary.csv`
  (jamica's JAX XLA pool — the reserved-equivalent shown for jamica in the reserved chart / full-batch table) and
  `raw/narval_nostop_i250_summary.csv` (CPU, whole-node exclusive, iteration-matched) — **the
  authoritative raw aggregates** the report is built from. The older `raw/chunk_{gpu3000,gpumem,cpu1000}_*`
  are the earlier (early-stop / contended) runs, retained for provenance only.
- `NOTES_measurement.md` — caveats: iteration-budget ≠ convergence, GPU vs CPU budgets differ, NVML vs
  allocator, the two jamica keys.

## What this answers
1. **Each implementation's batch/chunk knob is a real dial for fit *time*** — up to ~25× within one
   implementation over 1K–262K (amica-python, GPU; others 8–16×, and larger still out to full-batch). For the torch implementations it also moves peak
   VRAM (~2.7–3.6× NVML); jamica's GPU memory is flat in the median (~5.4 GiB) on its chunked path
   (per-subject 3.4–5.4 GiB, rising to 5.4–7.4 at 262K for the longest recordings).
2. **Small chunks win on neither device; the CPU optimum is implementation-specific.** On the GPU large
   chunks win; on whole (contention-free) CPU nodes jamica and amica-python minimize at the largest chunk,
   but pAMICA and pyamica minimize at a mid chunk (16K) — so "bigger is always better" is wrong on CPU.
   The earlier "small/mid wins on CPU" flip did not replicate under whole-node isolation (contention a
   likely confound). (The single-threaded Fortran reference is roughly flat, best at the smallest chunk.)
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
- **`262144` is the largest chunk of the core sweep** — recordings are 785k–1.36M samples, so 262144 is
  ~19–33% of the data. The GPU extension adds 512K, 1M and a full-batch pass on top (CPU stops at 262K).

## Provenance
- GPU fit @3000 (matched): `raw/nostop_gpu3000_summary.csv` (Trillium H100, iteration-matched, 25 subj/cell).
- GPU convergence ladder: `raw/nostop_ladder_i{100,250,500,1000,2000,3000}_summary.csv` (chunk 65536, 25 subj).
- GPU memory: `raw/nostop_gpumem_summary.csv` + `raw/nostop_gpumem_decomp.csv` — iteration-independent;
  full-batch key from the earlier `raw/chunk_gpumem_*.csv` memory run.
- CPU @250 (matched): `raw/narval_nostop_i250_summary.csv` (Narval 64-core Zen2, whole-node exclusive;
  iteration-matched; per-subject median over 25 subj; all 5 impls incl Fortran).
- Builds (main): jamica `df18b5e` · amica-python `e15e158` · pyamica `a8a4d7e` · pAMICA `0c4da39` ·
  Fortran ref `665b577`. 64 PCA components.
