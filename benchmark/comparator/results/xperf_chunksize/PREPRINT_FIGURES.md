# Preprint figures — AMICA chunk-size / iteration benchmark (ds004505)

These figures reproduce every chart in the two HTML reports
(`xperf_chunk_report.html`, `xperf_chunk_tldr.html`) as publication-quality
**vector PDF + 300 dpi PNG**. They are what to drop into the preprint.

## Regenerate (no LLM needed)

```bash
python make_paper_figures.py        # needs only matplotlib + numpy
```

Outputs land in `figures/`. The script reads **only committed CSVs** — no network,
no cluster, no seaborn. If the underlying numbers ever change, refresh the data
first, then the figures:

```bash
python gen_report.py                # single source of truth: writes the CSVs below
python make_paper_figures.py        # reads them, writes figures/
```

## The figures

| file | layout | shows |
|------|--------|-------|
| `figures/fig_chunk_time_memory.{pdf,png}` | 2×2 | fit time (log) and peak memory vs chunk size, GPU (4 parallel impls) on top, CPU (5 impls incl. the reference) below. GPU memory panel carries 24/40/80 GiB reference lines. Shaded band = middle 50% of subjects. Full-batch (FB) is a star off the chunk axis. |
| `figures/fig_iterations.{pdf,png}` | 2×2 | fit time and final log-likelihood vs iteration count at a fixed chunk (65536). GPU (100–3000 iters) on top, CPU (50–500 iters) below. Higher log-likelihood is better. |
| `figures/fig_gpu_memory_decomp.{pdf,png}` | 2×2 | GPU memory vs chunk decomposed: in active use ≤ reserved (allocator pool) ≤ total (NVML), plus total÷active. Reserved is drawn only where it was measured (262K and up). |
| `figures/fig_signal_duration.{pdf,png}` | 1 panel | per-subject recording length (25 subjects) against the tested chunk sizes as a fraction of the median recording. |

## Data files (schemas)

All written by `gen_report.py`:

- **`chunk_sweep_data.csv`** — `dataset, impl, knob, chunk, value, unit, note`.
  Datasets used by the figures: `gpu_fit_s_median` / `gpu_fit_s_p25` / `gpu_fit_s_p75`,
  `gpu_vram_gib_nvml`, `cpu_fit_s_bysubj_median` / `_p25` / `_p75`, `cpu_rss_gib_median`.
- **`chunk_sweep_fullbatch.csv`** — same schema, `chunk = "fullbatch"` (kept off the chunk axis).
- **`iter_ladder_data.csv`** — `device, impl, iters, fit_s, ll, mem_gib` (fixed chunk 65536).
- **`gpu_memory_decomp_data.csv`** — `impl, knob, chunk, active_gib, reserved_gib, total_gib, ratio_total_over_active`.
- **`subject_durations.csv`** — `subject_index, n_samples`.

## Implementation identity (colours + repos, same as the reports)

| impl | label | colour | knob | repo |
|------|-------|--------|------|------|
| jamica | jamica | `#6366f1` | `chunk_size` | github.com/snesmaeili/jamica |
| amica_python | amica-python | `#e11d48` | `batch_size` | github.com/scott-huberty/amica-python |
| pamica | pAMICA | `#d97706` | `block_size` | github.com/sccn/pAMICA |
| pyamica | pyamica | `#0d9488` | `chunk_t` | github.com/DerAndereJohannes/pyamica |
| fortran | Fortran (1 thread) | `#111827` | `block_size` | github.com/sccn/amica |

## Method / caveats (keep these true in any caption)

- **Dataset:** ds004505 (real EEG), 64 components, 25 subjects, 785,328–1,364,633 samples/subject @250 Hz.
- **GPU:** NVIDIA H100, fixed **3000** iterations, early-stops disabled (iteration-matched), per-subject median.
- **CPU:** 64-core machine, one fit per machine, fixed **250** iterations, per-subject median.
- GPU and CPU use **different iteration counts**, so their seconds are **not directly comparable**.
- Fit time is wall time at a fixed iteration budget, **not** time to an equivalent solution.
- The **1M** chunk point uses only the **20** recordings longer than 1,048,576 samples; other on-axis
  points cover 25 (CPU large-chunk points cover 19–25 — see the report tables).
- **Full-batch** = one pass over the whole recording; it scales with recording length, so it is a star
  off the chunk axis, not an on-axis point. amica-python's full-batch uses a per-subject batch equal to
  each recording's length.
- The single-threaded **Fortran** build is a reference footprint, not a like-for-like comparison against
  the 64-core / GPU runs (it does not run on the GPU here).

## If you'd rather have an LLM regenerate or restyle them

Paste this, plus the five CSVs above and `make_paper_figures.py`, into any coding LLM:

> Here is a matplotlib script (`make_paper_figures.py`) and the CSVs it reads. Regenerate the four
> figures as PDF + PNG. Keep the per-implementation colours and labels exactly as defined at the top of
> the script. Do not invent or alter any numbers — plot only what is in the CSVs. Keep the caveats in
> `PREPRINT_FIGURES.md` true (GPU 3000 iters vs CPU 250 iters are not comparable; 1M uses 20 recordings;
> full-batch is off the chunk axis; Fortran is single-threaded). If I ask for a restyle (fonts, sizes,
> one-column width, combining panels), change only presentation, never the data.
