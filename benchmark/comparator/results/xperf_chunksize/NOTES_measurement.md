# Measurement note — AMICA chunk-sweep campaign

> **Update (2026-08 — corrected campaign supersedes the numbers below).** The published report now
> uses a **realistic iteration budget** (GPU @3000, CPU @1000, not 100), **NVML** whole-GPU memory as
> the headline VRAM, and the **corrected jamica chunked path**. Two measurement bugs from the earlier
> 100-iter draft were found and fixed — see *The two jamica keys* and *NVML vs the allocator counters*
> below. The §"node contention" / best-of-5 / 100-iter material further down is the **origin story**,
> retained for provenance; where it conflicts with the two corrected sections, the corrected sections win.

## The two jamica keys (the "chunk-invariant" artifact)
The orchestrator exposes jamica under **two keys**:
- `amica_python_jax` — the **full-batch** path. It **ignores `--amica-chunk-size`** (materialises the
  full-width arrays regardless).
- `amica_python_jax_chunked` — the **chunked** path. It **applies** `--amica-chunk-size`.

The early chunk/memory sweeps drove jamica through `amica_python_jax`, so every "chunk" ran the same
full-batch program — making jamica look **falsely chunk-invariant** (flat time, a fixed ~2 GB-allocator
memory point). That was a harness-key artifact, **not** a property of jamica. All jamica chunk numbers
in the corrected report come from **`amica_python_jax_chunked`**. Competitor keys
(`scott_huberty_torch` / `pyamica_torch` / `pamica_torch` / `fortran_amica17`) were always correct —
only jamica had the two-key trap. Corrected result: **jamica is a normal, device-dependent
time/memory dial** like the others (GPU: big chunk faster, 763 s→62 s @3000; CPU: small chunk faster
*and* leaner, 1982 s/2.2 GB @1024 → 2731 s/7.0 GB @full — a device flip).

## NVML vs the allocator counters (the ~2× memory gap)
Peak-VRAM was reported three inconsistent ways in the 100-iter draft because each framework's
**allocator counter measures only its own live-tensor bytes** — JAX `peak_bytes_in_use`, torch
`max_memory_allocated` — omitting the CUDA/cuDNN context and pool the driver actually holds, and the
two frameworks count differently. They **understate the real footprint ~2×** and are **not comparable
across implementations**. The corrected headline is **NVML whole-GPU `used`** on a dedicated GPU
(`AMICA_NVML_CROSSCHECK=1`), which is framework-neutral and reflects what would actually fit on a card.
Example: jamica chunked ≈ **1.6 GB allocator vs ≈ 5.4 GB NVML**. A per-framework allocator bug (a
`bytes_in_use` fallback) had additionally produced a spurious 2.19 / 5.77 / 8.14 GB spread for jamica;
fixed by requiring `peak_bytes_in_use` + `jax.block_until_ready`.

**jamica memory is two-level.** On the chunked path jamica sits ≈ **5.4 GB NVML** across chunk sizes
(a chunk-independent full-width array dominates the peak until the block buffer overtakes it only at
262144). Its **full-batch path** (`chunk_size=None`, the *other* key) materialises the full-width
arrays for ≈ **8.2 GB allocator / 11.4 GB NVML** — at **no speed benefit** over chunked-at-full. So a
wrapper should always pass a chunk and never leave jamica on the full-batch path; a ~5 GB chunked
footprint fits the 8–12 GB cards many users have, the ~11 GB full-batch path may not.

## Corrected CPU campaign (contention, @1000, median-over-reps)
The corrected CPU sweep ran at **1000 iterations** with **5 repetitions per cell as separate array
tasks** and a background node-contention sampler. The cluster was **busy throughout** — the
quiet-window filter (`node_busy_mean ≤ 25`) found essentially **no clean reps** — so we report the
**median over 5 subj × 5 reps** and treat absolute CPU seconds as **contention-inflated**. (This
replaces the earlier "best-of-5" statistic; both are honest handling of the same DRAM-bandwidth
contention documented below, just at the realistic budget.) `pyamica@1024` (~767 eager blocks/iter)
exceeds the 12 h wall and is absent; nothing else OOMed at this budget. Trust the **curve shapes,
per-device optima, and NVML memory**; lean on CPU *ordering*, not exact CPU seconds.

---

# (Historical) node contention in the atomic 100-iter CPU sweep

**Status:** origin-story analysis of the earlier 100-iter draft (superseded by the corrected campaign
above, which it motivated). The *ordering* and *optima* it found are trustworthy; its *absolute* CPU
times carry a contention bias (see estimate). A clean rerun with `--exclusive` remains future work.

## The design tradeoff that causes it
The real-data sweep fans out **atomic `(impl, chunk)` cells** (one job fits one implementation at
one chunk size, loading a cached PCA projection). This buys **failure isolation** (a hang/OOM/diverge
wastes only that cell — e.g. Fortran diverging at `block=1024` over 100 iters) and **wall-clock
parallelism** (the slow scott@1024 cell no longer blocks amica@1024).

The cost: with 5 subjects × 5 impls × 5 chunks there are ~125 cells eligible to run at once, versus
~25 for a bundled (subject×chunk, impls-sequential) design. ~5× more concurrent jobs → the scheduler
packs more per node → contention. So the noise is **amplified by the atomic split**, though the root
cause (below) is present in any non-exclusive design.

## Root cause: shared DRAM bandwidth, not file I/O
AMICA fits are **memory-bandwidth-bound**: each iteration streams the whole ~400 MB data matrix
(64 × 785k × f64) through the cores several times with low arithmetic intensity (few FLOPs per byte).
Throughput is gated by RAM→CPU bandwidth, not FLOPs. Co-located cells hammer the node's shared memory
controllers and starve each other; L3-cache eviction compounds it (ironic, since `chunk_size` is
about keeping a block *in* cache). It is **not** filesystem contention — the cached `.npz` is read
once at startup (~seconds); the multi-minute fit is pure in-RAM compute. Two other node-sharing
effects can contribute the same "loaded node is slower" symptom and are not separated here: **core
oversubscription** and **turbo/frequency throttling**.

## Noise estimate (from the GPU 25-subject spread)
Per-cell fit time is **heavy-tailed**. Median vs. the least-contended run (min) and the worst outlier
(max), representative cells:

| cell            | median | min  | max  | median vs min | max vs median |
|-----------------|-------:|-----:|-----:|--------------:|--------------:|
| amica @16384    |  6.8s  |  6.0 | 43.0 |        +13%   |      6.3×      |
| amica @full     |  5.2s  |  4.1 | 30.5 |        +27%   |      5.9×      |
| amica @1024     | 29.2s  | 20.8 | 34.7 |        +40%   |      1.2×      |
| scott @16384    | 15.4s  | 13.3 | 85.7 |        +16%   |      5.6×      |
| pAMICA @1024    |139.6s  | 99.2 |170.4 |        +41%   |      1.2×      |

- The reported **median carries ≈ +15–40%** over the cleanest observation.
- **Individual cells spike 2–6×** (right tail = a cell that shared a node).
- The **median is robust** (the tail doesn't move it much at n=25) and the **relative ordering is
  unaffected** — every cell contends equally, so who-beats-whom and each impl's optimum are reliable.
- **CPU is expected to be worse than the GPU numbers above**, because CPU AMICA is *directly*
  DRAM-bandwidth-bound whereas the GPU's compute is not (the GPU tail here is mostly XLA-compile
  under CPU contention at small chunks). To be quantified from the CPU cell spread.

## Fix (future work)
Keep the atomic split (isolation) and add **`--exclusive` per cell** — each cell owns a whole node,
so it gets the full memory bandwidth with no neighbours. This gives clean absolute timings *and*
retains failure isolation. Cost: concurrency becomes node-limited (fewer cells run at once → slower
campaign), which is the correct price for a benchmark. A controlled-concurrency job array (`%N`
limit) is a cheaper middle ground. The GPU side was already close to clean because each GPU cell
owned its GPU (`--gpus-per-node=1`).

## Bottom line for readers of the curves
Trust the **shapes, the ordering, and each implementation's optimum**. Treat **absolute CPU
fit-times as upper bounds** carrying a ~tens-of-percent contention inflation until an `--exclusive`
rerun replaces them.

## Outcome of the throttled rerun (job array %4, 5 subjects)

Throttling helped only **modestly**: the per-cell **median stayed non-monotonic** (e.g. pyamica@4096
= 2345 s next to @16384 = 970 s), because (a) `%4` still co-locates some cells on `bycore` nodes, and
(b) the 5 subjects differ in length, so a cross-subject median mixes data sizes. The **min across
subjects** (best observed ≈ least-contended) *does* recover a clean chunk trend and is what the report
plots. Clean finding it exposes: **CPU optima are at small/mid chunks** (jamica 1024, scott/Fortran
~4096, pyamica 16384) — the opposite of the GPU (full-batch), a cache effect. jamica is fastest on CPU
too (~155 s best). pyamica@1024 exceeds the 1 h runner timeout; scott full-batch OOMs.

For truly clean CPU *absolutes* (not just the trend) the remaining lever is `--exclusive`/`bynode`
allocation — rejected here as fair-share-hostile (reserves a 192-core node for an ~8-core job that only
uses ~4). The best-of-5 (least-contended) figure is the honest compromise — the tightest of these
approximate, contention-inflated absolutes. Consistent with the "treat CPU absolutes as upper bounds"
caveat above: all CPU seconds here are contention-inflated; best-of-5 is simply the tightest such
estimate, so we lean on the ordering and optima rather than the exact seconds.
