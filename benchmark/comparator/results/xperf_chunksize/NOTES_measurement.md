# Measurement note — AMICA chunk-sweep campaign

> **Update (2026-08 — corrected campaign supersedes the numbers below).** The published report now
> uses a **realistic iteration budget** (GPU @3000 matched, CPU @250 matched whole-node, not 100), **NVML** whole-GPU memory as
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
full-batch program — making jamica look **falsely chunk-invariant** (flat time, a fixed ~2 GiB-allocator
memory point). That was a harness-key artifact, **not** a property of jamica. All jamica chunk numbers
in the corrected report come from **`amica_python_jax_chunked`**. Competitor keys
(`scott_huberty_torch` / `pyamica_torch` / `pamica_torch` / `fortran_amica17`) were always correct —
only jamica had the two-key trap. Corrected result: **jamica is a normal, device-dependent
time/memory dial** like the others (GPU: big chunk faster, 775 s→61 s @3000 matched; on the whole-node CPU
re-run big chunk is fastest for jamica too — 1269 s→753 s @250, see the CPU-campaign section below — the
old "small chunk wins on CPU" reading was from the contended fir run and did not replicate).

## NVML vs the allocator counters (the ~1.2–3.3× memory gap)
Peak-VRAM was reported three inconsistent ways in the 100-iter draft because each framework's
**allocator counter measures only its own live-tensor bytes** — JAX `peak_bytes_in_use`, torch
`max_memory_allocated` — omitting the CUDA/cuDNN context and pool the driver actually holds, and the
two frameworks count differently. They **understate the real footprint** and are **not comparable
across implementations**. The corrected headline is **NVML whole-GPU `used`** on a dedicated GPU
(`AMICA_NVML_CROSSCHECK=1`), which is framework-neutral and reflects what would actually fit on a card.
Across all 25 measured impl×chunk pairs (allocator + NVML both in `raw/chunk_gpumem_summary.csv`) the
gap is **~1.2–3.3×**: largest for jamica at small/mid chunks (chunked ≈ 1.94 GiB allocator vs 5.37 GiB
NVML = 2.8×; jamica@65536 = 3.3×), smallest for the torch impls at the largest chunk (pyamica@262144 ≈
9.0 vs 10.9 GiB = 1.2×), because the gap shrinks as live tensors grow to dominate the fixed CUDA context.
A single average is not claimed. A per-framework allocator bug (a
`bytes_in_use` fallback) had additionally produced a spurious 2.19 / 5.77 / 8.14 GiB spread for jamica;
fixed by requiring `peak_bytes_in_use` + `jax.block_until_ready`. All memory values are GiB
(bytes / 1024³). NVML is a 50 ms poll of whole-GPU `used`; JAX runs with pre-allocation off and torch
with its caching allocator on, so NVML is a neutral meter over slightly different allocator protocols.

**jamica memory is two-level.** On the chunked path jamica sits ≈ **5.37 GiB NVML** across chunk sizes
(per-subject ≈ 3.4–5.4 GiB below 262K, rising to 5.4–7.4 GiB at 262K for the longest recordings —
flat across *chunk* in the median, not a single constant; a chunk-independent full-width array
dominates the peak). Its **full-batch path** (`chunk_size=None`, the *other* key) materialises the
full-width arrays for ≈ **8.22 GiB allocator / 13.37 GiB NVML median (per-subject up to ~21.4)** — at
**no GPU speed benefit** over chunked-at-full (≈ 0.020 vs 0.021 s/iter, same i3000 run, measured). On CPU chunking helps
*both* axes (≈ 1985 s / 2.2 GiB chunked vs ≈ 4300 s / 19.8 GiB full-batch). So under these tested
conditions a wrapper should pass a chunk. In fairness the flip side: jamica's ~5.4 GiB chunked *floor*
is higher than the torch impls' small-chunk footprint (~1.8–3.1 GiB NVML), so on a small card the torch
impls at a small chunk fit where jamica may not; the "fits an 8–12 GiB card" reading is an extrapolation
(H100 NVML, JAX pre-allocation off), not measured on such a card.

## CPU campaign — whole-node exclusive, iteration-matched @250 (Narval)
The CPU sweep was re-run on **Narval whole nodes (64-core Zen2), one fit per node (exclusive)** so there is
**no memory-bandwidth contention**, **iteration-matched to 250** (early-stops disabled), **25 subjects**,
all five implementations including the single-threaded Fortran reference
(`raw/narval_nostop_i250_summary.csv`; per-subject median). Because each fit owned its node, the
per-subject bands are tight and the **absolute seconds and per-cell optima are trustworthy** — this
supersedes the earlier contended, unequal-coverage fir run (all 25 subjects). Headline: **jamica is
fastest on CPU too** (~753 s at the largest chunk), and **large chunks are fastest on CPU as well**
(jamica 1269→753 s, amica-python 5027→1053 s from 1K→262K) — the same
direction as the GPU, **overturning** the earlier "small/mid wins on CPU" flip, which was a contention
artifact. Fortran (single-threaded) is a reference footprint, not a fair-thread comparison on a whole node.
GPU and CPU iteration budgets differ (3000 vs 250) — do not compare GPU to CPU seconds. `pyamica@1024`,
which timed out on the contended fir run, completes here (~3099 s).

## GPU is iteration-matched (and so is the CPU re-run)
The GPU fit-time comparison was re-run **iteration-matched**: every implementation's early-stops were
disabled, so all run the full 3000 iterations (`raw/nostop_gpu3000_summary.csv`, `n_iter_min = median =
max = 3000`, 25 subj/cell). GPU wall time is therefore directly per-iteration-comparable
(s/iter × 3000 = wall). On GPU at the largest chunk (262144):

| impl | wall time | s/iter | iters run | ll_final (median) |
|------|----------:|-------:|-----------|------------------:|
| jamica  |  61 s | 0.0203 | 3000 | −1.1005 |
| amica-python | 230 s | 0.0768 | 3000 | −1.1002 |
| pAMICA | 262 s | 0.0875 | 3000 | −1.1107 (lowest) |
| pyamica | 296 s | 0.0986 | 3000 | −1.0995 (highest) |

At matched iterations jamica is fastest per iteration by ~4–5×. Three impls land within ~0.001 nats
(−1.0995 to −1.1005); pAMICA is ~0.011 nats lower and — per the measured iteration ladder
(`raw/nostop_ladder_i*_summary.csv`) — is still slowly improving at 3000 (~0.0016/1000 iters) but far
short of the others: a genuine convergence-quality gap at matched iterations, not an early-stop artifact
(beyond 3000 was not tested). These are *reported* LLs; component matching against the Fortran reference
was not run this pass, so this is not proof of numerically equivalent decompositions.

The **CPU** section is now the whole-node exclusive, iteration-matched (250-iter) Narval run — clean
absolutes across 25 subjects and all five implementations (see the CPU section above). It differs from the
GPU only in iteration budget (250 vs 3000), so GPU and CPU seconds are not directly comparable.

## "Largest tested chunk," not full-batch
`FULL = 262144` in the generator is the **largest chunk tested**, not a single full-batch pass:
per-subject sample counts are **785,328–1,364,633** (from the result JSONs), so 262144 is ~19–33% of a
recording. The competitors at 262144 still process 3–6 blocks per iteration. Only jamica's full-batch
*key* (`chunk_size=None`) is a genuine single-pass program, and it is reported separately (memory note).

---

# (Historical) node contention in the atomic 100-iter CPU sweep

**Status:** origin-story analysis of the earlier 100-iter draft (superseded by the corrected campaign
above, which it motivated). The *ordering* and *optima* it found are trustworthy; its *absolute* CPU
times carry a contention bias (see estimate). A clean rerun with `--exclusive` was subsequently done on
Narval (whole nodes, one fit each) — see the CPU-campaign section above — and supersedes this.

## The design tradeoff that causes it
The real-data sweep fans out **atomic `(impl, chunk)` cells** (one job fits one implementation at
one chunk size, loading a cached PCA projection). This buys **failure isolation** (a hang/OOM/diverge
wastes only that cell — e.g. Fortran diverging at `block=1024` over 100 iters) and **wall-clock
parallelism** (the slow amica-python@1024 cell no longer blocks amica@1024).

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
| amica-python @16384    | 15.4s  | 13.3 | 85.7 |        +16%   |      5.6×      |
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

## Bottom line for readers of the curves (of the 100-iter draft — superseded)
*This was the 100-iter draft's bottom line; the corrected campaign at the top does NOT name exact
per-cell optima (unresolved under contention + unequal coverage) and does not use best-of-5.* That draft
said: trust the **shapes and ordering** (and, it then claimed, each implementation's optimum). Treat
**absolute CPU fit-times as upper bounds** carrying a ~tens-of-percent contention inflation until an
`--exclusive`
rerun replaces them.

## Outcome of the throttled rerun (job array %4, 5 subjects)

Throttling helped only **modestly**: the per-cell **median stayed non-monotonic** (e.g. pyamica@4096
= 2345 s next to @16384 = 970 s), because (a) `%4` still co-locates some cells on `bycore` nodes, and
(b) the 5 subjects differ in length, so a cross-subject median mixes data sizes. In *that historical
100-iter draft* the **min across subjects** (best observed ≈ least-contended) recovered a clean chunk
trend and was what that draft plotted. (The current corrected report does NOT use best-of-5 — it plots
the median over subjects × reps at the realistic budget, with p25–p75 bands; see the corrected campaign
section at the top.) NOTE: the "small/mid wins on CPU" direction claimed here **did NOT survive** the
whole-node exclusive re-run (see the CPU-campaign section) — small chunks are best for none of the Python
impls; it was a contention artifact of the shared-cluster runs. The exact 100-iter optima quoted here
(jamica 1024, amica-python/Fortran ~4096, pyamica 16384) and the "~155 s best" figure are that old draft's;
the corrected @1000 numbers differ and carry wide contention bands. pyamica@1024 exceeded the runner
wall in both campaigns; amica-python full-batch did not OOM in the corrected GPU/CPU runs.

For truly clean CPU *absolutes* (not just the trend) the remaining lever is `--exclusive`/`bynode`
allocation — deprioritised at the time as fair-share-hostile, though later used on Narval for the clean CPU re-run above (reserves a full node for an ~8-core job that only
uses ~4). The best-of-5 (least-contended) figure is the honest compromise — the tightest of these
approximate, contention-inflated absolutes. Consistent with the "treat CPU absolutes as upper bounds"
caveat above: all CPU seconds here are contention-inflated; best-of-5 is simply the tightest such
estimate, so we lean on the ordering and optima rather than the exact seconds.
