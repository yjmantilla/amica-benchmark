# iter_ladder campaign state / resume note (2026-08-14)

Durable hand-off so a fresh session (or the user) can finish without the in-session cron.
Access is DIRECT ssh (`ssh -o BatchMode=yes <host> '<cmd>'`) — `cluster-run` is DEPRECATED.

## THE key bug (do not re-introduce)
jamica has TWO orchestrator keys: `amica_python_jax` = **full-batch** (ignores `--amica-chunk-size`)
and `amica_python_jax_chunked` = **applies the chunk**. Early campaigns used the full-batch key for
jamica, so jamica looked "chunk-invariant" — an ARTIFACT. All jamica chunk/memory numbers must come
from **`amica_python_jax_chunked`**. Competitor keys (scott_huberty_torch / pyamica_torch /
pamica_torch / fortran_amica17) were always correct.

## Corrected jamica results (DONE, reported)
GPU (Trillium `770290`, chunked, @3000, NVML): time 763s@1024 -> 62s@full (big chunk faster);
memory chunked ~1.6GB alloc / ~5.4GB NVML vs true full-batch (`chunk=None`) ~8.2 / ~11.4 GB.
Flat memory 1024-65536 because a chunk-independent ~0.55GB full-width array dominates until the
block buffer overtakes it at 262144; small chunks slow due to millions of tiny allocs (6.5M @1024).
CPU (fir `54723173`, chunked, @1000): small chunk faster AND leaner (1982s/2.2GB @1024 ->
2731s/7.0GB @full) -- device flip vs GPU. So jamica = a real device-dependent time/memory dial.

## Still in flight (competitor columns; jamica cols there are full-batch -> ignore)
- CPU chunk-sweep fit-time: fir array **54696224** (sentinel reported_cpuchunk)
- CPU memory: fir array **54698790** (sentinel reported_cpumem; pyamica@1024-type cells time out @12h)
GPU memory competitor (Trillium `768693`) already reported.

## Resume: aggregate the two remaining competitor jobs
```bash
# fit-time vs chunk (competitors) — on fir, cpu root:
ssh -o BatchMode=yes fir 'cd /scratch/yorguin/amica-benchmark-repro/benchmark/cc_benchmark; \
  python3 iter_ladder/aggregate_ladder.py --gpu-root /nonexistent \
  --cpu-root /scratch/yorguin/iter_ladder/cpu --out /scratch/yorguin/iter_ladder/cpu_data.csv \
  --summary-out /scratch/yorguin/iter_ladder/cpu_summary.csv'
# then extract, per (impl,chunk) at max_iter==1000: median fit_time_s (competitors),
# and cgroup_peak_gb/peak_rss_gb from c*_i1000_r1/ (job B). jamica -> use the _chunked results above.
```
Result JSONs: `/scratch/yorguin/iter_ladder/{gpu,cpu}/c<chunk>_i<iter>[_r<rep>]/<impl>_sub-NN_seed0_result.json`.
Fields added this session: `nvml_peak_vram_gb` (framework-neutral headline VRAM), `peak_vram_reserved_gb`
(torch), `cgroup_peak_gb`, `vram_stats` (raw jax memory_stats). NVML enabled via AMICA_NVML_CROSSCHECK=1.

## Next (report work)
1. DONE (2026-08-15, commit dfc347c): xperf_chunksize report rewritten around NVML + corrected chunked
   jamica (@3000 GPU / @1000 CPU); "chunk-invariant"/"2.19GB" claims dropped; published to artifact
   5c1007ae (replaced the old "hidden variable" draft).
2. DONE (2026-08-15): two-key gotcha + NVML/allocator gap logged in NOTES_measurement.md.
3. DONE (2026-08-16): report hardened over THREE independent review panels (codex=gpt-5.6-sol /
   grok-4.6 / claude-fable-5, via ~/Projects/agent-utilities/reviews/run_reviewers.sh; rounds under
   reviews/report-{corrected-dfc347c,panel2-33c6d9e,panel3-e14611a,panel4-431515d}/). Verdicts:
   do-not-ship -> ship-with-fixes -> ship-with-fixes (numbers verified against raw/). Commits:
   33c6d9e (convergence framing: n_iter+ll_final+s/iter, budget!=convergence box), e14611a (by-subject
   CPU + coverage disclosure, GPU-mem provenance, 262K-not-full-batch relabel, fixed run_fortran.py
   key+import), 431515d (README/NOTES reconcile, raw/chunk_gpumem_*.csv NVML+alloc traceable, per-subject
   memory spread, sibling README obsolete-ranking removed). Raw evidence: raw/chunk_{gpu3000,gpumem,
   cpu1000}_*.csv aggregated from cluster JSONs (Trillium GPU / fir CPU). Round-4 = confirmatory.
   KEY corrected facts: n_samples 785k-1.36M (262144 = 19-33%, not full-batch); GPU s/iter jamica 0.021
   < scott 0.076 < pAMICA 0.089 < pyamica 0.098; early-stop heterogeneous (Fortran off, pyamica full-cap,
   jamica 2572-3000, scott 786-1654, pAMICA 151-3000); jamica NVML chunked ~5.4 (up to 7.4@262K) vs
   full-batch key ~13.4 (up to 21.4); allocator understates 1.6-2.8x.
STILL TODO: the MNE-note (#13819, artifact 62fc4258) MEMORY/chunk section needs the same NVML +
chunked-jamica + convergence correction the xperf_chunksize report now has.
3. Memory panel (fable/gpt-5.6/grok) confirmed allocator counters aren't cross-framework comparable;
   NVML is the headline. jamica-chunk-memory panel: Fable caught the key bug (dissent was right).
Reviews kept local under reviews/ (excluded from commits).
