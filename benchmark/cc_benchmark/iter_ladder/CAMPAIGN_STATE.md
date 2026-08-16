# iter_ladder campaign state / resume note (2026-08-14)

Durable hand-off so a fresh session (or the user) can finish without the in-session cron.
Access is DIRECT ssh (`ssh -o BatchMode=yes <host> '<cmd>'`) — `cluster-run` is DEPRECATED.

## IN FLIGHT (2026-08-16): iteration-matched (stops-off) GPU re-run
Full 3000-iter GPU sweep with ALL early-stops DISABLED (AMICA_DISABLE_EARLYSTOP=1) + NVML bracketing
(nvml_post_init_gb) + ll_history, for the "time-to-iterations" table. Runners have env-gated disable
knobs (commit 4427fed + run_pamica fit-kwarg fix): jamica use_min_dll=False+minlrate=0; pyamica
use_min_dll=False+use_grad_norm=False+minlrate=0+min_nd=0; pAMICA fit(minlrate=0.0); amica-python
tol=-1e30. Smoke-validated (jobs 778832/778888): all 4 run to max_iter, earlystop_disabled=True.
- Output dir: /scratch/yorguin/iter_ladder/gpu_nostop/c<chunk>_i3000/  (SEPARATE from the @3000 baseline).
- Submit: iter_ladder/submit_iter_gpu_nostop.sh + manifest_gpu_nostop.txt (500 cells = 25 subj x 4 impls
  x 5 chunks). WAVE 1 = job 778907 (--array=1-250%8, ~subjects 1-12). QOS MaxSubmit=500 so 2 waves.
- RESUME: when wave 1 drains, submit WAVE 2:
  `ssh trillium-gpu 'cd /scratch/yorguin/amica-benchmark-repro/benchmark/cc_benchmark && sbatch --array=251-500%8 iter_ladder/submit_iter_gpu_nostop.sh iter_ladder/manifest_gpu_nostop.txt'`
  (ASK the user before submitting wave 2 — they want to approve job submissions.)
- AGGREGATE (after both waves): agg_chunk2.py on gpu_nostop @3000 -> iteration-matched fit-time + s/iter
  (everyone n_iter=3000); agg_mem.py-style for nvml_post_init. Then add the "time-to-iterations" table to
  the report next to the existing "time to comparable quality" (already added, from the baseline traces).

### QUEUED next: stops-off iteration-LADDER (user-approved to run AFTER the chunk campaign, all 25 subj)
Clean "wall time to reach N iterations" curves at fixed chunk 65536 (the current ladder's high-N points
are confounded: amica-python/pAMICA early-stop before the cap). Manifest: manifest_gpu_ladder_nostop.txt
= 500 cells (25 subj x 4 impls x N{100,250,500,1000,2000}; N=3000 comes free from c65536_i3000 of the
chunk run). Reuses submit_iter_gpu_nostop.sh (stops-off + bracketing; out-tag c65536_i<N> -> distinct
dirs, no collision). Submit in 2 waves (MaxSubmit=500) once the chunk campaign has DRAINED:
  `ssh trillium-gpu 'cd /scratch/yorguin/amica-benchmark-repro/benchmark/cc_benchmark && sbatch --array=1-250%8 iter_ladder/submit_iter_gpu_nostop.sh iter_ladder/manifest_gpu_ladder_nostop.txt'`
  then `--array=251-500%8 ...` once wave A drains. Aggregate per (impl, N) at chunk 65536 -> time-vs-N
  curve (compile intercept + slope). This ladder is user pre-approved; no need to re-ask before it.

### Full submission pipeline (serialize on MaxSubmit=500): chunk W1 (778907) + W2 (778977) both
### submitted (485/500 queued) -> [stale-pAMICA re-run] -> ladder WA (1-250) -> ladder WB (251-500).

### AUTONOMOUS orchestrator (2026-08-16, user asleep, approved unattended): a detached bash on the
### Trillium LOGIN node drives the rest via local sbatch/squeue (survives ssh-master expiry).
- Script: /scratch/yorguin/orchestrator.sh (source in session scratchpad); log:
  /scratch/yorguin/iter_ladder/orchestrator.log (tail it to see progress).
- Does: wait chunk drain -> re-run stale pAMICA cells (n_iter<3000; the ~5 sub-01 cells from the old
  minlrate-only build) -> ladder WA -> WB -> aggregate to /scratch/yorguin/nostop_{gpu3000,gpumem,
  ladder_i*}_summary.csv. Uses /scratch/yorguin/{agg_chunk2,agg_mem}.py (amica_python relabel baked in).
- pAMICA disable was FIXED mid-flight (commit: full stop family use_min_dll+use_grad_norm+minlrate+min_nd;
  minlrate-alone left a min_dll stop at ~495). Queued pAMICA cells use the fixed runner; only ~5 already
  done are stale (orchestrator re-runs them).
- IF the login process was reaped (check `pgrep -f orchestrator.sh` / the log): resume manually — submit
  ladder WA/WB (see QUEUED section) then run the 3 aggregations. The CHUNK campaign completes via SLURM
  regardless. On reconnect: pull the nostop_* summary CSVs, add the "time-to-iterations" table + NVML
  context split to the report (label by impl name), commit + republish artifact 5c1007ae.
- Aggregators (scratchpad, IMPL map already relabels scott_huberty_torch->amica_python): agg_chunk2.py,
  agg_mem.py, posthoc_conv.py (also committed at iter_ladder/posthoc_conv.py).

## IN FLIGHT (2026-08-16): CPU stops-off campaign (fir, rrg-kjerbi_cpu, user-scoped, autonomous)
Redesign to fix coverage + characterize contention: stops-off (AMICA_DISABLE_EARLYSTOP=1), 250 iters,
ALL 25 subjects, 10 reps/cell, high parallelism (%60), 12h wall, node-load monitor -> iter_ladder/cpu_nostop.
- chunk-sweep: chunks {1024,4096,16384,65536,262144} @250 = 6250 cells (manifest_cpu_sweep_nostop.txt).
- iter-ladder: chunk 65536, N{50,100,500} @ = 3750 cells (manifest_cpu_ladder_nostop.txt); the 250 point
  is reused from the sweep's 65536 cell -> ladder curve {50,100,250,500}.
- Prereq: fir had only 5 PCA caches; submit_preprocess_fir.sh (array 1-25) builds the missing 20
  (BIDS /project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids). Validating with a 1-cell test
  (sub-06, job 54939617) before the orchestrator launches.
- fir-side orchestrator: /scratch/yorguin/orchestrator_cpu.sh (detached on fir login; log
  iter_ladder/orchestrator_cpu.log). Does preprocess(1-25) -> chunk-sweep -> iter-ladder -> aggregate
  to /scratch/yorguin/cpu_nostop_i{50,100,250,500}_summary.csv (agg_chunk2.py; @250 = full sweep all
  chunks, @50/100/500 = ladder at 65536). 5 impls incl fortran (already stops-off in its runner).
- fir has NO MaxSubmit cap (MaxArraySize=10000) so single arrays are fine. Cost is large (~10k+ cells,
  pyamica@1024 slowest ~hours) but user-approved ("leverage parallelism, contended anyway", rrg alloc).
- RESUME if orchestrator reaped: sbatch the sweep then ladder (see submit script headers), then run the
  4 agg_chunk2 calls. On reconnect: pull cpu_nostop_i*_summary.csv, refresh the report's CPU section
  (by-subject median over 25 subj x 10 reps + contention distribution from nodemon/).

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
4. DONE (2026-08-16, commit 7dd720f): the MNE-note (#13819, artifact 62fc4258) rebuilt as a committed
   generator (results/xperf_chunksize/gen_mne_note.py) with the same corrections — 3000-iter budget, NVML
   memory, chunked jamica + its higher ~5.4GiB floor / ~13GiB full-batch default, convergence (s/iter +
   n_iter + ll), 262K-not-full-batch, AMD-EPYC-not-Xeon, no false OOM. Artifact 62fc4258 updated in place.
   The xperf_chunksize report itself went through a 4th confirmatory panel (round-4, commit 2ba7774,
   all ship) — see reviews/report-panel4-431515d/. Both deliverables now corrected + panel-hardened.
3. Memory panel (fable/gpt-5.6/grok) confirmed allocator counters aren't cross-framework comparable;
   NVML is the headline. jamica-chunk-memory panel: Fable caught the key bug (dissent was right).
Reviews kept local under reviews/ (excluded from commits).
