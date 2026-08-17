# amica-benchmark — reproduction TODO

Status as of 2026-08-11. Working model: 5 focused PR branches (`pr1`…`pr5`) →
open PRs #2–#6 on the `yjmantilla` fork; `repro-integration` is a disposable
test-merge of all five. Cluster: `fir` (Alliance), clone at
`/scratch/yorguin/amica-benchmark-repro`.

---

## PENDING (2026-08-17) — CPU large-chunk fold-in to xperf_chunksize report

GPU large-chunk / full-batch extension is **done, deployed, pushed** (branch
`iter-ladder-campaign`, commit `e75f307`; artifact 5c1007ae…). The CPU half is
**running on Narval but fairshare-throttled** (def-kjerbi_cpu EffectvUsage ~0.998
→ ~1–2+ days). User decision: **let it run; user will say when it's ready** (CPU
monitor stopped on purpose).

- [ ] When user says CPU is ready: verify jobs **1146546** (1M+full, 250),
      **1147196** (524K, 125), **1149409** (scott per-subject full-batch, 25) are
      COMPLETED on Narval.
- [ ] Aggregate CPU results from `/scratch/yorguin/iter_ladder/cpu_nostop/c<chunk>_i250_r1/`
      (and scott full-batch from per-subject `c<n_samples>_i250_r1/`), same script
      pattern as `scratchpad/agg_gpu2.py` but for CPU fit_s + RSS (peak_rss_gb).
- [ ] Fold into `gen_report.py` CPU charts (c_ct/c_cr): extend to log axis 1K→1M,
      add a **CPU full-batch table** (mirror `fullbatchrows()`); apply the same
      short-subject methodology — **1M restricted to the 22 subjects >1M**; scott
      full-batch uses per-subject batch = n_samples (its `BatchLoader` guard is
      device-independent). Remove the "GPU-only in this version" CPU note.
- [ ] Pull a CPU-ext raw CSV into `raw/`, update README/NOTES provenance, regen,
      redeploy to the SAME artifact URL, `git push fork iter-ladder-campaign`,
      refresh the htmlpreview link.
- Full context (paths, job IDs, the batch-guard finding, GPU memory table):
      memory file `narval-cpu-campaign.md`.

---

## DONE — reproducibility infrastructure

- [x] PR1 provenance stamping at the `write_result` choke point
- [x] PR2 central `pins.toml` + `check_env.py` (specs/verify/lock/pin/strict)
- [x] PR3 aggregator / CSV / schema fixes (content-hashed run_id, impl identity)
- [x] PR4 correctness (Fortran result gate, sha gate, identity block)
- [x] PR5 docs
- [x] 3-model panel re-reviews; soundness gaps R1–R6 closed
- [x] DIY reference: all 4 venvs code + stack pinned (competitors,
      neuromechanist, pamica, fir), reference locks committed, `--strict`
      enforcing — **no dependency on Sina's account**
- [x] Fortran amica17 built from vendored source (our sha `180301398f`, pinned)
- [x] `scipy-stack/2026a` module pinned for fir (numpy 2.4.2 / scipy 1.17.0)
- [x] 46 tests green both legs (jax + `AMICA_NO_JAX=1`)

**We have made the benchmark reproducible. We have not yet reproduced anything.**

---

## OPEN — gating questions (answer these; they change the order)

- [ ] Is there a hard deadline / specific deliverable the advisor needs?
- [ ] Are the datasets staged on the cluster (`/project/rrg-kjerbi/...`), or is
      staging part of the work?
- [ ] Goal = reproduce the *existing* benchmark, or *produce new* results with
      the new pAMICA? (changes what "done" means)
- [ ] Is landing PRs #2–#6 upstream a goal on its own?
- [ ] Compute constraints — allocation budget? avoid sbatch until X confirmed?

---

## TRACK A — Smoke run (de-risk environments)   [RAN 2026-08-11, job 54280617]
Prove the pinned venvs actually execute end-to-end and emit provenance-stamped
results, before spending real cluster hours.
- [x] Datasets staged — ds004505 sub-01 present at fir_env default path
- [x] Tiny run: ds004505 sub-01, 16 comps, 12 iter, 1 seed, all venvs + fortran
- [x] Provenance block present on amica results; content-hashed run_id path OK
- [x] `submit_smoke_all.sh` written (scratchpad + cluster clone, not yet committed)

RESULT: 6 / 8 implementations green in 4 min, and **all 6 agree numerically**
(pairwise Hungarian-matched |W corr| = 1.000 across every pair):
  fir: amica_python_jax / jax_chunked / numpy .......... OK
  competitors: pyamica_torch, scott_huberty_torch ...... OK
  pamica: pamica_torch (THE NEW pAMICA) ................ OK  <- advisor's ask
  neuromechanist_numpy ................................. FAIL (packaging)
  fortran_amica17 ...................................... FAIL (segfault rc=139)

### A.1 — neuromechanist_numpy FAIL -> FIXED 2026-08-12
`FileNotFoundError: .venv_neuromechanist/.../pyAMICA/params.json`.
- Root cause: at pinned commit 526aa32 the package is flat `pyAMICA` (dist
  `pyamica 0.1.dev0`) with params.json beside the modules, but pyproject
  `package-data = {"pyAMICA": ["data/*"]}` does NOT list params.json, so a
  non-editable `pip install git+...@526aa32` (wheel build) drops it. Sina's
  original install was editable (`pip install -e .`), which kept it via the
  source tree; ours is a git wheel install. (My earlier "installed flat pyAMICA
  doesn't match repo" confusion was a failed probe checkout showing the DEFAULT
  branch, not 526aa32 -- the plain install DOES reproduce the venv correctly.)
- Fix (pr2-pin-and-check ada1eef): setup_neuromechanist.sh fetches params.json
  from the pinned commit (DRY via `check_env pin`) after install, and the setup
  verify now CONSTRUCTS AMICA so a regression fails at setup, not mid-job.
- Validated: smoke re-run (job 54339763) -> neuromechanist_numpy OK.

### SMOKE MATRIX NOW 8/8 GREEN (job 54339763, 2026-08-12)
amica jax/jax_chunked/numpy, pyamica_torch, scott_huberty_torch, pamica_torch
(new pAMICA), neuromechanist_numpy, fortran_amica17 -- all OK, provenance stamp
firing. The full reproducible benchmark runs end-to-end on fir.

### A.2 — fortran_amica17 FAIL -> FIXED 2026-08-11
Segfault (signal 11, rc=139) during variable init, AFTER data load / mean / cov /
sphere. A/B on fir (job 54293118) localized it: OUR build crashed, Sina's
reference build (sha c02f22c3) RAN OK on the identical input -> our BUILD, not the
config. Root cause via Sina's BUILD_PROVENANCE.md: our vendored amica17_patched.f90
was missing the 3rd documented fix -- the fix_init init branch left `comp_list`
unset, so get_unmixing_matrices read it uninitialized under fix_init=1 (the mode
our runner always uses) -> segfault. Added `comp_list(i,h)=(h-1)*nw+i` to the
fix_init loop. Rebuilt (sha 665b5771); re-A/B (job 54307214): ours RAN OK 12 iters
17.5s vs Sina 17.7s. Committed: pr4-correctness 632052d (source+docs),
pr2-pin-and-check cdf25d2 (pins sha + test). STILL TODO: push both; rebuild
integration; Track B parity re-validation (submit_parity.sbatch) on the new build.

## TRACK B — Fortran parity re-validation   [DONE 2026-08-12, job 54322230]
Confirm the from-source amica17 (with fix#3) matches amica-python numerically,
not just that it runs.
- [x] Ran parity (synth6 6ch x 50k, m=3, Newton, 1000 iter, shared init) via
      adapted submit_parity_yorguin.sbatch (repo-relative paths + PYTHONPATH=REPO,
      since amica_python is a vendored dir here, not a pip package like in Sina's
      amica-python repo).
- [x] Deterministic rebuild -> same sha 665b5771. PASSED, matches Sina's ref:
      iter-0 ΔLL 2.66e-15 | final ΔLL 7.49e-8 | W matched |r| 0.99999999993 |
      matched-src |r| 0.99999999993 | W Frobenius rel 1.17e-5 | 1000 iters.
  => our from-source Fortran reference is numerically correct to optimizer precision.
- [x] Committed repo-relative + PYTHONPATH fix for submit_mklfix_parity.sbatch
      (pr4-correctness b1dc0a6) — parity now runs from the repo alone.
- [ ] NOTE: the sibling submit_ds004505_parity.sbatch has the same sesma-paths +
      no-PYTHONPATH bug; apply the same treatment if/when that path is used
      (not yet validated, so left untouched for now).

## TRACK C — Cross-impl perf campaign (new pAMICA)   [COMPLETE 2026-08-12 night]
FINAL: 250/250 cells, 0 failures, cold-consistent. Report artifact:
https://claude.ai/code/artifact/91386087-89fb-4c5c-ad88-b607216e76a2
CSV: /scratch/yorguin/xperf_campaign/xperf_times.csv (per-subject, both devices).
Headline (mean fit, 25 subj, 100 iter):
  GPU: amica 9.6s = scott 10.1s = pyamica 11.0s (fast tier) <<< pAMICA 265.7s (27x slower)
  CPU: amica_chunked 218s (fastest) < amica_jax 458 < scott 558 < pAMICA 673 < fortran 914 < pyamica 1125
  amica vs scott GPU = statistical TIE (t p=0.097, Wilcoxon p=0.048); amica sig faster than
  pyamica (p=.01) + pAMICA (p<1e-25). pAMICA barely accelerates CPU->GPU (2.5x) vs amica 23x.
  All agree numerically (|W corr| 0.997-1.000). amica-JAX-GPU via .venv_fir_gpu.
--- original plan notes below ---
User authorized unattended pilot->full. Cross-impl perf/memory on ds004505,
CPU (submit_mem_compare) + GPU (submit_mem_gpu), 64 comps, 100 iter, 1 seed.
Results under /scratch/yorguin/xperf_campaign/{cpu,gpu}/<sub-tag>/.
- [x] PILOT (jobs 54341115-118) surfaced TWO issues:
   1. CPU jobs FAILED: the clone's submit_mem_compare.sh (older than my PR work)
      defaults AMICA17_BIN to Sina's staged binary (sha c02f22c3); pins now expect
      OUR 665b5771 -> sha gate FATAL. FIX: pass AMICA17_BIN=<our binary> in --export.
      (The committed pr-branch submit_mem_compare already defaults to our binary;
      the /scratch clone is just stale. No repo change needed.)
   2. GPU amica_python_jax_chunked FAILED: RuntimeError "Backend 'cuda' not in
      ['cpu','tpu']" -- the fir venv jaxlib is CPU-only. amica-JAX cannot run on
      GPU without a cuda jaxlib. DECISION: not fixing tonight (risks the shared
      .venv_fir the CPU jobs use). Torch impls (pyamica/scott/PAMICA) run fine on
      CUDA. amica-JAX-GPU is a documented gap -> follow-up: build .venv_fir_gpu with
      [jax-gpu] + point AMICA_PYTHON_VENV at it.
- PILOT GPU numbers already valid (sub-01/02): pamica ~204-341s/100iter vs
  pyamica ~13-19s, scott ~11-16s (torch, CUDA). pamica notably slower -- a real finding.
- [~] FULL launched: GPU sub-03..25 (jobs 54341948-970, reuse pilot 01/02);
  CPU sub-01 validation (job 54341971, AMICA17_BIN fix). Monitoring CPU validation.
- [ ] IF CPU validation clean -> launch CPU sub-02..25 (24 jobs).

### amica-JAX-GPU gap -> BUILDING FIX (user asked re: uv)
Repo clue: reval_env.sh does the Alliance-native GPU jax = jax+jaxlib PLUS
`jax_cuda12_plugin` + `jax_cuda12_pjrt` (wheelhouse --no-index) + module cuda/12.6
cudnn. (jax[cuda12] from PyPI is the uv-friendly equivalent; uv is NOT on fir.)
Built .venv_fir_gpu (separate, safe): jax/jaxlib/plugins all 0.10.2+computecanada
(matching!), numpy 2.4.2, mne 1.12.1, amica@92003b4 -- mirrors .venv_fir + CUDA
backend. GPU test job 54342407 checking jax.devices() sees CUDA.
- [x] GPU venv WORKS (test 54342407): H100 detected, jax.devices()=[CudaDevice(id=0)],
      matmul on cuda:0. (Benign ptxas 12.6.2 clamping warning only.)
- [~] Cancelled torch-only GPU jobs; relaunched GPU sub-01..25 with
      AMICA_PYTHON_VENV=.venv_fir_gpu (jobs 54342672-696, name xperfg2_*). Early
      check on sub-01 confirms amica_python_jax_chunked runs on cuda. Now the GPU
      comparison has amica + pyamica + scott + pamica all on GPU.
- FOLLOW-UP for repo: add a setup_fir_gpu.sh (jax+jaxlib+jax_cuda12_plugin+
  jax_cuda12_pjrt wheelhouse + amica) so .venv_fir_gpu is reproducible; currently
  built ad hoc on the clone. uv-equivalent: uv venv + `uv pip install jax[cuda12]`.

### CPU validation clean + JAX cache-race fix
- CPU sub-01 (job 54341971) COMPLETED clean: all 6 impls OK incl fortran
  (AMICA17_BIN fix worked; W-corr vs fortran: pyamica 1.000, scott 0.999,
  amica 0.998, pamica 0.997). CPU sub-01 times: amica_jax 265s / jax_chunked 149s,
  pyamica 932s, scott 351s, pamica 405s, fortran 773s (peak RSS amica_jax 14.9GB!).
- BUG found mid-run: amica_jax GPU failed intermittently (~10%: sub-13,19) with
  JaxRuntimeError on `~/.cache/amica/jax_cache/xla_gpu_per_fusion_autotune_cache_dir`.
  Root cause: amica/backend.py defaults the JAX compilation cache to a SHARED
  ~/.cache/amica/jax_cache ($HOME) -> concurrent jobs race the autotune cache.
  amica respects env JAX_COMPILATION_CACHE_DIR (backend.py:49). run_subprocess
  copies os.environ, so a job-level value reaches amica.
  FIX: cancelled + relaunched CPU sub-02..25 (54345114-137) and GPU
  sub-13,19,20-25 (54345138-145) each with a PER-SUBJECT
  JAX_COMPILATION_CACHE_DIR=/scratch/yorguin/jax_cache/<dev>_<sub> (on scratch,
  no race, off $HOME). Kept 17 good GPU + CPU sub-01. Final monitor b0tchfob1.
- FOLLOW-UP for repo: submit_mem_* should set a per-job JAX_COMPILATION_CACHE_DIR
  (the shared $HOME default races under array/concurrent runs + hits $HOME I/O).

### Cache effect on timings -> RECOMPUTE for consistency
Controlled A/B (sub-05 GPU, job 54345928): same subject, fresh vs old shared cache.
  torch impls IDENTICAL (pamica 248->250, pyamica 9.07->9.06 = <1% noise, confirms
  they're unaffected + GPU timing noise <1%). amica_jax_chunked 7.71 (warm) -> 8.89
  (cold) = +15%. REAL cache effect: the old shared cache let later subjects skip
  some compile. Only amica_jax affected (torch/fortran untouched).
Decision: recompute the WARM amica cells cold for consistency + fairness (cold =
  amica pays compile like torch pays warmup). Relaunched GPU sub-1-12,14-18
  (54346754-770) + CPU sub-01 (54346771) with per-subject cold caches.
NOW ALL 25 CPU + 25 GPU are cold-consistent (per-subject JAX cache). Final
  aggregation monitor b2eel7hkl -> morning report.
Note: effect is small; qualitative headline unchanged (amica ~20-30x faster than
  pamica on GPU; pamica barely accelerates CPU->GPU ~1.4x while amica ~20x).
- [ ] Aggregate: comparator/aggregate_pilot.py --root {cpu,gpu} --impls all +
      plot_pilot.py. Results: /scratch/yorguin/xperf_campaign/{cpu,gpu}/<sub>/.
- [ ] Morning report.
- ds004504/ds004621 comparators NOT runnable (not staged) -- ds004505 only (25 subj).

## TRACK D — Merge PRs upstream
Land the harness improvements (independent of compute).
- [ ] Shepherd PRs #2–#6 to merge against `snesmaeili/amica-benchmark`
- [ ] Address any upstream review

## TRACK E — Remaining hardening (non-blocking gold-plating)
- [ ] Wheel `--require-hashes`
- [ ] Dataset content-hashing
- [ ] `delta_rss` labeling
- [ ] Container image
- [ ] Pin scipy-stack module for reval env (if applicable)

## TRACK F — Optional reruns (higher rigor; non-blocking)
None of these change the current report's claims — the GPU chunk study is
contention-free and valid, and the CPU absolutes are already labelled approximate.
They upgrade rigor only. Infrastructure now exists in `benchmark/cc_benchmark/iter_ladder/`.

- [ ] **CPU chunk-size study — re-run with the new contention methodology.** The
  published CPU chunk timings are best-of-5-*subjects* (min), which conflates the
  shortest subject with the least-contended run (panel-flagged). Re-run just the CPU
  chunk sweep with the iter_ladder handling: N=10 reps as *separate* array tasks +
  `node_monitor.sh` per cell (co-tenant / node-busy% trace) + quiet-filter
  aggregation (`node_busy% <= 25`) → contention-certified CPU absolutes. Adapt
  `iter_ladder/submit_iter_cpu.sh` to sweep chunk at fixed iters. **GPU chunk study
  needs no rerun** (each cell holds a GPU; no shared-node bandwidth contention).
- [ ] **Increase the iteration budget 100 → 1000 across the campaign.** The chunk
  study used a 100-iteration budget (short — compile-influenced for jamica, not
  steady-state). A 1000-iteration variant (GPU + CPU) gives more realistic
  steady-state timings. GPU is cheap; CPU is ~10× the 100-iter cost, so scope
  accordingly (fewer subjects/impls if needed). Partially overlaps the iteration
  ladder, which already covers chunk=65536 up to 3000 iters.
- [ ] (stretch) Re-render the chunk report from the contention-certified CPU numbers
  once the above land; document the methodology upgrade in `NOTES_measurement.md`.

---

## RECOMMENDED EXECUTION ORDER

1. **A — Smoke run** — cheapest, gates everything; catches gross breakage in the
   pinned envs before any real spend.
2. **B — Fortran parity** — validates the reference; can batch alongside/just
   after the smoke jobs since both are small compute.
3. **C — Full campaign** — only after A is green and B confirms the Fortran
   reference; this produces the deliverable.
4. **D — Merge PRs upstream** — start once the smoke run has exercised the code;
   can run concurrently with C (no compute dependency).
5. **E — Hardening** — opportunistic, last; blocks nothing.

Critical path: **A → C**. B feeds trust into C. D and E parallelize off to the
side. All compute (A/B/C) is gated on sbatch + Duo approval and staged datasets.
