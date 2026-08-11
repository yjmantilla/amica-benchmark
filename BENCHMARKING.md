# Running the AMICA benchmarks

How to reproduce the cross-implementation comparison on your own data, on a
laptop or on a Slurm cluster. Every number in the manuscript's
cross-implementation table and runtime figures comes from the scripts below.

What gets compared: `amica` (JAX, this package, full-batch and blocked),
AMICA-Python (PyTorch), `pyamica`, `pAMICA`, and Fortran AMICA 1.7 where a
binary is available. All of them fit the *same* PCA-projected array, for the
same iteration budget, in one invocation. Most receive the same seed; **`pyamica`
and Fortran AMICA ignore it and use a fixed initialization** (they record
`seed_respected: false`), so a seed sweep shows zero spread for those two — that
is fixed init, not robustness. Each result also carries an `effective_config`
block with the exact hyperparameters that runner used.

## Install

The implementations cannot share one environment — `pAMICA` needs Python ≥3.12
and a newer torch than the older packages were built against — so there are
two or three:

Do NOT `pip install pyamica pamica amica torch` unpinned: those URLs float to
upstream HEAD (and *two different projects* both import as `amica` — scott-huberty's
and Sina's — so a bare `pip install amica` is ambiguous). Use the pinned setup
scripts, which install every implementation by full commit SHA from
`benchmark/cc_benchmark/pins.toml` and then assert installed == intended:

```bash
cd benchmark/cc_benchmark
bash setup_competitors.sh      # .venv_competitors: pyamica + scott-huberty (py3.11)
bash setup_pamica.sh           # .venv_pamica:      sccn/pAMICA v0.3.1     (py3.12)
bash setup_neuromechanist.sh   # .venv_neuromechanist: the pyAMICA snapshot (optional)
# the package under test (`amica`) is installed into the fir venv by fir_env.sh,
# pinned in pins.toml; on the cluster AMICA_SRC can override it with a checkout.
```

Run these on the **login node**: they are a one-time environment build (git+pip
needs internet, which compute nodes lack), which is the supported use of pip on a
login node — not inside a job allocation. The actual benchmark *fits* run in an
allocation; only the env build is a login-node step.

## Datasets

Start with **mne_sample**. It needs no manual download, no BIDS tooling and no
allocation — MNE fetches it on first use (~1.5 GB) — and it is the fixture the
local panels in the manuscript use. Everything below works on it, so a first
run needs nothing from this section beyond the first block.

```bash
# mne_sample: fetched automatically, but fetch it once up front so a benchmark
# run is not timing a download
python -c "import mne; print(mne.datasets.sample.data_path())"
export MNE_DATASETS_SAMPLE_PATH=/path/to/mne_data   # where you want it
```

Set `MNE_DATASETS_SAMPLE_PATH` explicitly rather than relying on MNE's stored
config. That config goes stale the moment the data moves — including moving
between drives — and then fails in a way that reads as a benchmark bug rather
than a path problem.

The three OpenNeuro EEG datasets are larger (~10 GB each) and need
`datalad` or `git-annex`, which fetch only the files actually required rather
than the whole repository:

| accession | what it is | used for |
|---|---|---|
| `ds004505` | table tennis, 25 subjects, 120 channels | the cluster panels and the memory campaign |
| `ds004504` | eyes-closed rest, clinical | cross-recording replication |
| `ds004621` | eyes-open rest, 128 channels | cross-recording replication |

```bash
cd benchmark/cc_benchmark
cp env.template env.local          # set BIDS_ROOT_DS4505 etc. to where data should land
module load git-annex              # on Alliance clusters; otherwise install datalad

bash download_ds004505.sh          # ~10 GB, expect 25 .set files
bash download_ds004504.sh
bash download_ds004621.sh
```

Each script prints how many recordings it materialised, so a partial fetch is
visible immediately rather than surfacing later as a missing-subject error. Run
them on a machine with internet — on a cluster that means a login node, which is
fine because downloading is I/O, not compute.

If neither `datalad` nor `git-annex` is available, the scripts fall back to
`openneuro-py`, which produces a different directory layout; in that case set
`AMICA_INPUT_LEVEL=merged` instead of the default `bids`.

## Local runs

```bash
cd benchmark/local_bench
export AMICA_PYTHON_VENV=/path/to/amica/.venv-dev/bin/python
export COMPETITORS_VENV=/path/to/amica-venvs/comp/bin/python

export MNE_DATASETS_SAMPLE_PATH=/path/to/mne_data

bash run_seed_comparison.sh      # 5 seeds x 100 iterations   (~45 min)
bash run_iter_curve.sh           # 4 iteration caps, 1 seed   (~3.5 h)
```

Both default to `mne_sample` at 30 components (166,800 samples), which is what
makes them runnable on a laptop without downloading anything. Override with
`DATASET` and `N_COMPONENTS`:

```bash
DATASET=ds004505 N_COMPONENTS=64 bash run_seed_comparison.sh my_tag
```

`run_seed_comparison.sh` answers "which is fastest here, and is the gap bigger
than the run-to-run spread" — one run per implementation cannot answer the
second half. `run_iter_curve.sh` produces the runtime-vs-iterations curve.

**Two things that will silently ruin a run**, both learned the hard way:

- **Run nothing else on the machine.** The first iteration-curve campaign was
  discarded because other work ran alongside it and 1000 iterations came out
  *faster* than 700 for three implementations. The tell is that every
  implementation's cost moves together — contention moves all lines at once, a
  real difference does not. `run_iter_curve.sh` now times a fixed canary fit
  before each block so drift is recorded rather than inferred afterwards.
- **Never compare absolute times across campaigns.** On the reference laptop
  the same configuration has produced 66.9 s and 47.0 s in separate clean runs.
  Orderings and ratios *within* one back-to-back campaign have been stable every
  time, which is why every script measures all implementations in a single
  invocation.

See `benchmark/local_bench/README.md` for the full detail.

## Cluster runs (Slurm)

```bash
cd benchmark/cc_benchmark
cp env.template env.local          # set account, BIDS_ROOT, results dir
sbatch submit_iter_curve_cpu.sh    # runtime vs iterations, CPU, one task per implementation
sbatch submit_iter_curve_gpu.sh    # same on one GPU
sbatch submit_mem_recheck.sh       # peak memory, full batch vs blocked, six recordings
```

Parameterised through the environment, so a different dataset or size needs no
edit:

| variable | meaning | default |
|---|---|---|
| `AMICA_MEM_DATASET` | `ds004505`, `ds004504`, `ds004621`, `mne_sample` | `ds004505` |
| `AMICA_MEM_SUBJECT` | subject number | `1` |
| `AMICA_MEM_NCOMP` | PCA rank | `64` |
| `AMICA_ITERS` | iteration caps to visit | `100 400 700 1000` |
| `AMICA_ITER_TAG` | results subdirectory | `itercurve_cpu` |
| `AMICA_SRC` | source checkout of `amica` to measure | `/scratch/$USER/amica-blocked` |

```bash
# same recording at a different rank, into its own directory
sbatch --export=ALL,AMICA_MEM_NCOMP=30,AMICA_ITER_TAG=itercurve_c30 submit_iter_curve_cpu.sh
```

Each job asserts on the compute node, before its first fit, that `amica`
imports from `$AMICA_SRC` and that its default `chunk_size` is `"auto"`, and
exits otherwise. That guard exists because an older checkout imports and fits
perfectly well — it would simply produce a plausible curve for a different
implementation, and nothing downstream would reveal it. Each job also logs the
commit it measured.

## Figures and tables

```bash
python benchmark/comparator/plot_iter_curve.py \
    --panel "Local CPU=results/comparator/itercurve_local_cpu" \
    --panel "Cluster CPU=results/comparator/cluster/cpu/itercurve_cpu" \
    --ncols 2 --ybreak "Cluster GPU=250,3" \
    --out results/figures/fig_iter_curve.pdf

python scripts/paper/figures/make_tab_cross_implementation.py
```

The plotter uses iterations *actually run* rather than the cap requested, so an
implementation that converges early is not drawn with a downward bend that looks
like sublinear scaling, and takes per-iteration cost from a least-squares fit
over all points rather than a two-point difference — that estimator is what
produced a *negative* per-iteration cost in an earlier GPU campaign.

## Profiling the package itself

These ship with the package rather than living here, because they profile it
rather than compare it:

```bash
python -m amica.benchmark.profile_cpu       # where an iteration's time goes
python -m amica.benchmark.profile_memory    # where a fit's peak memory goes
python -m amica.benchmark.profile_scaling   # how cost scales with samples and rank
python scripts/regression_vs_ref.py         # this checkout vs any baseline commit
```

`regression_vs_ref.py` is the one to reach for when asking whether an
optimisation changed the numbers: it fits identical data under two checkouts, in
separate processes, and reports the Hungarian-matched unmixing correlation and
the final log-likelihood against a baseline commit of your choosing.
