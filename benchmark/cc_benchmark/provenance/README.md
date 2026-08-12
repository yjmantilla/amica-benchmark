# Recovered publication provenance

These are the exact competitor pins + environment behind the **published** numbers,
recovered on 2026-08-11 from the surviving `.venv_competitors` on Compute Canada
`fir` (Sina Esmaeili / `rrg-kjerbi`). Package versions were **not** captured
per-run at publication time, so this is reconstructed from the intact venv and
cross-checked against the paper's `references.bib` — solid, but reconstructed.

Durable read-only copy on fir: `/project/rrg-kjerbi/sesma-shared/amica-repro/`.

## Competitors (published `.venv_competitors`, CPython 3.11.5)

| benchmark runner | project | published pin | upstream commit behind it |
|---|---|---|---|
| `pyamica_torch` | DerAndereJohannes/pyamica | PyPI `pyamica==0.3.0` | `a8a4d7e0ad14a88cf2cabeff5094cd0c8a262536` |
| `scott_huberty_torch` | scott-huberty/amica-python | PyPI `amica-python==0.1.1` | `cad98a6cc98782ffb6f1bff22c99b31431ee5832` (the 0.1.0→0.1.1 bump) |
| `neuromechanist_numpy` | neuromechanist/pyAMICA | git `526aa3231623490ea21ef9c45acbb50730929622` | (same) |

The two PyPI packages have no VCS commit in their metadata — a **version pin is
the stricter identity**, so `pins.toml` pins them by version and records the
upstream commit only for reference. `check_env.py verify` checks the installed
version for these.

As-built numerical stack (Alliance `+computecanada` wheels — site-specific, not on
PyPI, hence captured as-built rather than pinned): `torch 2.12.0+computecanada`,
`numpy 2.4.2+computecanada`, `scipy 1.17.0+computecanada`, `mne 1.12.1`.

## The `pyamica` / `pyAMICA` name collision

`neuromechanist/pyAMICA`'s distribution name (`pyAMICA`) canonicalizes to `pyamica`
(PEP 503) — the same project name as `DerAndereJohannes/pyamica`. In the original
single-venv setup Sina resolved this with a local rename patch
(`neuromechanist_pyamica_rename.diff`, archived here): `pyAMICA` → `neuromechanist_pyamica`.

This repo instead isolates the snapshot in its **own** venv (`setup_neuromechanist.sh`),
so there is no collision and **no patch is needed** — the snapshot installs
non-editable by SHA (`git+…@526aa3…`), which `check_env.py verify` can check. The
patch is kept here only to document what the original single-venv run applied
(it changes the distribution name, not the algorithm — identical numbers).

## Fortran AMICA 1.7 reference binary

`amica17` sha256 `c02f22c37cb259364e921d1e1b42f7181ce9fb7baae6a716c2ade261b49771fe`
(verified). Built on fir 2026-06-09, dynamically linked against fir's CVMFS
gentoo/2023 libraries — **runs on fir compute nodes only**. Durable copy +
`BUILD_PROVENANCE.md` (3 required patches vs upstream, build flags, parity result)
+ sources + `SHA256SUMS` live at `/project/rrg-kjerbi/sesma-shared/amica-repro/`
and `/project/rrg-kjerbi/sesma/amica_fortran_reference/`. See `fortran/README.md`.
