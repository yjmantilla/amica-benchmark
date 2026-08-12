"""Runner for the reference Fortran AMICA 1.7 binary (CPU; memory + W-parity).

Writes the comparator's already-projected X (.npz key 'X') to the Fortran on-disk
format (data.fdt float32 col-major + amica.param) via the vendored _fortran_io, runs
``/usr/bin/time -v mpirun -np 1 $AMICA17_BIN amica.param``, and parses the child's
peak RSS from GNU time ("Maximum resident set size (kbytes)"). The final W is read
back for the Hungarian-parity sanity. Fortran is CPU-only and allocates its working
arrays up front with a ~zero import baseline, so absolute peak RSS ~= delta RSS;
peak_vram_gb is None.

amica17 runs its standard sphere/mean/PCA path (do_sphere=1/do_mean=1/doPCA=1, pcakeep=n_comp)
on the comparator's pre-projected input — PCA there is just a rotation, and this is the validated
parity config (do_sphere=0/doPCA=0 makes amica17 exit at init with 0 iterations). Same
hyperparameters as the Python runners are passed in --config.

Environment (cluster):
  AMICA17_BIN   path to amica17 (default the validated reference build below)
  MPIRUN_BIN    mpi launcher (default 'mpirun')
  GNU_TIME_BIN  GNU time with -v (default '/usr/bin/time')
mpirun + the binary's BLAS/OpenMP modules must be loaded by the sbatch wrapper.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fortran_io as fio
from _common import load_data, parse_runner_args, write_result

_DEFAULT_BIN = "amica17"  # portable default: set AMICA17_BIN to an absolute path, or have amica17 on PATH


def _parse_maxrss_kb(stderr_text: str) -> float | None:
    """Peak RSS (KiB) from GNU `/usr/bin/time -v` 'Maximum resident set size (kbytes)'."""
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text or "")
    return float(m.group(1)) if m else None


def _resolve_bin(amica_bin: str) -> str:
    """Absolute path of the amica17 binary (resolve a bare name via PATH)."""
    if os.path.sep in amica_bin or os.path.isabs(amica_bin):
        return os.path.abspath(amica_bin)
    found = shutil.which(amica_bin)
    return found or amica_bin


def _sha256(path: str) -> str | None:
    """SHA-256 of the resolved binary, so a row is tied to the exact build.

    Two different amica17 builds produced a matched |r| of 0.94 vs 0.28 on the
    same input — the digest is the only thing that tells those rows apart.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape
    n_mix = cfg.get("n_mix", 3)

    amica_bin = _resolve_bin(os.environ.get("AMICA17_BIN", _DEFAULT_BIN))
    mpirun = os.environ.get("MPIRUN_BIN", "mpirun")
    gnu_time = os.environ.get("GNU_TIME_BIN", "/usr/bin/time")

    # Effective config actually passed to amica17 (frozen literals below +
    # cfg-overridable knobs). Serialized in every row so a version-default change
    # is visible. amica17 forces fix_init and ignores cfg["seed"] (see below).
    effective_config = {
        "num_mix_comps": n_mix, "max_iter": cfg["max_iter"],
        "do_newton": int(bool(cfg.get("do_newton", True))),
        "newt_start": 50, "newt_ramp": 10,
        "lrate": cfg.get("lrate", 0.1), "rholrate": 0.05,
        "rho0": 1.5, "minrho": 1.0, "maxrho": 2.0, "pdftype": 0, "num_models": 1,
        "do_sphere": 1, "do_mean": 1, "doPCA": 1, "pcakeep": n_comp,
        "use_min_dll": 0, "use_grad_norm": 0, "fix_init": 1,
    }
    # amica17 forces fix_init=1 and takes no seed argument: cfg["seed"] does NOT
    # change its initialization. Record that so seed-sweep tables don't read its
    # zero spread as robustness.
    identity = {
        "fortran_bin": amica_bin,
        "fortran_sha256": _sha256(amica_bin),
        "seed_respected": False,
        "init": "fix_init",
        "requested_seed": cfg.get("seed", 0),
        "effective_config": effective_config,
    }

    # Gate 0: if the expected binary sha is provided (AMICA17_SHA_EXPECTED, which
    # the submit scripts set from pins.toml [fortran]), refuse to measure the wrong
    # build — a different amica17 gave a matched |r| of 0.28 vs 0.94. Require FULL
    # equality; tolerate only an explicit prefix of >= 16 hex (an 8-char prefix is
    # too weak to identify a build). Fails fast (before the fit) with an error row.
    _expected_sha = os.environ.get("AMICA17_SHA_EXPECTED", "").strip().lower()
    if _expected_sha:
        _got_sha = (identity["fortran_sha256"] or "").lower()
        if len(_expected_sha) < 16:
            write_result(args.output, {
                "implementation": "fortran_amica17",
                "error": f"expected_sha_too_short ({_expected_sha!r}; use the full sha or >=16 hex)",
                **identity,
            })
            return
        if not _got_sha or not (_got_sha == _expected_sha or _got_sha.startswith(_expected_sha)):
            write_result(args.output, {
                "implementation": "fortran_amica17",
                "error": f"binary_sha_mismatch (got {_got_sha or None}, expected {_expected_sha})",
                **identity,
            })
            return

    workdir = Path(tempfile.mkdtemp(prefix="fortran_amica_"))

    def _cleanup():
        # amica17 writes large working arrays (data.fdt + outputs); leaving one
        # mkdtemp per run behind fills the node's tmpfs over a sweep. Keep it only
        # for diagnosis via AMICA_KEEP_WORKDIR=1.
        if os.environ.get("AMICA_KEEP_WORKDIR") != "1":
            shutil.rmtree(workdir, ignore_errors=True)

    # Any exception during setup or launch (e.g. a missing mpirun/gnu_time raising
    # in subprocess.run, or an fio write failure) must not leak the workdir.
    try:
        data_dir = workdir / "data"
        out_dir = workdir / "out"
        data_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        fio.write_fdt(X, data_dir / "data.fdt")
        fio.write_param(
            workdir / "amica.param",
            files=str(data_dir / "data.fdt"),
            outdir=str(out_dir) + "/",
            n_channels=n_comp, n_samples=n_samples,
            block_size=min(int(n_samples), 100000),
            # Run amica17's standard sphere/mean/PCA path (the validated parity config). On the
            # already-projected, unit-variance input, PCA(pcakeep=n_comp) is just a rotation. NOTE:
            # do_sphere=0/doPCA=0 makes amica17 exit at init with 0 iterations.
            do_sphere=1, do_mean=1, doPCA=1, pcakeep=n_comp,
            # same hyperparameters as the Python runners (base_cfg):
            num_mix_comps=n_mix,
            max_iter=cfg["max_iter"],
            do_newton=int(bool(cfg.get("do_newton", True))),
            newt_start=50, newt_ramp=10,
            lrate=cfg.get("lrate", 0.1), rholrate=0.05,
            rho0=1.5, minrho=1.0, maxrho=2.0, pdftype=0, num_models=1,
            max_threads=1, writestep=1, write_LLt=0, fix_init=1,
            use_min_dll=0, use_grad_norm=0,   # run full max_iter (no early stop)
        )

        cmd = [gnu_time, "-v", mpirun, "-np", "1", amica_bin, str(workdir / "amica.param")]
        run_env = dict(os.environ, OMP_NUM_THREADS="1")  # match parity recipe (param max_threads=1)
        t0 = time.perf_counter()
        cp = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
        elapsed = time.perf_counter() - t0
    except BaseException:
        _cleanup()
        raise

    def _error(reason: str):
        # A failed run must NOT be written as a normal row: the old code emitted
        # W=null / ll=NaN with no error, corrupting perf aggregates. Gate on it.
        row = {
            "implementation": "fortran_amica17",
            "error": reason,
            "returncode": cp.returncode,
            "cmd": " ".join(cmd),
            "fit_time_s": float(elapsed),
            "stderr": (cp.stderr or "")[-2000:],
            "stdout": (cp.stdout or "")[-1000:],
        }
        row.update(identity)
        write_result(args.output, row)
        _cleanup()

    # Gate 1: the process must have exited cleanly. GNU time prints max RSS even
    # when its child exits nonzero, so a returncode check is what actually
    # distinguishes a completed fit from a crashed one.
    if cp.returncode != 0:
        _error(f"nonzero_exit (returncode={cp.returncode})")
        return

    maxrss_kb = _parse_maxrss_kb(cp.stderr)
    if maxrss_kb is None:
        _error("no_maxrss (GNU /usr/bin/time -v unavailable or run failed)")
        return
    peak_gb = maxrss_kb / 1024 ** 2  # KiB -> GiB

    # Gate 2: the expected outputs must exist and have the expected shape before
    # this counts as a success. A parse failure or a truncated W is an error row.
    try:
        res = fio.read_fortran_results(out_dir, n_components=n_comp, n_mixtures=n_mix)
        W = np.asarray(res["W"], dtype=float)
        ll = list(np.asarray(res.get("LL_clean", []), dtype=float).flatten())
    except Exception as exc:
        _error(f"output_read_failed ({type(exc).__name__}: {exc})")
        return
    if W is None or W.shape != (n_comp, n_comp):
        _error(f"bad_W_shape (got {None if W is None else W.shape}, "
               f"expected {(n_comp, n_comp)})")
        return
    if not np.isfinite(W).all():
        # A square-but-all-NaN W with one finite LL would otherwise pass as a
        # normal row and corrupt the W-parity aggregate.
        _error("nonfinite_W (unmixing matrix has NaN/Inf entries)")
        return
    if not ll or not np.all(np.isfinite(ll)):
        _error("no_finite_ll (log-likelihood trace missing or non-finite)")
        return

    out = {
        "implementation": "fortran_amica17",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": float(peak_gb),
        # Fortran allocates up front with ~zero import baseline -> delta ~= absolute peak.
        "baseline_rss_gb": 0.0,
        "delta_rss_gb": float(peak_gb),
        "peak_vram_gb": None,
        "nvml_peak_vram_gb": None,
        "ll_final": float(ll[-1]) if ll else float("nan"),
        "ll_history": ll,
        "W": W.tolist() if W is not None else None,
        "device": "cpu",
        "dtype": "float64",   # AMICA 1.7 computes in double precision (float32 .fdt input)
        "n_iter": int(len(ll)),
    }
    out.update(identity)  # fortran_bin, sha256, seed_respected/init, effective_config
    write_result(args.output, out)
    _cleanup()


if __name__ == "__main__":
    main()
