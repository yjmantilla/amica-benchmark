"""Runner for sccn/pAMICA (PyTorch natural-gradient EM).

pAMICA is the SCCN implementation by Shirazi, Delorme and Makeig. It is the
same project as the repository previously benchmarked here as
``neuromechanist/pyAMICA`` -- that URL now redirects to ``sccn/pAMICA`` (same
GitHub repository id) -- but the package renamed to ``pamica``, moved to the
PyTorch backend, and gained CUDA and Fortran-parity work. Pin the install; see
``setup_pamica.sh``.

``AMICA.fit`` takes ``(n_channels, n_samples)``, which is already the
orchestrator's layout, so unlike run_pyamica there is no transpose. The
orchestrator pre-PCA-projects, hence ``do_mean=False, do_sphere=False``.

Two behaviours worth knowing when reading the numbers this writes:

* ``keep_best`` (pAMICA's default, left on) restores the best-likelihood
  iterate when the fit ends, so the delivered model is not necessarily the last
  one. ``ll_history`` is documented to remain the true per-iteration
  trajectory, so both are recorded: ``ll_final`` is what the implementation
  hands back, ``ll_last_iterate`` is where the trajectory actually ended.
* ``get_unmixing_matrix()`` returns the raw model ``W`` in Fortran convention,
  not composed with the sphering transform. With ``do_sphere=False`` the sphere
  is identity, so this is directly comparable to the ``W`` the other runners
  report.

**What "matched fixture" does and does not mean here.** The orchestrator imposes
the shared experimental protocol on every implementation -- iteration budget,
``n_mix``, ``lrate``, Newton on -- and that is what makes the rows comparable.
It does not mean copying one library's internal tuning onto another. An earlier
version of this runner did exactly that, carrying run_pyamica's ``newt_start=50``,
``invsigmin=1e-8`` and ``invsigmax=100.0`` across, and it cost pamica most of its
accuracy: worst matched row correlation against the amica JAX reference was
0.8217, against 0.95-0.98 for the other implementations. Restoring pamica's own
three constants, with the shared protocol unchanged, brings it to 0.9524 --
level with pyamica's 0.9523 on the same input. The constants below are therefore
pamica's documented defaults, and the difference is a measurement artefact worth
remembering rather than a property of the implementation.

Note that pamica's *full* library defaults are not the right comparison either:
at ``do_newton=False`` and ``lrate=0.05`` it reaches only 0.1867 within a
100-iteration budget, because that configuration is meant for the long
Fortran-parity runs its documentation describes, not a short fixed budget. The
shared protocol stays imposed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    baseline_rss_gb,
    load_data,
    parse_runner_args,
    peak_rss_gb,
    start_nvml_sampler,
    stop_nvml_sampler,
    write_result,
)


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples) -- pamica's own layout
    n_comp, n_samples = X.shape

    import torch
    from pamica import AMICA

    device = os.environ.get("TORCH_DEVICE", "cpu")

    model = AMICA(n_models=1, n_mix=cfg.get("n_mix", 3), device=device, verbose=False)

    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    _nvml = start_nvml_sampler(_use_nvml)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    model.fit(
        X,
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        do_mean=False,             # orchestrator already centred
        do_sphere=False,           # orchestrator already PCA-projected
        do_newton=cfg.get("do_newton", True),
        # --- passed through to the AMICATorchNG constructor ---
        # pamica's own defaults. Do not substitute another implementation's
        # values here: see the module docstring, it costs ~0.13 of matched row
        # correlation and reads as pamica being inaccurate.
        newt_start=cfg.get("newt_start", 20),
        newt_ramp=cfg.get("newt_ramp", 10),
        rho0=cfg.get("rho0", 1.5),
        minrho=cfg.get("minrho", 1.0),
        maxrho=cfg.get("maxrho", 2.0),
        rholrate=cfg.get("rholrate", 0.05),
        invsigmin=cfg.get("invsigmin", 1e-4),
        invsigmax=cfg.get("invsigmax", 1000.0),
        doscaling=cfg.get("doscaling", True),
        seed=cfg.get("seed", 0),
        dtype=torch.float64,
    )
    elapsed = time.perf_counter() - t0

    # Torch device peak: bytes in live tensors (max_memory_allocated = true demand, NOT the
    # cached/reserved pool). The caching allocator stays ON so this counter is tracked.
    peak_vram_gb = None
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    ll = [float(v) for v in np.asarray(model.ll_history_, dtype=float).ravel()]
    ll_final = float(model.final_ll_) if model.final_ll_ is not None else (
        ll[-1] if ll else float("nan"))

    # pamica's stop_reason vocabulary is max_iter / lrate_floor / nan_ll /
    # singular_ll. Its `converged_` attribute means "not degenerate" -- i.e. not
    # nan_ll or singular_ll -- which is NOT the sense the stopping-status table
    # uses. There, "converged before the cap" means the stopping criterion fired
    # before the iteration budget ran out, so derive it from stop_reason and do
    # not pass `converged_` through under that name.
    stop_reason = model.stop_reason_
    usable = bool(model.converged_)
    converged_before_cap = usable and stop_reason != "max_iter"

    # A degenerate fit holds non-finite parameters and pamica refuses to return
    # an unmixing matrix for it (its issue #50). Report that as a result rather
    # than dying with a traceback, so one bad subject does not take down an
    # array job and the reason survives into the aggregate.
    if not usable:
        write_result(args.output, {
            "implementation": "pamica_torch",
            "error": f"degenerate fit (stop_reason={stop_reason!r})",
            "n_components": int(n_comp),
            "n_samples": int(n_samples),
            "max_iter": cfg["max_iter"],
            "fit_time_s": float(elapsed),
            "ll_history": ll,
            "n_iter": int(len(ll)),
            "stop_reason": stop_reason,
            "device": device,
            "pamica_version": _pamica_version(),
        })
        return

    W = np.asarray(model.get_unmixing_matrix(model_idx=0), dtype=float)

    peak = peak_rss_gb()
    out = {
        "implementation": "pamica_torch",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak,
        "baseline_rss_gb": baseline,
        "delta_rss_gb": peak - baseline,
        "peak_vram_gb": peak_vram_gb,
        "nvml_peak_vram_gb": nvml_peak_vram_gb,
        "ll_final": ll_final,
        "ll_history": ll,
        "W": W.tolist(),
        "device": device,
        "dtype": "float64",
        "n_iter": int(len(ll)),
        # --- pamica-specific, beyond RESULT_KEYS ---
        # ll_final is the kept-best model; this is where the trajectory ended.
        # They differ only when keep_best actually restored an earlier iterate.
        "ll_last_iterate": ll[-1] if ll else None,
        "stop_reason": stop_reason,
        "converged_before_cap": converged_before_cap,
        "pamica_version": _pamica_version(),
        # pamica DOES take the seed and re-seeds per sweep.
        "seed_respected": True,
        "requested_seed": cfg.get("seed", 0),
        # Effective hyperparameters. NB: unlike pyamica/scott's frozen literals,
        # pamica's constants are cfg.get(...)-overridable — a --config that sets
        # e.g. invsigmin changes pamica only. Serialized so that asymmetry is
        # visible in the row rather than implicit in the source.
        "effective_config": {
            "n_mix": cfg.get("n_mix", 3), "max_iter": cfg["max_iter"],
            "lrate": cfg.get("lrate", 0.1), "do_newton": cfg.get("do_newton", True),
            "newt_start": cfg.get("newt_start", 20), "newt_ramp": cfg.get("newt_ramp", 10),
            "rho0": cfg.get("rho0", 1.5), "minrho": cfg.get("minrho", 1.0),
            "maxrho": cfg.get("maxrho", 2.0), "rholrate": cfg.get("rholrate", 0.05),
            "invsigmin": cfg.get("invsigmin", 1e-4), "invsigmax": cfg.get("invsigmax", 1000.0),
            "doscaling": cfg.get("doscaling", True), "seed": cfg.get("seed", 0),
        },
    }
    write_result(args.output, out)


def _pamica_version() -> str:
    """Recorded per run: the parity and performance claims are version-specific."""
    try:
        from importlib.metadata import version
        return version("pamica")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
