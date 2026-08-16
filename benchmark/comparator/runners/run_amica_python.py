"""Runner for amica-python (Sina's, JAX or NumPy depending on AMICA_NO_JAX).

Reads (n_components, n_samples) from --input. Sina's amica-python expects
data in (n_channels=n_components, n_samples) shape since the orchestrator
already PCA-projected.
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
    cgroup_peak_gb,
    peak_rss_gb,
    start_nvml_sampler,
    stop_nvml_sampler,
    nvml_used_gb,
    write_result,
)


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape

    no_jax = os.environ.get("AMICA_NO_JAX", "0") == "1"
    # chunk_size from env: "auto" (VRAM/RAM-aware) or an integer; unset -> None (full-batch).
    _chunk_env = os.environ.get("AMICA_CHUNK_SIZE", "").strip()
    if _chunk_env.lower() == "auto":
        chunk_size = "auto"
    elif _chunk_env:
        chunk_size = int(_chunk_env)
    else:
        chunk_size = None
    if no_jax:
        impl = "amica_python_numpy"
    elif chunk_size is not None:
        impl = "amica_python_jax_chunked"
    else:
        impl = "amica_python_jax"

    # The package was renamed amica_python -> amica. Only the cluster venvs still
    # carry the old name, from editable installs that predate the rename, so
    # importing it unconditionally meant this runner could measure the archived
    # checkout and not the package anyone can install today. Prefer the current
    # name and keep the old one working, so both an existing cluster environment
    # and a fresh `pip install amica` are measurable.
    try:
        from amica import Amica, AmicaConfig
    except ImportError:  # pragma: no cover - exercised by the legacy venvs
        from amica_python import Amica, AmicaConfig

    # Report the device JAX actually placed work on (not a hardcoded label).
    device = "cpu"
    if not no_jax:
        try:
            import jax
            device = "gpu" if any(
                getattr(d, "platform", "") in ("gpu", "cuda", "rocm")
                for d in jax.devices()
            ) else "cpu"
        except Exception:
            device = "cpu"

    # NVML post-init floor (harness-only): force the XLA/CUDA context, read whole-GPU used
    # BEFORE the model/data are on device. peak - floor = XLA pool + executables + live tensors.
    _use_nvml = (os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1"
                 and not no_jax and device == "gpu")
    nvml_post_init_gb = None
    if _use_nvml:
        try:
            import jax.numpy as jnp
            jnp.zeros(1).block_until_ready()
            nvml_post_init_gb = nvml_used_gb(True)
        except Exception:
            nvml_post_init_gb = None

    # Iteration-matched mode: disable every early-stop so the fit runs the full max_iter.
    # jamica has TWO stops -- the ΔLL/patience path (use_min_dll) AND an ungated lrate<=minlrate
    # exit -- so both must be neutralised. See NOTES_measurement.md / the earlystop-feasibility panel.
    _disable_es = os.environ.get("AMICA_DISABLE_EARLYSTOP", "0") == "1"
    _es_kw = dict(use_min_dll=False, minlrate=0.0) if _disable_es else {}
    config = AmicaConfig(
        max_iter=cfg["max_iter"],
        num_mix_comps=cfg.get("n_mix", 3),
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        do_sphere=False,    # already PCA-projected by orchestrator
        do_mean=False,
        chunk_size=chunk_size,   # None = full-batch; "auto"/int = chunked (lower peak memory)
        **_es_kw,
    )
    model = Amica(config, random_state=cfg.get("seed", 0))

    _nvml = start_nvml_sampler(_use_nvml)
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    result = model.fit(X)
    elapsed = time.perf_counter() - t0

    # Block on the fit's async device work before reading the memory high-water mark.
    if not no_jax and device == "gpu":
        try:
            import jax
            jax.block_until_ready((result.unmixing_matrix_white_, result.log_likelihood))
        except Exception:
            pass

    # GPU device peak: XLA's high-water of live-buffer bytes (peak_bytes_in_use), prealloc disabled.
    # Do NOT fall back to bytes_in_use (an INSTANTANEOUS live count) -- storing that under a "peak"
    # field was a bug that produced inconsistent VRAM numbers. If peak_bytes_in_use is absent, leave
    # peak_vram_gb=None (metric unavailable). NVML (nvml_peak_vram_gb) is the framework-NEUTRAL
    # cross-check; peak_bytes_in_use is a JAX-allocator-local diagnostic, not directly comparable to
    # torch max_memory_allocated. Raw stats saved for provenance.
    peak_vram_gb = None
    vram_stats = None
    if not no_jax and device == "gpu":
        try:
            import jax
            gpus = [d for d in jax.devices()
                    if getattr(d, "platform", "") in ("gpu", "cuda", "rocm")]
            if gpus:
                stats = gpus[0].memory_stats() or {}
                vram_stats = {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in stats.items()}
                pk = stats.get("peak_bytes_in_use")   # require the true peak; NO fallback
                if pk is not None:
                    peak_vram_gb = float(pk) / 1024 ** 3
        except Exception:
            peak_vram_gb = None
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    W = np.asarray(result.unmixing_matrix_white_)
    ll_history = np.asarray(result.log_likelihood).tolist()

    # best-effort native per-iteration wall times (used by the iteration-ladder plots for a
    # fine-grained curve; absent on older jamica builds, in which case it stays None).
    iteration_times = None
    for _src in (result, getattr(result, "convergence", None)):
        if _src is None:
            continue
        for _attr in ("iteration_times", "iter_times", "per_iter_times"):
            _v = getattr(_src, _attr, None)
            if _v is not None:
                try:
                    iteration_times = [float(x) for x in np.asarray(_v).ravel().tolist()]
                except Exception:
                    iteration_times = None
                break
        if iteration_times is not None:
            break

    peak = peak_rss_gb()
    out = {
        "implementation": impl,
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak,
        "baseline_rss_gb": baseline,
        "delta_rss_gb": peak - baseline,
        "cgroup_peak_gb": cgroup_peak_gb(),
        "peak_vram_gb": peak_vram_gb,
        "peak_vram_reserved_gb": None,          # JAX has no allocator-reserved concept (torch does)
        "nvml_peak_vram_gb": nvml_peak_vram_gb,  # framework-neutral cross-check (whole-GPU used)
        "nvml_post_init_gb": nvml_post_init_gb,  # whole-GPU used after init, before model/data (context floor)
        "earlystop_disabled": _disable_es,       # AMICA_DISABLE_EARLYSTOP: iteration-matched mode
        "vram_stats": vram_stats,                # raw jax memory_stats() for provenance
        "ll_final": float(ll_history[-1]) if ll_history else float("nan"),
        "ll_history": ll_history,
        "iteration_times": iteration_times,
        "W": W.tolist(),
        "device": device,
        "dtype": "float64",
        "n_iter": int(result.n_iter),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
