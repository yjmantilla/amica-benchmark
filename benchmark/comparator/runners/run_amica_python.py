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
    peak_rss_gb,
    start_nvml_sampler,
    stop_nvml_sampler,
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

    config = AmicaConfig(
        max_iter=cfg["max_iter"],
        num_mix_comps=cfg.get("n_mix", 3),
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        do_sphere=False,    # already PCA-projected by orchestrator
        do_mean=False,
        chunk_size=chunk_size,   # None = full-batch; "auto"/int = chunked (lower peak memory)
    )
    model = Amica(config, random_state=cfg.get("seed", 0))

    _use_nvml = (os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1"
                 and not no_jax and device == "gpu")
    _nvml = start_nvml_sampler(_use_nvml)
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    result = model.fit(X)
    elapsed = time.perf_counter() - t0

    # GPU device peak: XLA's high-water of bytes handed to live buffers
    # (peak_bytes_in_use), measured with prealloc disabled by the orchestrator.
    peak_vram_gb = None
    if not no_jax and device == "gpu":
        try:
            import jax
            gpus = [d for d in jax.devices()
                    if getattr(d, "platform", "") in ("gpu", "cuda", "rocm")]
            if gpus:
                stats = gpus[0].memory_stats() or {}
                vram_bytes = stats.get("peak_bytes_in_use", stats.get("bytes_in_use"))
                if vram_bytes is not None:
                    peak_vram_gb = float(vram_bytes) / 1024 ** 3
        except Exception:
            peak_vram_gb = None
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    W = np.asarray(result.unmixing_matrix_white_)
    ll_history = np.asarray(result.log_likelihood).tolist()

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
        "peak_vram_gb": peak_vram_gb,
        "nvml_peak_vram_gb": nvml_peak_vram_gb,
        "ll_final": float(ll_history[-1]) if ll_history else float("nan"),
        "ll_history": ll_history,
        "W": W.tolist(),
        "device": device,
        "dtype": "float64",
        "n_iter": int(result.n_iter),
        # amica-python takes random_state, so the seed is honored per sweep.
        "seed_respected": True,
        "requested_seed": cfg.get("seed", 0),
        "effective_config": {
            "max_iter": cfg["max_iter"], "num_mix_comps": cfg.get("n_mix", 3),
            "lrate": cfg.get("lrate", 0.1), "do_newton": cfg.get("do_newton", True),
            "do_sphere": False, "do_mean": False, "chunk_size": chunk_size,
            "random_state": cfg.get("seed", 0),
        },
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
