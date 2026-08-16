"""Runner for scott-huberty/amica-python (PyTorch, sklearn-style).

The amica-python `amica.AMICA` is sklearn-compatible: fit(X) where X is
(n_samples, n_features). We pass already-PCA-projected data and disable
its internal whitening via whiten=None / batching=None.
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

    import torch
    from amica import AMICA  # amica-python sklearn-style class

    device = os.environ.get("TORCH_DEVICE", "cpu")
    # sklearn fits on (n_samples, n_features); transpose
    Xt = X.T

    # NVML post-init floor (harness-only, no impl change): force the CUDA context, then read
    # whole-GPU used BEFORE the model/data are on device. peak - this floor = allocator pool +
    # lazily-loaded cuBLAS/cuDNN + live tensors. See results/xperf_chunksize/ memory note.
    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    nvml_post_init_gb = None
    if _use_nvml and torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        nvml_post_init_gb = nvml_used_gb(True)

    # Iteration-matched mode: amica-python's use_min_dll is a module constant that raises if flipped;
    # its single public `tol` sets BOTH the ΔLL and grad-norm thresholds (min_dll = min_nd = tol), so a
    # large-negative tol makes neither stop fire -> runs the full max_iter. See the earlystop panel.
    _disable_es = os.environ.get("AMICA_DISABLE_EARLYSTOP", "0") == "1"
    _es_kw = dict(tol=-1e30) if _disable_es else {}
    model = AMICA(
        n_components=n_comp,
        n_mixtures=cfg.get("n_mix", 3),
        device=device,
        n_models=1,
        mean_center=False,
        whiten=None,                     # already projected
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        newt_start=50,
        random_state=cfg.get("seed", 0),
        verbose=0,
        # Optional chunk-size override for the chunk-size study (unset -> scott's
        # default full batch). See results/xperf_chunksize/.
        **({"batch_size": int(os.environ["AMICA_SCOTT_BATCH"])}
           if os.environ.get("AMICA_SCOTT_BATCH") else {}),
        **_es_kw,
    )

    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    _nvml = start_nvml_sampler(_use_nvml)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = baseline_rss_gb()
    t0 = time.perf_counter()
    model.fit(Xt)
    elapsed = time.perf_counter() - t0

    # Torch device peak: bytes in live tensors (max_memory_allocated = true demand, NOT the
    # cached/reserved pool). The caching allocator stays ON so this counter is tracked.
    peak_vram_gb = None
    peak_vram_reserved_gb = None
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_vram_reserved_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
    nvml_peak_vram_gb = stop_nvml_sampler(_nvml)

    # Scott's sklearn-style attributes: components_ (unmixing), ll_ (per-iter), n_iter_
    W = np.asarray(model.components_)
    if W.ndim == 3:
        W = W[0]
    ll = np.asarray(model.ll_).flatten().tolist()
    n_iter = int(model.n_iter_)

    peak = peak_rss_gb()
    out = {
        "implementation": "scott_huberty_torch",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": peak,
        "baseline_rss_gb": baseline,
        "delta_rss_gb": peak - baseline,
        "cgroup_peak_gb": cgroup_peak_gb(),
        "peak_vram_gb": peak_vram_gb,
        "peak_vram_reserved_gb": peak_vram_reserved_gb,
        "nvml_peak_vram_gb": nvml_peak_vram_gb,
        "nvml_post_init_gb": nvml_post_init_gb,
        "earlystop_disabled": _disable_es,
        "ll_final": float(ll[-1]) if ll else float("nan"),
        "ll_history": ll,
        "W": W.tolist() if W is not None else None,
        "device": device,
        "dtype": str(np.asarray(W).dtype) if W is not None else "float64",
        "n_iter": n_iter,
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
