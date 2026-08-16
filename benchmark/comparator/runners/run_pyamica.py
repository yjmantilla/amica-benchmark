"""Runner for DerAndereJohannes/pyamica (PyTorch).

Pyamica's AMICA accepts (T, n_components) tensors and applies its own
sphering when do_sphere=True. Our orchestrator pre-PCA-projects, so we
pass do_sphere=False and feed already-whitened data.
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
    from pyamica import AMICA

    device = os.environ.get("TORCH_DEVICE", "cpu")
    # NVML post-init floor (harness-only): force the CUDA context, read whole-GPU used BEFORE
    # data/model are on device. peak - floor = pool + lazily-loaded libs + live tensors.
    _use_nvml = os.environ.get("AMICA_NVML_CROSSCHECK", "0") == "1" and device == "cuda"
    nvml_post_init_gb = None
    if _use_nvml and torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        nvml_post_init_gb = nvml_used_gb(True)
    Xt = torch.from_numpy(X.T).to(device)  # (n_samples, n_components)
    # Iteration-matched mode: pyamica has THREE stop families (use_min_dll, use_grad_norm, minlrate);
    # disable all so the fit runs the full max_iter. See the earlystop-feasibility panel.
    _disable_es = os.environ.get("AMICA_DISABLE_EARLYSTOP", "0") == "1"
    _es_kw = dict(use_min_dll=False, use_grad_norm=False, minlrate=0.0, min_nd=0.0) if _disable_es else {}
    model = AMICA(
        n_components=n_comp,
        n_models=1,
        n_mix=cfg.get("n_mix", 3),
        max_iter=cfg["max_iter"],
        lrate=cfg.get("lrate", 0.1),
        lrate0=cfg.get("lrate", 0.1),
        do_newton=cfg.get("do_newton", True),
        newt_start=50,
        newt_ramp=10,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
        rholrate=0.05,
        invsigmin=1e-8,
        invsigmax=100.0,
        do_sphere=False,
        doscaling=True,
        verbose=False,
        dtype=torch.float64,
        device=device,
        fix_init=True,
        # Optional chunk-size override for the chunk-size study (unset -> pyamica's
        # default full batch). See results/xperf_chunksize/.
        **({"chunk_t": int(os.environ["AMICA_PYAMICA_CHUNK"])}
           if os.environ.get("AMICA_PYAMICA_CHUNK") else {}),
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

    W = model.W_[0].cpu().numpy()
    ll = model.LL_.cpu().numpy().tolist()

    peak = peak_rss_gb()
    out = {
        "implementation": "pyamica_torch",
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
        "W": W.tolist(),
        "device": device,
        "dtype": "float64",
        "n_iter": int(len(ll)),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
