#!/usr/bin/env python3
"""Time-to-N-iterations plot from the iteration-ladder data.

Total fit time vs iteration cap, one line per implementation, a panel per device.
Across subjects we show the median (line) and min-max band; a dashed linear reference
(a + b*iters through the two largest caps) makes the fixed cost a (intercept, the
compile/setup) and steady per-iter b (slope) legible.

    python plot_time_vs_iter.py --csv iter_ladder_data.csv --out time_vs_iter.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"amica_python_jax": "jamica", "scott_huberty_torch": "scott-huberty",
         "pyamica_torch": "pyamica", "pamica_torch": "pAMICA", "fortran_amica17": "Fortran amica17"}
COLOR = {"amica_python_jax": "#3b5bdb", "scott_huberty_torch": "#e11d48",
         "pyamica_torch": "#0d9488", "pamica_torch": "#d97706", "fortran_amica17": "#7c3aed"}
ORDER = ["amica_python_jax", "scott_huberty_torch", "pyamica_torch", "pamica_torch", "fortran_amica17"]


def _cc(r):
    try:
        return float(r.get("cotenant_cores_mean") or 0.0)
    except ValueError:
        return 0.0


def load(csv_path, filter_quiet=False, quiet_max=8.0):
    # data[device][impl][max_iter] = [fit_time_s, ...] over subjects/reps
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or not r["fit_time_s"]:
                continue
            if filter_quiet and _cc(r) > quiet_max:
                continue
            data[r["device"]][r["impl"]][int(r["max_iter"])].append(float(r["fit_time_s"]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "iter_ladder_data.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "time_vs_iter.png")
    ap.add_argument("--logy", action="store_true", help="log-scale the time axis")
    ap.add_argument("--stat", choices=["median", "mean", "min"], default="median",
                    help="central line: use-all median/mean, or min (least-contended)")
    ap.add_argument("--filter-quiet", action="store_true",
                    help="keep only reps the sampler saw run on a lightly-loaded node")
    ap.add_argument("--quiet-cotenant-max", type=float, default=8.0)
    args = ap.parse_args()

    stat_fn = {"median": np.median, "mean": np.mean, "min": np.min}[args.stat]
    data = load(args.csv, args.filter_quiet, args.quiet_cotenant_max)
    devices = [d for d in ("gpu", "cpu") if d in data]
    if not devices:
        raise SystemExit(f"no usable rows in {args.csv}")

    plt.rcParams.update({"font.size": 10, "axes.facecolor": "white", "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, len(devices), figsize=(5.4 * len(devices), 4.2), squeeze=False)

    for ax, dev in zip(axs[0], devices):
        impls = [im for im in ORDER if im in data[dev]]
        for im in impls:
            caps = sorted(data[dev][im])
            med = np.array([stat_fn(data[dev][im][c]) for c in caps])
            lo = np.array([np.min(data[dev][im][c]) for c in caps])
            hi = np.array([np.max(data[dev][im][c]) for c in caps])
            x = np.array(caps, float)
            c = COLOR[im]
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(x, med, "-o", color=c, lw=1.8, ms=4, label=LABEL[im])
            # dashed a + b*iters reference through the two largest caps
            if len(caps) >= 2:
                (x1, y1), (x2, y2) = (caps[-2], med[-2]), (caps[-1], med[-1])
                b = (y2 - y1) / (x2 - x1)
                a = y2 - b * x2
                xr = np.array([0, caps[-1]], float)
                ax.plot(xr, a + b * xr, ls=":", color=c, lw=1.0, alpha=0.7)
                ax.annotate(f"{b*1000:.0f} ms/it", (caps[-1], med[-1]), color=c,
                            fontsize=7.5, xytext=(4, 0), textcoords="offset points", va="center")
        ax.set_title(f"{dev.upper()} · chunk 65536", fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("iteration cap (max_iter)")
        ax.set_ylabel("total fit time (s)")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    mode = f"{args.stat}" + (f", quiet-only (≤{args.quiet_cotenant_max:g} co-tenant cores)"
                             if args.filter_quiet else ", all reps")
    fig.suptitle(f"Time to reach N iterations — ds004505, per implementation [{mode}] "
                 "(dashed = a + b·iters through top two caps: a=compile/setup, b=steady per-iter)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    # also drop an SVG sibling for crisp embedding
    fig.savefig(args.out.with_suffix(".svg"), bbox_inches="tight")
    print(f"wrote {args.out} and {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
