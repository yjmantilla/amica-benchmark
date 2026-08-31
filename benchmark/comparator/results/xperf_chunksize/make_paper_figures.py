#!/usr/bin/env python3
"""Generate publication figures for the AMICA chunk-size / iteration benchmark (ds004505).

Reproduces every chart in the two HTML reports (xperf_chunk_report / _tldr) as
vector PDF + raster PNG, reading ONLY the committed CSVs that gen_report.py exports:

    chunk_sweep_data.csv        fit time + memory vs chunk (GPU 3000-iter, CPU 250-iter)
    chunk_sweep_fullbatch.csv   the full-batch (one-pass) point, kept off the chunk axis
    iter_ladder_data.csv        fit time + log-likelihood vs iterations (fixed chunk 65536)
    gpu_memory_decomp_data.csv  GPU memory: active <= reserved <= total, vs chunk
    subject_durations.csv       per-subject recording lengths (for the duration figure)

Figures written to ./figures/:
    fig_chunk_time_memory.{pdf,png}   2x2  GPU/CPU x fit-time/peak-memory vs chunk
    fig_iterations.{pdf,png}          2x2  GPU/CPU x fit-time/convergence vs iterations
    fig_gpu_memory_decomp.{pdf,png}   2x2  active / reserved / total / (total/active) vs chunk
    fig_signal_duration.{pdf,png}     per-subject recording length vs the tested chunk sizes

No network, no seaborn: matplotlib + numpy only. Run:  python make_paper_figures.py
"""
import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- house style (matches scripts/paper/generate_publication_figures.py) ----
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

# impl identity — colours + labels are the same as the HTML reports
COLOR = {"jamica": "#6366f1", "pamica": "#d97706", "pyamica": "#0d9488",
         "amica_python": "#e11d48", "fortran": "#111827"}
LABEL = {"jamica": "jamica", "pamica": "pAMICA", "pyamica": "pyamica",
         "amica_python": "amica-python", "fortran": "Fortran (1 thread)"}
GPU_IMPLS = ["jamica", "amica_python", "pamica", "pyamica"]            # 4 parallel impls (no GPU Fortran)
CPU_IMPLS = ["jamica", "amica_python", "pamica", "pyamica", "fortran"]  # + single-threaded reference

CHUNKS = [1024, 4096, 16384, 65536, 262144, 524288, 1048576]
CLAB = {1024: "1K", 4096: "4K", 16384: "16K", 65536: "64K",
        262144: "262K", 524288: "512K", 1048576: "1M"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_chunk():
    """dataset -> impl -> chunk(int or 'fullbatch') -> value."""
    d = {}
    for fn in ("chunk_sweep_data.csv", "chunk_sweep_fullbatch.csv"):
        with open(os.path.join(HERE, fn)) as fh:
            for r in csv.DictReader(fh):
                v = _f(r["value"])
                if v is None:
                    continue
                ch = r["chunk"]
                if ch != "fullbatch":
                    try:
                        ch = int(ch)
                    except ValueError:
                        continue   # skip non-axis rows (e.g. pAMICA block_size sensitivity labels)
                d.setdefault(r["dataset"], {}).setdefault(r["impl"], {})[ch] = v
    return d


def load_ladder():
    """device -> impl -> sorted [(iters, fit_s, ll)]."""
    d = {}
    with open(os.path.join(HERE, "iter_ladder_data.csv")) as fh:
        for r in csv.DictReader(fh):
            d.setdefault(r["device"], {}).setdefault(r["impl"], []).append(
                (int(r["iters"]), _f(r["fit_s"]), _f(r["ll"])))
    for dev in d:
        for im in d[dev]:
            d[dev][im].sort()
    return d


def load_decomp():
    """impl -> chunk(int) -> (active, reserved, total, ratio)."""
    d = {}
    with open(os.path.join(HERE, "gpu_memory_decomp_data.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                c = int(r["chunk"])
            except ValueError:
                continue
            d.setdefault(r["impl"], {})[c] = (
                _f(r["active_gib"]), _f(r["reserved_gib"]),
                _f(r["total_gib"]), _f(r["ratio_total_over_active"]))
    return d


def load_durations():
    ns = []
    with open(os.path.join(HERE, "subject_durations.csv")) as fh:
        for r in csv.DictReader(fh):
            ns.append(int(r["n_samples"]))
    return ns


CH = load_chunk()
LAD = load_ladder()
DEC = load_decomp()
DUR = load_durations()

X = {c: math.log2(c) for c in CHUNKS}
XFB = math.log2(1048576) + 1.15   # off-axis slot for the full-batch marker


def _chunk_axis(ax, show_fb=True):
    ax.set_xticks([X[c] for c in CHUNKS] + ([XFB] if show_fb else []))
    ax.set_xticklabels([CLAB[c] for c in CHUNKS] + (["FB"] if show_fb else []),
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("chunk / block size (samples)")
    if show_fb:
        ax.axvline((X[1048576] + XFB) / 2, color="#c9ccd6", lw=0.8, ls=":", zorder=0)


def _series(ax, impls, ds, band_lo=None, band_hi=None, logy=False, fb_ds=None):
    """Plot value-vs-chunk lines for `impls` from dataset `ds`, with optional band + full-batch marker."""
    for im in impls:
        pts = CH.get(ds, {}).get(im, {})
        xs = [X[c] for c in CHUNKS if c in pts]
        ys = [pts[c] for c in CHUNKS if c in pts]
        if not xs:
            continue
        ax.plot(xs, ys, "-o", color=COLOR[im], ms=3.2, lw=1.5, label=LABEL[im], zorder=3)
        if band_lo and band_hi:
            lo = CH.get(band_lo, {}).get(im, {})
            hi = CH.get(band_hi, {}).get(im, {})
            bx = [X[c] for c in CHUNKS if c in lo and c in hi]
            bl = [lo[c] for c in CHUNKS if c in lo and c in hi]
            bh = [hi[c] for c in CHUNKS if c in lo and c in hi]
            if bx:
                ax.fill_between(bx, bl, bh, color=COLOR[im], alpha=0.12, lw=0, zorder=1)
        fb = pts.get("fullbatch")
        if fb is not None:
            ax.plot([XFB], [fb], marker="*", color=COLOR[im], ms=8, lw=0, zorder=4)
    if logy:
        ax.set_yscale("log")


def _legend(fig, impls, show_fb=True, ncol=None):
    handles = [Line2D([0], [0], color=COLOR[im], marker="o", ms=4, lw=1.6, label=LABEL[im])
               for im in impls]
    if show_fb:
        handles.append(Line2D([0], [0], color="#555", marker="*", ms=8, lw=0, label="full-batch (FB)"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote figures/%s.pdf + .png" % name)


# ============================ Figure 1: chunk size ============================
def fig_chunk():
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (a00, a01), (a10, a11) = ax

    _series(a00, GPU_IMPLS, "gpu_fit_s_median", "gpu_fit_s_p25", "gpu_fit_s_p75", logy=True)
    a00.set_title("GPU · fit time vs chunk"); a00.set_ylabel("fit time (s, log)")

    _series(a01, GPU_IMPLS, "gpu_vram_gib_nvml")
    for gib, lab in [(24, "24 GiB"), (40, "40 GiB"), (80, "80 GiB")]:
        a01.axhline(gib, color="#9aa0ad", lw=0.8, ls="--", zorder=0)
        a01.text(X[1024], gib + 0.6, lab, fontsize=7, color="#7a8090")
    a01.set_title("GPU · peak memory vs chunk"); a01.set_ylabel("GPU memory used (GiB)")
    a01.set_ylim(0, 84)

    _series(a10, CPU_IMPLS, "cpu_fit_s_bysubj_median", "cpu_fit_s_bysubj_p25", "cpu_fit_s_bysubj_p75", logy=True)
    a10.set_title("CPU · fit time vs chunk"); a10.set_ylabel("fit time (s, log)")

    _series(a11, CPU_IMPLS, "cpu_rss_gib_median")
    a11.set_title("CPU · peak memory vs chunk"); a11.set_ylabel("peak RSS (GiB)")

    for a in (a00, a01, a10, a11):
        _chunk_axis(a)
        a.grid(axis="y", color="#eceef2", lw=0.7, zorder=0)
    fig.suptitle("Fit time and peak memory vs the batch/chunk-size setting  ·  ds004505 (real EEG), 25 subjects",
                 fontsize=11, y=0.995)
    _legend(fig, CPU_IMPLS)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "fig_chunk_time_memory")


# ============================ Figure 2: iterations ============================
def fig_iters():
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (a00, a01), (a10, a11) = ax

    def plot(ax_, dev, impls, idx, ylab, title):
        for im in impls:
            pts = LAD.get(dev, {}).get(im, [])
            xs = [p[0] for p in pts if p[idx] is not None]
            ys = [p[idx] for p in pts if p[idx] is not None]
            if xs:
                ax_.plot(xs, ys, "-o", color=COLOR[im], ms=3.2, lw=1.5, label=LABEL[im])
        ax_.set_xlabel("iterations"); ax_.set_ylabel(ylab); ax_.set_title(title)
        ax_.grid(color="#eceef2", lw=0.7, zorder=0)

    plot(a00, "gpu", GPU_IMPLS, 1, "fit time (s)", "GPU · fit time vs iterations")
    plot(a01, "gpu", GPU_IMPLS, 2, "final log-likelihood", "GPU · convergence vs iterations")
    plot(a10, "cpu", CPU_IMPLS, 1, "fit time (s)", "CPU · fit time vs iterations")
    plot(a11, "cpu", CPU_IMPLS, 2, "final log-likelihood", "CPU · convergence vs iterations")
    a00.set_xlim(0, 3100); a01.set_xlim(0, 3100)
    a10.set_xlim(0, 520); a11.set_xlim(0, 520)

    fig.suptitle("Fit time and convergence vs iterations  ·  fixed chunk 65536  ·  higher log-likelihood is better",
                 fontsize=11, y=0.995)
    _legend(fig, CPU_IMPLS, show_fb=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "fig_iterations")


# ==================== Figure 3: GPU memory decomposition =====================
def fig_decomp():
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (a00, a01), (a10, a11) = ax
    panels = [(a00, 0, "in active use (GiB)", "GPU · memory in active use"),
              (a01, 1, "reserved (GiB)", "GPU · reserved (allocator pool)"),
              (a10, 2, "total GPU memory (GiB)", "GPU · total memory used (NVML)"),
              (a11, 3, "total ÷ active (×)", "GPU · how much 'active' understates total")]
    for ax_, idx, ylab, title in panels:
        for im in GPU_IMPLS:
            pts = DEC.get(im, {})
            xs = [X[c] for c in CHUNKS if c in pts and pts[c][idx] is not None]
            ys = [pts[c][idx] for c in CHUNKS if c in pts and pts[c][idx] is not None]
            if xs:
                ax_.plot(xs, ys, "-o", color=COLOR[im], ms=3.2, lw=1.5, label=LABEL[im])
        _chunk_axis(ax_, show_fb=False)
        ax_.set_ylabel(ylab); ax_.set_title(title)
        ax_.grid(axis="y", color="#eceef2", lw=0.7, zorder=0)
    fig.suptitle("GPU memory decomposition vs chunk  ·  active ≤ reserved ≤ total (NVML)  ·  per-subject median",
                 fontsize=11, y=0.995)
    _legend(fig, GPU_IMPLS, show_fb=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save(fig, "fig_gpu_memory_decomp")


# ======================= Figure 4: signal duration ==========================
def fig_duration():
    ns = np.array(sorted(DUR))
    med = float(np.median(ns))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(1, len(ns) + 1), ns / 1e6, color="#6366f1", alpha=0.55, width=0.8)
    for c in (65536, 262144, 1048576):
        ax.axhline(c / 1e6, color="#e11d48", lw=1.2, ls="--")
        ax.text(len(ns) + 0.8, c / 1e6, "%s (%d%%)" % (CLAB[c], round(100 * c / med)),
                fontsize=7.5, color="#e11d48", va="center", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlabel("subjects, sorted by recording length →")
    ax.set_ylabel("samples (millions)")
    ax.set_title("Per-subject recording length (ds004505, 25 subjects @250 Hz) vs the tested chunk sizes")
    ax.set_xlim(0.3, len(ns) + 5.5)
    ax.grid(axis="y", color="#eceef2", lw=0.7, zorder=0)
    fig.tight_layout()
    save(fig, "fig_signal_duration")


if __name__ == "__main__":
    fig_chunk()
    fig_iters()
    fig_decomp()
    fig_duration()
    print("\nAll figures written to", OUT)
