#!/usr/bin/env python3
"""Aggregate the iteration-ladder cells into a tidy CSV for the time-vs-iterations plot.

Reads the per-runner result JSONs the orchestrator wrote under each device root:
    <root>/c<CHUNK>_i<MAXITER>/<impl>_<sub-NN>_seed<K>_result.json
and emits one row per (device, impl, subject, chunk, max_iter) with total fit_time_s.

Usage (after pulling results locally, or on the cluster):
    python aggregate_ladder.py \
        --gpu-root /scratch/yorguin/iter_ladder/gpu \
        --cpu-root /scratch/yorguin/iter_ladder/cpu \
        --out iter_ladder_data.csv
Missing roots are skipped, so you can aggregate one device at a time.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median

CELL = re.compile(r"^c(?P<chunk>\d+)_i(?P<iter>\d+)(?:_r(?P<rep>\d+))?$")
FILE = re.compile(r"^(?P<impl>.+)_(?P<subj>sub-\d+)_seed(?P<seed>\d+)_result\.json$")


def summarize_nodemon(mondir: Path, subj: str, impl: str, chunk: int, it: int, rep) -> dict:
    """Mean/max co-tenant cores + mean node busy% from this cell's contention trace(s)."""
    if not mondir or not mondir.exists():
        return {}
    tag = f"{subj}_{impl}_c{chunk}_i{it}" + (f"_r{rep}" if rep is not None else "")
    cc, busy, load = [], [], []
    for f in mondir.glob(tag + "_*.csv"):
        try:
            for r in csv.DictReader(f.open()):
                cc.append(float(r["cotenant_cores"]))
                busy.append(float(r["node_cpu_busy_pct"]))
                load.append(float(r["loadavg1"]))
        except Exception:
            continue
    if not cc:
        return {}
    return {"cotenant_cores_mean": round(sum(cc) / len(cc), 1), "cotenant_cores_max": max(cc),
            "node_busy_mean": round(sum(busy) / len(busy), 1),
            "load1_mean": round(sum(load) / len(load), 2)}


def scan(root: Path, device: str, rows: list, itrows: list):
    if not root or not root.exists():
        print(f"[skip] {device} root not found: {root}")
        return 0
    mondir = root / "nodemon"
    n = 0
    for cell_dir in sorted(root.glob("c*_i*")):
        m = CELL.match(cell_dir.name)
        if not m:
            continue
        chunk, it = int(m["chunk"]), int(m["iter"])
        rep = int(m["rep"]) if m["rep"] else None
        for f in sorted(cell_dir.glob("*_result.json")):
            fm = FILE.match(f.name)
            if not fm:
                continue
            impl, subj, seed = fm["impl"], fm["subj"], int(fm["seed"])
            try:
                d = json.loads(f.read_text())
            except Exception as e:
                print(f"[warn] unreadable {f}: {e}")
                continue
            ft = d.get("fit_time_s")
            status = "ok" if isinstance(ft, (int, float)) else (d.get("error") or "no_fit_time")
            row = {
                "device": device, "impl": impl, "subject": subj,
                "chunk": chunk, "max_iter": it, "rep": rep if rep is not None else 1, "seed": seed,
                "fit_time_s": ft if isinstance(ft, (int, float)) else "",
                "status": status,
                "cotenant_cores_mean": "", "cotenant_cores_max": "",
                "node_busy_mean": "", "load1_mean": "",
            }
            row.update(summarize_nodemon(mondir, subj, impl, chunk, it, rep))
            rows.append(row)
            # bonus: native per-iteration times (jamica), if the runner persisted them
            itimes = d.get("iteration_times") or (d.get("convergence") or {}).get("iteration_times")
            if isinstance(itimes, list) and itimes:
                cum = 0.0
                for i, dt in enumerate(itimes, 1):
                    try:
                        cum += float(dt)
                    except (TypeError, ValueError):
                        continue
                    itrows.append({"device": device, "impl": impl, "subject": subj,
                                   "chunk": chunk, "max_iter": it, "iteration": i,
                                   "iter_time_s": float(dt), "cum_time_s": cum})
            n += 1
    print(f"[{device}] {n} result files under {root}")
    return n


def _cc(r):
    """co-tenant cores for a row; missing/blank (e.g. GPU, no sampler) counts as quiet=0."""
    v = r.get("cotenant_cores_mean")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pct(vals, q):
    if not vals:
        return ""
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 2)


def summarize(rows, quiet_max):
    """Per (device, impl, max_iter): both aggregation modes side by side, so the plot/report can
    pick one without re-running. 'all' = every rep (median/mean/IQR); 'quiet' = only reps the
    sampler saw run on a lightly-loaded node (cotenant_cores <= quiet_max), then min/median."""
    groups = {}
    for r in rows:
        if r["status"] != "ok" or r["fit_time_s"] == "":
            continue
        groups.setdefault((r["device"], r["impl"], int(r["max_iter"])), []).append(r)
    out = []
    for (dev, impl, it), rs in sorted(groups.items()):
        t = [float(r["fit_time_s"]) for r in rs]
        quiet = [float(r["fit_time_s"]) for r in rs if _cc(r) <= quiet_max]
        out.append({
            "device": dev, "impl": impl, "max_iter": it,
            "n": len(t), "t_min": round(min(t), 2), "t_median": round(median(t), 2),
            "t_mean": round(mean(t), 2), "t_p25": _pct(t, .25), "t_p75": _pct(t, .75),
            "n_quiet": len(quiet),
            "t_quiet_min": round(min(quiet), 2) if quiet else "",
            "t_quiet_median": round(median(quiet), 2) if quiet else "",
            "cotenant_cores_mean": round(mean([_cc(r) for r in rs]), 1),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-root", type=Path, default=Path("/scratch/yorguin/iter_ladder/gpu"))
    ap.add_argument("--cpu-root", type=Path, default=Path("/scratch/yorguin/iter_ladder/cpu"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "iter_ladder_data.csv")
    ap.add_argument("--summary-out", type=Path,
                    default=Path(__file__).resolve().parent / "iter_ladder_summary.csv")
    ap.add_argument("--quiet-cotenant-max", type=float, default=8.0,
                    help="a rep is 'quiet' if its mean co-tenant cores <= this (default 8 ~ near-idle)")
    args = ap.parse_args()

    rows, itrows = [], []
    scan(args.gpu_root, "gpu", rows, itrows)
    scan(args.cpu_root, "cpu", rows, itrows)

    rows.sort(key=lambda r: (r["device"], r["impl"], r["subject"], r["max_iter"], r.get("rep", 1)))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "impl", "subject", "chunk", "max_iter", "rep",
                                          "seed", "fit_time_s", "status", "cotenant_cores_mean",
                                          "cotenant_cores_max", "node_busy_mean", "load1_mean"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    summ = summarize(rows, args.quiet_cotenant_max)
    with args.summary_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["device", "impl", "max_iter", "n", "t_min", "t_median",
                                          "t_mean", "t_p25", "t_p75", "n_quiet", "t_quiet_min",
                                          "t_quiet_median", "cotenant_cores_mean"])
        w.writeheader(); w.writerows(summ)
    print(f"wrote {args.summary_out} ({len(summ)} summary rows; quiet = cotenant_cores <= "
          f"{args.quiet_cotenant_max:g})")

    if itrows:
        itp = args.out.with_name(args.out.stem + "_periter.csv")
        with itp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["device", "impl", "subject", "chunk",
                                              "max_iter", "iteration", "iter_time_s", "cum_time_s"])
            w.writeheader(); w.writerows(itrows)
        print(f"wrote {itp} ({len(itrows)} per-iteration rows)")


if __name__ == "__main__":
    main()
