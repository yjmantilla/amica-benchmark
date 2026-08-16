#!/usr/bin/env python3
"""Post-hoc convergence analysis from EXISTING @3000 GPU traces (zero new GPU).

For each per-cell result JSON (ll_history + fit_time + n_iter), compute the honest
convergence-matched comparisons the harmonized re-runs would target, WITHOUT re-running:
 - time-to-DeltaLL: iterations until the per-iteration LL increase first drops below tau
   (signed test LL[i]-LL[i-1] < tau, matching the impls' own criterion), x measured s/iter.
 - time-to-LL-level: iterations until LL reaches within eps of the per-(subject,chunk) BEST
   final LL across impls -> "time to comparable *quality*", the thing 2b only proxies.
Aggregated BY SUBJECT (median within subject over any reps -> median across subjects).
Usage: posthoc_conv.py <gpu_root> <target_iter> <out_prefix>
"""
import os, sys, json, glob, re, statistics as st, csv, collections
root, target_iter, outpre = sys.argv[1], int(sys.argv[2]), sys.argv[3]
IMPL = {"amica_python_jax_chunked":"jamica","scott_huberty_torch":"amica_python",
        "pyamica_torch":"pyamica","pamica_torch":"pamica"}
CELL = re.compile(r"c(\d+)_i(\d+)$")
TAUS = [1e-6, 1e-7, 1e-8]; EPS = 1e-3

def med(xs): return st.median(xs) if xs else None

# gather per (chunk, subject, impl): ll_history, s_per_iter, ll_final
cells = {}
for jf in glob.glob(os.path.join(root, "*", "*result*.json")):
    m = CELL.match(os.path.basename(os.path.dirname(jf)))
    if not m: continue
    chunk, it = int(m.group(1)), int(m.group(2))
    if it != target_iter: continue
    impl = IMPL.get(json.load(open(jf)).get("implementation")) if False else None
    try: d = json.load(open(jf))
    except Exception: continue
    impl = IMPL.get(d.get("implementation"));
    if impl is None: continue
    sub = re.search(r"sub-(\d+)", os.path.basename(jf)); sub = sub.group(0) if sub else "?"
    h = d.get("ll_history") or []; t = d.get("fit_time_s"); ni = d.get("n_iter")
    if not h or not t or not ni: continue
    cells[(chunk, sub, impl)] = {"h": [float(x) for x in h], "spi": t/ni, "llf": float(h[-1])}

def iters_to_tau(h, tau):
    for i in range(1, len(h)):
        if h[i] - h[i-1] < tau:
            return i
    return None  # never flattened within the trace

def iters_to_level(h, target):
    for i in range(len(h)):
        if h[i] >= target:
            return i + 1
    return None  # never reached that LL within the trace

# per-(chunk,subject) best final LL across impls present
best = {}
for (chunk, sub, impl), c in cells.items():
    best[(chunk, sub)] = max(best.get((chunk, sub), -1e18), c["llf"])

rows = []
per = collections.defaultdict(lambda: collections.defaultdict(list))  # (impl,chunk,metric)->subj->val
for (chunk, sub, impl), c in cells.items():
    tgt = best[(chunk, sub)] - EPS
    lvl_i = iters_to_level(c["h"], tgt)
    per[(impl, chunk, "level_iters")][sub].append(lvl_i if lvl_i else None)
    per[(impl, chunk, "level_time")][sub].append(lvl_i * c["spi"] if lvl_i else None)
    for tau in TAUS:
        ti = iters_to_tau(c["h"], tau)
        per[(impl, chunk, f"tau{tau:.0e}_iters")][sub].append(ti if ti else None)
        per[(impl, chunk, f"tau{tau:.0e}_time")][sub].append(ti * c["spi"] if ti else None)

# aggregate by subject: reached-fraction + median over subjects that reached
agg = {}
for (impl, chunk, metric), subs in per.items():
    vals = [v[0] for v in subs.values()]
    reached = [x for x in vals if x is not None]
    agg[(impl, chunk, metric)] = (med(reached), len(reached), len(vals))

impls = ["jamica", "amica_python", "pamica", "pyamica"]; chunks = sorted({c for (_, c, _) in agg})
with open(outpre + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["impl","chunk","metric","median","n_reached","n_total"])
    for (impl, chunk, metric), (m0, nr, nt) in sorted(agg.items()):
        w.writerow([impl, chunk, metric, ("%.4f" % m0) if m0 is not None else "NA", nr, nt])

# compact console table at the podium chunk (262144)
C = 262144
print("== Post-hoc convergence @ chunk 262144 (from existing @%d traces; EPS=%g) ==" % (target_iter, EPS))
print("%-13s | time-to-LL-level(s)  iters  reached | time-to-DLL<1e-7(s) iters reached" % "impl")
for impl in impls:
    lt = agg.get((impl, C, "level_time"), (None,0,0)); li = agg.get((impl, C, "level_iters"), (None,0,0))
    tt = agg.get((impl, C, "tau1e-07_time"), (None,0,0)); ti = agg.get((impl, C, "tau1e-07_iters"), (None,0,0))
    f = lambda x: ("%.0f" % x) if x is not None else "NA"
    print("%-13s | %-8s %-14s %d/%d | %-8s %-8s %d/%d" % (
        impl, f(lt[0])+"s", f(li[0]), lt[1], lt[2], f(tt[0])+"s", f(ti[0]), tt[1], tt[2]))
print("wrote", outpre + ".csv", "with", len(agg), "rows")
