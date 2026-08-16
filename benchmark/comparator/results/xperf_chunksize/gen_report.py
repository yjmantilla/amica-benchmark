#!/usr/bin/env python3
"""Render the AMICA cross-implementation timing/memory report (self-contained HTML), real data.

Outsider-facing measurement report: fit time, convergence, and peak memory for each Python AMICA
implementation on a real EEG dataset (ds004505), swept across each one's batch/chunk-size setting on
GPU and CPU. Neutral and even-handed — it reports measurements, not a product verdict.

Data provenance (all dicts below are the median of the per-cell result JSONs, cross-checked against
benchmark/comparator/results/xperf_chunksize/raw/chunk_{gpu3000,cpu1000}_{summary,percell}.csv, which
are aggregated straight from the cluster JSONs):
 - GPU @3000 iters: /scratch/yorguin/iter_ladder/gpu/c<chunk>_i3000/  (Trillium H100, 20-25 subj/cell)
 - CPU @1000 iters: /scratch/yorguin/iter_ladder/cpu/c<chunk>_i1000_r<rep>/  (fir, 8 cores, 5 subj x reps)
VRAM headline = NVML whole-GPU 'used' (framework-neutral). jamica = the CHUNKED path
(amica_python_jax_chunked); its full-batch key (amica_python_jax, chunk_size=None) is a *separate*
program shown only in the memory note. Memory values are GiB (bytes / 1024**3).

IMPORTANT MEASUREMENT CAVEAT (see the "How to read this" box in the report): the fit-time comparison is
wall time to a fixed ITERATION BUDGET (GPU 3000 / CPU 1000), not time to an equivalent solution. The
implementations run very different actual iteration counts within that budget (recorded here as
n_iter) and reach final log-likelihoods (ll_final) in a tight but non-identical band. Equal budget is
not equal work and not equal convergence.
"""
import math, os, csv

FULL = 262144                       # the LARGEST TESTED chunk (samples); NOT a full-batch pass.
N_SAMP_MIN, N_SAMP_MAX = 785328, 1364633   # per-subject sample counts (ds004505); 262144 = ~15-33% of a recording
IMPLS = ["jamica", "pamica", "pyamica", "scott"]
LABEL = {"jamica":"jamica","pamica":"pAMICA (sccn)","pyamica":"pyamica","scott":"scott-huberty"}
KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","scott":"batch_size","fortran":"block_size"}
COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","scott":"e15e158","fortran":"665b577"}
COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","scott":"#e11d48","fortran":"#7c3aed"}

# ===== GPU @3000, per-subject median : chunk -> (fit_s, nvml_vram_gib). jamica = chunked path. =====
GPU = {
 "jamica":  {1024:(763.1,5.37),4096:(227.9,5.37),16384:(97.6,5.37),65536:(71.9,5.37),FULL:(61.6,5.37)},
 "pamica":  {1024:(4086.5,1.83),4096:(1043.7,1.89),16384:(370.8,2.13),65536:(313.6,3.09),FULL:(244.7,6.57)},
 "pyamica": {1024:(2414.4,3.05),4096:(608.8,3.05),16384:(365.9,3.05),65536:(339.0,4.46),FULL:(294.0,10.92)},
 "scott":   {1024:(2023.8,1.82),4096:(502.0,1.87),16384:(163.2,2.05),65536:(110.4,2.77),FULL:(79.3,4.88)},
}
# GPU fit-time IQR (p25,p75) across subjects, for the band
GPU_BAND = {
 "jamica":  {1024:(713,777),4096:(215,234),16384:(92,100),65536:(70,76),FULL:(58,63)},
 "pamica":  {1024:(3296,4229),4096:(836,1088),16384:(274,398),65536:(169,325),FULL:(64,267)},
 "pyamica": {1024:(2291,2458),4096:(580,620),16384:(350,376),65536:(321,347),FULL:(276,301)},
 "scott":   {1024:(1872,2395),4096:(485,586),16384:(155,190),65536:(102,127),FULL:(76,100)},
}
# GPU convergence at chunk=262144 (largest tested): ll_final median, actual iterations-run (min-max).
GPU_CONV = {  # impl -> (ll_median, n_iter_min, n_iter_max)
 "jamica":  (-1.101, 2444, 3000), "scott": (-1.100, 784, 1654),
 "pamica":  (-1.120, 151, 3000),  "pyamica": (-1.0995, 3000, 3000),
}
# ===== CPU @1000, median over 5 subj x reps : chunk -> fit_s / rss_gib. jamica = chunked path. =====
CPU_FIT = {
 "jamica":  {1024:1982,4096:2078,16384:2138,65536:2603,FULL:2731},
 "pamica":  {1024:4275,4096:3577,16384:4397,65536:9101,FULL:8694},
 "pyamica": {4096:24301,16384:10173,65536:9307,FULL:11091},
 "scott":   {1024:3208,4096:2787,16384:2809,65536:4138,FULL:4397},
 "fortran": {1024:5627,4096:5301,16384:6049,65536:8222,FULL:7171},
}
CPU_RSS = {
 "jamica":  {1024:2.2,4096:2.2,16384:2.2,65536:3.7,FULL:7.0},
 "pamica":  {1024:1.6,4096:1.7,16384:2.4,65536:2.7,FULL:6.0},
 "pyamica": {4096:1.9,16384:2.7,65536:3.6,FULL:9.8},
 "scott":   {1024:2.0,4096:2.0,16384:2.0,65536:2.3,FULL:4.2},
 "fortran": {1024:0.6,4096:0.6,16384:0.7,65536:1.3,FULL:3.1},
}
# CPU fit-time IQR (p25,p75) — wide, because the cluster was contended throughout (see note)
CPU_BAND = {
 "jamica":  {1024:(1313,2706),4096:(1761,2537),16384:(2039,2367),65536:(2295,2773),FULL:(2428,2926)},
 "pamica":  {1024:(3264,5356),4096:(3043,4440),16384:(3836,5730),65536:(7358,9846),FULL:(8270,9508)},
 "pyamica": {4096:(20976,29237),16384:(9279,11644),65536:(7808,10549),FULL:(10149,14659)},
 "scott":   {1024:(2411,4046),4096:(2277,3369),16384:(2498,3641),65536:(3356,4610),FULL:(3718,4784)},
 "fortran": {1024:(5610,6330),4096:(5292,5383),16384:(5906,6421),65536:(7555,9021),FULL:(7045,7172)},
}
CPU_FIT_MISS = {("pyamica",1024):("timeout","bad")}  # pyamica@1024 (~767-1333 eager blocks/iter): >12h wall
# jamica's two-level memory (two orchestrator keys). The chunked path is what's swept above; the
# full-batch key (chunk_size=None) is a separate program. Numbers below are measured (GPU time from the
# fit sweep, GPU NVML from the memory campaign, CPU from the CPU sweep):
J_CHUNKED_GPU_T, J_FULLBATCH_GPU_T = 61.6, 61.0      # s @3000 -> essentially identical (no GPU speed benefit)
J_CHUNKED_GPU_NVML, J_FULLBATCH_GPU_NVML = 5.4, 11.4 # GiB NVML
J_CHUNKED_ALLOC = 1.6                                # GiB JAX allocator (chunked); vs 5.4 NVML
J_CHUNKED_CPU_T, J_FULLBATCH_CPU_T = 1982, 4000      # s @1000 (CPU: chunking helps time too)
J_CHUNKED_CPU_RSS, J_FULLBATCH_CPU_RSS = 2.2, 19.8   # GiB (CPU: and memory)
# pAMICA block_size sensitivity, GPU @3000 (the ~17x within one impl):
REAL_PAM = [("1024 (near 512 default)",4086.5,1.83,"artifact"),("16384 (tuned)",370.8,2.13,"tuned"),
            ("262144 = largest tested",244.7,6.57,"best")]

def xlog(c): return math.log2(c)
XT=[1024,4096,16384,65536,FULL]; XMIN,XMAX=math.log2(1024)-0.4,xlog(FULL)+0.4

def chart(series, band, ylab, ylog, title, sub, impls, mark, oom=None):
    W,H=520,340; ml,mr,mt,mb=56,14,32,52; pw,ph=W-ml-mr,H-mt-mb
    def X(c): return ml+(xlog(c)-XMIN)/(XMAX-XMIN)*pw
    allv=[v for im in impls for v in series[im].values()]
    if band: allv+=[b for im in impls for c in series[im] if im in band and c in band[im] for b in band[im][c]]
    vmax=max(allv); vmin=min(v for v in allv if v>0)
    if ylog:
        lo,hi=math.log10(vmin*0.8),math.log10(vmax*1.25)
        def Y(v): return mt+ph-(math.log10(max(v,vmin*0.5))-lo)/(hi-lo)*ph
    else:
        hi=vmax*1.12; lo=0
        def Y(v): return mt+ph-(v-lo)/(hi-lo)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    s.append(f'<text x="{ml}" y="16" class="ct">{title}</text>')
    s.append(f'<text x="{ml}" y="{H-6}" class="cx">chunk / block size (samples) →</text>')
    s.append(f'<text transform="translate(14,{mt+ph/2}) rotate(-90)" class="cy">{ylab}</text>')
    if ylog:
        ticks=[]; d0=1
        while d0<=vmax*1.25:
            for m0 in (1,2,5):
                if vmin*0.7<=d0*m0<=vmax*1.25: ticks.append(d0*m0)
            d0*=10
    else:
        raw=hi/4; e=10**math.floor(math.log10(raw)); f=raw/e
        step=(1 if f<1.5 else 2 if f<3 else 5 if f<7 else 10)*e
        ticks=[]; t0=0.0
        while t0<=hi: ticks.append(t0); t0+=step
    for t in ticks:
        y=Y(t); s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        lab=f'{t:.0f}' if t>=1 else f'{t:.1f}'
        s.append(f'<text x="{ml-6}" y="{y+3:.1f}" class="cyt">{lab}</text>')
    for c in XT:
        x=X(c); lab="262K" if c==FULL else f'{c//1024}K'
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+16}" class="cxt">{lab}</text>')
    for im in impls:
        cs=sorted(series[im])
        if not cs: continue
        if band and im in band:
            up=" ".join(f"{X(c):.1f},{Y(band[im][c][1]):.1f}" for c in cs if c in band[im])
            dn=" ".join(f"{X(c):.1f},{Y(band[im][c][0]):.1f}" for c in reversed(cs) if c in band[im])
            if up: s.append(f'<polygon points="{up} {dn}" fill="{COLOR[im]}" opacity="0.09"/>')
        path=" ".join((("M" if i==0 else "L")+f"{X(c):.1f},{Y(series[im][c]):.1f}") for i,c in enumerate(cs))
        s.append(f'<path d="{path}" fill="none" stroke="{COLOR[im]}" stroke-width="2.4"/>')
        for c in cs: s.append(f'<circle cx="{X(c):.1f}" cy="{Y(series[im][c]):.1f}" r="3" fill="{COLOR[im]}"/>')
        bc=min(cs,key=lambda k:series[im][k])
        s.append(f'<circle cx="{X(bc):.1f}" cy="{Y(series[im][bc]):.1f}" r="6" fill="none" stroke="{COLOR[im]}" stroke-width="2"/>')
    if oom:
        for im,cc in oom.items():
            s.append(f'<text x="{X(FULL):.1f}" y="{mt+13}" class="oom">{im} ✕ OOM</text>')
    s.append('</svg>')
    cap=f'{sub} · shaded = p25–p75 across subjects · ◯ = {mark} setting'
    return f'<figure class="cf"><figcaption>{cap}</figcaption>{"".join(s)}</figure>'

gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 3000-iter budget · median",IMPLS,"fastest")
c_gv=chart(gpu_v,None,"peak VRAM · NVML (GiB)",False,"GPU · memory vs chunk","real ds004505 · H100 · whole-GPU NVML peak",IMPLS,"leanest")
CPU_IMPLS=["jamica","pamica","pyamica","scott","fortran"]
c_cr=chart(CPU_RSS,None,"peak RSS (GiB)",False,"CPU · memory vs chunk","real ds004505 · 8 cores · median peak RSS",CPU_IMPLS,"leanest")
c_ct=chart(CPU_FIT,CPU_BAND,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · 8 cores · 1000-iter budget · median",CPU_IMPLS,"fastest")

def legend(impls):
    it="".join(f'<span class="lg"><i style="background:{COLOR[i]}"></i>{LABEL.get(i,"Fortran amica17 (1 thread)")} <code>{KNOB[i]}</code> <span class="cm">@{COMMIT[i]}</span></span>' for i in impls)
    return f'<div class="legend">{it}</div>'

def convrows():
    # wall time to the iteration budget at chunk=262144 (largest tested), with iterations-run + final LL
    order=["jamica","scott","pamica","pyamica"]; base=GPU["jamica"][FULL][0]; r=""
    for im in order:
        t=GPU[im][FULL][0]; ll,n0,n1=GPU_CONV[im]
        nit=f'{n0:,}' if n0==n1 else f'{n0:,}–{n1:,}'
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td>'
            f'<td class="num">{t:.0f}s</td><td class="num">{t/base:.1f}×</td>'
            f'<td class="num">{nit}</td><td class="num">{ll:.4f}</td></tr>')
    return r
def pamrows():
    bd={"artifact":'<span class="badge bad">near default</span>',"tuned":'<span class="badge ok">tuned</span>',"best":'<span class="badge best">largest tested</span>'}
    return "".join(f'<tr><td><code>{cfg}</code></td><td class="num">{t:.0f}s</td><td class="num">{v:.2f} GiB</td><td>{bd[tag]}</td></tr>' for cfg,t,v,tag in REAL_PAM)
def cpufitrows():
    r=""
    for im in CPU_IMPLS:
        cells=""
        for c in XT:
            if c in CPU_FIT[im]:
                cells+=f'<td class="num">{CPU_FIT[im][c]:.0f}s</td>'
            elif (im,c) in CPU_FIT_MISS:
                lab,cls=CPU_FIT_MISS[(im,c)]
                cells+=f'<td class="num"><span class="badge {cls}">{lab}</span></td>'
            else:
                cells+='<td class="num">—</td>'
        lab=LABEL.get(im,"Fortran amica17 (1 thread)")
        r+=f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{lab}</td>{cells}</tr>'
    return r

HTML=f"""<title>AMICA implementations — fit time, convergence and peak memory (ds004505)</title>
<style>
:root{{--bg:#ffffff;--panel:#ffffff;--ink:#1a1d29;--mut:#5b6172;--line:#e5e7eb;--accent:#2563eb;--good:#0d9488;--bad:#e11d48;--warn:#d97706;--code:#f1f3f8;--shadow:0 1px 2px rgba(20,24,45,.05),0 8px 22px rgba(20,24,45,.06)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 96px}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:var(--code);padding:.06em .4em;border-radius:5px}}
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
header.hero{{padding:60px 0 28px;border-bottom:1px solid var(--line)}}
.kick{{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}}
h1{{font-size:clamp(2rem,4.4vw,3rem);line-height:1.05;letter-spacing:-.02em;margin:.28em 0 .3em;text-wrap:balance;font-weight:800}}
h1 em{{font-style:normal;color:var(--accent)}}
.lede{{font-size:1.14rem;color:var(--mut);max-width:66ch;margin:0}}
.stamp{{display:inline-flex;flex-wrap:wrap;gap:6px 14px;margin:16px 0 0;padding:10px 14px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;font-size:.83rem}}
.stamp b{{color:var(--accent)}}
section{{padding:44px 0 6px;border-bottom:1px solid var(--line)}}
h2{{font-size:1.5rem;letter-spacing:-.01em;margin:0 0 4px;font-weight:750}}
.sub{{color:var(--mut);margin:.1em 0 1.3em;max-width:68ch}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:12px 12px 4px}}
.cf{{margin:0}}.cf figcaption{{font-size:.82rem;color:var(--mut);padding:3px 5px 8px}}
svg.chart{{width:100%;height:auto;display:block;overflow:visible}}
.chart .grid{{stroke:var(--line)}}.chart .vg{{stroke-dasharray:2 3;opacity:.7}}
.chart .ct{{fill:var(--ink);font:700 13px ui-sans-serif}}.chart .cx,.chart .cy{{fill:var(--mut);font:11px ui-sans-serif}}
.chart .cyt,.chart .cxt{{fill:var(--mut);font:10px ui-monospace,monospace}}.chart .cyt{{text-anchor:end}}.chart .cxt{{text-anchor:middle}}
.chart .oom{{fill:#e11d48;font:600 10px ui-sans-serif;text-anchor:middle}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;margin:14px 2px 2px;font-size:.82rem;color:var(--mut);align-items:center}}
.lg{{display:inline-flex;align-items:center;gap:6px}}.lg i{{width:15px;height:3px;border-radius:2px}}.cm{{font-family:ui-monospace,monospace;font-size:.78em;opacity:.7}}
table{{width:100%;border-collapse:collapse;font-size:.95rem;margin:.2em 0 1em}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}}
th{{font-size:.74rem;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);font-weight:650}}
tbody tr:last-child td{{border-bottom:none}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.badge{{font-size:.72rem;padding:2px 9px;border-radius:20px;font-weight:600;white-space:nowrap}}
.badge.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}}.badge.ok{{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}}.badge.best{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}}
.callout{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0 20px}}@media(max-width:720px){{.callout{{grid-template-columns:1fr}}}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:var(--shadow)}}
.stat .big{{font-size:1.9rem;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1}}
.stat .lab{{color:var(--mut);font-size:.86rem;margin-top:6px}}
.stat.bad .big{{color:var(--bad)}}.stat.warn .big{{color:var(--warn)}}
p{{max-width:68ch}}.note{{font-size:.9rem;color:var(--mut)}}
.warn-box{{background:color-mix(in srgb,var(--warn) 8%,var(--panel));border:1px solid var(--line);border-left:3px solid var(--warn);border-radius:10px;padding:14px 18px;margin:6px 0 14px;font-size:.92rem}}
.info-box{{background:color-mix(in srgb,var(--accent) 6%,var(--panel));border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:14px 18px;margin:6px 0 14px;font-size:.92rem}}
.info-box.big{{border-left-width:4px}}
ul.tk{{max-width:68ch;padding-left:0;list-style:none}}ul.tk li{{padding:7px 0 7px 24px;position:relative;border-bottom:1px solid var(--line)}}
ul.tk li:before{{content:"";position:absolute;left:3px;top:14px;width:8px;height:8px;border-radius:50%;background:var(--accent)}}
footer{{padding:34px 0 0;color:var(--mut);font-size:.86rem}}
.prov{{display:grid;grid-template-columns:160px 1fr;gap:4px 16px;font-size:.9rem;margin-top:8px}}.prov dt{{color:var(--mut)}}.prov dd{{margin:0;font-family:ui-monospace,monospace;font-size:.86em}}
</style>
<div class="wrap">
<header class="hero">
  <div class="kick">Cross-implementation AMICA · ds004505 · real EEG</div>
  <h1>Fit time, convergence, and peak memory across AMICA implementations</h1>
  <p class="lede">We measured how long each Python AMICA implementation runs, how far it converges, and
  how much memory it needs on a real EEG dataset — sweeping each one across its batch/chunk-size
  setting on GPU and CPU. That single setting changes fit time by up to ~25× within one implementation;
  it also changes memory (for the torch implementations), and the fastest setting flips between GPU and
  CPU. <b>The fit times are wall time to a fixed iteration budget, not time to an equivalent
  solution</b> — read them with the convergence box below.</p>
  <div class="stamp"><span><b>Builds (main):</b></span><span>jamica <code>df18b5e</code></span><span>scott-huberty <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 64 comp · GPU 3000-iter / CPU 1000-iter budget · H100 + 8-core Xeon</span></div>
</header>

<section>
  <h2>How to read this</h2>
  <div class="info-box big"><b>These are wall times to a fixed iteration budget — not a convergence
  race.</b> Every fit was capped at 3000 iterations on GPU / 1000 on CPU. But the implementations do
  <em>not</em> all use that budget the same way. At the largest chunk on GPU, the actual iterations run
  before each one stopped were: <b>pyamica 3000</b> (always runs the full cap), <b>jamica 2,444–3,000</b>,
  <b>scott-huberty 784–1,654</b> (early-converges and stops well before the cap), <b>pAMICA 151–3,000</b>
  (its early-stop can fire very early). They reach final log-likelihoods in a tight but non-identical
  band (about −1.10 to −1.12; pAMICA's is the lowest). So a shorter wall time can mean a faster
  implementation, an earlier stop, or fewer iterations of work — not necessarily a better or
  equally-finished decomposition. Equal budget is not equal work, and not equal convergence. We report
  <code>n_iter</code> and <code>ll_final</code> alongside time so you can see which is which. We also do
  not verify that the four decompositions are numerically equivalent (component matching against the
  Fortran reference is not part of this pass).</div>
  <div class="callout">
    <div class="stat warn"><div class="big">~25×</div><div class="lab">Widest fit-time range across the setting within a single implementation (scott-huberty, GPU). The others span 8–17×; every implementation is chunk-sensitive on time.</div></div>
    <div class="stat"><div class="big">grows</div><div class="lab">Peak VRAM grows with chunk for the torch impls (~2.7–3.6× NVML). jamica's GPU memory is flat (~5.4 GiB) on its chunked path — a floor, not a dial.</div></div>
    <div class="stat"><div class="big">GPU ⇄ CPU</div><div class="lab">The fastest setting flips by device: large chunks on the H100, small/mid on CPU. Budgets differ (3000 vs 1000) — do not compare GPU seconds to CPU seconds.</div></div>
  </div>
  <p class="note"><b>On "the largest tested chunk."</b> The biggest setting we swept is 262,144 samples.
  Each recording is {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples, so 262,144 is only ~15–33% of the data — it
  is the largest chunk tested, <em>not</em> a single full-batch pass. We label that axis point
  <code>262K</code>, not "full-batch." True single-pass full-batch is a separate thing, discussed only
  for jamica in the memory note.</p>
</section>

<section>
  <h2>GPU — fit time &amp; memory</h2>
  <p class="sub">Each implementation swept across its setting on real ds004505 (per-subject median,
  3000-iter budget, H100). Shaded band = p25–p75 across subjects. VRAM is the <b>NVML whole-GPU peak</b>
  (framework-neutral; see the memory note).</p>
  <div class="grid2"><div class="card">{c_gt}</div><div class="card">{c_gv}</div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:20px">
    <li><b>Larger chunks are faster on the GPU — for all four.</b> Median fit time falls steeply as the
    chunk grows (jamica 763&nbsp;s → 62&nbsp;s; scott-huberty 2024&nbsp;s → 79&nbsp;s from 1024 to
    262K). Small chunks starve the device.</li>
    <li><b>Memory grows with chunk for the torch implementations</b> (scott-huberty ~1.8→4.9, pAMICA
    ~1.8→6.6, pyamica ~3.0→10.9&nbsp;GiB NVML). <b>jamica's GPU memory is flat at ~5.4&nbsp;GiB</b> on
    its chunked path — a chunk-independent full-width array sets a floor, so for jamica the chunk is a
    time dial but not a GPU-memory dial. Nothing exceeded the 80&nbsp;GiB card at this budget (peak
    ~10.9&nbsp;GiB).</li>
    <li><b>The bands are informative.</b> pAMICA's wide 262K band reflects its early-stop firing at very
    different iteration counts across subjects (see the convergence box).</li>
  </ul>
</section>

<section>
  <h2>Wall time to the iteration budget (GPU, largest chunk)</h2>
  <p class="sub">Each implementation at chunk 262K on the H100 (per-subject median, 3000-iter budget).
  This is <b>not</b> a convergence-equalised ranking: the "iters run" and "final LL" columns show that
  the implementations did different amounts of work. Read all three columns together.</p>
  <table><thead><tr><th>Implementation</th><th>Wall time</th><th>Relative</th><th>Iters run</th><th>Final LL (median)</th></tr></thead><tbody>{convrows()}</tbody></table>
  <p class="note">jamica has the shortest wall time and runs near the full budget; scott-huberty is
  ~1.3× longer but stops at roughly half the iterations (early convergence); pyamica runs the full 3000
  every time for the highest final LL; pAMICA is longest and lands at the lowest LL. "Faster" here means
  "less wall time to its own stopping point," which is not the same as "converges sooner" or "to a
  better solution." As a separate illustration of how far one setting sits from an implementation's
  best, pAMICA's <code>block_size</code> spans ~17× end to end:</p>
  <table><thead><tr><th><code>block_size</code></th><th>Wall time</th><th>Peak VRAM (NVML)</th><th></th></tr></thead><tbody>{pamrows()}</tbody></table>
  <p class="note">pAMICA's shipped default is <code>block_size=512</code>, not re-measured here; the
  nearest tested point, 1024, is already ~17× off its fastest tested setting. (jamica's own shipped
  default — <code>chunk_size=None</code>, the full-batch path — is discussed in the memory note; it was
  not the path swept here.)</p>
</section>

<section>
  <h2>CPU — fit time &amp; memory</h2>
  <p class="sub">Real ds004505, 8 cores, 5 subjects × reps, 1000-iter budget. Shaded band = p25–p75.
  Includes the Fortran amica17 reference (CPU-only, <b>single-threaded</b>, so not a like-for-like
  wall-time comparator against the 8-thread Python impls).</p>
  <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
  {legend(CPU_IMPLS)}
  <table style="margin-top:16px"><thead><tr><th>fit time (s)</th><th class="num">1K</th><th class="num">4K</th><th class="num">16K</th><th class="num">64K</th><th class="num">262K</th></tr></thead><tbody>{cpufitrows()}</tbody></table>
  <ul class="tk" style="margin-top:8px">
    <li><b>The fastest setting flips on CPU.</b> CPU is fastest at <em>small/mid</em> chunks (jamica
    1024, scott-huberty ~4096) — the opposite of the GPU, where large chunks won. Small blocks fit CPU
    cache. <b>But the CPU bands are wide</b> (contended cluster): several apparent optima sit inside the
    noise — scott-huberty 4096 vs 16384 differ &lt;1%, jamica 1024 vs 4096 ~5%. Trust the broad
    small-chunk tendency, not the exact per-impl optimum.</li>
    <li><b>CPU AMICA is a heavy method</b> (tens of minutes to hours at 1000 iterations): jamica
    ~1980&nbsp;s at its best CPU setting, scott-huberty ~2790&nbsp;s, pAMICA ~3580&nbsp;s, then pyamica
    and Fortran in the thousands of seconds. Compare absolute CPU seconds only coarsely.</li>
    <li><b>Memory grows with chunk on CPU too</b> (pyamica ~9.8, jamica ~7&nbsp;GiB at 262K); small
    chunks bring everyone to ~1.6–2.2&nbsp;GiB. The single-threaded Fortran reference is leanest
    everywhere (0.6→3.1&nbsp;GiB).</li>
    <li><b>One setting failed:</b> pyamica at 1024 (hundreds–thousands of eager blocks/iter) exceeded the
    12&nbsp;h wall — absent from the curve.</li>
  </ul>
  <div class="warn-box" style="margin-top:14px"><b>On the CPU timing numbers.</b> CPU fits ran on shared
  cluster nodes; AMICA is memory-bandwidth-bound, so co-located jobs interfere. The cluster was busy
  throughout (repeated cells found no quiet window), which is why the p25–p75 bands are wide and some
  cells are non-monotonic. We report the median over 5 subjects × reps; treat <em>absolute</em> CPU
  seconds as contention-inflated and rely on the broad shapes and the device flip, not exact seconds or
  fine ordering. CPU <em>memory</em> is allocation-driven and clean. Fortran is single-threaded
  (<code>OMP_NUM_THREADS=1</code>); per core-second it is actually more efficient than the 8-thread
  Python impls, so keep it as a reference footprint, not a head-to-head. Detail:
  <code>NOTES_measurement.md</code>.</div>
</section>

<section>
  <h2>A note on the memory numbers</h2>
  <div class="info-box"><b>NVML is the framework-neutral VRAM figure.</b> The per-framework allocator
  counters (JAX <code>peak_bytes_in_use</code>, torch <code>max_memory_allocated</code>) measure only
  live-tensor bytes in each framework's own allocator — they omit the CUDA/cuDNN context and the pool
  the driver actually holds, and the two frameworks count differently, so they are <b>not comparable
  across implementations</b> and understate the real footprint (by ~1.4–3.4× in the pairs we can check;
  e.g. jamica chunked ~{J_CHUNKED_ALLOC:.1f}&nbsp;GiB allocator vs ~5.4&nbsp;GiB NVML = 3.4×; jamica
  full-batch 8.2 vs 11.4&nbsp;GiB = 1.4×). The charts use <b>NVML whole-GPU 'used'</b> on a dedicated
  GPU. Two caveats on that meter: it is a 50&nbsp;ms poll (a sub-interval spike could be missed), and
  the frameworks run under different allocator settings (JAX with pre-allocation off; torch with its
  caching allocator on), so NVML is a neutral <em>meter</em> over somewhat different <em>protocols</em>.</div>
  <p class="note"><b>jamica memory is two-level, and it cuts both ways.</b> On its chunked path jamica
  sits at ~5.4&nbsp;GiB NVML across chunk sizes. Its <em>full-batch</em> path
  (<code>chunk_size=None</code> — a different orchestrator key, and jamica's shipped default) uses
  ~{J_FULLBATCH_GPU_NVML:.0f}&nbsp;GiB NVML for essentially the <b>same GPU wall time</b>
  (~{J_FULLBATCH_GPU_T:.0f}&nbsp;s vs ~{J_CHUNKED_GPU_T:.0f}&nbsp;s at 3000 iters — measured, no speed
  benefit). On CPU, chunking helps <em>both</em> axes (~{J_CHUNKED_CPU_T:,}&nbsp;s /
  ~{J_CHUNKED_CPU_RSS:.1f}&nbsp;GiB chunked vs ~{J_FULLBATCH_CPU_T:,}&nbsp;s /
  ~{J_FULLBATCH_CPU_RSS:.0f}&nbsp;GiB full-batch). So a wrapper should always pass a chunk. The flip
  side, in fairness: jamica's ~5.4&nbsp;GiB chunked <em>floor</em> is higher than the torch impls'
  small-chunk footprint (~1.8–3.1&nbsp;GiB NVML) — on a small card the torch impls at a small chunk fit
  where jamica may not. (The "fits an 8–12&nbsp;GiB card" reading is an extrapolation from H100 NVML
  with JAX pre-allocation off; it was not measured on such a card.)</p>
</section>

<footer>
  <div class="kick" style="color:var(--mut)">Provenance &amp; reproduction</div>
  <dl class="prov">
    <dt>Dataset</dt><dd>ds004505 · 64 PCA components · {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples/subject</dd>
    <dt>GPU cells</dt><dd>/scratch/yorguin/iter_ladder/gpu/c&lt;chunk&gt;_i3000/ · Trillium H100 · NVML peak · 20-25 subj/cell</dd>
    <dt>CPU cells</dt><dd>/scratch/yorguin/iter_ladder/cpu/c&lt;chunk&gt;_i1000_r&lt;rep&gt;/ · fir 8 cores · getrusage peak_rss · 5 subj × reps</dd>
    <dt>Raw aggregate</dt><dd>raw/chunk_{{gpu3000,cpu1000}}_{{summary,percell}}.csv (per-cell fit/mem/ll_final/n_iter, straight from the JSONs)</dd>
    <dt>Budget</dt><dd>GPU 3000-iter cap · CPU 1000-iter cap (memory is iteration-independent; time is not)</dd>
    <dt>jamica path</dt><dd>amica_python_jax_chunked (chunked); full-batch key amica_python_jax shown only in the memory note</dd>
    <dt>Units</dt><dd>memory in GiB (bytes / 1024³); times in seconds, per-subject median</dd>
    <dt>Commits</dt><dd>jamica df18b5e · scott e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
    <dt>Caveats</dt><dd>NOTES_measurement.md (iteration-budget ≠ convergence · CPU contention · NVML vs allocator · the two jamica keys)</dd>
  </dl>
  <p class="note" style="margin-top:16px">Trust the curve shapes, the per-device fastest settings, the
  NVML memory figures, and the convergence columns read together. GPU times are per-subject medians;
  CPU times are contention-inflated approximations. Fit times are wall time to a fixed iteration budget,
  <b>not</b> time to an equivalent solution — <code>n_iter</code> and <code>ll_final</code> are reported
  so the difference is visible. jamica's chunk is a real GPU-time / CPU-time+memory dial; its GPU memory
  on the chunked path is flat.</p>
</footer>
</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
open(os.path.join(HERE,"xperf_chunk_report.html"),"w").write(HTML)
_title = HTML.split("<title>",1)[1].split("</title>",1)[0] if "<title>" in HTML else "AMICA chunk-size report"
STANDALONE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              f'<title>{_title}</title></head><body>\n{HTML}\n</body></html>\n')
open(os.path.join(HERE,"xperf_chunk_report_standalone.html"),"w").write(STANDALONE)

# tidy dataset — regenerated from the report dicts; the authoritative raw per-cell data lives in raw/.
def _cn(c): return str(c)
_rows = [("dataset", "impl", "knob", "chunk", "value", "unit", "note")]
for im in IMPLS:
    for c, (t, v) in sorted(GPU[im].items()):
        _rows.append(("gpu_fit_s_median", im, KNOB[im], _cn(c), t, "s", "GPU H100 3000-iter budget, per-subj median"))
        _rows.append(("gpu_vram_gib_nvml", im, KNOB[im], _cn(c), v, "GiB", "NVML whole-GPU peak"))
    for c, (lo, hi) in sorted(GPU_BAND[im].items()):
        _rows.append(("gpu_fit_s_p25", im, KNOB[im], _cn(c), lo, "s", ""))
        _rows.append(("gpu_fit_s_p75", im, KNOB[im], _cn(c), hi, "s", ""))
for im, (ll, n0, n1) in GPU_CONV.items():
    _rows.append(("gpu_ll_final_median", im, KNOB[im], _cn(FULL), ll, "nats", "at chunk 262144"))
    _rows.append(("gpu_n_iter_min", im, KNOB[im], _cn(FULL), n0, "iters", "actual iterations run"))
    _rows.append(("gpu_n_iter_max", im, KNOB[im], _cn(FULL), n1, "iters", ""))
for im in CPU_IMPLS:
    for c, v in sorted(CPU_RSS[im].items()):
        _rows.append(("cpu_rss_gib_median", im, KNOB[im], _cn(c), v, "GiB", "fir 8 cores 1000-iter budget, median"))
    for c, v in sorted(CPU_FIT[im].items()):
        _rows.append(("cpu_fit_s_median", im, KNOB[im], _cn(c), v, "s", "median over subj x reps (contention-inflated)"))
    for c, (lo, hi) in sorted(CPU_BAND[im].items()):
        _rows.append(("cpu_fit_s_p25", im, KNOB[im], _cn(c), lo, "s", ""))
        _rows.append(("cpu_fit_s_p75", im, KNOB[im], _cn(c), hi, "s", ""))
for (im, c), (lab, _cls) in CPU_FIT_MISS.items():
    _rows.append(("cpu_fit_s_median", im, KNOB[im], _cn(c), lab, "", "no value: 12h wall timeout"))
for cfg, t, v, tag in REAL_PAM:
    _rows.append(("gpu_pamica_sensitivity", "pamica", "block_size", cfg, t, "s", tag))
with open(os.path.join(HERE, "chunk_sweep_data.csv"), "w", newline="") as _f:
    csv.writer(_f).writerows(_rows)

print("wrote CORRECTED report", len(HTML), "bytes (+ standalone", len(STANDALONE),
      "bytes) + chunk_sweep_data.csv", len(_rows) - 1, "rows")
