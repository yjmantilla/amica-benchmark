#!/usr/bin/env python3
"""Render the AMICA cross-implementation timing/convergence/memory report (self-contained HTML).

Fit time, convergence (n_iter + ll_final + seconds/iter), and peak memory for each Python AMICA
implementation on real EEG (ds004505), swept across each one's batch/chunk-size setting on GPU and CPU.
Neutral and even-handed — measurements, not a product verdict.

Provenance (every dict below is the aggregate of the per-cell result JSONs; the aggregates are shipped
in raw/ and this generator is cross-checked against them):
 - GPU FIT + convergence @3000, ITERATION-MATCHED (early-stops DISABLED, every impl runs the full 3000):
   -> raw/nostop_gpu3000_summary.csv (t_subj_median, s_per_iter_median, ll_median, n_iter_min/med/max=3000;
   25 subj/cell; Trillium H100). s_per_iter x 3000 = wall time exactly.
 - GPU CONVERGENCE LADDER (chunk 65536, early-stops disabled): raw/nostop_ladder_i{100,250,500,1000,2000,
   3000}_summary.csv (LL + wall at each iteration count; all 25 subj).
 - GPU MEMORY: raw/nostop_gpumem_summary.csv (NVML + allocator) + raw/nostop_gpumem_decomp.csv (measured
   decomposition: context=nvml_post_init pre-fit baseline, live=allocator peak, nvml_total; medians over
   25 subj). Per-chunk NVML also appears as mem_median in nostop_gpu3000_summary.csv. Memory is
   iteration-independent. jamica's FULL-BATCH key (amica_python_jax, chunk_size=None) memory (13.37 GiB
   etc.) is from the earlier memory run (raw/chunk_gpumem_summary.csv) -- a separate program shown only in
   the memory note.
 - CPU FIT + RSS @1000 (the EARLIER, non-iteration-matched, contended run -- superseded by the Narval
   whole-node re-run in progress): raw/chunk_cpu1000_{summary,percell}.csv (BY-SUBJECT median; n_subjects
   disclosed because cells have UNEQUAL subject coverage).

jamica = the CHUNKED path (amica_python_jax_chunked). Memory in GiB (bytes/1024**3).

CAVEATS baked into the report (see NOTES_measurement.md):
 - GPU fit time is wall time at a MATCHED 3000 iterations (early-stops disabled, all impls run the full
   3000), so it is directly per-iteration-comparable (s/iter x 3000 = wall). The CPU section is the EARLIER
   run at a 1000-iter cap with each impl's own early-stop ON -- NOT iteration-matched.
 - CPU cells have unequal subject coverage (Fortran is sub-01-only at 4 of 5 chunks) AND ran on a
   contended cluster -> absolute CPU seconds and exact per-cell optima are not resolved; only the broad
   small/mid-vs-large device flip is trustworthy.
"""
import math, os, csv

FULL = 262144                       # the LARGEST TESTED chunk (samples); NOT a full-batch pass.
N_SAMP_MIN, N_SAMP_MAX = 785328, 1364633   # per-subject sample counts; 262144 = ~19-33% of a recording
IMPLS = ["jamica", "pamica", "pyamica", "amica_python"]
LABEL = {"jamica":"jamica","pamica":"pAMICA","pyamica":"pyamica","amica_python":"amica-python"}
KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","amica_python":"batch_size","fortran":"block_size"}
COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","amica_python":"e15e158","fortran":"665b577"}
COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","amica_python":"#e11d48","fortran":"#7c3aed"}

# ===== GPU @3000, per-subject median : chunk -> (fit_s, nvml_vram_gib). jamica = chunked path.
# fit_s from the i3000 run. nvml: jamica-chunked from i3000 (logged NVML for jamica only),
# torch impls + fullbatch from i1000 (iteration-independent). All in raw/chunk_gpumem_summary.csv.
GPU = {
 "jamica":  {1024:(775.3,5.31),4096:(227.8,5.37),16384:(97.0,5.37),65536:(70.8,5.37),FULL:(61.0,5.37)},
 "pamica":  {1024:(4155.5,1.82),4096:(1058.0,1.89),16384:(392.7,2.13),65536:(320.3,3.08),FULL:(262.4,6.54)},
 "pyamica": {1024:(2417.7,3.05),4096:(615.3,3.05),16384:(366.2,3.05),65536:(338.9,4.46),FULL:(295.7,10.92)},
 "amica_python":   {1024:(5646.0,1.82),4096:(1410.0,1.88),16384:(459.1,2.09),65536:(303.3,2.83),FULL:(230.3,4.89)},
}
GPU_BAND = {  # GPU fit-time p25,p75 across subjects (iteration-matched @3000)
 "jamica":  {1024:(743,788),4096:(221,232),16384:(96,99),65536:(69,73),FULL:(58,63)},
 "pamica":  {1024:(3902,4250),4096:(1008,1079),16384:(369,404),65536:(313,330),FULL:(258,267)},
 "pyamica": {1024:(2308,2450),4096:(576,626),16384:(348,378),65536:(323,348),FULL:(278,301)},
 "amica_python":   {1024:(5443,5832),4096:(1365,1479),16384:(441,475),65536:(290,315),FULL:(220,242)},
}
# GPU convergence at chunk=262144, ITERATION-MATCHED (early-stops disabled -> all run the full 3000):
# impl -> (ll_median, n_iter_min, n_iter_median, n_iter_max, s_per_iter). raw/nostop_gpu3000_summary.csv.
GPU_CONV = {
 "jamica":  (-1.1005, 3000, 3000, 3000, 0.0203), "amica_python":  (-1.1002, 3000, 3000, 3000, 0.0768),
 "pamica":  (-1.1107, 3000, 3000, 3000, 0.0875),  "pyamica": (-1.0995, 3000, 3000, 3000, 0.0986),
}
# ===== ITERATION LADDER (measured), chunk fixed at 65536, early-stops disabled so every point is the
# full iteration count. impl -> {iters: (wall_s_median, ll_median)}. raw/nostop_ladder_i*_summary.csv
# (i3000 point = the 65536 cell of nostop_gpu3000_summary.csv). All 25 subjects at every point.
LAD_ITERS = [100, 250, 500, 1000, 2000, 3000]
LADDER = {
 "jamica":       {100:(4.6,-1.11197),250:(7.9,-1.10503),500:(13.7,-1.10126),1000:(25.2,-1.10056),2000:(48.0,-1.10050),3000:(70.8,-1.10047)},
 "amica_python": {100:(11.2,-1.11537),250:(26.1,-1.10503),500:(51.2,-1.10283),1000:(102.0,-1.10075),2000:(204.1,-1.10058),3000:(303.3,-1.10056)},
 "pyamica":      {100:(11.8,-1.11518),250:(28.8,-1.10445),500:(57.1,-1.100545),1000:(113.6,-1.09987),2000:(226.2,-1.09958),3000:(338.9,-1.099548)},
 "pamica":       {100:(12.2,-1.12929),250:(28.2,-1.12025),500:(54.3,-1.11791),1000:(107.5,-1.11506),2000:(215.7,-1.11223),3000:(320.3,-1.11067)},
}
# per-iteration GPU memory at chunk 65536 (mem_median GiB; iteration-independent -> ~flat).
LADDER_MEM = {
 "jamica":       {100:5.37,250:5.37,500:5.37,1000:5.37,2000:5.37,3000:5.37},
 "amica_python": {100:2.73,250:2.73,500:2.73,1000:2.73,2000:2.80,3000:2.83},
 "pyamica":      {100:4.46,250:4.46,500:4.46,1000:4.46,2000:4.46,3000:4.46},
 "pamica":       {100:3.08,250:3.08,500:3.08,1000:3.08,2000:3.08,3000:3.08},
}
# ===== CPU @1000, BY-SUBJECT median (median within subject over reps, then across subjects) =====
CPU_FIT = {
 "jamica":  {1024:1985,4096:1850,16384:2140,65536:2729,FULL:2793},
 "pamica":  {1024:4664,4096:3858,16384:4397,65536:9447,FULL:8811},
 "pyamica": {4096:24452,16384:10225,65536:10941,FULL:10980},
 "amica_python":   {1024:3025,4096:3080,16384:2646,65536:4037,FULL:4397},
 "fortran": {1024:5627,4096:5301,16384:6049,65536:8015,FULL:7171},
}
CPU_BAND = {  # CPU by-subject p25,p75 (wide: contended cluster + unequal coverage)
 "jamica":  {1024:(1471,2135),4096:(1546,2332),16384:(2078,2434),65536:(2601,2770),FULL:(2514,3055)},
 "pamica":  {1024:(3861,4915),4096:(3043,4389),16384:(4006,4783),65536:(9115,9506),FULL:(8337,8826)},
 "pyamica": {4096:(23342,29939),16384:(9434,10886),65536:(10340,11733),FULL:(10149,11654)},
 "amica_python":   {1024:(2449,4000),4096:(2578,3592),16384:(2533,3534),65536:(3622,4423),FULL:(4213,4549)},
}
CPU_NSUB = {  # subjects contributing to each cell (out of 5) -- disclose the unequal coverage
 "jamica":  {1024:5,4096:4,16384:5,65536:4,FULL:5},
 "pamica":  {1024:5,4096:5,16384:5,65536:5,FULL:5},
 "pyamica": {4096:5,16384:5,65536:5,FULL:5},
 "amica_python":   {1024:5,4096:5,16384:5,65536:5,FULL:5},
 "fortran": {1024:1,4096:1,16384:1,65536:5,FULL:1},
}
CPU_RSS = {  # peak RSS (GiB) BY-SUBJECT median (also subject-length dependent -> same coverage caveat)
 "jamica":  {1024:2.2,4096:2.2,16384:2.2,65536:3.7,FULL:7.2},
 "pamica":  {1024:1.6,4096:1.7,16384:2.4,65536:2.9,FULL:6.2},
 "pyamica": {4096:1.9,16384:2.7,65536:3.8,FULL:9.8},
 "amica_python":   {1024:2.0,4096:2.0,16384:2.0,65536:2.3,FULL:4.2},
 "fortran": {1024:0.6,4096:0.6,16384:0.7,65536:1.3,FULL:3.1},
}
CPU_FIT_MISS = {("pyamica",1024):("timeout","bad")}  # pyamica@1024 (~767-1333 eager blocks/iter): >12h wall
# jamica two orchestrator keys (all traceable to raw/chunk_gpumem_summary.csv):
J_CHUNKED_GPU_NVML, J_FULLBATCH_GPU_NVML = 5.37, 13.37   # GiB NVML median (full-batch key from the prior memory run)
J_CHUNKED_GPU_NVML_MAX, J_FULLBATCH_GPU_NVML_MAX = 7.37, 21.37  # per-subject max (longest recording)
J_CHUNKED_ALLOC, J_FULLBATCH_ALLOC = 1.93, 8.22          # GiB JAX allocator (peak_bytes_in_use), at 262144
J_CHUNKED_GPU_SPI, J_FULLBATCH_GPU_SPI = 0.0203, 0.0204  # s/iter @3000 matched (same run) -> no GPU chunk benefit
J_CHUNKED_CPU_T, J_FULLBATCH_CPU_T = 1985, 4300          # s @1000 (CPU: chunking helps time too)
J_CHUNKED_CPU_RSS, J_FULLBATCH_CPU_RSS = 2.2, 19.8       # GiB
# pAMICA block_size sensitivity, GPU @3000 matched (the ~16x within one impl):
REAL_PAM = [("1024 (near 512 default)",4155.5,1.82,"artifact"),("16384 (tuned)",392.7,2.13,"tuned"),
            ("262144 = largest tested",262.4,6.54,"best")]
# Measured GPU memory DECOMPOSITION (median across 25 subjects) from the per-cell JSONs:
#   context = nvml_post_init_gb (whole-GPU used right after the CUDA context is forced, BEFORE model+data)
#   live    = peak_vram_gb (framework allocator live-tensor peak: JAX peak_bytes_in_use / torch max_allocated)
#   total   = nvml_peak_vram_gb (whole-GPU NVML peak)   -- impl -> chunk -> (context, live, total), GiB.
MEMDECOMP = {
 "jamica":       {65536:(1.02,1.61,5.37), 262144:(1.02,1.93,5.37)},
 "amica_python": {65536:(1.08,1.38,2.83), 262144:(1.08,3.28,4.89)},
 "pamica":       {65536:(1.08,1.75,3.08), 262144:(1.08,4.95,6.54)},
 "pyamica":      {65536:(1.08,3.07,4.46), 262144:(1.08,8.98,10.92)},
}

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
    bandtxt = " · shaded = p25–p75" if band else ""
    cap=f'{sub}{bandtxt} · ◯ = {mark} setting'
    return f'<figure class="cf"><figcaption>{cap}</figcaption>{"".join(s)}</figure>'

def chart_iters(series, ylab, title, sub, impls, y0zero=True):
    # x-axis = iterations (linear), for the chunk-65536 ladder.
    W,H=520,320; ml,mr,mt,mb=64,14,30,48; pw,ph=W-ml-mr,H-mt-mb
    XMAXI=3120.0
    def X(n): return ml+n/XMAXI*pw
    allv=[v for im in impls for v in series[im].values()]
    vmx=max(allv); vmn=min(allv)
    if y0zero: lo,hi=0.0,vmx*1.12
    else:
        pad=(vmx-vmn)*0.15 or abs(vmx)*0.01; lo,hi=vmn-pad,vmx+pad
    def Y(v): return mt+ph-(v-lo)/(hi-lo)*ph
    rng=hi-lo; raw=rng/4; e=10**math.floor(math.log10(raw)); f=raw/e
    step=(1 if f<1.5 else 2 if f<3 else 5 if f<7 else 10)*e
    fmt=("%.0f" if step>=1 else "%.1f" if step>=0.1 else "%.2f" if step>=0.01 else "%.3f")
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    s.append(f'<text x="{ml}" y="16" class="ct">{title}</text>')
    s.append(f'<text x="{ml}" y="{H-6}" class="cx">iterations →</text>')
    s.append(f'<text transform="translate(14,{mt+ph/2}) rotate(-90)" class="cy">{ylab}</text>')
    t0=math.ceil(lo/step)*step
    while t0<=hi+1e-9:
        y=Y(t0); s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{ml-6}" y="{y+3:.1f}" class="cyt">{fmt%t0}</text>'); t0+=step
    for n in (0,1000,2000,3000):
        x=X(n); s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+16}" class="cxt">{n}</text>')
    for im in impls:
        cs=sorted(series[im])
        path=" ".join((("M" if i==0 else "L")+f"{X(n):.1f},{Y(series[im][n]):.1f}") for i,n in enumerate(cs))
        s.append(f'<path d="{path}" fill="none" stroke="{COLOR[im]}" stroke-width="2.4"/>')
        for n in cs: s.append(f'<circle cx="{X(n):.1f}" cy="{Y(series[im][n]):.1f}" r="3" fill="{COLOR[im]}"/>')
    s.append('</svg>')
    return f'<figure class="cf"><figcaption>{sub}</figcaption>{"".join(s)}</figure>'

LAD_TIME={im:{n:LADDER[im][n][0] for n in LADDER[im]} for im in IMPLS}
LAD_LL  ={im:{n:LADDER[im][n][1] for n in LADDER[im]} for im in IMPLS}
c_lt=chart_iters(LAD_TIME,"fit time (s)","GPU · fit time vs iterations","chunk 65536 · per-subject median · slope = s/iter",IMPLS,y0zero=True)
c_ll=chart_iters(LAD_LL,"final log-likelihood","GPU · convergence vs iterations","chunk 65536 · median final LL (higher = better)",IMPLS,y0zero=False)
c_lm=chart_iters(LADDER_MEM,"peak VRAM · NVML (GiB)","GPU · memory vs iterations","chunk 65536 · NVML whole-GPU median (iteration-independent)",IMPLS,y0zero=True)

gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
CPU_CHART=["jamica","pamica","pyamica","amica_python"]   # Fortran excluded from CPU plots (sub-01-only at 4/5 chunks)
c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 3000-iter budget · median",IMPLS,"fastest")
c_gv=chart(gpu_v,None,"peak VRAM · NVML (GiB)",False,"GPU · memory vs chunk","real ds004505 · H100 · whole-GPU NVML peak",IMPLS,"leanest")
c_cr=chart(CPU_RSS,None,"peak RSS (GiB)",False,"CPU · memory vs chunk","real ds004505 · 8 cores · by-subject median RSS",CPU_CHART,"leanest")
c_ct=chart(CPU_FIT,CPU_BAND,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · 8 cores · 1000-iter budget · by-subject median",CPU_CHART,"fastest observed (per-cell optimum unresolved)")
# convergence is now the MEASURED iteration ladder (table below), not a post-hoc estimate.

def ladderrows():
    # measured ladder at chunk 65536: LL at each iteration count + wall time at the endpoint.
    order=["jamica","pyamica","amica_python","pamica"]; r=""
    for im in order:
        cells="".join(f'<td class="num">{LADDER[im][n][1]:.4f}</td>' for n in LAD_ITERS)
        w3=LADDER[im][3000][0]
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td>'
            f'{cells}<td class="num">{w3:.0f}s</td></tr>')
    return r

def memdecomprows(chunk):
    order=["jamica","pyamica","pamica","amica_python"]; r=""
    for im in order:
        ctx,liv,tot=MEMDECOMP[im][chunk]
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td>'
            f'<td class="num">{ctx:.2f}</td><td class="num">{liv:.2f}</td><td class="num">{tot:.2f}</td>'
            f'<td class="num">{tot/liv:.1f}×</td></tr>')
    return r

def memdecomp_chart(chunk):
    order=["pyamica","pamica","amica_python","jamica"]
    W,H=520,210; ml,mr,mt,mb=94,26,30,30; pw=W-ml-mr
    maxv=max(MEMDECOMP[im][chunk][2] for im in order)*1.10
    def X(v): return ml+v/maxv*pw
    rowh=(H-mt-mb)/len(order); bh=min(20,rowh*0.5)
    clab="262K" if chunk>=262144 else f"{chunk//1024}K"
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="GPU memory decomposition at chunk {clab}">']
    s.append(f'<text x="{ml-86}" y="16" class="ct">GPU memory decomposition · chunk {clab} · median</text>')
    v=0.0
    while v<=maxv:
        x=X(v); s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{H-mb}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{H-mb+14:.0f}" class="cxt">{v:.0f}</text>'); v+=2
    s.append(f'<text x="{ml}" y="{H-4}" class="cx">GiB →</text>')
    for i,im in enumerate(order):
        ctx,liv,tot=MEMDECOMP[im][chunk]; y=mt+i*rowh+(rowh-bh)/2
        s.append(f'<text x="{ml-8}" y="{y+bh*0.75:.1f}" class="cyt">{LABEL[im]}</text>')
        s.append(f'<rect x="{ml}" y="{y:.1f}" width="{X(tot)-ml:.1f}" height="{bh}" fill="{COLOR[im]}" opacity="0.14"/>')
        s.append(f'<rect x="{ml}" y="{y:.1f}" width="{max(X(ctx)-ml,1):.1f}" height="{bh}" fill="{COLOR[im]}" opacity="0.9"/>')
        s.append(f'<rect x="{X(ctx):.1f}" y="{y:.1f}" width="{X(ctx+liv)-X(ctx):.1f}" height="{bh}" fill="{COLOR[im]}" opacity="0.45"/>')
        s.append(f'<text x="{X(tot)+4:.1f}" y="{y+bh*0.75:.1f}" class="cxt" style="text-anchor:start">{tot:.1f}</text>')
    s.append('</svg>')
    cap=('dark = context floor (measured, before model+data) · mid = allocator live-tensor peak · '
         'light = remaining NVML (JAX pool / resident data / overhead) · number = NVML total')
    return f'<figure class="cf"><figcaption>{cap}</figcaption>{"".join(s)}</figure>'

def legend(impls):
    it="".join(f'<span class="lg"><i style="background:{COLOR[i]}"></i>{LABEL.get(i,"Fortran amica17 (1 thread)")} <code>{KNOB[i]}</code> <span class="cm">@{COMMIT[i]}</span></span>' for i in impls)
    return f'<div class="legend">{it}</div>'

def convrows():
    # GPU wall time to the budget at chunk=262144, with seconds/iter, iterations-run, final LL.
    order=["jamica","amica_python","pamica","pyamica"]; r=""
    for im in order:
        t=GPU[im][FULL][0]; ll,n0,nmed,n1,spi=GPU_CONV[im]
        nit=f'{nmed:,}' if n0==n1 else f'{nmed:,} <span class="mut">[{n0:,}–{n1:,}]</span>'
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td>'
            f'<td class="num">{t:.0f}s</td><td class="num">{spi:.4f}</td>'
            f'<td class="num">{nit}</td><td class="num">{ll:.4f}</td></tr>')
    return r
def pamrows():
    bd={"artifact":'<span class="badge bad">near default</span>',"tuned":'<span class="badge ok">tuned</span>',"best":'<span class="badge best">largest tested</span>'}
    return "".join(f'<tr><td><code>{cfg}</code></td><td class="num">{t:.0f}s</td><td class="num">{v:.2f} GiB</td><td>{bd[tag]}</td></tr>' for cfg,t,v,tag in REAL_PAM)
def cpufitrows():
    r=""
    for im in ["jamica","pamica","pyamica","amica_python","fortran"]:
        cells=""
        for c in XT:
            ns=CPU_NSUB.get(im,{}).get(c)
            star="" if (ns is None or ns==5) else f'<sup class="st">*{ns}</sup>'
            if c in CPU_FIT[im]:
                cells+=f'<td class="num">{CPU_FIT[im][c]:.0f}s{star}</td>'
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
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}.mut{{color:var(--mut);font-size:.85em}}.st{{color:var(--warn);font-size:.7em}}
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
  <div class="stamp"><span><b>Builds (main):</b></span><span>jamica <code>df18b5e</code></span><span>amica-python <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 64 comp · GPU 3000-iter matched (early-stops off) · CPU 1000-iter (earlier run) · H100 + 8-core AMD EPYC (fir)</span></div>
</header>

<section>
  <h2>How to read this</h2>
  <div class="info-box big"><b>The GPU fits are iteration-matched: every implementation's early-stops
  were disabled, so all run the full 3000 iterations.</b> That removes the earlier confound of
  implementations quitting at different points — GPU wall time is now directly comparable per iteration
  (<b>seconds/iteration × 3000 = wall time</b>, exactly), so time differences reflect genuine
  per-iteration throughput, not who stopped early. At the largest chunk all four run 3,000/3,000
  iterations. Three of the four land within ~0.001 nats of each other in final log-likelihood (jamica
  −1.1005, amica-python −1.1002, pyamica −1.0995); <b>pAMICA is ~0.011 nats lower (−1.1107)</b> in
  median — consistently, at <em>matched</em> iterations — i.e. a small but real
  <em>convergence-quality</em> gap, not an early-stop artifact (it runs the same 3000 iterations as the
  others and still lands ~0.011 nats lower; the measured iteration ladder below shows it still slowly
  improving at 3000 but far short of the others, and we did not test a larger budget). We do not
  verify that the four decompositions are numerically equivalent (component matching against the Fortran
  reference is not part of this pass); equal iterations is not proof of equal solutions. <b>The CPU
  section further down is the earlier, non-iteration-matched run</b> (its own caveats apply; a
  contention-free, iteration-matched CPU re-run is in progress — see that section).</div>
  <div class="callout">
    <div class="stat warn"><div class="big">~25×</div><div class="lab">Widest fit-time range across the setting within a single implementation (amica-python, GPU, iteration-matched). The others span 8–16×; every implementation is chunk-sensitive on time.</div></div>
    <div class="stat"><div class="big">grows</div><div class="lab">Peak VRAM grows with chunk for the torch impls (~2.7–3.6× NVML). jamica's GPU memory is flat in the median (~5.4 GiB; per-subject up to ~7.4 at 262K) on its chunked path — mostly a floor, not a dial.</div></div>
    <div class="stat"><div class="big">GPU ⇄ CPU</div><div class="lab">The fastest setting flips by device: large chunks on the H100, small/mid on CPU. Budgets differ (3000 vs 1000) — do not compare GPU seconds to CPU seconds.</div></div>
  </div>
  <p class="note"><b>On "the largest tested chunk."</b> The biggest setting we swept is 262,144 samples.
  Each recording is {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples, so 262,144 is only ~19–33% of the data — it
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
    chunk grows (jamica 775&nbsp;s → 61&nbsp;s; amica-python 5646&nbsp;s → 230&nbsp;s from 1024 to
    262K, all at matched 3000 iterations). Small chunks under-utilise the device (utilisation not directly
    measured).</li>
    <li><b>Memory grows with chunk for the torch implementations</b> (amica-python ~1.8→4.9, pAMICA
    ~1.8→6.5, pyamica ~3.0→10.9&nbsp;GiB NVML median). <b>jamica's GPU memory is flat in the median
    (~5.4&nbsp;GiB)</b> across chunk on its chunked path — a chunk-independent full-width array sets the
    floor — though at 262K the longest recordings do rise (per-subject 5.4→7.4&nbsp;GiB), so it is
    mostly, not entirely, a memory-flat dial. Nothing exceeded the 80&nbsp;GiB card; the heaviest cell
    median is pyamica ~10.9&nbsp;GiB (per-subject up to ~11.4).</li>
    <li><b>The bands are tight now.</b> With early-stops disabled every subject runs the same 3000
    iterations, so the p25–p75 fit-time spread reflects only per-subject recording length (e.g. pAMICA at
    262K is 258–267&nbsp;s) — the wide, early-stop-driven scatter of the previous run is gone.</li>
  </ul>
</section>

<section>
  <h2>Wall time at matched 3000 iterations (GPU, largest chunk)</h2>
  <p class="sub">Each implementation at chunk 262K on the H100 (per-subject median), <b>iteration-matched
  to 3000</b> (early-stops disabled). Because every implementation now runs the full 3000 iterations,
  <b>seconds/iteration × 3000 = wall time</b> and this ranking is a clean per-iteration-throughput
  comparison — no early-stop confound. <b>Seconds/iteration</b> is still a descriptive per-iteration cost
  (it folds in fixed/compile overhead, and the update rules differ, so it is not an isolated kernel
  speed). "Iters run" is 3,000 for all four; "final LL" is where each landed at 3000 iterations.</p>
  <table><thead><tr><th>Implementation</th><th>Wall time</th><th>s / iter</th><th>Iters run</th><th>Final LL (median)</th></tr></thead><tbody>{convrows()}</tbody></table>
  <p class="note">jamica has both the shortest wall time and by far the lowest cost per iteration
  (~0.020&nbsp;s/iter vs 0.077–0.099 for the others — ~4–5× faster per iteration). At matched iterations
  pyamica is the longest wall time (and lands at the highest final LL), pAMICA is close behind on time but
  lands ~0.011 nats lower, and amica-python sits in between. Equal iterations is not proof of equal
  solutions — but with the budget matched, the wall-time differences here are per-iteration speed, not
  early stopping. As a separate illustration of how far one setting sits from an implementation's best,
  pAMICA's <code>block_size</code> spans ~16× end to end:</p>
  <table><thead><tr><th><code>block_size</code></th><th>Wall time</th><th>Peak VRAM (NVML)</th><th></th></tr></thead><tbody>{pamrows()}</tbody></table>
  <p class="note">pAMICA's shipped default is <code>block_size=512</code>, not re-measured here; the
  nearest tested point, 1024, is already ~16× off its fastest tested setting. (jamica's own shipped
  default — <code>chunk_size=None</code>, the full-batch path — is discussed in the memory note; it was
  not the path swept here.)</p>
</section>

<section>
  <h2>Convergence vs iterations — the measured iteration ladder</h2>
  <p class="sub">A direct, <em>measured</em> ladder (not a post-hoc estimate): at a fixed chunk (65536)
  with early-stops disabled, we ran every implementation to 100, 250, 500, 1000, 2000 and 3000 iterations
  and recorded wall time, final log-likelihood, and peak memory at each — all 25 subjects at every point.
  Fit time is linear in iterations (slope = s/iter); peak memory is flat (iteration-independent);
  convergence (LL) is the quality story — pAMICA is the outlier.</p>
  <div class="grid2"><div class="card">{c_lt}</div><div class="card">{c_ll}</div></div>
  <div class="grid2" style="margin-top:16px"><div class="card">{c_lm}</div>
  <div class="card"><table><thead><tr><th>Impl</th><th class="num">LL@100</th><th class="num">@250</th><th class="num">@500</th><th class="num">@1000</th><th class="num">@2000</th><th class="num">@3000</th><th class="num">wall@3000</th></tr></thead><tbody>{ladderrows()}</tbody></table></div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:16px">
    <li><b>jamica converges fastest in both iterations and wall time.</b> It is within ~0.001 nats of its
    own final LL by ~500 iterations (−1.1013 at 500 → −1.1005 at 3000), and because it is ~4–5× cheaper
    per iteration it gets there in ~14&nbsp;s — versus ~51–57&nbsp;s for the others to run the same 500
    iterations.</li>
    <li><b>pyamica reaches the best (least-negative) reported LL</b> (−1.0995) but uses most of the budget to squeeze the last
    fraction; amica-python tracks it and plateaus around −1.1006.</li>
    <li><b>pAMICA stays ~0.011 nats lower across the whole ladder.</b> Its LL climbs from −1.1293 (100) to
    −1.1107 (3000) but never reaches the other three's ~−1.100. It is <em>still improving</em> at the end
    (~0.0016 nats over the last 1000 iters, vs ≤0.0001 for the others), so this is a genuine
    convergence-quality gap at matched iterations — not an early-stop artifact. We did not run past 3000, so
    the honest claim is that the gap persists through 3000 and is not closing at any practical budget, not
    that it can never close.</li>
  </ul>
</section>

<section>
  <h2>CPU — fit time &amp; memory</h2>
  <p class="sub">Real ds004505, 8 cores, 1000-iter budget. Values are the <b>by-subject median</b>
  (median within each subject over reps, then across the subjects present). <b>Cells have unequal
  subject coverage</b> — see the <code>*n</code> marks in the table and the caveat below. The Fortran
  amica17 reference (single-threaded) is in the table but <b>not plotted</b>, because 4 of its 5 chunk
  cells are a single subject.</p>
  <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
  {legend(CPU_CHART)}
  <table style="margin-top:16px"><thead><tr><th>fit time (s) · by-subject median</th><th class="num">1K</th><th class="num">4K</th><th class="num">16K</th><th class="num">64K</th><th class="num">262K</th></tr></thead><tbody>{cpufitrows()}</tbody></table>
  <p class="note"><code>*n</code> = fewer than 5 subjects in that cell (n shown). Fortran is sub-01 only
  except at 64K; jamica@4K/64K are missing one subject.</p>
  <ul class="tk" style="margin-top:8px">
    <li><b>The fastest setting flips on CPU.</b> The Python implementations are fastest at
    <em>small/mid</em> chunks (roughly 1K–16K) — the opposite of the GPU, where large chunks won —
    consistent with a cache effect (small blocks stay resident; not directly measured). <b>The exact
    per-cell winner is not resolved and we do not name a per-impl optimum:</b> the by-subject bands are
    wide (contention) and coverage is unequal (a cell missing the longest recording looks artificially
    fast — e.g. jamica's 4K cell drops one subject), so read only the broad small/mid-vs-large flip.</li>
    <li><b>CPU AMICA is a heavy method</b> (tens of minutes to hours at 1000 iterations): jamica ~2000&nbsp;s
    at its best CPU setting, amica-python ~2600&nbsp;s, pAMICA ~3900&nbsp;s, pyamica ~10⁴&nbsp;s. Compare
    absolute CPU seconds only coarsely.</li>
    <li><b>Memory grows with chunk on CPU too</b> (pyamica ~9.8, jamica ~7&nbsp;GiB at 262K); small
    chunks bring the Python impls to ~1.6–2.2&nbsp;GiB. The single-threaded Fortran reference is leanest
    in absolute terms (0.6→3.1&nbsp;GiB), though those cells are largely single-subject.</li>
    <li><b>One setting failed:</b> pyamica at 1024 (hundreds–thousands of eager blocks/iter) exceeded the
    12&nbsp;h wall — absent from the curve.</li>
  </ul>
  <div class="warn-box" style="margin-top:14px"><b>On the CPU numbers.</b> Three confounds, all disclosed
  here: (1) <b>contention</b> — CPU fits ran on shared cluster nodes, AMICA is memory-bandwidth-bound,
  and the cluster was busy throughout (no quiet window), which is why the p25–p75 bands are wide;
  (2) <b>unequal subject coverage</b> — cells contain 1–5 subjects (see the <code>*n</code> marks), so a
  cell missing the longest recording looks artificially fast. We therefore aggregate by subject (median
  within subject, then across subjects) and still treat absolute CPU seconds and exact per-cell optima
  as unresolved — only the broad device flip is trustworthy. CPU memory scales with chunk (and with
  recording length). Fortran is single-threaded (<code>OMP_NUM_THREADS=1</code>); per core-second it is
  actually more efficient than the 8-thread Python impls, so it is a reference footprint, not a
  head-to-head. And (3) <b>these CPU fits are not iteration-matched</b> — unlike the GPU section above, they
  used each implementation's own early-stop under a 1000-iteration cap, so CPU wall times are not directly
  per-iteration-comparable and are read only for the coarse device flip. Detail + per-cell data:
  <code>NOTES_measurement.md</code>, <code>raw/</code>.</div>
  <div class="info-box"><b>A contention-free, iteration-matched CPU re-run is in progress.</b> The CPU
  numbers above are from the earlier shared-cluster run (contended, unequal subject coverage, early-stop
  on). A clean re-run — <b>whole-node exclusive</b> (one fit per node, so no memory-bandwidth contention),
  <b>iteration-matched</b> (early-stops disabled), all 25 subjects — is currently executing on the Narval
  cluster and will replace this section when complete.</div>
</section>

<section>
  <h2>A note on the memory numbers</h2>
  <div class="info-box"><b>NVML is the framework-neutral VRAM figure.</b> The per-framework allocator
  counters (JAX <code>peak_bytes_in_use</code>, torch <code>max_memory_allocated</code>) measure only
  live-tensor bytes in each framework's own allocator — they omit the CUDA/cuDNN context and the pool
  the driver actually holds, and the two frameworks count differently, so they are <b>not comparable
  across implementations</b> and understate the real footprint by <b>~1.2–3.3×</b> across the 25
  measured impl×chunk pairs (all in <code>raw/chunk_gpumem_summary.csv</code>) — e.g. jamica chunked
  ~1.6&nbsp;GiB allocator vs ~5.3&nbsp;GiB NVML ≈ 3.3× at small chunks, shrinking to pyamica@262K ~1.2× at
  the largest chunk (the gap shrinks as live tensors grow to dominate the fixed context + pool overhead). The charts use
  <b>NVML whole-GPU 'used'</b> on a dedicated GPU. Two caveats on that meter: it is a 50&nbsp;ms poll (a
  sub-interval spike could be missed), and the frameworks run under different allocator settings (JAX
  with pre-allocation off; torch with its caching allocator on), so NVML is a neutral <em>meter</em>
  over somewhat different <em>protocols</em>.</div>
  <div class="grid2" style="margin:10px 0 6px"><div class="card">{memdecomp_chart(FULL)}</div>
  <div class="card"><table><thead><tr><th>@ 262K</th><th class="num">context</th><th class="num">live-alloc</th><th class="num">NVML total</th><th class="num">NVML÷alloc</th></tr></thead><tbody>{memdecomprows(FULL)}</tbody></table>
  <p class="note" style="padding:0 6px">Measured medians (GiB) across 25 subjects: <b>context</b> = whole-GPU
  used right after the CUDA context is forced, before model/data; <b>live-alloc</b> = framework allocator
  live-tensor peak; <b>NVML total</b> = whole-GPU peak. <b>This pre-fit baseline is ~1&nbsp;GiB for all
  four</b> (JAX 1.02, torch 1.08) — essentially equal. It is measured before the fit, so any kernel/cuDNN
  state loaded lazily during fitting falls into the remainder below, not this baseline; we therefore claim
  only that the measured <em>pre-fit</em> floor does not differ across frameworks, not that the complete
  fixed framework context is ~1&nbsp;GiB. What differs is the resident/pool memory: torch's NVML ≈ baseline
  + its live/reserved pool (and scales with chunk), whereas jamica's allocator peak stays small (~1.9)
  while its NVML is ~5.4 — JAX holds a larger, chunk-independent resident/pool footprint the allocator
  counter undercounts most.</p></div></div>
  <p class="note"><b>jamica memory is two-level, and it cuts both ways.</b> On its chunked path jamica's
  median NVML is ~5.4&nbsp;GiB across chunk sizes (per-subject ~3.4–5.4&nbsp;GiB below 262K, rising to
  5.4–{J_CHUNKED_GPU_NVML_MAX:.1f}&nbsp;GiB at 262K for the longest recordings — flat in the median
  across chunk, a floor, not a single constant). Its <em>full-batch</em> path
  (<code>chunk_size=None</code> — a different orchestrator key, and jamica's shipped default) uses
  ~{J_FULLBATCH_GPU_NVML:.1f}&nbsp;GiB NVML median (per-subject up to ~{J_FULLBATCH_GPU_NVML_MAX:.0f} for
  the longest recording) for essentially the <b>same GPU speed</b> (~{J_FULLBATCH_GPU_SPI:.3f} vs
  ~{J_CHUNKED_GPU_SPI:.3f}&nbsp;s/iter — no chunk benefit on GPU; the full-batch figure is from the earlier
  memory run). On CPU, chunking helps
  <em>both</em> axes (~{J_CHUNKED_CPU_T:,}&nbsp;s /
  ~{J_CHUNKED_CPU_RSS:.1f}&nbsp;GiB chunked vs ~{J_FULLBATCH_CPU_T:,}&nbsp;s /
  ~{J_FULLBATCH_CPU_RSS:.0f}&nbsp;GiB full-batch). So under these tested conditions a wrapper should pass
  a chunk. The flip side, in fairness: jamica's ~5.4&nbsp;GiB chunked <em>floor</em> is higher than the
  torch impls' small-chunk footprint (~1.8–3.1&nbsp;GiB NVML) — on a small card the torch impls at a
  small chunk fit where jamica may not. (The "fits an 8–12&nbsp;GiB card" reading is an extrapolation
  from H100 NVML with JAX pre-allocation off; it was not measured on such a card.)</p>
  <div class="info-box"><b>"Flat jamica memory" is not the earlier full-batch measurement bug.</b> Flat
  memory was the signature of a fixed bug where jamica was accidentally run full-batch at every chunk, so
  we checked. Two things rule it out here: (1) jamica's <b>fit time varies ~13× with the chunk</b>
  (775&nbsp;s→61&nbsp;s) — only possible if the chunk is actually applied; the true full-batch path is
  chunk-<em>independent</em> in time too (~61&nbsp;s at every chunk). (2) The chunked path's memory
  (~5.4&nbsp;GiB NVML / ~{J_CHUNKED_ALLOC:.1f}&nbsp;GiB allocator) is less than half the full-batch path's
  (~{J_FULLBATCH_GPU_NVML:.1f}&nbsp;/&nbsp;{J_FULLBATCH_ALLOC:.1f}&nbsp;GiB) — different numbers, different
  code path (both in <code>raw/chunk_gpumem_summary.csv</code>). The chunked NVML is flat because of what
  it's made of: the measured context floor is only ~1&nbsp;GiB (chunk-independent, and about the same for
  JAX and torch — see the decomposition above), and the rest of the ~5.4&nbsp;GiB is JAX's memory pool plus
  <em>chunk-independent resident arrays</em> (the whitened data and source outputs, sized by
  <code>n_samples</code>); the framework allocator counter (<code>peak_bytes_in_use</code>
  ~{J_CHUNKED_ALLOC:.1f}&nbsp;GiB) undercounts that resident/pooled footprint the most for JAX, which is
  why jamica's NVML sits far above its allocator peak. The chunk-scaled E-step block buffer is small next
  to the resident arrays until 262K, where it begins to add (allocator 1.6→1.9&nbsp;GiB; per-subject NVML
  up to ~7.4). So chunk is a real <em>time</em> dial for jamica but not a GPU-<em>memory</em> dial — a
  genuine property of the chunked path, distinct from the full-batch key. <b>And it is by design, not a
  leak:</b> the chunked E-step accumulates small per-chunk sufficient statistics
  (<code>O(n_comp²)</code>) and never materialises the full-width <code>(n_comp, n_samples)</code>
  per-sample tensors — those appear only on the full-batch path. Of the ~5.4&nbsp;GiB whole-GPU NVML, only
  ~1&nbsp;GiB is the measured pre-fit context floor and ~1.6&nbsp;GiB is the allocator's live-tensor peak
  (resident whitened data + accumulators); the remaining ~2.4&nbsp;GiB is JAX pool / resident bytes the
  allocator counter does not report / post-init runtime, which we do not separate further — it is
  chunk-independent (see the decomposition above). Chunking bounds the chunk-scaled working set as
  intended; it simply cannot go below that context + pool + resident-data floor (which is <em>not</em> a
  ~3.8&nbsp;GiB context — the context alone is ~1&nbsp;GiB).</div>
</section>

<footer>
  <div class="kick" style="color:var(--mut)">Provenance &amp; reproduction</div>
  <dl class="prov">
    <dt>Dataset</dt><dd>ds004505 · 64 PCA components · {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples/subject</dd>
    <dt>GPU fit @3000 (matched)</dt><dd>iteration-matched, early-stops disabled → raw/nostop_gpu3000_summary.csv (Trillium H100, 25 subj/cell, all n_iter=3000)</dd>
    <dt>GPU convergence ladder</dt><dd>raw/nostop_ladder_i{100,250,500,1000,2000,3000}_summary.csv (chunk 65536, all 25 subj, early-stops disabled)</dd>
    <dt>GPU memory (NVML+alloc)</dt><dd>raw/chunk_gpumem_*.csv + raw/nostop_gpumem_summary.csv (iteration-independent; per-chunk NVML also confirmed by the matched i3000 run; nvml_min/max = per-subject range; full-batch key from the prior memory run)</dd>
    <dt>CPU @1000 (earlier)</dt><dd>raw/chunk_cpu1000_*.csv (fir 8 cores; by-subject median; n_subjects in the summary; NOT iteration-matched — superseded by the Narval re-run in progress)</dd>
    <dt>Budget</dt><dd>GPU 3000-iter MATCHED (early-stops disabled, all impls run full 3000) · CPU 1000-iter (earlier run, each impl's early-stop on) · memory is iteration-independent</dd>
    <dt>jamica path</dt><dd>amica_python_jax_chunked (chunked); full-batch key amica_python_jax in the memory note only</dd>
    <dt>Units</dt><dd>memory in GiB (bytes / 1024³); times in seconds; GPU per-subject median, CPU by-subject median</dd>
    <dt>Commits</dt><dd>jamica df18b5e · amica-python e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
    <dt>Caveats</dt><dd>NOTES_measurement.md (iteration-budget ≠ convergence · CPU contention + unequal coverage · NVML vs allocator · the two jamica keys)</dd>
  </dl>
  <p class="note" style="margin-top:16px">Trust the curve shapes, the NVML memory figures, the GPU
  per-iteration speeds, and the convergence columns read together. Trust the <em>broad</em> GPU-large /
  CPU-small-to-mid device flip — not the exact CPU per-cell optima (contention + unequal subject
  coverage). Fit times are wall time to a fixed iteration budget, <b>not</b> time to an equivalent
  solution. jamica's chunk is a real GPU-time / CPU-time+memory dial; its GPU memory on the chunked path
  is flat in the median across chunk (per-subject it rises at 262K for the longest recordings).</p>
</footer>
</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
open(os.path.join(HERE,"xperf_chunk_report.html"),"w").write(HTML)
_title = HTML.split("<title>",1)[1].split("</title>",1)[0] if "<title>" in HTML else "AMICA chunk-size report"
STANDALONE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              f'<title>{_title}</title></head><body>\n{HTML}\n</body></html>\n')
open(os.path.join(HERE,"xperf_chunk_report_standalone.html"),"w").write(STANDALONE)

# tidy dataset — regenerated from the report dicts; authoritative raw per-cell data lives in raw/.
def _cn(c): return str(c)
_rows = [("dataset", "impl", "knob", "chunk", "value", "unit", "note")]
for im in IMPLS:
    for c, (t, v) in sorted(GPU[im].items()):
        _rows.append(("gpu_fit_s_median", im, KNOB[im], _cn(c), t, "s", "GPU H100 3000-iter MATCHED (early-stops disabled), per-subj median"))
        _rows.append(("gpu_vram_gib_nvml", im, KNOB[im], _cn(c), v, "GiB", "NVML whole-GPU median, iteration-matched i3000 run (raw/nostop_gpu3000_summary.csv)"))
    for c, (lo, hi) in sorted(GPU_BAND[im].items()):
        _rows.append(("gpu_fit_s_p25", im, KNOB[im], _cn(c), lo, "s", ""))
        _rows.append(("gpu_fit_s_p75", im, KNOB[im], _cn(c), hi, "s", ""))
for im, (ll, n0, nmed, n1, spi) in GPU_CONV.items():
    _rows.append(("gpu_ll_final_median", im, KNOB[im], _cn(FULL), ll, "nats", "at chunk 262144"))
    _rows.append(("gpu_s_per_iter_median", im, KNOB[im], _cn(FULL), spi, "s/iter", "at chunk 262144"))
    _rows.append(("gpu_n_iter_min", im, KNOB[im], _cn(FULL), n0, "iters", "at chunk 262144"))
    _rows.append(("gpu_n_iter_median", im, KNOB[im], _cn(FULL), nmed, "iters", "at chunk 262144"))
    _rows.append(("gpu_n_iter_max", im, KNOB[im], _cn(FULL), n1, "iters", "at chunk 262144"))
for im in ["jamica","pamica","pyamica","amica_python","fortran"]:
    for c, v in sorted(CPU_RSS[im].items()):
        _rows.append(("cpu_rss_gib_median", im, KNOB[im], _cn(c), v, "GiB", "fir 8 cores 1000-iter, by-subject median"))
    for c, v in sorted(CPU_FIT[im].items()):
        ns = CPU_NSUB.get(im,{}).get(c,"")
        _rows.append(("cpu_fit_s_bysubj_median", im, KNOB[im], _cn(c), v, "s", f"by-subject median, n_subjects={ns}"))
    for c, (lo, hi) in sorted(CPU_BAND.get(im, {}).items()):
        _rows.append(("cpu_fit_s_bysubj_p25", im, KNOB[im], _cn(c), lo, "s", ""))
        _rows.append(("cpu_fit_s_bysubj_p75", im, KNOB[im], _cn(c), hi, "s", ""))
for (im, c), (lab, _cls) in CPU_FIT_MISS.items():
    _rows.append(("cpu_fit_s_bysubj_median", im, KNOB[im], _cn(c), lab, "", "no value: 12h wall timeout"))
for cfg, t, v, tag in REAL_PAM:
    _rows.append(("gpu_pamica_sensitivity", "pamica", "block_size", cfg, t, "s", tag))
with open(os.path.join(HERE, "chunk_sweep_data.csv"), "w", newline="") as _f:
    csv.writer(_f).writerows(_rows)

print("wrote CORRECTED report", len(HTML), "bytes (+ standalone", len(STANDALONE),
      "bytes) + chunk_sweep_data.csv", len(_rows) - 1, "rows")
