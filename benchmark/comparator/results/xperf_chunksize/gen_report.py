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
 - CPU FIT + RSS @250, WHOLE-NODE exclusive (one fit per node -> no contention), iteration-matched
   (early-stops disabled), 25 subjects, all 5 impls incl Fortran (Narval 64-core Zen2).
   -> raw/narval_nostop_i250_summary.csv (per-subject median; t_subj_median, t_subj_p25/p75, mem_median).

jamica = the CHUNKED path (amica_python_jax_chunked). Memory in GiB (bytes/1024**3).

CAVEATS baked into the report (see NOTES_measurement.md):
 - GPU fit time is wall time at a MATCHED 3000 iterations (early-stops disabled, all impls run the full
   3000), so it is directly per-iteration-comparable (s/iter x 3000 = wall). CPU is a separate MATCHED run
   at 250 iters, whole-node exclusive (no contention) -> CPU absolute seconds and per-cell optima ARE
   trustworthy; only GPU-vs-CPU comparison is off-limits (different iteration budgets).
 - All CPU cells cover 25 subjects; Fortran (single-threaded) is a reference footprint, not a fair-thread
   comparison on a whole node.
"""
import math, os, csv

FULL = 262144                       # the LARGEST TESTED chunk (samples); NOT a full-batch pass.
N_SAMP_MIN, N_SAMP_MAX = 785328, 1364633   # per-subject sample counts; 262144 = ~19-33% of a recording
# per-subject sample counts (25 subjects, ds004505 @250 Hz) for the duration-distribution plot
SUBJ_SAMPLES = [785328,1364633,1099796,1057036,1038926,1042909,1174503,1159910,1146679,1139466,1104857,
                1111932,1120666,1129764,1105757,1151916,1119559,1155503,1117345,916388,1128361,1108947,
                1053821,1146791,877574]
SFREQ = 250.0
IMPLS = ["jamica", "pamica", "pyamica", "amica_python"]
LABEL = {"jamica":"jamica","pamica":"pAMICA","pyamica":"pyamica","amica_python":"amica-python"}
KNOB  = {"jamica":"chunk_size","pamica":"block_size","pyamica":"chunk_t","amica_python":"batch_size","fortran":"block_size"}
COMMIT= {"jamica":"df18b5e","pamica":"0c4da39","pyamica":"a8a4d7e","amica_python":"e15e158","fortran":"665b577"}
COLOR = {"jamica":"#6366f1","pamica":"#d97706","pyamica":"#0d9488","amica_python":"#e11d48","fortran":"#111827"}

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
# per-chunk GPU memory decomposition (median over 25 subj, from cell JSONs): context = nvml_post_init
# (pre-fit baseline), live = framework allocator peak. NVML total per chunk = gpu_v (from the GPU dict).
MEM_CTX = {
 "jamica":{1024:1.02,4096:1.02,16384:1.02,65536:1.02,FULL:1.02},
 "pamica":{1024:1.08,4096:1.08,16384:1.08,65536:1.08,FULL:1.08},
 "pyamica":{1024:1.08,4096:1.08,16384:1.08,65536:1.08,FULL:1.08},
 "amica_python":{1024:1.08,4096:1.08,16384:1.08,65536:1.08,FULL:1.08},
}
MEM_LIVE = {
 "jamica":{1024:1.61,4096:1.61,16384:1.61,65536:1.61,FULL:1.93},
 "pamica":{1024:0.58,4096:0.64,16384:0.86,65536:1.75,FULL:4.95},
 "pyamica":{1024:1.66,4096:1.66,16384:1.66,65536:3.07,FULL:8.98},
 "amica_python":{1024:0.58,4096:0.62,16384:0.77,65536:1.38,FULL:3.28},
}
# ===== CPU @250, WHOLE-NODE exclusive (one fit per node -> no memory-bandwidth contention),
# iteration-matched (early-stops disabled), per-subject median. Narval 64-core Zen2. All 5 impls incl
# Fortran, 25 subjects (all cells, after the repair). raw/narval_nostop_i250_summary.csv
CPU_FIT = {
 "jamica":  {1024:1269,4096:976,16384:1277,65536:938,FULL:753},
 "pamica":  {1024:7400,4096:2806,16384:1527,65536:2064,FULL:1536},
 "pyamica": {1024:3099,4096:1921,16384:1211,65536:2409,FULL:1328},
 "amica_python":   {1024:5027,4096:1710,16384:1295,65536:1084,FULL:1053},
 "fortran": {1024:3676,4096:3679,16384:4577,65536:4004,FULL:4118},
}
CPU_BAND = {  # CPU per-subject p25,p75 (whole-node exclusive -> tight, no contention)
 "jamica":  {1024:(1168,1326),4096:(946,1009),16384:(1099,1438),65536:(877,1206),FULL:(730,792)},
 "pamica":  {1024:(6892,8042),4096:(2718,3015),16384:(1309,1815),65536:(1917,2158),FULL:(1402,1684)},
 "pyamica": {1024:(2408,3622),4096:(1850,2072),16384:(1153,1278),65536:(2223,2510),FULL:(1272,1464)},
 "amica_python":   {1024:(4351,6011),4096:(1522,1779),16384:(1204,1394),65536:(1040,1150),FULL:(982,1074)},
 "fortran": {1024:(3466,3752),4096:(3451,3794),16384:(4415,4735),65536:(3814,4239),FULL:(3988,4264)},
}
CPU_NSUB = {  # subjects per cell -- all 25 after the repair
 "jamica":  {1024:25,4096:25,16384:25,65536:25,FULL:25},
 "pamica":  {1024:25,4096:25,16384:25,65536:25,FULL:25},
 "pyamica": {1024:25,4096:25,16384:25,65536:25,FULL:25},
 "amica_python":   {1024:25,4096:25,16384:25,65536:25,FULL:25},
 "fortran": {1024:25,4096:25,16384:25,65536:25,FULL:25},
}
CPU_RSS = {  # peak RSS (GiB) per-subject median (whole-node; iteration-independent)
 "jamica":  {1024:2.35,4096:2.34,16384:2.85,65536:6.26,FULL:10.34},
 "pamica":  {1024:1.63,4096:1.74,16384:2.25,65536:2.76,FULL:6.08},
 "pyamica": {1024:2.05,4096:2.05,16384:2.80,65536:3.84,FULL:9.59},
 "amica_python":   {1024:2.13,4096:2.14,16384:2.13,65536:2.36,FULL:4.25},
 "fortran": {1024:0.84,4096:0.84,16384:0.85,65536:1.31,FULL:3.09},
}
CPU_FIT_MISS = {}  # no timeouts on the whole-node run (pyamica@1024 completed: ~3099 s)
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

# ===== LARGE-CHUNK / FULL-BATCH EXTENSION (GPU, 2026-08-17) — raw/nostop_gpu_ext_summary.csv.
# Adds 512K (=262K x2), 1M (=262K x4), and full-batch. Two subtleties, both disclosed in the report:
#  * Above the shortest recording (785,328 samples) a chunk can exceed a subject's data. jamica/pyamica/
#    pAMICA silently CLAMP the chunk to full-batch for that subject; amica-python REJECTS it
#    (BatchLoader raises: "batch_size N exceeds data size M"), i.e. it cannot run a single-chunk pass with
#    an oversized batch. So the 1M point is restricted to the 22 subjects longer than 1M for ALL impls
#    (apples-to-apples on genuinely-chunked data); the short subjects appear only at the full-batch point.
#  * Full-batch = one pass over the whole recording (per-subject batch = n_samples for amica-python; the
#    others clamp their chunk to n_samples). n contributing subjects is carried per point (preemption +
#    the n22 restriction => some n<25) and shown in the coverage table.
# chunk -> (fit_s, nvml_gib, alloc_gib, reserved_gib|None, n_subjects)
FB = 4194304                                     # x-axis sentinel for the full-batch point (label "full")
C512, C1M = 524288, 1048576
GEXT = {
 "jamica":       {C512:(59.9,13.37,5.75,12.00,25), C1M:(58.8,13.37,5.76,12.00,20), FB:(61.7,13.37,8.18,12.00,24)},
 "amica_python": {C512:(211.3,7.57,5.77,6.33,20), C1M:(209.0,13.32,10.78,12.09,20), FB:(197.9,14.10,11.46,12.87,25)},
 "pyamica":      {C512:(287.0,19.06,16.86,17.82,20), C1M:(283.7,28.06,26.63,26.82,20), FB:(278.5,29.56,28.16,28.33,21)},
 "pamica":       {C512:(248.8,10.57,9.32,10.09,19), C1M:(247.9,19.33,18.09,18.09,17), FB:(242.5,20.50,19.25,19.26,22)},
}
GEXT_BAND = {  # fit p25,p75 at the extension chunks (iteration-matched @3000)
 "jamica":       {C512:(59.0,61.3), C1M:(58.0,59.9), FB:(58.7,69.7)},
 "amica_python": {C512:(202.0,215.9), C1M:(204.6,211.6), FB:(193.2,202.7)},
 "pyamica":      {C512:(267.4,293.2), C1M:(280.1,289.3), FB:(262.7,285.2)},
 "pamica":       {C512:(236.5,254.0), C1M:(243.7,252.8), FB:(237.3,247.6)},
}
# allocator reserved-pool high-water — the OOM-relevant counter. torch: max_memory_reserved; JAX/jamica:
# peak_pool_bytes (the XLA BFC pool — the JAX analog). 262K from the i3000 run; 512K/1M from the extension
# (full-batch is in the table, not charted). jamica pool steps 4->12 GiB at 512K, mirroring its NVML step.
MEM_RESV = {
 "jamica":       {262144:4.00, C512:12.00, C1M:12.00},
 "amica_python": {262144:3.66, C512:6.33,  C1M:12.09},
 "pyamica":      {262144:9.68, C512:17.82, C1M:26.82},
 "pamica":       {262144:5.34, C512:10.09, C1M:18.09},
}
# fold ONLY the on-axis chunks (512K, 1M) into the charted dicts. Full-batch (FB) is a regime, not a chunk
# size, so it is kept OUT of the vs-chunk charts and shown in its own table (fullbatchrows) — GEXT[im][FB].
for _im in IMPLS:
    for _c in (C512, C1M):
        _fit,_nvml,_alloc,_resv,_n = GEXT[_im][_c]
        GPU[_im][_c] = (_fit, _nvml)
        GPU_BAND[_im][_c] = GEXT_BAND[_im][_c]
        MEM_LIVE[_im][_c] = _alloc
        MEM_CTX[_im][_c]  = MEM_CTX[_im][FULL]      # context floor ~1 GiB, chunk-independent
GX_N = {_im:{_c:GEXT[_im][_c][4] for _c in (C512, C1M)} for _im in IMPLS}   # subjects per on-axis large-chunk point
GPU_CEILINGS = [(24,"24"),(40,"40"),(80,"80")]   # GiB card capacities (labels kept short; devices in caption)

def xlog(c): return math.log2(c)
XT=[1024,4096,16384,65536,FULL]                                  # original CPU / ladder axis (1K–262K)
GXT=[1024,4096,16384,65536,FULL,C512,C1M]                        # extended GPU chunk axis (log; 1K–1M, no full-batch)
CLAB={1024:"1K",4096:"4K",16384:"16K",65536:"64K",FULL:"262K",C512:"512K",C1M:"1M",FB:"full"}
XMIN,XMAX=math.log2(1024)-0.4,xlog(FULL)+0.4

def chart(series, band, ylab, ylog, title, sub, impls, mark, oom=None, xt=None, hlines=None):
    xt = xt or XT
    xmn, xmx = math.log2(min(xt))-0.4, math.log2(max(xt))+0.4
    W,H=520,340; ml,mr,mt,mb=56,(34 if hlines else 14),32,52; pw,ph=W-ml-mr,H-mt-mb
    def X(c): return ml+(xlog(c)-xmn)/(xmx-xmn)*pw
    allv=[v for im in impls for v in series[im].values()]
    if band: allv+=[b for im in impls for c in series[im] if im in band and c in band[im] for b in band[im][c]]
    if hlines: allv+=[hv for hv,_ in hlines]
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
    for c in xt:
        x=X(c); lab=CLAB.get(c, f'{c//1024}K')
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+16}" class="cxt">{lab}</text>')
    if hlines:
        for hv,hl in hlines:
            if not (vmin*0.5 <= hv <= vmax*1.3): continue
            y=Y(hv)
            s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="#e11d48" stroke-width="1.2" stroke-dasharray="5 3" opacity="0.8"/>')
            s.append(f'<text x="{W-mr+3}" y="{y+3:.1f}" class="cxt" style="text-anchor:start;fill:#e11d48">{hl}</text>')
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

def chart_durations():
    vals=sorted(SUBJ_SAMPLES); n=len(vals); med=vals[n//2]
    W,H=520,300; ml,mr,mt,mb=54,66,30,44; pw,ph=W-ml-mr,H-mt-mb
    ymax=max(vals)*1.06
    def Y(v): return mt+ph-v/ymax*ph
    def X(i): return ml+(i+0.5)*pw/n
    bw=pw/n*0.72
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="signal duration distribution">']
    s.append(f'<text x="{ml-2}" y="16" class="ct">Signal duration per subject (25) vs chunk sizes</text>')
    s.append(f'<text transform="translate(13,{mt+ph/2}) rotate(-90)" class="cy">samples (millions)</text>')
    s.append(f'<text x="{ml}" y="{H-6}" class="cx">subjects, sorted by length →</text>')
    t=0.0
    while t<=ymax:
        y=Y(t); s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{ml-6}" y="{y+3:.1f}" class="cyt">{t/1e6:.2f}</text>'); t+=250000
    for i,v in enumerate(vals):
        s.append(f'<rect x="{X(i)-bw/2:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{mt+ph-Y(v):.1f}" fill="#6366f1" opacity="0.5"/>')
    for c,lab in [(65536,"64K"),(262144,"262K"),(1048576,"1M")]:
        y=Y(c)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="4 3"/>')
        s.append(f'<text x="{W-mr+3}" y="{y+3:.1f}" class="cxt" style="text-anchor:start;fill:#e11d48">{lab} ({c/med*100:.0f}%)</text>')
    s.append('</svg>')
    cap=(f'25 subjects, {N_SAMP_MIN/SFREQ/60:.0f}–{N_SAMP_MAX/SFREQ/60:.0f} min (median {med/SFREQ/60:.0f} min '
         f'@{SFREQ:.0f} Hz). Dashed = tested chunk sizes as % of the median recording; full-batch = the whole bar.')
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

def chart_wallbar():
    order=sorted(IMPLS, key=lambda im: GPU[im][FULL][0])   # ascending wall time at 262K
    W,H=520,200; ml,mr,mt,mb=96,46,32,30; pw=W-ml-mr
    maxv=max(GPU[im][FULL][0] for im in order)*1.14
    def X(v): return ml+v/maxv*pw
    rowh=(H-mt-mb)/len(order); bh=min(24,rowh*0.55)
    s=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="wall time at matched 3000 iterations">']
    s.append(f'<text x="{ml-90}" y="16" class="ct">GPU · wall time @ matched 3000 iters · chunk 262K</text>')
    v=0.0
    while v<=maxv:
        x=X(v); s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{H-mb}" class="grid vg"/>')
        s.append(f'<text x="{x:.1f}" y="{H-mb+14:.0f}" class="cxt">{v:.0f}</text>'); v+=100
    s.append(f'<text x="{ml}" y="{H-4}" class="cx">wall time (s) →</text>')
    for i,im in enumerate(order):
        t=GPU[im][FULL][0]; y=mt+i*rowh+(rowh-bh)/2
        s.append(f'<text x="{ml-8}" y="{y+bh*0.72:.1f}" class="cyt">{LABEL[im]}</text>')
        s.append(f'<rect x="{ml}" y="{y:.1f}" width="{X(t)-ml:.1f}" height="{bh}" fill="{COLOR[im]}" opacity="0.85" rx="2"/>')
        s.append(f'<text x="{X(t)+5:.1f}" y="{y+bh*0.72:.1f}" class="cxt" style="text-anchor:start">{t:.0f}s</text>')
    s.append('</svg>')
    return f'<figure class="cf"><figcaption>wall time = s/iter × 3000 (early-stops disabled); jamica ~4–5× faster per iteration</figcaption>{"".join(s)}</figure>'

LAD_TIME={im:{n:LADDER[im][n][0] for n in LADDER[im]} for im in IMPLS}
LAD_LL  ={im:{n:LADDER[im][n][1] for n in LADDER[im]} for im in IMPLS}
c_lt=chart_iters(LAD_TIME,"fit time (s)","GPU · fit time vs iterations","chunk 65536 · per-subject median · slope = s/iter",IMPLS,y0zero=True)
c_ll=chart_iters(LAD_LL,"final log-likelihood","GPU · convergence vs iterations","chunk 65536 · median final LL (higher = better)",IMPLS,y0zero=False)
c_lm=chart_iters(LADDER_MEM,"peak VRAM · NVML (GiB)","GPU · memory vs iterations","chunk 65536 · NVML whole-GPU median (iteration-independent)",IMPLS,y0zero=True)

gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
CPU_CHART=["jamica","pamica","pyamica","amica_python","fortran"]   # Fortran now has full 25-subject CPU coverage (whole-node run)
TORCH=["amica_python","pyamica","pamica"]   # impls with a torch reserved counter (jamica=JAX, none)
c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 3000-iter matched · per-subject median",IMPLS,"fastest",xt=GXT)
# GPU section memory chart = NVML (true whole-GPU ceiling), log-y, with real GPU-capacity reference lines
c_gv=chart(gpu_v,None,"peak VRAM · NVML (GiB, log)",True,"GPU · memory vs chunk (with GPU capacities)","real ds004505 · whole-GPU NVML peak · dashed = card VRAM (24 RTX/A10 · 40 A100-40 · 80 H100), GiB",IMPLS,"leanest",xt=GXT,hlines=GPU_CEILINGS)
c_cr=chart(CPU_RSS,None,"peak RSS (GiB)",False,"CPU · memory vs chunk","real ds004505 · Narval whole-node · per-subject median RSS",CPU_CHART,"leanest")
c_ct=chart(CPU_FIT,CPU_BAND,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · Narval whole-node exclusive · 250-iter matched · per-subject median",CPU_CHART,"fastest")
# GPU memory decomposition, 2x2 vs chunk (allocator-live ⊆ reserved ⊆ NVML total; + understatement factor)
c_mliv=chart(MEM_LIVE,None,"allocator live peak (GiB)",False,"GPU · allocator live-tensor vs chunk","framework counter (peak_bytes_in_use / max_allocated) · per-subj median",IMPLS,"leanest",xt=GXT)
c_mresv=chart(MEM_RESV,None,"reserved / pool peak (GiB)",False,"GPU · reserved allocator pool vs chunk","OOM-relevant: torch max_memory_reserved · JAX peak_pool_bytes (XLA BFC pool)",IMPLS,"leanest",xt=GXT)
c_mtot=chart(gpu_v,None,"NVML whole-GPU peak (GiB)",False,"GPU · NVML total vs chunk","framework-neutral whole-GPU peak · per-subj median",IMPLS,"leanest",xt=GXT)
MEM_RATIO={im:{c:round(gpu_v[im][c]/MEM_LIVE[im][c],2) for c in gpu_v[im]} for im in IMPLS}
c_mrat=chart(MEM_RATIO,None,"NVML ÷ allocator (×)",False,"GPU · allocator understatement vs chunk","NVML total ÷ allocator peak · per-subj median",IMPLS,"lowest",xt=GXT)
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
def covrows():
    # subjects contributing to each on-axis GPU large-chunk point (262K & below = all 25).
    order=["jamica","amica_python","pyamica","pamica"]; r=""
    for im in order:
        cells="".join(f'<td class="num">{GX_N[im][c]}</td>' for c in (C512,C1M))
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}</td>'
            f'<td class="num">25</td>{cells}</tr>')
    return r

def fullbatchrows():
    # full-batch (one pass over the whole recording) — its own table, not on the chunk axis.
    order=["jamica","amica_python","pamica","pyamica"]; r=""
    for im in order:
        fit,nvml,alloc,resv,n = GEXT[im][FB]
        resv_s = f'{resv:.1f}' if resv is not None else '—'
        note = ' <span class="mut">· per-subject batch</span>' if im=="amica_python" else ''
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{LABEL[im]}{note}</td>'
            f'<td class="num">{fit:.0f}s</td><td class="num">{alloc:.1f}</td>'
            f'<td class="num">{resv_s}</td><td class="num">{nvml:.1f}</td><td class="num">{n}</td></tr>')
    return r

def cpufitrows():
    r=""
    for im in ["jamica","pamica","pyamica","amica_python","fortran"]:
        cells=""
        for c in XT:
            ns=CPU_NSUB.get(im,{}).get(c)
            star="" if (ns is None or ns==25) else f'<sup class="st">*{ns}</sup>'
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
  <p class="lede">Every Python AMICA implementation exposes one batch/chunk-size knob. We swept it on
  real EEG (ds004505), GPU and CPU, at matched iterations — and it is a big dial. Three takeaways: on the
  GPU the speed-up <b>saturates by a ~262K chunk</b> (bigger buys almost no time and a lot of memory);
  <b>peak VRAM climbs steeply toward the card ceiling</b> (pyamica ~30&nbsp;GiB at full-batch); and
  <b>small chunks are fastest on neither device</b>. Fit times are wall time to a fixed iteration budget,
  not time to an equivalent solution — read them with the convergence section.</p>
  <div class="stamp"><span><b>Builds (main):</b></span><span>jamica <code>df18b5e</code></span><span>amica-python <code>e15e158</code></span><span>pyamica <code>a8a4d7e</code></span><span>pAMICA <code>0c4da39</code></span><span>Fortran ref <code>665b577</code></span><span>· 64 comp · GPU 3000-iter matched · CPU 250-iter matched, whole-node · H100 (Trillium) + 64-core Zen2 (Narval)</span></div>
</header>

<section>
  <h2>How to read this</h2>
  <div class="info-box big"><b>The GPU fits are iteration-matched</b> — early-stops disabled, so all four
  run the full 3000 iterations and GPU wall time is directly comparable per iteration
  (<b>seconds/iteration × 3000 = wall</b>, exactly). At matched iterations three of the four land within
  ~0.001 nats in final log-likelihood (jamica −1.1005, amica-python −1.1002, pyamica −1.0995);
  <b>pAMICA sits ~0.011 nats lower (−1.1107)</b> — a small but real convergence-quality gap (detailed in
  the iteration ladder), not an early-stop artifact. Equal iterations is not proof of equal solutions — we
  did not component-match the decompositions. <b>The CPU section is a separate whole-node run at 250
  iterations</b>: clean absolutes, but a different budget from the GPU's — don't compare GPU and CPU
  seconds.</div>
  <div class="callout">
    <div class="stat warn"><div class="big">~25×</div><div class="lab">Widest fit-time range across the setting within a single implementation (amica-python, GPU, iteration-matched). The others span 8–16×; every implementation is chunk-sensitive on time.</div></div>
    <div class="stat"><div class="big">grows</div><div class="lab">Peak VRAM grows with chunk for the torch impls — pyamica reaches ~29.6 GiB NVML at full-batch (would OOM any card &lt;32 GiB). jamica is flat (~5.4 GiB) only through 262K, then climbs to ~13.4 GiB once the chunk is large — its chunked path converges to the full-batch footprint.</div></div>
    <div class="stat"><div class="big">no CPU flip</div><div class="lab">The old "small/mid wins on CPU" flip did not survive contention-free whole nodes — small chunks are best on neither device. But the CPU optimum is impl-specific (262K for jamica/amica-python, 16K for pAMICA/pyamica, 1K for Fortran), not uniformly large. Budgets differ (GPU 3000, CPU 250) — don't compare GPU vs CPU seconds.</div></div>
  </div>
  <div class="warn-box"><b>Reading the large end of the chunk axis (512K · 1M · full-batch).</b> Each
  recording is {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples, so 262K is only ~19–33% of the data. We extended the
  GPU sweep past it — <code>512K</code> (262K×2) and <code>1M</code> (262K×4) on the log chunk axis, plus
  <b>full-batch</b> (one pass over the whole recording) in a separate table, since full-batch is a regime,
  not a chunk size. Two things to know above ~785K samples (the shortest recording), where a chunk can
  exceed a subject's data:
  <ul style="margin:6px 0 0">
    <li><b>Implementations diverge on an over-long chunk.</b> jamica / pyamica / pAMICA silently
    <em>clamp</em> the chunk to the recording (that subject runs full-batch); <b>amica-python instead
    rejects it</b> — its <code>BatchLoader</code> raises <code>batch_size N exceeds data size M</code>, so
    it cannot run a single-chunk pass with one oversized value. Its full-batch point here therefore uses a
    <em>per-subject</em> batch equal to each recording's length.</li>
    <li><b>The 1M point is restricted to the 22 subjects longer than 1M</b>, for <em>every</em>
    implementation, so all four are compared on genuinely-chunked data (no clamp-to-full bias). The 3 short
    subjects (sub-01/20/25) appear only at the full-batch point. Contributing <em>n</em> is shown per point
    in the coverage table below the GPU charts (a few cells lost subjects to preemption).</li>
  </ul></div>
  <div class="card" style="max-width:560px;margin:8px 0 4px">{chart_durations()}</div>
</section>

<section>
  <h2>GPU — fit time &amp; memory</h2>
  <p class="sub">Each implementation swept across its setting on real ds004505 (per-subject median,
  3000-iter matched, H100), now extended to <b>512K, 1M and full-batch</b>. Shaded band = p25–p75 across
  subjects. Memory is the <b>NVML whole-GPU peak</b> (framework-neutral; see the memory note), on a
  <b>log</b> axis with real GPU-capacity lines so you can read where each impl would OOM.</p>
  <div class="grid2"><div class="card">{c_gt}</div><div class="card">{c_gv}</div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:20px">
    <li><b>Larger chunks are faster on the GPU — but the win saturates by ~262K.</b> Fit time falls
    steeply over the small chunks (jamica 775&nbsp;s → 61&nbsp;s; amica-python 5646&nbsp;s → 230&nbsp;s
    from 1024 to 262K), then is essentially flat from 262K on (across 262K→1M: jamica ~61→59&nbsp;s,
    amica-python 230→209, pyamica 296→284, pAMICA 262→248&nbsp;s; full-batch in the table below). So 262K
    already captures almost all of the GPU speedup; going bigger buys little time and costs a lot of memory.</li>
    <li><b>Memory climbs steeply toward the device ceiling.</b> NVML rises with chunk for every impl (262K→1M:
    pyamica 10.9→<b>28.1</b>, pAMICA 6.5→19.3, amica-python 4.9→13.3, jamica 5.4→13.4&nbsp;GiB; full-batch in
    the table). Everything fits the 80&nbsp;GiB H100, but the capacity lines tell the portable story:
    <b>pyamica crosses 24&nbsp;GiB by ~1M</b> (would OOM an RTX/A10-class card) and needs a ≥32&nbsp;GiB card at
    full-batch (29.6&nbsp;GiB). <b>jamica is memory-flat (~5.4&nbsp;GiB) only through 262K</b>, then jumps to
    ~13.4&nbsp;GiB at 512K and holds — its chunked path converges to the full-batch footprint once the chunk
    is large, so "chunk isn't a memory dial for jamica" holds at small/mid chunks only.</li>
    <li><b>amica-python cannot run a single-chunk / full-batch pass with an oversized batch.</b> Its
    <code>BatchLoader</code> rejects any batch larger than the recording, so a fixed full-batch value fails
    on every subject; the full-batch point here uses a per-subject batch = recording length (14.1&nbsp;GiB
    NVML — no OOM, it was a library guard, not a memory limit). The others clamp such a chunk to full-batch
    silently.</li>
    <li><b>The bands are tight</b> (matched 3000 iterations → spread reflects only recording length). A few
    large-chunk cells lost subjects to preemption and the 1M point is restricted to the 22 subjects longer
    than 1M — contributing <em>n</em> per point:</li>
  </ul>
  <table style="margin-top:6px;max-width:480px"><thead><tr><th>subjects per point</th><th class="num">≤262K</th><th class="num">512K</th><th class="num">1M *</th></tr></thead><tbody>{covrows()}</tbody></table>
  <p class="note">* 1M restricted to the 22 subjects with &gt;1M samples (all impls), so the comparison is
  on genuinely-chunked data; the 3 shorter subjects appear only at full-batch (table below). Other
  shortfalls are preemption on the shared GPU partition.</p>
  <p style="margin-top:22px"><b>Full-batch — one pass over the whole recording.</b> Full-batch is a
  <em>regime</em>, not a chunk size, so it is tabulated here rather than placed on the log chunk axis
  above. Per-subject median @3000 iters. amica-python uses a per-subject batch = recording length (its
  <code>BatchLoader</code> rejects an oversized fixed value); the others clamp their chunk to the recording.</p>
  <table style="max-width:620px"><thead><tr><th>full-batch</th><th class="num">fit</th><th class="num">alloc (GiB)</th><th class="num">reserved</th><th class="num">NVML</th><th class="num">n</th></tr></thead><tbody>{fullbatchrows()}</tbody></table>
  <p class="note">Full-batch NVML: jamica 13.4 · amica-python 14.1 · pAMICA 20.5 · <b>pyamica 29.6</b> GiB —
  pyamica needs a ≥32&nbsp;GiB card; the other three fit 24&nbsp;GiB. Fit time is within a few percent of the
  1M point for all four (the GPU time win already saturated by 262K), so full-batch buys no speed and only
  costs memory.</p>
</section>

<section>
  <h2>A note on the memory numbers</h2>
  <div class="info-box"><b>NVML is the framework-neutral VRAM figure, and it answers "why does a low line
  OOM?"</b> Each framework's own counter (JAX <code>peak_bytes_in_use</code>, torch
  <code>max_memory_allocated</code>) measures only live-tensor bytes — it omits the CUDA context and the
  pool the driver holds, counts differently across frameworks, and understates the real footprint by
  <b>~1.2–3.3×</b> — so it is not comparable across implementations. Crucially, <b>the OOM-relevant counter is
  the allocator's <em>reserved pool</em></b> (torch <code>max_memory_reserved</code>, JAX
  <code>peak_pool_bytes</code> — the XLA BFC pool): the driver OOMs when it cannot grow that pool,
  <em>not</em> on live-tensor bytes. So the allocated line you'd naively plot is not what hits the ceiling
  — the reserved pool (plus a little out-of-pool cuSOLVER workspace, visible only to NVML) is. The charts use <b>NVML whole-GPU 'used'</b> on a dedicated GPU; caveat: it is a
  50&nbsp;ms poll, so a sub-interval spike can be missed.</div>
  <div class="grid2" style="margin:10px 0 6px"><div class="card">{c_mliv}</div><div class="card">{c_mresv}</div></div>
  <div class="grid2" style="margin:8px 0 6px"><div class="card">{c_mtot}</div><div class="card">{c_mrat}</div></div>
  {legend(IMPLS)}
  <p class="note" style="margin-top:12px">Four measured views vs chunk (per-subject median), on the log
  chunk axis through 1M (full-batch is in the table above): the three counters nest as
  <b>allocator-live ⊆ reserved ⊆ NVML total</b>.
  <b>Allocator-live</b> is each framework's own live-tensor counter (understates the footprint).
  <b>Reserved / pool</b> (torch <code>max_memory_reserved</code>, JAX <code>peak_pool_bytes</code>) is the
  allocator pool the driver actually holds — the OOM-relevant number — and tracks between allocated and NVML
  for all four impls. <b>NVML total</b> is the
  whole-GPU peak (the GPU-section chart adds card-capacity lines); it climbs steeply for the torch impls and
  for jamica once the chunk passes 262K. <b>NVML ÷ allocator</b> shows how much each framework's own counter
  understates the whole-GPU footprint (~1.2–3.3×): largest for jamica at small chunks (~3.3×, ~5.4&nbsp;GiB
  NVML over a ~1.6&nbsp;GiB allocator peak — JAX pool + resident data, not context), shrinking toward ~1.05×
  as live tensors grow to dominate.</p>
  <p class="note"><b>jamica's memory is a two-level step — and the extended sweep shows the two levels are
  one curve.</b> On its chunked path jamica holds a flat ~5.4&nbsp;GiB NVML floor through 262K (per-subject
  ~3.4–{J_CHUNKED_GPU_NVML_MAX:.1f}&nbsp;GiB), then <b>steps to ~{J_FULLBATCH_GPU_NVML:.1f}&nbsp;GiB at 512K
  and holds</b>, converging to its full-batch key's footprint (<code>chunk_size=None</code>, jamica's
  shipped default: per-subject up to ~{J_FULLBATCH_GPU_NVML_MAX:.0f}&nbsp;GiB) at the <b>same GPU speed</b>
  either way (~{J_CHUNKED_GPU_SPI:.3f}&nbsp;s/iter — no GPU time benefit from chunking). So chunk <em>is</em>
  a memory dial for jamica — a step function, not the torch impls' smooth climb. The flat floor is by
  design, not the old full-batch-at-every-chunk bug (ruled out: jamica's fit time varies ~13× with the
  chunk, 775→61&nbsp;s, impossible unless the chunk is applied): the chunked E-step accumulates
  <code>O(n_comp²)</code> sufficient statistics and never materialises the full-width
  <code>(n_comp, n_samples)</code> tensors. Of the ~5.4&nbsp;GiB NVML, the XLA BFC pool holds ~4&nbsp;GiB
  (measured — <code>peak_pool_bytes</code>), of which ~1.6&nbsp;GiB is live tensors (resident whitened data +
  accumulators); the remaining ~1.4&nbsp;GiB is context + scratch outside the pool — all chunk-independent
  until the block buffer grows past 262K, where the pool steps to ~12&nbsp;GiB. It cuts both ways: jamica's ~5.4&nbsp;GiB floor is <em>higher</em> than the torch impls'
  ~1.8–3.1&nbsp;GiB at small chunks (they fit a small card where jamica may not), but jamica never climbs
  the way pyamica does. (On CPU jamica's full-batch key is likewise far heavier —
  ~{J_FULLBATCH_CPU_RSS:.0f}&nbsp;GiB RSS vs a few GiB chunked, earlier fir measurement — so a wrapper
  should pass a chunk.)</p>
</section>

<section>
  <h2>Wall time at matched 3000 iterations (GPU, chunk 262K)</h2>
  <p class="sub">Each implementation at chunk 262K on the H100 (per-subject median), <b>iteration-matched
  to 3000</b> (early-stops disabled). Because every implementation now runs the full 3000 iterations,
  <b>seconds/iteration × 3000 = wall time</b> and this ranking is a clean per-iteration-throughput
  comparison — no early-stop confound. <b>Seconds/iteration</b> is still a descriptive per-iteration cost
  (it folds in fixed/compile overhead, and the update rules differ, so it is not an isolated kernel
  speed). "Iters run" is 3,000 for all four; "final LL" is where each landed at 3000 iterations.</p>
  <div class="grid2"><div class="card">{chart_wallbar()}</div>
  <div class="card"><table><thead><tr><th>Impl</th><th>Wall</th><th>s / iter</th><th>Iters</th><th>Final LL</th></tr></thead><tbody>{convrows()}</tbody></table></div></div>
  <p class="note">jamica has both the shortest wall time and by far the lowest cost per iteration
  (~0.020&nbsp;s/iter vs 0.077–0.099 for the others — ~4–5× faster per iteration). At matched iterations
  pyamica is the longest wall time (and lands at the highest final LL), pAMICA is close behind on time but
  lands ~0.011 nats lower, and amica-python sits in between. With the budget matched, these wall-time
  differences are per-iteration speed, not early stopping. As a separate illustration of how far one
  setting sits from an implementation's best, pAMICA's <code>block_size</code> spans ~16× end to end:</p>
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
  <p class="sub">Real ds004505 on <b>Narval whole nodes (64-core Zen2), one fit per node (exclusive) — no
  memory-bandwidth contention</b>, <b>iteration-matched to 250</b> (early-stops disabled), per-subject
  median over 25 subjects. All five implementations — including the single-threaded Fortran reference —
  now have full 25-subject coverage. Because each fit owned its node, these are clean absolute times, not
  the contention-blurred numbers of the earlier shared-cluster run.</p>
  <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
  {legend(CPU_CHART)}
  <p class="note" style="margin-top:8px"><b>Large-chunk extension (512K · 1M · full-batch) is GPU-only in
  this version.</b> The matching CPU run is completing on Narval whole nodes and will be folded into these
  two charts; the same short-subject handling applies (amica-python's batch guard is device-independent, so
  its CPU full-batch also uses a per-subject batch = recording length).</p>
  <table style="margin-top:16px"><thead><tr><th>fit time (s) · per-subject median · 250 iters</th><th class="num">1K</th><th class="num">4K</th><th class="num">16K</th><th class="num">64K</th><th class="num">262K</th></tr></thead><tbody>{cpufitrows()}</tbody></table>
  <p class="note">All cells cover all 25 subjects (5 implementations × 5 chunks).</p>
  <ul class="tk" style="margin-top:8px">
    <li><b>jamica is fastest at its best CPU setting</b> (~753&nbsp;s at chunk 262K), ahead of
    amica-python (~1053&nbsp;s @262K), pyamica (~1211&nbsp;s @16K), pAMICA (~1527&nbsp;s @16K) and the
    single-threaded Fortran reference (~3676&nbsp;s @1K). Bands overlap between adjacent chunks, so read
    these as lowest measured medians, not certified optima.</li>
    <li><b>No small-chunk advantage on CPU — but the optimum is implementation-specific, not uniformly
    large.</b> Small chunks (1K) are best for none of the Python/JAX impls: jamica and amica-python
    minimize at the <em>largest</em> chunk (262K), while <b>pAMICA and pyamica minimize at a mid chunk
    (16K)</b> — pyamica is ~10% slower at 262K and ~2× slower at 64K, and pAMICA also bumps at 64K, so the
    curves are non-monotonic, not a smooth "bigger is better". The earlier shared-cluster result (small/mid
    chunks win on CPU) <b>did not replicate</b> under whole-node isolation — contention was a likely
    confound, though the runs also differ in platform and budget, so we don't claim it was the sole cause.
    Fortran (single-threaded) is roughly flat, marginally best at 1K.</li>
    <li><b>CPU AMICA is a heavy method</b> — even at 250 iterations it is ~13–25&nbsp;min per fit for the
    Python impls and ~1&nbsp;h for the single-threaded Fortran; at a full 3000-iteration budget that is
    hours. A GPU remains far preferable.</li>
    <li><b>Memory grows with chunk on CPU</b> (jamica ~2.3→10.4, pyamica ~2.1→9.6, pAMICA
    ~1.6→6.1&nbsp;GiB from 1K→262K); the single-threaded Fortran reference is leanest (0.8→3.1&nbsp;GiB).
    RSS is iteration-independent, like VRAM on the GPU.</li>
  </ul>
  <div class="info-box" style="margin-top:14px"><b>On the CPU numbers.</b> These are the clean re-run:
  <b>whole-node exclusive</b> (one fit per node) removes the memory-bandwidth contention that blurred the
  earlier shared-cluster measurement, so the absolute seconds are meaningful. The per-subject bands are
  tighter than the contended run but <em>not</em> uniformly small (relative IQR ranges from ~8% to
  ~35–39% across cells), so adjacent-chunk differences of a few percent are not robust — trust the curve
  shapes and the death of the small-chunk story, not an exact certified optimum. <b>Iteration-matched to
  250</b> (early-stops disabled), all 25 subjects, all five implementations. The four Python/JAX impls were
  given the whole node (64 threads via OMP/MKL/OpenBLAS); Fortran is single-threaded
  (<code>OMP_NUM_THREADS=1</code>), so a whole 64-core node is a fair-<em>isolation</em>, not a
  fair-<em>thread</em>, comparison for it — read it as a reference footprint. GPU and CPU iteration budgets
  differ (GPU 3000, CPU 250), so do not compare GPU seconds to CPU seconds directly. Detail + per-cell
  data: <code>NOTES_measurement.md</code>, <code>raw/narval_nostop_*.csv</code>.</div>
</section>


<footer>
  <div class="kick" style="color:var(--mut)">Provenance &amp; reproduction</div>
  <dl class="prov">
    <dt>Dataset</dt><dd>ds004505 · 64 PCA components · {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples/subject</dd>
    <dt>GPU fit @3000 (matched)</dt><dd>iteration-matched, early-stops disabled → raw/nostop_gpu3000_summary.csv (Trillium H100, 25 subj/cell, all n_iter=3000)</dd>
    <dt>GPU convergence ladder</dt><dd>raw/nostop_ladder_i{100,250,500,1000,2000,3000}_summary.csv (chunk 65536, all 25 subj, early-stops disabled)</dd>
    <dt>GPU memory (NVML+alloc)</dt><dd>raw/chunk_gpumem_*.csv + raw/nostop_gpumem_summary.csv + raw/nostop_gpu_ext_summary.csv (512K/1M/full: fit + allocator/reserved/NVML, 1M restricted to 22 subj >1M; iteration-independent; full-batch key from the prior memory run)</dd>
    <dt>CPU @250 (matched)</dt><dd>raw/narval_nostop_i250_summary.csv (Narval 64-core Zen2, whole-node exclusive; iteration-matched; per-subject median over 25 subj; all 5 impls incl Fortran)</dd>
    <dt>Budget</dt><dd>GPU 3000-iter MATCHED (Trillium H100) · CPU 250-iter MATCHED, whole-node exclusive (Narval) · memory iteration-independent · GPU/CPU budgets differ — don't compare seconds</dd>
    <dt>jamica path</dt><dd>amica_python_jax_chunked (chunked); full-batch key amica_python_jax in the memory note only</dd>
    <dt>Units</dt><dd>memory in GiB (bytes / 1024³); times in seconds; per-subject median (GPU and CPU)</dd>
    <dt>Commits</dt><dd>jamica df18b5e · amica-python e15e158 · pyamica a8a4d7e · pAMICA 0c4da39 · Fortran 665b577</dd>
    <dt>Caveats</dt><dd>NOTES_measurement.md (iteration-budget ≠ convergence · GPU vs CPU budgets differ · NVML vs allocator · the two jamica keys)</dd>
  </dl>
  <p class="note" style="margin-top:16px"><b>What to trust:</b> the curve shapes, the NVML memory figures,
  the per-iteration speeds, and the convergence columns read together. <b>Takeaways:</b> the GPU speed-up
  saturates by ~262K (bigger buys little time and costs memory); memory climbs toward the card ceiling
  (pyamica ~29.6&nbsp;GiB at full-batch; jamica a ~5.4→13.4&nbsp;GiB step); small chunks win on neither
  device, and the CPU optimum is implementation-specific (262K for jamica/amica-python, 16K for
  pAMICA/pyamica); amica-python cannot run a single-chunk pass with an oversized batch (a library guard,
  not a memory limit). Fit times are wall time to a fixed iteration budget, <b>not</b> time to an
  equivalent solution — and GPU and CPU budgets differ, so don't compare their seconds.</p>
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
        _rows.append(("cpu_rss_gib_median", im, KNOB[im], _cn(c), v, "GiB", "Narval whole-node 250-iter, per-subject median"))
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
