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

FULL = 262144                       # largest chunk of the core axis (samples); the extension adds 512K/1M/full-batch.
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
REPO  = {"jamica":"https://github.com/snesmaeili/jamica","pamica":"https://github.com/sccn/pAMICA",
         "pyamica":"https://github.com/DerAndereJohannes/pyamica","amica_python":"https://github.com/scott-huberty/amica-python",
         "fortran":"https://github.com/sccn/amica"}
def rlink(im, text):   # implementation name as a link to its repository
    return f'<a class="rl" href="{REPO[im]}" target="_blank" rel="noopener">{text}</a>' if im in REPO else text
# Cross-links between the two reports (published artifacts). TLDR_URL is filled once the summary is deployed.
_GH = "https://htmlpreview.github.io/?https://github.com/snesmaeili/jamica-benchmark/blob/main/benchmark/comparator/results/xperf_chunksize"
DETAILED_URL = f"{_GH}/xperf_chunk_report_standalone.html"
TLDR_URL     = f"{_GH}/xperf_chunk_tldr_standalone.html"

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
            ("262144 (large chunk)",262.4,6.54,"best")]
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
#    an oversized batch. So the 1M point is restricted to the 20 subjects longer than 1,048,576 for ALL impls
#    (apples-to-apples on genuinely-chunked data); the short subjects are excluded from the 1M point only.
#  * Full-batch = one pass over the whole recording (per-subject batch = n_samples for amica-python; the
#    others clamp their chunk to n_samples). n contributing subjects is carried per point (preemption +
#    the n20 restriction at 1M => some n<25) and shown in the coverage table.
# chunk -> (fit_s, nvml_gib, alloc_gib, reserved_gib|None, n_subjects)
FB = 4194304                                     # x-axis sentinel for the full-batch point (label "full")
C512, C1M = 524288, 1048576
GEXT = {
 "jamica":       {C512:(59.9,13.37,5.75,12.00,25), C1M:(58.8,13.37,5.76,12.00,20), FB:(61.7,13.37,8.18,12.00,24)},
 "amica_python": {C512:(211.5,7.57,5.77,6.33,25), C1M:(209.0,13.32,10.78,12.09,20), FB:(197.9,14.10,11.46,12.87,25)},
 "pyamica":      {C512:(286.2,19.05,16.86,17.81,25), C1M:(283.7,28.06,26.63,26.82,20), FB:(278.5,29.56,28.16,28.33,21)},
 "pamica":       {C512:(250.2,10.59,9.33,10.09,25), C1M:(247.7,19.33,18.08,18.09,20), FB:(242.5,20.50,19.25,19.26,22)},
}
GEXT_BAND = {  # fit p25,p75 at the extension chunks (iteration-matched @3000)
 "jamica":       {C512:(59.0,61.3), C1M:(58.0,59.9), FB:(58.7,69.7)},
 "amica_python": {C512:(203.9,217.2), C1M:(204.6,211.6), FB:(193.2,202.7)},
 "pyamica":      {C512:(268.4,293.2), C1M:(280.1,289.3), FB:(262.7,285.2)},
 "pamica":       {C512:(237.5,258.3), C1M:(243.5,252.9), FB:(237.3,247.6)},
}
# allocator reserved-pool high-water — the OOM-relevant counter. torch: max_memory_reserved; JAX/jamica:
# peak_pool_bytes (the XLA BFC pool — the JAX analog). 262K from the i3000 run; 512K/1M from the extension
# (full-batch is in the table, not charted). jamica pool steps 4->12 GiB at 512K, mirroring its NVML step.
MEM_RESV = {
 "jamica":       {262144:4.00, C512:12.00, C1M:12.00},
 "amica_python": {262144:3.66, C512:6.33,  C1M:12.09},
 "pyamica":      {262144:9.68, C512:17.81, C1M:26.82},
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
    bandtxt = " · shaded = middle 50% of subjects" if band else ""
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

def chart_iters(series, ylab, title, sub, impls, y0zero=True, xmaxi=3120.0, xticks=(0,1000,2000,3000)):
    # x-axis = iterations (linear), for the chunk-65536 ladder.
    W,H=520,320; ml,mr,mt,mb=64,14,30,48; pw,ph=W-ml-mr,H-mt-mb
    XMAXI=float(xmaxi)
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
    for n in xticks:
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
    return f'<figure class="cf"><figcaption>wall time = seconds per iteration × 3000; jamica ~3.8–4.9× faster per iteration</figcaption>{"".join(s)}</figure>'

LAD_TIME={im:{n:LADDER[im][n][0] for n in LADDER[im]} for im in IMPLS}
LAD_LL  ={im:{n:LADDER[im][n][1] for n in LADDER[im]} for im in IMPLS}
c_lt=chart_iters(LAD_TIME,"fit time (s)","GPU · fit time vs iterations","chunk 65536 · per-subject median · slope = seconds per iteration",IMPLS,y0zero=True)
c_ll=chart_iters(LAD_LL,"final log-likelihood","GPU · convergence vs iterations","chunk 65536 · median final LL (higher = better)",IMPLS,y0zero=False)
c_lm=chart_iters(LADDER_MEM,"GPU memory used (GiB)","GPU · memory vs iterations","chunk 65536 · total GPU memory · essentially the same regardless of the number of iterations",IMPLS,y0zero=True)

gpu_t={im:{c:v[0] for c,v in GPU[im].items()} for im in IMPLS}
gpu_v={im:{c:v[1] for c,v in GPU[im].items()} for im in IMPLS}
CPU_CHART=["jamica","pamica","pyamica","amica_python","fortran"]   # Fortran now has full 25-subject CPU coverage (whole-node run)
TORCH=["amica_python","pyamica","pamica"]   # impls with a torch reserved counter (jamica=JAX, none)
c_gt=chart(gpu_t,GPU_BAND,"fit time (s, log)",True,"GPU · fit time vs chunk","real ds004505 · H100 · 3000-iter matched · per-subject median",IMPLS,"fastest",xt=GXT)
# GPU section memory chart = NVML (true whole-GPU ceiling), log-y, with real GPU-capacity reference lines
c_gv=chart(gpu_v,None,"GPU memory used (GiB, log)",True,"GPU · memory vs chunk (with card capacities)","actual GPU memory used · dashed lines = card capacity (24, 40, 80 GiB)",IMPLS,"leanest",xt=GXT,hlines=GPU_CEILINGS)
c_cr=chart(CPU_RSS,None,"memory used (GiB)",False,"CPU · memory vs chunk","real ds004505 · 64-core machine · per-subject median",CPU_CHART,"leanest")
c_ct=chart(CPU_FIT,CPU_BAND,"fit time (s, log)",True,"CPU · fit time vs chunk","real ds004505 · 64-core machine, one fit per machine · 250 iterations · per-subject median",CPU_CHART,"fastest")
# CPU iteration ladder (chunk 65536; iters 50/100/250/500) from raw/narval_nostop_i*_summary.csv.
CPU_LAD_ITERS=[50,100,250,500]
CPU_LAD={
 "jamica":       {50:(222.7,-1.2320),100:(419.2,-1.1145),250:(937.8,-1.1050),500:(2235.9,-1.1020)},
 "amica_python": {50:(183.0,-1.2280),100:(409.9,-1.1152),250:(1084.2,-1.1040),500:(2278.4,-1.1027)},
 "pyamica":      {50:(478.5,-1.2238),100:(980.0,-1.1152),250:(2408.5,-1.1045),500:(4769.9,-1.1012)},
 "pamica":       {50:(412.4,-1.1754),100:(814.2,-1.1293),250:(2063.7,-1.1202),500:(4148.3,-1.1179)},
 "fortran":      {50:(847.8,-1.2233),100:(1665.6,-1.1151),250:(4004.4,-1.1048),500:(8230.1,-1.1013)},
}
CPU_LAD_TIME={im:{n:CPU_LAD[im][n][0] for n in CPU_LAD[im]} for im in CPU_CHART}
CPU_LAD_LL  ={im:{n:CPU_LAD[im][n][1] for n in CPU_LAD[im]} for im in CPU_CHART}
c_clt=chart_iters(CPU_LAD_TIME,"fit time (s)","CPU · fit time vs iterations","chunk 65536 · 64-core machine · per-subject median",CPU_CHART,y0zero=True,xmaxi=520,xticks=(0,100,250,500))
c_cll=chart_iters(CPU_LAD_LL,"final log-likelihood","CPU · convergence vs iterations","chunk 65536 · median (higher = better)",CPU_CHART,y0zero=False,xmaxi=520,xticks=(0,100,250,500))
# GPU memory decomposition, 2x2 vs chunk (allocator-live ⊆ reserved ⊆ NVML total; + understatement factor)
c_mliv=chart(MEM_LIVE,None,"in active use (GiB)",False,"GPU · memory in active use vs chunk","what each framework reports as actively used · per-subject median",IMPLS,"leanest",xt=GXT)
c_mresv=chart(MEM_RESV,None,"reserved (GiB)",False,"GPU · reserved memory vs chunk","the memory the software holds in reserve (what runs out first) · per-subject median",IMPLS,"leanest",xt=GXT)
c_mtot=chart(gpu_v,None,"total GPU memory (GiB)",False,"GPU · total memory used vs chunk","actual memory used on the card · per-subject median",IMPLS,"leanest",xt=GXT)
MEM_RATIO={im:{c:round(gpu_v[im][c]/MEM_LIVE[im][c],2) for c in gpu_v[im]} for im in IMPLS}
c_mrat=chart(MEM_RATIO,None,"total ÷ active (×)",False,"GPU · how much 'active' understates the total","total GPU memory ÷ actively-used memory · per-subject median",IMPLS,"lowest",xt=GXT)
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
    it="".join(f'<span class="lg"><i style="background:{COLOR[i]}"></i>{rlink(i, LABEL.get(i,"Fortran amica17 (1 thread)"))} <code>{KNOB[i]}</code> <span class="cm">@{COMMIT[i]}</span></span>' for i in impls)
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
    bd={"artifact":'<span class="badge bad">near default</span>',"tuned":'<span class="badge ok">tuned</span>',"best":'<span class="badge best">large chunk</span>'}
    return "".join(f'<tr><td><code>{cfg}</code></td><td class="num">{t:.0f}s</td><td class="num">{v:.2f} GiB</td><td>{bd[tag]}</td></tr>' for cfg,t,v,tag in REAL_PAM)
# Neutral big-picture summary (plain rounded values, drawn from the sections below). Each implementation's
# best fit time and its memory range (smallest chunk -> full-batch/largest). Alphabetical order.
MAIN = {   # GPU memory / CPU memory = range from smallest chunk to a full-batch pass (GPU mem floors are the 1K NVML values)
 "amica_python": ("~200 s", "2–14 GiB", "~18 min", "2–4 GiB"),
 "jamica":       ("~60 s",  "5–13 GiB", "~13 min", "2–10 GiB"),
 "pamica":       ("~245 s", "2–20 GiB", "~25 min", "2–6 GiB"),
 "pyamica":      ("~280 s", "3–30 GiB", "~20 min", "2–10 GiB"),
 "fortran":      ("—",      "—",        "~60 min", "1–3 GiB"),
}
MAIN_ORDER = ["jamica", "amica_python", "pamica", "pyamica", "fortran"]   # by GPU fit time (fastest first; Fortran has no GPU run → last)
def mainrows():
    r=""
    for im in MAIN_ORDER:
        g_t,g_m,c_t,c_m = MAIN[im]
        lab = LABEL.get(im, "Fortran (reference)")
        r+=(f'<tr><td><span class="dot" style="background:{COLOR[im]}"></span>{rlink(im, lab)}</td>'
            f'<td class="num">{g_t}</td><td class="num">{g_m}</td>'
            f'<td class="num">{c_t}</td><td class="num">{c_m}</td></tr>')
    return r

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

HTML=f"""<title>AMICA implementations: fit time, convergence, and peak memory (ds004505)</title>
<style>
:root{{--bg:#ffffff;--panel:#ffffff;--ink:#1a1d29;--mut:#5b6172;--line:#e5e7eb;--accent:#2563eb;--good:#0d9488;--bad:#e11d48;--warn:#d97706;--code:#f1f3f8;--shadow:0 1px 2px rgba(20,24,45,.05),0 8px 22px rgba(20,24,45,.06)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 96px}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85em;background:var(--code);padding:.06em .4em;border-radius:5px}}
a.rl{{color:inherit;text-decoration:underline;text-decoration-color:var(--mut);text-underline-offset:2px}}a.rl:hover{{color:var(--accent);text-decoration-color:var(--accent)}}
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
.pg{{display:grid;grid-template-columns:46px 1fr 1fr;gap:12px 14px;margin:6px 0 2px}}
.pg .ch{{font:700 .74rem/1 ui-sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);text-align:center;align-self:end;padding-bottom:4px}}
.pg .rh{{display:flex;align-items:center;justify-content:center;border-radius:12px;color:#fff}}
.pg .rh span{{writing-mode:vertical-rl;transform:rotate(180deg);font:800 1.05rem ui-sans-serif;letter-spacing:.12em}}
.pg .rh.gpu{{background:var(--accent)}}.pg .rh.cpu{{background:var(--ink)}}
@media(max-width:760px){{.pg{{grid-template-columns:1fr}}.pg .ch{{display:none}}.pg .rh{{padding:8px 0;margin-top:8px}}.pg .rh span{{writing-mode:horizontal-tb;transform:none}}}}
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
  real EEG (ds004505), GPU and CPU, at matched iterations. It turns out to be a big dial. Three takeaways: on the
  GPU <b>most of the speed-up is in by a ~262K chunk</b> (a little more out to 1M, paid for in memory);
  <b>GPU memory climbs steeply with the chunk</b> (pyamica reaches ~30&nbsp;GiB at full-batch); and
  <b>for the four parallel implementations, small chunks are not the fastest setting on either device</b>. Fit times are wall time to a fixed iteration budget,
  not time to an equivalent solution, so read them alongside the convergence section.</p>
  <div class="stamp"><span><b>Builds:</b></span><span>{rlink("jamica","jamica")} <code>df18b5e</code></span><span>{rlink("amica_python","amica-python")} <code>e15e158</code></span><span>{rlink("pyamica","pyamica")} <code>a8a4d7e</code></span><span>{rlink("pamica","pAMICA")} <code>0c4da39</code></span><span>{rlink("fortran","Fortran ref")} <code>665b577</code></span><span>· 64 components · GPU: 3000 iterations · CPU: 250 iterations · NVIDIA H100 GPU + 64-core CPU</span></div>
  {f'<p class="note" style="margin:14px 0 0">Short on time? Read the <a class="rl" href="{TLDR_URL}">one-page summary →</a></p>' if TLDR_URL else ''}
</header>

<section>
  <h2>Main findings</h2>
  <p class="sub">The big picture at each implementation's best setting, before the detailed charts. Fit
  time is for a fixed number of iterations (3000 on the GPU, 250 on the CPU), and memory is the peak used.
  GPU and CPU times are measured at different iteration counts, so they are not directly comparable.</p>
  <table style="max-width:780px"><thead><tr><th>Implementation</th><th class="num">GPU fit</th><th class="num">GPU memory</th><th class="num">CPU fit</th><th class="num">CPU memory</th></tr></thead><tbody>{mainrows()}</tbody></table>
  <p class="note">GPU fit is each implementation's fastest setting (a large chunk or a full-batch pass);
  the GPU memory column spans the smallest chunk to a full-batch pass, the CPU memory column the smallest
  to the largest tested chunk (262K). Patterns shared by all implementations: on the GPU a bigger chunk is faster
  but the gain flattens at large chunks, while memory keeps rising; on the CPU the best chunk depends on
  the implementation. None of the four parallel implementations is fastest at a small chunk, on either device. The single-threaded Fortran build is a
  reference point (it does not run on the GPU here), not a fair comparison against the 64-core runs.</p>
</section>

<section>
  <h2>How to read this</h2>
  <div class="info-box big"><b>The GPU fits are iteration-matched:</b> early-stops disabled, so all four
  run the full 3000 iterations and GPU wall time is directly comparable per iteration
  (<b>seconds/iteration × 3000 = wall</b>, exactly). At matched iterations three of the four land within
  ~0.001 nats in final log-likelihood (jamica −1.1005, amica-python −1.1002, pyamica −1.0995);
  <b>pAMICA sits ~0.011 nats lower (−1.1107)</b>, a small but real gap in convergence quality (detailed in
  the iteration ladder), not just an effect of stopping at different points. Equal iterations is not proof of equal solutions; we
  did not check that the results are identical decompositions. <b>The CPU section is a separate run on a
  64-core machine at 250 iterations</b>: clean absolute times, but the iteration count differs from the
  GPU's, so the two sets of seconds aren't comparable.</div>
  <div class="callout">
    <div class="stat warn"><div class="big">~25×</div><div class="lab">GPU fit-time range from the smallest chunk to 262K within a single implementation (amica-python); the others span 8–16× over that range (larger still out to full-batch).</div></div>
    <div class="stat"><div class="big">grows</div><div class="lab">GPU memory grows with the chunk for three of the four. pyamica reaches ~29.6 GiB at full-batch, more than a 24 GiB card holds. jamica stays flat (~5.4 GiB) through a 262K chunk, then steps up to ~13.4 GiB.</div></div>
    <div class="stat"><div class="big">not small</div><div class="lab">Small chunks never win for the four parallel implementations. On the GPU the largest chunks are fastest; on the CPU it depends on the implementation (the largest for two of them, a mid-size chunk for the other two). The GPU and CPU runs use different iteration counts, so don't compare their raw seconds.</div></div>
  </div>
  <div class="warn-box"><b>Reading the large end of the chunk axis (512K, 1M, full-batch).</b> Each
  recording is {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples, so a 262K chunk is only about a fifth to a third of
  the data. We extended the GPU sweep past it: 512K and 1M on the chunk axis, plus <b>full-batch</b> (one
  pass over the whole recording) in a separate table, since full-batch is a different regime, not a chunk
  size. Two things to know once a chunk is larger than the shortest recording:
  <ul style="margin:6px 0 0">
    <li><b>The implementations react differently.</b> Three of them quietly fall back to a single pass for
    that subject; <b>amica-python instead refuses a chunk larger than the recording</b>, so it cannot use
    one oversized value for every subject. Its full-batch result uses a chunk equal to each recording's own
    length.</li>
    <li><b>The 1M point covers only the 20 recordings longer than 1M samples</b>, for every implementation,
    so all four are compared on the same real chunking. The 5 shorter recordings are left out of the 1M
    point only; they appear at every other point, including the full-batch table. The number of subjects
    behind each point is shown in a small table under the GPU charts.</li>
  </ul></div>
  <div class="card" style="max-width:560px;margin:8px 0 4px">{chart_durations()}</div>
</section>

<section>
  <h2>GPU fit time and memory</h2>
  <p class="sub">Each implementation swept across its setting on real ds004505 (per-subject median,
  3000-iter matched, H100), now extended to <b>512K, 1M and full-batch</b>. Shaded band = the middle 50% of
  subjects. Memory is the <b>total GPU memory actually used</b> (see the memory note), on a <b>log</b> axis
  with real card-capacity lines so you can compare each footprint to common card sizes.</p>
  <div class="grid2"><div class="card">{c_gt}</div><div class="card">{c_gv}</div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:20px">
    <li><b>Larger chunks are faster on the GPU, but the win saturates by ~262K.</b> Fit time falls
    steeply over the small chunks (jamica 775&nbsp;s → 61&nbsp;s; amica-python 5646&nbsp;s → 230&nbsp;s
    from 1024 to 262K), then is essentially flat from 262K on (across 262K→1M: jamica ~61→59&nbsp;s,
    amica-python 230→209, pyamica 296→284, pAMICA 262→248&nbsp;s; full-batch in the table below). Going bigger buys little time and costs a lot of memory.</li>
    <li><b>Memory rises steeply with the chunk.</b> Total GPU memory grows for every implementation (262K→1M:
    pyamica 10.9→<b>28.1</b>, pAMICA 6.5→19.3, amica-python 4.9→13.3, jamica 5.4→13.4&nbsp;GiB; full-batch in
    the table). Everything fit the card used here (80&nbsp;GiB), but the capacity lines put the numbers in
    context: <b>pyamica's footprint passes 24&nbsp;GiB by ~1M</b> and reaches ~29.6&nbsp;GiB at full-batch, so
    it would not fit a 24&nbsp;GiB card if those numbers carry over to smaller hardware (not tested here).
    <b>jamica stays flat (~5.4&nbsp;GiB) only through 262K</b>, then steps up to ~13.4&nbsp;GiB at 512K and
    holds. For jamica the chunk setting only starts to move memory once the chunk is large.</li>
    <li><b>amica-python will not accept a chunk larger than the recording.</b> A single oversized value
    fails for every subject, so its full-batch result uses a chunk equal to each recording's own length
    (14.1&nbsp;GiB of GPU memory, comfortably within the card; this was a setting limit, not running out of
    memory). The other three simply fall back to a single pass in that case.</li>
    <li><b>The spread across subjects is small.</b> With every subject run for the same number of
    iterations, the shaded band mainly reflects how long each recording is. The 1M point covers only the 20
    recordings longer than 1M (so every implementation is compared on the same real chunking); the count of
    subjects behind each point:</li>
  </ul>
  <table style="margin-top:6px;max-width:480px"><thead><tr><th>subjects per point</th><th class="num">≤262K</th><th class="num">512K</th><th class="num">1M *</th></tr></thead><tbody>{covrows()}</tbody></table>
  <p class="note">* 1M covers only the 20 recordings longer than 1,048,576 samples (for every
  implementation), so the comparison is on genuinely chunked data; the other 5 recordings appear at every
  other point, including the full-batch table.</p>
  <p style="margin-top:22px"><b>Full-batch: one pass over the whole recording.</b> This is a different
  regime, not a chunk size, so it is shown in a table rather than on the chunk axis above (per-subject
  median). amica-python uses a chunk equal to each recording's length; the others fall back to a single
  pass.</p>
  <table style="max-width:620px"><thead><tr><th>full-batch</th><th class="num">fit</th><th class="num">in use (GiB)</th><th class="num">reserved</th><th class="num">GPU total</th><th class="num">n</th></tr></thead><tbody>{fullbatchrows()}</tbody></table>
  <p class="note">Full-batch GPU memory (per-subject median): jamica 13.4 · amica-python 14.1 · pAMICA 20.5 ·
  <b>pyamica 29.6</b> GiB. Everything fit the card used here; pyamica's ~29.6&nbsp;GiB would not fit a
  24&nbsp;GiB card if it carries over to other hardware (not tested on one). Past 262K the remaining
  speed-up is small (about 4–9% out to 1M), and full-batch is within ~5% of the 1M time, slightly faster
  for the three torch implementations, most of all amica-python.</p>
</section>

<section>
  <h2>A note on the memory numbers</h2>
  <div class="info-box"><b>Three memory numbers, and why the smallest one can still crash a run.</b> The
  charts report the memory actually used on the card ("total GPU memory"), measured independently of any
  framework. Each framework <em>also</em> reports how much it has <em>actively in use</em>, but that number
  understates the real footprint by about <b>1.2–3.3×</b> at small-to-mid chunks (shrinking toward ~1× at the largest chunks, where the working set dominates). It leaves out fixed start-up cost and the memory
  the software holds in reserve, and the frameworks count it differently, so it isn't comparable across
  implementations. The number that matters for running out of memory is the <b>reserved</b> memory (the pool
  the software holds), not the actively-used figure. That is why an implementation with a modest "in use"
  number can still be close to the limit. The three nest: in active use ≤ reserved ≤ total.</div>
  <div class="grid2" style="margin:10px 0 6px"><div class="card">{c_mliv}</div><div class="card">{c_mresv}</div></div>
  <div class="grid2" style="margin:8px 0 6px"><div class="card">{c_mtot}</div><div class="card">{c_mrat}</div></div>
  {legend(IMPLS)}
  <p class="note" style="margin-top:12px">The four panels above (per-subject median, chunk axis through
  1M; full-batch is in the table earlier): memory <b>in active use</b> (each framework's own count), memory
  <b>reserved</b> (what the software holds, and what runs out first), <b>total GPU memory</b> used,
  and <b>how much the "active" number understates the total</b> (largest for jamica at small chunks, about
  3×, shrinking as the working memory grows). All climb with the chunk for the three torch implementations;
  jamica stays flat until the chunk passes a mid size, then steps up.</p>
  <p class="note"><b>jamica's memory comes in two levels.</b> Through a 262K chunk it holds a flat floor
  of about 5&nbsp;GiB, then steps up to about 13&nbsp;GiB for large chunks and stays there. The step is not a measurement quirk: jamica's fit time changes about 13× across the chunk range (roughly
  775&nbsp;s down to 61&nbsp;s), which could not happen unless the chunk setting were actually applied. The floor exists
  because jamica keeps small running summaries per chunk instead of holding the whole recording at once, so
  its working memory is bounded: about 1&nbsp;GiB of fixed start-up cost plus a few GiB of resident data
  that do not depend on the chunk, until the chunk itself becomes large. It cuts both ways: jamica's
  ~5&nbsp;GiB floor is <em>higher</em> than the others' ~2–3&nbsp;GiB at small chunks (on a small card they
  may fit where jamica does not), but jamica never climbs the way pyamica does.</p>
</section>

<section>
  <h2>Wall time at matched 3000 iterations (GPU, chunk 262K)</h2>
  <p class="sub">Each implementation at chunk 262K on the H100 (per-subject median), all run for the same
  3000 iterations. Because the iteration count is the same, <b>seconds per iteration × 3000 = wall
  time</b>, so this ranking is a fair per-step speed comparison. Seconds per iteration folds in fixed
  start-up cost, and the implementations' update rules differ, so it is not a pure kernel-speed number.
  "Iters run" is 3,000 for all four; "final LL" is the log-likelihood each reached at 3000 iterations.</p>
  <div class="grid2"><div class="card">{chart_wallbar()}</div>
  <div class="card"><table><thead><tr><th>Impl</th><th>Wall</th><th>s / iter</th><th>Iters</th><th>Final LL</th></tr></thead><tbody>{convrows()}</tbody></table></div></div>
  <p class="note">jamica has both the shortest wall time and by far the lowest cost per iteration
  (~0.020&nbsp;s/iter vs 0.077–0.099 for the others, ~3.8–4.9× faster per iteration). At matched iterations
  pyamica has the longest wall time (and the highest final LL), while pAMICA is close behind on time but
  lands ~0.011 nats lower, and amica-python sits in between. With the budget matched, these wall-time
  differences are per-iteration speed, not early stopping. One more example of how far a single setting can sit from an implementation's best: pAMICA's <code>block_size</code> spans ~16× across the
  1K–262K range:</p>
  <table><thead><tr><th><code>block_size</code></th><th>Wall time</th><th>Peak GPU memory</th><th></th></tr></thead><tbody>{pamrows()}</tbody></table>
  <p class="note">pAMICA's shipped default is <code>block_size=512</code>, not re-measured here; the nearest
  tested point, 1024, is already ~16× off its fastest setting in that range.</p>
</section>

<section>
  <h2>Convergence vs iterations: the measured ladder</h2>
  <p class="sub">A directly measured ladder: at a fixed chunk (65536), we ran every implementation to 100,
  250, 500, 1000, 2000 and 3000 iterations and recorded wall time, final log-likelihood, and memory at each,
  with all 25 subjects at every point. Fit time grows in a straight line with iterations; memory is essentially flat
  (it doesn't depend on iterations); convergence (how good the fit is) is the interesting part: pAMICA is
  the outlier.</p>
  <div class="grid2"><div class="card">{c_lt}</div><div class="card">{c_ll}</div></div>
  <div class="grid2" style="margin-top:16px"><div class="card">{c_lm}</div>
  <div class="card"><table><thead><tr><th>Impl</th><th class="num">LL@100</th><th class="num">@250</th><th class="num">@500</th><th class="num">@1000</th><th class="num">@2000</th><th class="num">@3000</th><th class="num">wall@3000</th></tr></thead><tbody>{ladderrows()}</tbody></table></div></div>
  {legend(IMPLS)}
  <ul class="tk" style="margin-top:16px">
    <li><b>jamica converges fastest in both iterations and wall time.</b> It is within ~0.001 in
    log-likelihood (measured in nats, the natural-log units; 0.001 is a very small difference) of its
    own final LL by ~500 iterations (−1.1013 at 500 → −1.1005 at 3000), and because it is ~3.8–4.9× cheaper
    per iteration it gets there in ~14&nbsp;s, versus ~51–57&nbsp;s for the others to run the same 500
    iterations.</li>
    <li><b>pyamica reaches the best (least-negative) reported LL</b> (−1.0995) but uses most of the budget to squeeze the last
    fraction; amica-python tracks it and plateaus around −1.1006.</li>
    <li><b>pAMICA sits below the other three at every iteration count.</b> Its log-likelihood climbs from
    −1.1293 (100) to −1.1107 (3000) but never reaches the other three's ~−1.100. The gap is wider early
    (~0.015 at 100 iterations) and narrows to ~0.011 by 3000. It is <em>still improving</em> at the end
    (~0.0016 over the last 1000 iterations, versus ≤0.0001 for the others), so the gap is genuine at the
    same iteration count, not an effect of stopping early. It is still narrowing at 3000; we did not
    measure beyond that, so whether it eventually closes is untested.</li>
  </ul>
</section>

<section>
  <h2>CPU fit time and memory</h2>
  <p class="sub">Real EEG (ds004505) on a 64-core machine, one fit per machine so nothing else competes for
  memory. Every implementation was run for the same 250 iterations; values are the median across 25
  subjects. All five, including the single-threaded reference, cover all 25 subjects, so these are clean
  absolute times.</p>
  <div class="grid2"><div class="card">{c_ct}</div><div class="card">{c_cr}</div></div>
  {legend(CPU_CHART)}
  <p class="note" style="margin-top:8px">The CPU sweep here covers chunks through 262K; the larger points
  (512K, 1M, full-batch) were measured on the GPU only.</p>
  <table style="margin-top:16px"><thead><tr><th>fit time (s) · per-subject median · 250 iters</th><th class="num">1K</th><th class="num">4K</th><th class="num">16K</th><th class="num">64K</th><th class="num">262K</th></tr></thead><tbody>{cpufitrows()}</tbody></table>
  <p class="note">All cells cover all 25 subjects (5 implementations × 5 chunks).</p>
  <ul class="tk" style="margin-top:8px">
    <li><b>jamica is fastest at its best CPU setting</b> (~753&nbsp;s at chunk 262K), ahead of
    amica-python (~1053&nbsp;s @262K), pyamica (~1211&nbsp;s @16K), pAMICA (~1527&nbsp;s @16K) and the
    single-threaded Fortran reference (~3676&nbsp;s @1K). Bands overlap between adjacent chunks, so read
    these as lowest measured medians, not certified optima.</li>
    <li><b>The smallest chunk is fastest for none of the four parallel implementations, and the best setting
    depends on the implementation.</b> jamica and amica-python are fastest at the largest chunk; pAMICA and pyamica are
    fastest at a mid-size chunk (pyamica is ~10% slower at the largest and about 2× slower at a 64K chunk).
    So the curves are not a smooth "bigger is better". The single-threaded reference is roughly flat,
    marginally best at the smallest chunk.</li>
    <li><b>AMICA is a heavy method on the CPU.</b> Even at 250 iterations it is ~13–25&nbsp;min per fit for
    the four main implementations and about an hour for the single-threaded reference; at a full
    3000-iteration run that is hours. A GPU is far preferable.</li>
    <li><b>Memory grows with the chunk on the CPU too</b> (jamica ~2.3→10.4, pyamica ~2.1→9.6, pAMICA
    ~1.6→6.1&nbsp;GiB from the smallest to the largest chunk); the single-threaded reference is leanest
    (0.8→3.1&nbsp;GiB). As on the GPU, memory does not depend on the number of iterations.</li>
  </ul>
  <div class="info-box" style="margin-top:14px"><b>How to read the CPU numbers.</b> Each fit had a whole
  machine to itself, so the absolute times are meaningful. The spread across subjects varies (from small to
  fairly wide depending on the setting), so differences of a few percent between neighbouring chunks are not
  reliable. Trust the overall shape rather than an exact best chunk. The four main implementations used all 64
  cores; the reference implementation is single-threaded, so treat it as a footprint rather than a
  like-for-like speed comparison. The GPU and CPU runs use different iteration counts, so don't compare
  their seconds.</div>
  <p style="margin-top:22px"><b>Convergence vs iterations (CPU).</b> Fit time and how good the fit is as the
  number of iterations grows (chunk 65536), measured at 50, 100, 250 and 500 iterations. Fit time grows in
  a straight line with iterations; the fit quality improves quickly and then levels off, with pAMICA the
  outlier, the same pattern as on the GPU. All points cover 25 subjects except pyamica at 50 iterations
  (24).</p>
  <div class="grid2"><div class="card">{c_clt}</div><div class="card">{c_cll}</div></div>
  {legend(CPU_CHART)}
</section>


<footer>
  <div class="kick" style="color:var(--mut)">How this was measured</div>
  <dl class="prov">
    <dt>Dataset</dt><dd>ds004505 (real EEG) · 64 components · {N_SAMP_MIN:,}–{N_SAMP_MAX:,} samples per subject · 25 subjects</dd>
    <dt>GPU</dt><dd>NVIDIA H100 · each implementation run for a fixed 3000 iterations · fit time, convergence and memory, measured per subject</dd>
    <dt>CPU</dt><dd>64-core machine, one fit per machine · fixed 250 iterations · fit time and memory, measured per subject</dd>
    <dt>Memory</dt><dd>reported as the actual GPU memory used (measured at the card) and, on the CPU, peak system memory · does not depend on the number of iterations</dd>
    <dt>Note</dt><dd>fit times are wall time at a fixed number of iterations, not time to a solution · the GPU and CPU use different iteration counts, so their seconds are not comparable</dd>
    <dt>Builds</dt><dd>{rlink("jamica","jamica")} df18b5e · {rlink("amica_python","amica-python")} e15e158 · {rlink("pyamica","pyamica")} a8a4d7e · {rlink("pamica","pAMICA")} 0c4da39 · {rlink("fortran","Fortran")} 665b577</dd>
  </dl>
  <p class="note" style="margin-top:16px"><b>What to trust:</b> the curve shapes, the actual GPU-memory
  figures, the per-iteration speeds, and the convergence columns read together. The fastest measured CPU setting is
  implementation-specific (262K for jamica/amica-python, 16K for pAMICA/pyamica), and amica-python cannot
  run a single-chunk pass with an oversized batch (a setting limit, not running out of memory). Fit times are wall time to a fixed iteration budget, <b>not</b> time to an
  equivalent solution, and GPU and CPU budgets differ, so don't compare their seconds.</p>
</footer>
</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
open(os.path.join(HERE,"xperf_chunk_report.html"),"w").write(HTML)
_title = HTML.split("<title>",1)[1].split("</title>",1)[0] if "<title>" in HTML else "AMICA chunk-size report"
STANDALONE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              f'<title>{_title}</title></head><body>\n{HTML}\n</body></html>\n')
open(os.path.join(HERE,"xperf_chunk_report_standalone.html"),"w").write(STANDALONE)

# ===== TL;DR (one-page summary) — reuses the detailed report's stylesheet + the Main-findings table and
# the GPU memory/OOM chart. Links to the detailed report for the full method and analysis.
_style = HTML[HTML.index("<style>"):HTML.index("</style>")+len("</style>")]
TLDR=f"""<title>AMICA implementations: summary (ds004505)</title>
{_style}
<div class="wrap">
<header class="hero">
  <div class="kick">Cross-implementation AMICA · ds004505 · real EEG · summary</div>
  <h1>AMICA implementations: speed and memory at a glance</h1>
  <p class="lede">A one-page summary of how five implementations of AMICA compare on fit time and memory,
  across GPU and CPU, using a real EEG dataset. Each exposes one batch/chunk-size knob that trades memory for
  speed: small chunks are never the fast setting for the four parallel implementations, and on the GPU the
  speed gain flattens once the chunk is reasonably large. Full charts, analysis and method are in the
  <a class="rl" href="{DETAILED_URL}">detailed report →</a>.</p>
</header>
<section>
  <h2>Main findings</h2>
  <p class="sub">Each implementation at its fastest setting (a large chunk or a full-batch pass). Fit time
  is for a fixed number of iterations (3000 on the GPU, 250 on the CPU; measured at different counts, the two are
  not directly comparable); memory is the peak used. Each name links to its repository.</p>
  <table style="max-width:820px"><thead><tr><th>Implementation</th><th class="num">GPU fit</th><th class="num">GPU memory</th><th class="num">CPU fit</th><th class="num">CPU memory</th></tr></thead><tbody>{mainrows()}</tbody></table>
  <p class="note">The GPU memory column spans the smallest chunk to a full-batch pass; the CPU memory
  column spans the smallest to the largest tested chunk (262K). Small and mid chunks cover all 25 subjects;
  the large-chunk and full-batch GPU points cover 20–25 (the 1M point uses the 20 recordings longer than
  1M). The single-threaded Fortran build is a reference point (it does not run on the GPU here).</p>
</section>
<section>
  <h2>Chunk size</h2>
  <p class="sub">Fit time and peak memory versus the batch/chunk-size setting, by device. On the GPU a
  bigger chunk is faster but uses more memory; on the CPU the best chunk depends on the implementation. The
  GPU memory panel marks common card capacities.</p>
  <div class="pg">
    <div class="ch"></div><div class="ch">Fit time</div><div class="ch">Peak memory</div>
    <div class="rh gpu"><span>GPU</span></div><div class="card">{c_gt}</div><div class="card">{c_gv}</div>
    <div class="rh cpu"><span>CPU</span></div><div class="card">{c_ct}</div><div class="card">{c_cr}</div>
  </div>
  {legend(CPU_CHART)}
</section>
<section>
  <h2>Iterations</h2>
  <p class="sub">Fit time and fit quality (log-likelihood, higher is better) versus the number of
  iterations, by device, at a fixed chunk. Fit time grows in a straight line with iterations; quality
  improves quickly and then levels off. The exception is pAMICA, which stays a little below the others and is
  still improving at the end.</p>
  <div class="pg">
    <div class="ch"></div><div class="ch">Fit time</div><div class="ch">Log-likelihood</div>
    <div class="rh gpu"><span>GPU</span></div><div class="card">{c_lt}</div><div class="card">{c_ll}</div>
    <div class="rh cpu"><span>CPU</span></div><div class="card">{c_clt}</div><div class="card">{c_cll}</div>
  </div>
  {legend(CPU_CHART)}
</section>
<footer>
  <p class="note"><b>How this was measured:</b> real EEG (ds004505), 64 components, 25 subjects · GPU:
  NVIDIA H100 · CPU: 64-core machine, one fit per machine · fit time is wall time at a fixed number of
  iterations, not time to a solution. Full method and per-implementation analysis:
  <a class="rl" href="{DETAILED_URL}">detailed report →</a>.</p>
</footer>
</div>"""
open(os.path.join(HERE,"xperf_chunk_tldr.html"),"w").write(TLDR)
_ttitle = TLDR.split("<title>",1)[1].split("</title>",1)[0]
TLDR_STANDALONE = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1">'
                   f'<title>{_ttitle}</title></head><body>\n{TLDR}\n</body></html>\n')
open(os.path.join(HERE,"xperf_chunk_tldr_standalone.html"),"w").write(TLDR_STANDALONE)

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
