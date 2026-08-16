#!/usr/bin/env python3
"""Render the MNE #13819 performance note (self-contained HTML content fragment for the artifact).

Outsider-facing note for the MNE-Python thread on adding AMICA. Authored by the jamica group (disclosed).
Uses the SAME corrected data as the xperf_chunksize report (raw/chunk_{gpu3000,gpumem,cpu1000}_*.csv):
realistic 3000-iter GPU budget, NVML whole-GPU memory (framework-neutral), jamica on its chunked path,
convergence (n_iter/ll_final) reported, and "262K = largest tested chunk", not a full-batch pass.
"""
import math, os

# ---- corrected GPU @3000 fit time (s, per-subject median) : chunk -> s ----
FIT = {
 "jamica":  {1024:763.1,4096:227.9,16384:97.6,65536:71.9,262144:61.6},
 "pamica":  {1024:4086.5,4096:1043.7,16384:370.8,65536:313.6,262144:244.7},
 "pyamica": {1024:2414.4,4096:608.8,16384:365.9,65536:339.0,262144:294.0},
 "scott":   {1024:2023.8,4096:502.0,16384:163.2,65536:110.4,262144:79.3},
}
LAB = {"jamica":"jamica","pamica":"pAMICA (SCCN)","pyamica":"pyamica","scott":"amica-python (scott-huberty)"}
COL = {"jamica":"#3b5bdb","pamica":"#b45309","pyamica":"#0d9488","scott":"#c2410c"}
# each impl at its best (=262K here) @3000: fit s, s/iter, n_iter med [min-max], NVML GiB, ll
BEST = [  # ordered by wall time
 ("jamica",  61.6, 0.021, "3000 [2572–3000]", 5.4, -1.1016),
 ("scott",   79.3, 0.076, "1106 [786–1654]",  4.9, -1.1004),
 ("pamica", 244.7, 0.089, "3000 [151–3000]",  6.6, -1.1204),
 ("pyamica",294.0, 0.098, "3000",             10.9, -1.0995),
]
XT = [1024,4096,16384,65536,262144]; XLAB = {1024:"1K",4096:"4K",16384:"16K",65536:"64K",262144:"262K"}
YT = [50,100,200,500,1000,2000,5000]
SMIN, SMAX = 50, 5000

def _chart():
    W,H=680,380; x0,x1=64,500; y0,y1=300,45
    def X(c): return x0 + (math.log2(c)-math.log2(1024))/(math.log2(262144)-math.log2(1024))*(x1-x0)
    def Y(s): return y0 - (math.log10(s)-math.log10(SMIN))/(math.log10(SMAX)-math.log10(SMIN))*(y0-y1)
    s=['<svg viewBox="0 0 %d %d" role="img" aria-label="Fit time vs chunk size" style="width:100%%;height:auto">'%(W,H)]
    s.append('<text transform="translate(18,%.1f) rotate(-90)" class="axl">fit time, seconds (log; 3000-iter budget)</text>'%((y0+y1)/2))
    s.append('<text x="%.1f" y="374" class="axl" text-anchor="middle">chunk / block size — samples processed per pass</text>'%((x0+x1)/2))
    for t in YT:
        y=Y(t); s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'%(x0,y,x1+38,y))
        s.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%ds</text>'%(x0-8,y+4,t))
    for c in XT:
        s.append('<text x="%.1f" y="352" class="tick" text-anchor="middle">%s</text>'%(X(c),XLAB[c]))
    for im in ["scott","pamica","pyamica","jamica"]:
        cs=sorted(FIT[im]); pts=" ".join(("M" if i==0 else "L")+"%.1f,%.1f"%(X(c),Y(FIT[im][c])) for i,c in enumerate(cs))
        s.append('<path d="%s" fill="none" stroke="%s" stroke-width="2.2"/>'%(pts,COL[im]))
        for c in cs: s.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>'%(X(c),Y(FIT[im][c]),COL[im]))
        lc=cs[-1]; s.append('<text x="%.1f" y="%.1f" class="lgl" fill="%s">%s</text>'%(X(lc)+8,Y(FIT[im][lc])+4,COL[im],LAB[im]))
    s.append('</svg>')
    return "".join(s)

def _bestrows():
    r=""
    for im,t,spi,nit,mem,ll in BEST:
        r+=('<tr><td><span class="dot" style="background:%s"></span>%s</td>'
            '<td class="n">%.0f s</td><td class="n">%.3f</td><td class="n">%s</td>'
            '<td class="n">~%.1f GiB</td><td class="n">%.4f</td></tr>'%(COL[im],LAB[im],t,spi,nit,mem,ll))
    return r

CHART=_chart(); BESTROWS=_bestrows()
HTML=f"""<title>A performance note on the Python AMICA implementations, for MNE #13819</title>
<style>
:root{{--bg:#ffffff;--paper:#fbfbfa;--ink:#1b1b1f;--soft:#565b66;--line:#e7e7ea;--rule:#d9d9dd;--accent:#2457d6;--warn:#c0392b;--code:#f2f2f0}}
:root[data-theme=dark]{{--bg:#14151a;--paper:#191b21;--ink:#e9e9ec;--soft:#a2a7b3;--line:#2a2c34;--rule:#33363f;--accent:#7aa2ff;--warn:#f0857a;--code:#20222a}}
@media(prefers-color-scheme:dark){{:root:not([data-theme]){{--bg:#14151a;--paper:#191b21;--ink:#e9e9ec;--soft:#a2a7b3;--line:#2a2c34;--rule:#33363f;--accent:#7aa2ff;--warn:#f0857a;--code:#20222a}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:19px/1.62 "Iowan Old Style","Charter","Georgia","Times New Roman",serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:720px;margin:0 auto;padding:56px 24px 96px}}
h1{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:1.92rem;line-height:1.16;letter-spacing:-.015em;font-weight:750;margin:.15em 0 .1em;text-wrap:balance}}
.dek{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:1rem;color:var(--soft);margin:0 0 6px;font-weight:450}}
.byline{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.82rem;color:var(--soft);border-bottom:1px solid var(--rule);padding-bottom:18px;margin-bottom:6px}}
.byline a{{color:var(--accent);text-decoration:none}}
h2{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:1.24rem;letter-spacing:-.01em;font-weight:700;margin:1.9em 0 .35em}}
p{{margin:0 0 1.02em}} a{{color:var(--accent)}}
code{{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.82em;background:var(--code);padding:.06em .38em;border-radius:4px}}
strong{{font-weight:650}} em{{font-style:italic}}
figure{{margin:1.5em 0;padding:16px 16px 10px;background:var(--paper);border:1px solid var(--line);border-radius:12px}}
figcaption{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.8rem;color:var(--soft);margin-top:8px;line-height:1.45}}
.grid{{stroke:var(--line);stroke-width:1}}
.axl{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11px;fill:var(--soft)}}
.tick{{font-family:ui-monospace,monospace;font-size:11px;fill:var(--soft)}}
.lgl{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11.5px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-family:ui-sans-serif,system-ui,sans-serif;font-size:.9rem;margin:1.2em 0}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}}
th{{font-size:.71rem;letter-spacing:.04em;text-transform:uppercase;color:var(--soft);font-weight:600}}
td.n{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;font-family:ui-monospace,monospace;font-size:.9em}}
tbody tr:last-child td{{border-bottom:none}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}}
.callout{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.92rem;line-height:1.5;background:var(--paper);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:12px 16px;margin:1.3em 0}}
.note{{font-size:.92rem;color:var(--soft)}}
ul{{margin:0 0 1.05em;padding-left:1.1em}}li{{margin:.32em 0}}
hr{{border:none;border-top:1px solid var(--rule);margin:2.3em 0 1.5em}}
.foot{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:.8rem;color:var(--soft);line-height:1.55}}
.foot strong{{color:var(--ink)}}
</style>
<div class="wrap">

<p class="dek">Notes for MNE issue #13819 — adding AMICA as an ICA method</p>
<h1>A performance note on the Python AMICA implementations</h1>
<p class="byline">Cross-implementation timing &amp; memory on real EEG (ds004505). Complements the accuracy
results already in the thread. Data, code, and an independent factual audit:
<a href="https://github.com/snesmaeili/jamica-benchmark/pull/7">jamica-benchmark #7</a>.</p>

<p><strong>Disclosure.</strong> We (the jamica group) maintain one of the implementations compared here;
another, <code>amica-python</code>, is maintained by @scott-huberty, who is active in this thread. We've
tried to compare like with like — each package at its best measured setting — and everything below is
reproducible at the link.</p>

<p><strong>Scope.</strong> This is only about <em>speed and memory across the AMICA implementations</em>.
It says nothing about source-separation quality (that's Sina's separate benchmark), and nothing about the
multi-model feature @cbrnr raised (see the end — we consider it the more important open question).</p>

<div class="callout"><strong>Read the numbers correctly.</strong> These are <em>wall times to a fixed
3000-iteration budget</em>, not times to an equivalent solution. The implementations do not use that
budget the same way — under their own early-stop rules, at the largest chunk the median iterations
actually run were pyamica 3000, jamica 3000 (2572–3000), pAMICA 3000 (but as few as 151), amica-python
1106 (786–1654, it converges and stops early). They reach final log-likelihoods in a tight band
(−1.0995 to −1.1204). So a shorter wall time can mean a faster implementation, an earlier stop, or fewer
iterations of work — we report iterations-run and final LL alongside time, and we do not claim the four
decompositions are numerically identical. Realistic convergence is far heavier than any single-config
5–10&nbsp;s number floated earlier in this thread: on one H100 it is tens of seconds to a few minutes
depending on implementation and chunk, and on CPU it is minutes to hours.</div>

<h2>Configuration, not the algorithm, dominates the cost</h2>

<p>The most useful thing we found is that performance is driven less by which implementation you pick than
by one setting they all expose: the <strong>batch / chunk size</strong> — how many samples each pass
processes. Sweeping it on real EEG, fit time moved by up to <strong>~25×</strong> within a single
implementation (and 8–17× for the others). Peak GPU memory also moves with the chunk for the three
PyTorch implementations (~2.7–3.6× on the framework-neutral NVML meter); jamica's GPU memory is instead
roughly flat in the median (~5.4&nbsp;GiB) on its chunked path. The shipped defaults are usually far from
each package's own optimum.</p>

<figure>
{CHART}
<figcaption>Fit time vs. chunk size on real EEG (ds004505, 64 components, <strong>3000-iteration
budget</strong>, per-subject median, one NVIDIA H100 80&nbsp;GB). Larger chunks are faster on the GPU for
all four; the fastest tested point is the largest chunk (262K samples) for every implementation. The
biggest setting swept, 262K, is ~19–33% of each 0.79–1.36&nbsp;M-sample recording — the largest chunk
tested, <em>not</em> a single full-batch pass. pAMICA's shipped default is <code>block_size=512</code>;
its nearest measured point (1024) is already ~17× off its own fastest tested setting, so the default is a
footgun. On CPU the fastest setting flips to small/mid chunks (a cache effect).</figcaption>
</figure>

<p>For an MNE wrapper the practical implication is that <strong>the wrapper should choose the chunk size</strong>
(and adapt it to the device — below), rather than leave a raw default exposed; otherwise two users running
“the same” method get very different experiences. It's the same idea as shipping a sensible
<code>max_iter</code>.</p>

<h2>Comparing the implementations, each at its best measured setting</h2>

<p>Run each package at its own best tested chunk (262K) on the real workload (per-subject median, one
H100, 3000-iteration budget). Because the implementations run different numbers of iterations under their
own early-stop, we show iterations-run and final log-likelihood next to wall time; <em>seconds/iteration</em>
normalizes for the differing iteration counts (it is the median of per-subject time÷iterations-run, so the
columns will not multiply back exactly):</p>

<table>
<thead><tr><th>Implementation</th><th>wall time</th><th>s / iter</th><th>iters run</th><th>peak VRAM (NVML)</th><th>final LL</th></tr></thead>
<tbody>{BESTROWS}</tbody>
</table>

<p>jamica has the shortest wall time and the lowest per-iteration cost (~0.021&nbsp;s/iter vs
0.076–0.098). amica-python's shorter wall time than pAMICA/pyamica is partly that it early-converges and
runs ~1,100 of the 3000 iterations; pyamica runs the full 3000 for the highest final LL; pAMICA lands at
a modestly but consistently lower LL (a paired per-subject delta of ~0.007 nats, negative for every
subject). <strong>Every one of these is a legitimate, working AMICA</strong>; the wall-time gaps are real
at a fixed budget but conflate per-iteration speed with how far each ran. For choosing an implementation,
license, API, install footprint, multi-model support, and maintenance are likely to matter more than
these gaps.</p>

<h2>Memory — quantifying @scott-huberty's point</h2>

<p>The concern that “a 1&nbsp;GiB EEG file can lead to 3&nbsp;GiB intermediates and &gt;10&nbsp;GiB peak” is
well-founded: AMICA's intermediates genuinely scale with <code>n_samples</code> (and with
<code>n_mixtures</code>). On the framework-neutral NVML meter (whole-GPU peak on a dedicated card), peak
VRAM at the largest tested chunk ran from ~4.9&nbsp;GiB (amica-python) to ~10.9&nbsp;GiB (pyamica), and it
<em>grows with the chunk</em> for the three PyTorch implementations — so a wrapper's chunk choice, not a
fixed property of the method, sets most of the peak a user sees. Two nuances worth flagging honestly:</p>
<ul>
<li><strong>jamica has a higher memory <em>floor</em>.</strong> Its chunked-path GPU memory is roughly
flat at ~5.4&nbsp;GiB (per-subject 3.4–5.4, rising to ~7.4 at the largest chunk for the longest
recordings) — it does not drop at small chunks the way the PyTorch implementations do (~1.8–3&nbsp;GiB).
So on a small card the PyTorch impls at a small chunk can fit where jamica may not; the trade is that they
are much slower at those small chunks.</li>
<li><strong>jamica's shipped default is the memory-heavy path.</strong> jamica has two entry points: the
chunked path (everything above) and a full-batch path (<code>chunk_size=None</code>, its default) that
materialises the full-width arrays for ~13&nbsp;GiB NVML median (per-subject up to ~21) at essentially the
<em>same</em> GPU speed — no benefit. A wrapper should always pass a chunk.</li>
</ul>
<p class="note"><strong>Why jamica's chunked memory looks flat (and why that is not a measurement
artifact).</strong> Flat memory was the signature of a bug we fixed — jamica accidentally run full-batch
at every chunk — so we checked. It is not that here: jamica's <em>fit time</em> varies ~12× with the
chunk (763→62&nbsp;s), which only happens if the chunk is applied (the true full-batch path is flat in
time too, ~61&nbsp;s everywhere), and the chunked path's ~5.4&nbsp;GiB is less than half the full-batch
path's ~13&nbsp;GiB — different code paths. The ~5.4&nbsp;GiB is flat because it is ~3.8&nbsp;GiB fixed
CUDA context plus ~1.6&nbsp;GiB of chunk-independent full-width arrays; the chunk-scaled block buffer is
small next to those until the largest chunk. So on the GPU the chunk is a real <em>time</em> dial for
jamica but not a memory dial — a genuine property, not the old bug.</p>
<p class="note">On the meter: we report <strong>NVML whole-GPU peak</strong> as the headline because it is
framework-neutral. Each backend's own allocator counter (JAX <code>peak_bytes_in_use</code>, Torch
<code>max_memory_allocated</code>) measures only its live-tensor bytes, omits the CUDA context/pool, and
is not comparable across frameworks — it understates the real footprint by ~1.2–3.3× in the pairs we can
check. An earlier draft of this note quoted those allocator numbers (and a jamica full-batch figure that
was too low); the NVML numbers here supersede them.</p>

<h2>The GPU-vs-CPU picture, plainly</h2>

<p>This matches what Sina and @cbrnr already said. <strong>AMICA has no algorithmic speed advantage over
Picard or Infomax</strong>; what makes it practical is the JAX GPU path. On a GPU a realistically
converged fit lands in the same ballpark as running the lighter methods on CPU (Sina reported AMICA
~134&nbsp;s vs Infomax 84&nbsp;s / FastICA 99&nbsp;s / Picard 150&nbsp;s in his 3000-iteration run).
Without a GPU it is far heavier: at our 1000-iteration CPU budget jamica's best setting was ~2,000&nbsp;s
(~33&nbsp;min) on 8 cores, and the others run to the thousands of seconds — per iteration the GPU is on
the order of ~100× faster. The optimum chunk <em>flips</em> to small/mid sizes on CPU (a cache effect,
the opposite of the GPU). Our CPU absolute seconds are contention-inflated (the cluster was busy
throughout) and the per-cell subject coverage is uneven, so we trust only the broad device flip and the
minutes-to-hours magnitude, not exact CPU seconds or a per-implementation CPU optimum. The honest summary
for MNE: <em>today</em>, AMICA wants a GPU; on CPU it is a minutes-to-hours method, and the labs involved
have noted CPU performance is an active work item.</p>

<h2>One benchmarking caution</h2>

<p>An early sweep made jamica look <em>flat</em> — as if uniquely insensitive to the chunk knob. It
wasn't: that was a measurement artifact (a very short synthetic run whose one-time JIT-compile cost
dominated the wall clock, compounded by a harness bug that had accidentally run jamica full-batch at every
“chunk”). Measured correctly — on the real workload, over a realistic iteration budget, with the chunk
actually applied — jamica's curve is a normal, steep one like everyone else's. The general point for a
thread that will see many benchmark numbers: short or misconfigured ICA timing runs can mislead, so it is
worth asking over how many iterations, on what data, and with which setting actually varied, a number was
measured.</p>

<h2>The question this note does <em>not</em> answer</h2>

<p>@cbrnr identified the real differentiator of AMICA over Extended Infomax: the <strong>multi-model</strong>
case — M&gt;1 unmixing matrices with a most-probable model per time point. We did <em>not</em> benchmark
M&gt;1; it plausibly changes compute, memory, and convergence, and it raises the API question of how a
wrapper would even return multiple models with their probabilities. Its runtime and memory scaling are open
and, in our view, worth measuring before settling wrapper defaults — arguably more decision-relevant than
the single-model timings here. Separation quality is likewise out of scope: Sina reported the MIR results in
this thread (AMICA highest, though he flagged those as in-sample, with a held-out run underway).</p>

<h2>Practical notes for a wrapper</h2>
<ul>
<li><strong>Let the wrapper set the chunk size</strong>, device-aware: a large chunk on GPU (largest is
fastest here, but a mid chunk such as 16K–65K is within a factor of ~1.5–2× of it and, for the PyTorch
impls, at lower memory), small/mid on CPU. This single choice governs both speed and, for the PyTorch
impls, peak memory.</li>
<li><strong>Mind the memory floor and the default:</strong> jamica's chunked GPU footprint is ~5.4&nbsp;GiB
regardless of chunk, and its full-batch <em>default</em> (<code>chunk_size=None</code>) is ~13&nbsp;GiB for
no speed gain — a wrapper should pass an explicit chunk. The PyTorch impls reach ~1.8–3&nbsp;GiB at small
chunks but are much slower there. None of these full-size operating points is comfortable on the 8–12&nbsp;GiB
cards many MNE users have, so a moderate chunk is the portable default.</li>
<li><strong>Set expectations:</strong> with a modern GPU (our numbers are from an H100; a consumer card
will be slower) a fit is tens of seconds to a few minutes; on CPU it is minutes to hours. AMICA is the
heavy, high-quality method — not a Picard replacement on speed.</li>
<li><strong>Among implementations,</strong> at a fixed budget the tuned differences are real but modest;
the decision should also weigh license, API, multi-model support, install footprint, and maintenance.</li>
</ul>

<hr>
<p class="foot"><strong>Method &amp; scope.</strong> Real ds004505 (Studnicki 2022, dual-layer EEG), 64 PCA
components, identical 1–100&nbsp;Hz + per-site notch + 250&nbsp;Hz preprocessing, single-model. GPU numbers
are per-subject medians over 20–25 subjects on one NVIDIA H100 80&nbsp;GB (SciNet Trillium), at a fixed
3000-iteration budget; peak VRAM is NVML whole-GPU peak (framework-neutral; memory is iteration-independent,
measured on the matched runs). jamica is measured on its chunked path; its full-batch key is reported only
in the memory section. CPU numbers are by-subject medians over 5 subjects on 8 cores (Alliance fir, AMD
EPYC) at a 1000-iteration budget and carry a documented node-contention + uneven-coverage caveat — trust
the device flip and magnitude, not the exact CPU seconds or a per-implementation CPU optimum. Fit times are
wall time to a fixed iteration budget, not time to convergence; iterations-run and final log-likelihood are
reported so the difference is visible. The AMICA-vs-Picard/Infomax/FastICA numbers are Sina's
realistic-convergence (3000-iteration) results from this thread. Runners, tidy data (with per-cell raw
aggregates), and a multi-model factual audit are in the study repository
(<a href="https://github.com/snesmaeili/jamica-benchmark/pull/7">jamica-benchmark #7</a>,
<code>results/xperf_chunksize/</code>). This is performance evidence to inform the integration discussion —
not a recommendation of which implementation MNE should adopt.</p>

</div>"""
HERE=os.path.dirname(os.path.abspath(__file__))
open(os.path.join(HERE,"mne_note_13819.html"),"w").write(HTML)
STANDALONE=('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>A performance note on the Python AMICA implementations, for MNE #13819</title>'
            f'</head><body>\n{HTML}\n</body></html>\n')
open(os.path.join(HERE,"mne_note_13819_standalone.html"),"w").write(STANDALONE)
print("wrote mne_note_13819.html", len(HTML), "bytes (+ standalone", len(STANDALONE), "bytes)")
