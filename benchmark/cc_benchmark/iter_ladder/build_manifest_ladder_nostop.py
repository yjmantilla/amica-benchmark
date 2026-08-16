#!/usr/bin/env python3
"""Build the iteration-LADDER (stops-off) GPU manifest: fixed chunk 65536, every impl x every N-cap x
N subjects, early-stops disabled (via submit_iter_gpu_nostop.sh). Gives clean 'wall time to reach N
iterations' curves -- the compile/setup INTERCEPT + per-iteration slope -- without the early-stop
confound (amica-python/pAMICA otherwise never reach the high-N caps). N=3000 comes free from the
chunk-sweep stops-off run (c65536_i3000), so the ladder only needs the lower caps.
Usage: build_manifest_ladder_nostop.py [n_subjects=25]
"""
import sys, os
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
IMPLS = ["amica_python_jax_chunked", "scott_huberty_torch", "pyamica_torch", "pamica_torch"]
CHUNK = 65536
ITERS = [100, 250, 500, 1000, 2000]   # 3000 supplied by the chunk-sweep stops-off run
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "manifest_gpu_ladder_nostop.txt")
lines = [f"{s} {impl} {CHUNK} {it}"
         for s in range(1, N + 1) for impl in IMPLS for it in ITERS]
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}: {len(lines)} cells "
      f"({N} subj x {len(IMPLS)} impls x {len(ITERS)} N-caps @ chunk {CHUNK}, stops off)")
