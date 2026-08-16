#!/usr/bin/env python3
"""Build the iteration-matched (stops-off) GPU re-run manifest: every impl x every chunk x N subjects
at max_iter=3000. One line "SUBJECT IMPL CHUNK MAXITER" per array task.
Usage: build_manifest_nostop.py [n_subjects=20]
"""
import sys, os
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
IMPLS = ["amica_python_jax_chunked", "scott_huberty_torch", "pyamica_torch", "pamica_torch"]
CHUNKS = [1024, 4096, 16384, 65536, 262144]
MAXITER = 3000
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "manifest_gpu_nostop.txt")
lines = []
for s in range(1, N + 1):
    for impl in IMPLS:
        for c in CHUNKS:
            lines.append(f"{s} {impl} {c} {MAXITER}")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}: {len(lines)} cells "
      f"({N} subj x {len(IMPLS)} impls x {len(CHUNKS)} chunks @ {MAXITER} iters, stops off)")
