#!/usr/bin/env python3
"""Build the CPU stops-off manifests (chunk-sweep + iteration-ladder), one line per array task
"SUBJECT IMPL CHUNK MAXITER REP". All 25 subjects x 10 reps; stops-off; 5 impls incl Fortran.
 - chunk-sweep: chunks {1024,4096,16384,65536,262144} @ 250 iters  -> manifest_cpu_sweep_nostop.txt
 - iter-ladder: chunk 65536 @ N {50,100,500} (250 reused from the sweep's 65536 cell)
                -> manifest_cpu_ladder_nostop.txt  (ladder curve = {50,100,250,500})
Usage: build_manifest_cpu_nostop.py [n_subjects=25] [n_reps=10]
"""
import sys, os
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 10
IMPLS = ["amica_python_jax_chunked", "scott_huberty_torch", "pyamica_torch",
         "pamica_torch", "fortran_amica17"]
here = os.path.dirname(os.path.abspath(__file__))

sweep = [f"{s} {impl} {c} 250 {r}"
         for s in range(1, NS + 1) for impl in IMPLS
         for c in [1024, 4096, 16384, 65536, 262144] for r in range(1, NR + 1)]
ladder = [f"{s} {impl} 65536 {n} {r}"
          for s in range(1, NS + 1) for impl in IMPLS
          for n in [50, 100, 500] for r in range(1, NR + 1)]

for name, lines in [("manifest_cpu_sweep_nostop.txt", sweep),
                    ("manifest_cpu_ladder_nostop.txt", ladder)]:
    with open(os.path.join(here, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {name}: {len(lines)} cells")
