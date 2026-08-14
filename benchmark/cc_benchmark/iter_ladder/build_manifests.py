#!/usr/bin/env python3
"""Build the iteration-ladder manifests (time-to-N-iterations campaign).

Design (locked with yorguin):
  - Hold chunk/batch CONSTANT for everyone at 65536 (common feasible: scott-huberty's
    GPU optimum, near-best for the rest, no full-batch OOM).
  - Sweep max_iter (the "ladder"); one complete fit run to each cap, timed end to end.
  - GPU (H100/Trillium): 25 subjects x 4 impls x caps {100,250,500,1000,2000,3000}.
  - CPU (fir):            5 subjects x 5 impls x caps {100,250,500,1000}  (+ Fortran, CPU-only).

Manifest lines:
  GPU:  SUBJECT  IMPL  CHUNK  MAXITER            (one GPU per cell -> no contention, 1 rep)
  CPU:  SUBJECT  IMPL  CHUNK  MAXITER  REP        (CPU_REPS separate array tasks per cell)
consumed by submit_iter_gpu.sh / submit_iter_cpu.sh (one array task per line).

CPU reps are SEPARATE array tasks (not sequential-in-one-job): each rep lands on a different
node/time, so min-across-reps ~= least-contended and the rep spread quantifies contention -- and
each rep fits the 3 h wall (5 sequential fits at cap 1000 would not). GPU needs no reps.

Impl tokens are the harness machine keys (what implementation_perf.py --skip expects):
  amica_python_jax (jamica) · scott_huberty_torch · pyamica_torch · pamica_torch · fortran_amica17
"""
import argparse
from pathlib import Path

CHUNK = 65536
GPU_IMPLS = ["amica_python_jax", "scott_huberty_torch", "pyamica_torch", "pamica_torch"]
CPU_IMPLS = GPU_IMPLS + ["fortran_amica17"]
GPU_SUBJECTS = list(range(1, 26))   # all 25
CPU_SUBJECTS = list(range(1, 6))    # 5 (the CPU chunk-campaign set)
GPU_LADDER = [100, 250, 500, 1000, 2000, 3000]
CPU_LADDER = [100, 250, 500, 1000]

HERE = Path(__file__).resolve().parent


def write(name, subjects, impls, ladder, reps=1):
    lines = []
    for s in subjects:
        for im in impls:
            for it in ladder:
                for r in range(1, reps + 1):
                    lines.append(f"{s} {im} {CHUNK} {it}" + (f" {r}" if reps > 1 else ""))
    p = HERE / name
    p.write_text("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} cells  ({len(subjects)} subj x {len(impls)} impl x "
          f"{len(ladder)} caps" + (f" x {reps} reps" if reps > 1 else "") + f")  -> {p}")
    return len(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-reps", type=int, default=10,
                    help="repetitions per CPU cell, as separate array tasks (default 10)")
    args = ap.parse_args()
    g = write("manifest_gpu.txt", GPU_SUBJECTS, GPU_IMPLS, GPU_LADDER)
    c = write("manifest_cpu.txt", CPU_SUBJECTS, CPU_IMPLS, CPU_LADDER, reps=args.cpu_reps)
    print(f"total: {g + c} cells (gpu {g} + cpu {c})")
    print("note: re-run ladder cost ~= sum(caps) iters/cell; GPU sum=%d, CPU sum=%d (x%d reps)"
          % (sum(GPU_LADDER), sum(CPU_LADDER), args.cpu_reps))
