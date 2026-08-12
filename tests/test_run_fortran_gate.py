"""run_fortran.py must not emit a failed run as a successful benchmark row.

The bug: GNU `time -v` prints "Maximum resident set size" even when its child
exits nonzero, so the old code parsed a peak RSS and wrote a normal-looking row
with W=null / ll=NaN / no error. This test drives a deliberately-failing command
(a fake gnu_time that prints the maxrss line AND exits 1) and asserts an ERROR
row, not a success row.

Hermetic: no mpirun, no amica17, no _fortran_io outputs — the returncode gate
fires before any of that. Not an AMICA compute path.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNERS = REPO / "benchmark" / "comparator" / "runners"


def load_run_fortran():
    path = RUNNERS / "run_fortran.py"
    spec = importlib.util.spec_from_file_location("run_fortran", path)
    module = importlib.util.module_from_spec(spec)
    # run_fortran inserts its own dir on sys.path for _common/_fortran_io.
    spec.loader.exec_module(module)
    return module


def _fake_gnu_time(tmp_path: Path, returncode: int) -> Path:
    """A stand-in for /usr/bin/time -v: prints a maxrss line, then exits `rc`."""
    script = tmp_path / "fake_time.sh"
    script.write_text(
        "#!/bin/bash\n"
        "echo 'Maximum resident set size (kbytes): 123456' >&2\n"
        f"exit {returncode}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _run(monkeypatch, tmp_path, returncode):
    rf = load_run_fortran()
    # tiny input
    X = np.random.default_rng(0).standard_normal((3, 200))
    npz = tmp_path / "in.npz"
    np.savez(npz, X=X)
    out = tmp_path / "result.json"
    gnu = _fake_gnu_time(tmp_path, returncode)
    monkeypatch.setenv("GNU_TIME_BIN", str(gnu))
    monkeypatch.setenv("MPIRUN_BIN", "true")     # /bin/true — ignores its args, exit 0
    monkeypatch.setenv("AMICA17_BIN", "/bin/true")  # resolvable, hashable
    monkeypatch.setattr(sys, "argv", [
        "run_fortran.py", "--input", str(npz), "--output", str(out),
        "--config", json.dumps({"max_iter": 3, "n_mix": 3, "seed": 5}),
    ])
    rf.main()
    return json.loads(out.read_text())


def test_nonzero_exit_writes_error_row_not_success(monkeypatch, tmp_path):
    doc = _run(monkeypatch, tmp_path, returncode=1)
    assert "error" in doc
    assert "nonzero_exit" in doc["error"]
    assert doc["returncode"] == 1
    # A success row would carry these; an error row must not.
    assert "ll_final" not in doc
    assert "W" not in doc


def test_wrong_binary_sha_is_rejected_before_any_fit(monkeypatch, tmp_path):
    """AMICA17_SHA_EXPECTED (from pins.toml) that doesn't match the binary => error
    row, before the subprocess even runs (measuring the wrong build is fatal)."""
    rf = load_run_fortran()
    X = np.random.default_rng(0).standard_normal((3, 200))
    npz = tmp_path / "in.npz"
    np.savez(npz, X=X)
    out = tmp_path / "result.json"
    monkeypatch.setenv("AMICA17_BIN", "/bin/true")     # resolvable, hashable
    monkeypatch.setenv("AMICA17_SHA_EXPECTED", "deadbeefdeadbeef")  # will not match
    monkeypatch.setattr(sys, "argv", [
        "run_fortran.py", "--input", str(npz), "--output", str(out),
        "--config", json.dumps({"max_iter": 3, "n_mix": 3, "seed": 5}),
    ])
    rf.main()
    doc = json.loads(out.read_text())
    assert "error" in doc and "binary_sha_mismatch" in doc["error"]
    assert "ll_final" not in doc
    assert doc["fortran_bin"] == "/bin/true"


def test_short_expected_sha_is_rejected(monkeypatch, tmp_path):
    """An 8-char prefix is too weak to identify a build — reject it, don't bless it."""
    rf = load_run_fortran()
    X = np.random.default_rng(0).standard_normal((3, 200))
    npz = tmp_path / "in.npz"
    np.savez(npz, X=X)
    out = tmp_path / "result.json"
    monkeypatch.setenv("AMICA17_BIN", "/bin/true")
    monkeypatch.setenv("AMICA17_SHA_EXPECTED", "c02f22c3")  # 8 hex — too short
    monkeypatch.setattr(sys, "argv", [
        "run_fortran.py", "--input", str(npz), "--output", str(out),
        "--config", json.dumps({"max_iter": 3, "n_mix": 3, "seed": 5}),
    ])
    rf.main()
    doc = json.loads(out.read_text())
    assert "error" in doc and "expected_sha_too_short" in doc["error"]


def test_clean_exit_with_nonfinite_W_is_error(monkeypatch, tmp_path):
    """returncode 0 + maxrss present, but an all-NaN W must NOT be a success row."""
    rf = load_run_fortran()
    X = np.random.default_rng(0).standard_normal((3, 200))
    npz = tmp_path / "in.npz"
    np.savez(npz, X=X)
    out = tmp_path / "result.json"
    gnu = _fake_gnu_time(tmp_path, 0)  # exit 0, maxrss printed
    monkeypatch.setenv("GNU_TIME_BIN", str(gnu))
    monkeypatch.setenv("MPIRUN_BIN", "true")
    monkeypatch.setenv("AMICA17_BIN", "/bin/true")
    # Fortran "succeeded" but the recovered W is all-NaN (one finite LL).
    monkeypatch.setattr(rf.fio, "read_fortran_results",
                        lambda *a, **k: {"W": np.full((3, 3), np.nan), "LL_clean": [-1.0]})
    monkeypatch.setattr(sys, "argv", [
        "run_fortran.py", "--input", str(npz), "--output", str(out),
        "--config", json.dumps({"max_iter": 3, "n_mix": 3, "seed": 5}),
    ])
    rf.main()
    doc = json.loads(out.read_text())
    assert "error" in doc and "nonfinite_W" in doc["error"]
    assert "W" not in doc


def test_error_row_still_records_binary_identity_and_seed_honesty(monkeypatch, tmp_path):
    doc = _run(monkeypatch, tmp_path, returncode=2)
    # binary identity is recorded on BOTH paths
    assert doc["fortran_bin"] == "/bin/true"
    assert doc["fortran_sha256"] and len(doc["fortran_sha256"]) == 64
    # seed honesty: fortran forces fix_init and ignores the requested seed
    assert doc["seed_respected"] is False
    assert doc["init"] == "fix_init"
    assert doc["requested_seed"] == 5
    assert doc["effective_config"]["fix_init"] == 1
