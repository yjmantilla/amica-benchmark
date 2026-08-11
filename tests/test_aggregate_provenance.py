"""aggregate.py: seed honesty, provenance columns, payload-derived method tag.

Loads amica_python/benchmark/{schema,aggregate}.py under a synthetic package so
the heavy amica_python/__init__ (jax/mne) is never imported — hermetic, no
AMICA compute path, backend-agnostic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "amica_python" / "benchmark"


def _load(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agg():
    pkg = types.ModuleType("aggpkg")
    pkg.__path__ = []  # mark as package so relative imports resolve
    sys.modules["aggpkg"] = pkg
    _load("aggpkg.schema", BENCH / "schema.py", package="aggpkg")
    return _load("aggpkg.aggregate", BENCH / "aggregate.py", package="aggpkg")


def make_doc(*, subject="sub-01", backend="jax", device="gpu", seed=0,
             schema_version="3.0", extra_payload=None, with_provenance=True):
    payload = {
        "method": "amica",
        "backend": backend,
        "device": device,
        "dtype": "float64",
        "runtime_s": 12.0,
        "n_iter": 5,
        "max_iter": 10,
        "n_components": 4,
        "random_state": seed,
        "jit_compile_s": 1.5,
        "steady_iter_s": 0.2,
    }
    if extra_payload:
        payload.update(extra_payload)
    run = {
        "timestamp": "2026-08-10T12:00:00Z",
        "hostname": "node42",
    }
    if with_provenance:
        run["provenance"] = {
            "python": "3.11.7",
            "packages": {"amica": {"version": "0.3.0", "commit": "a" * 40,
                                   "commit_source": "amica_src"}},
            "stack": {"jax": "0.9.1", "numpy": "2.0.0"},
        }
    return {
        "_schema_version": schema_version,
        "_run": run,
        "_data": {"subject": subject, "dataset": "ds004505", "n_samples": 1000},
        "amica": payload,
    }


def write(dirpath: Path, name: str, doc: dict):
    (dirpath / name).write_text(json.dumps(doc), encoding="utf-8")


def _one_row(agg, tmp_path, **kw):
    write(tmp_path, "benchmark_sub-01_hp1.0hz_jax_gpu.json", make_doc(**kw))
    runs = list(agg.discover_runs(tmp_path))
    assert len(runs) == 1
    return agg.benchmark_row(runs[0], agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})


def test_seed_roundtrips_and_is_not_constant(agg, tmp_path):
    row = _one_row(agg, tmp_path, seed=7)
    assert row["random_seed"] == 7  # not the old hard-coded 42


def test_absent_seed_is_none(agg, tmp_path, capsys):
    doc = make_doc()
    del doc["amica"]["random_state"]
    write(tmp_path, "benchmark_sub-01_hp1.0hz_jax_gpu.json", doc)
    run = next(iter(agg.discover_runs(tmp_path)))
    row = agg.benchmark_row(run, agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})
    assert row["random_seed"] is None
    assert "no recorded seed" in capsys.readouterr().out


def test_provenance_columns_populated(agg, tmp_path):
    row = _one_row(agg, tmp_path, seed=3)
    assert row["schema_version"] == "3.0"
    assert row["run_timestamp"] == "2026-08-10T12:00:00Z"
    assert row["implementation_version"] == "0.3.0"
    assert row["implementation_commit"] == "a" * 40
    assert row["aggregated_at"] == "T"
    assert row["aggregator_commit"] == "c"


def test_jit_steady_dtype_carried(agg, tmp_path):
    row = _one_row(agg, tmp_path)
    assert row["jit_compile_s"] == 1.5
    assert row["steady_iter_s"] == 0.2
    assert row["dtype"] == "float64"


def test_method_tag_from_payload_not_filename(agg, tmp_path):
    # Same "hp1.0hz_"-less style filename but different payload backends must NOT
    # collapse to one label.
    write(tmp_path, "benchmark_sub-01_hp2.5hz_a.json",
          make_doc(backend="numpy", device="cpu"))
    write(tmp_path, "benchmark_sub-01_hp2.5hz_b.json",
          make_doc(backend="jax", device="gpu"))
    runs = list(agg.discover_runs(tmp_path))
    # run_id now carries a content hash suffix, but the human prefix is the tag
    prefixes = sorted(r.run_id.rsplit("__", 1)[0] for r in runs)
    assert prefixes == ["sub-01__jax_gpu", "sub-01__numpy_cpu"]
    methods = {r.method for r in runs}
    assert len(methods) == 2  # distinct labels, not merged


def test_seed_sweep_gets_distinct_run_ids(agg, tmp_path):
    write(tmp_path, "benchmark_sub-01_hp1.0hz_seed0.json", make_doc(seed=0))
    # different filename, same backend/device, different seed
    (tmp_path / "benchmark_sub-01_hp1.0hz_seed1.json").write_text(json.dumps(make_doc(seed=1)))
    runs = list(agg.discover_runs(tmp_path))
    rows = [agg.benchmark_row(r, agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})
            for r in runs]
    assert sorted(r["random_seed"] for r in rows) == [0, 1]  # not [42, 42]
    # the two seeds must NOT collapse onto one run_id (they used to)
    assert len({r.run_id for r in runs}) == 2


def test_chunk_variants_get_distinct_run_ids(agg, tmp_path):
    """Full-batch (None) and chunked/auto runs of otherwise-equal config must not
    collapse onto one run_id (they used to, because chunk_size was dead in the hash)."""
    write(tmp_path, "benchmark_sub-01_hp1.0hz_jax_gpu_full.json",
          make_doc(extra_payload={"chunk_size": None}))
    write(tmp_path, "benchmark_sub-01_hp1.0hz_jax_gpu_chunked.json",
          make_doc(extra_payload={"chunk_size": "auto"}))
    runs = list(agg.discover_runs(tmp_path))
    assert len({r.run_id for r in runs}) == 2


def test_commit_source_reaches_the_csv(agg, tmp_path):
    row = _one_row(agg, tmp_path)
    assert row["implementation_commit_source"] == "amica_src"


def test_legacy_poisoned_comparator_doc_not_labelled_amica(agg, tmp_path):
    """A legacy Picard doc — comparator backend, NO library_versions, but a stray
    amica provenance block — must not be attributed to the amica package."""
    doc = make_doc(backend="picard", device="cpu")  # provenance has amica
    doc["amica"].pop("library_versions", None)       # legacy: none recorded
    write(tmp_path, "benchmark_sub-01_hp1.0hz_picard_cpu.json", doc)
    run = next(iter(agg.discover_runs(tmp_path)))
    row = agg.benchmark_row(run, agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})
    assert row["implementation_version"] is None      # NOT amica's 0.3.0
    assert row["implementation_commit"] is None


def test_comparator_row_not_labelled_as_amica(agg, tmp_path):
    """A Picard doc with library_versions must report Picard's version — even if a
    stray amica provenance block is present — not the amica package."""
    doc = make_doc(backend="picard", device="cpu")
    doc["amica"]["library_versions"] = {"mne": "1.7.0", "sklearn": "1.4.0", "picard": "0.7"}
    # simulate the poison: an amica provenance block on a comparator doc
    doc["_run"]["provenance"] = {"packages": {"amica": {"version": "9.9.9", "commit": "z" * 40}}}
    write(tmp_path, "benchmark_sub-01_hp1.0hz_picard_cpu.json", doc)
    run = next(iter(agg.discover_runs(tmp_path)))
    row = agg.benchmark_row(run, agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})
    assert row["implementation_version"] == "0.7"      # python-picard, not 9.9.9
    assert row["implementation_commit"] is None


def test_fastica_not_mislabelled_as_picard(agg, tmp_path):
    """FastICA is sklearn; a picard version merely installed in the env must not win."""
    doc = make_doc(backend="fastica", device="cpu")
    doc["amica"]["library_versions"] = {"mne": "1.7.0", "sklearn": "1.4.0", "picard": "0.7"}
    doc["_run"].pop("provenance", None)
    write(tmp_path, "benchmark_sub-01_hp1.0hz_fastica_cpu.json", doc)
    run = next(iter(agg.discover_runs(tmp_path)))
    row = agg.benchmark_row(run, agg_meta={"aggregated_at": "T", "aggregator_commit": "c"})
    assert row["implementation_version"] == "1.4.0"  # sklearn, not picard 0.7


def test_mixed_schema_dir_accepts_minor_skips_major(agg, tmp_path, capsys):
    write(tmp_path, "benchmark_sub-01_hp1.0hz_v30.json", make_doc(schema_version="3.0"))
    write(tmp_path, "benchmark_sub-02_hp1.0hz_v31.json", make_doc(subject="sub-02", schema_version="3.1"))
    write(tmp_path, "benchmark_sub-03_hp1.0hz_v20.json", make_doc(subject="sub-03", schema_version="2.0"))
    runs = list(agg.discover_runs(tmp_path))
    subjects = sorted(r.subject for r in runs)
    assert subjects == ["sub-01", "sub-02"]  # 3.0 and 3.1 kept, 2.0 dropped
    out = capsys.readouterr().out
    assert "incompatible" in out          # the 2.0 skip was reported, not silent
    assert "minor differs" in out         # the 3.1 acceptance was noted


def test_all_columns_present_and_ordered(agg, tmp_path):
    row = _one_row(agg, tmp_path)
    for col in ("schema_version", "run_timestamp", "harness_commit",
                "implementation_version", "implementation_commit",
                "aggregated_at", "aggregator_commit", "jit_compile_s",
                "steady_iter_s", "dtype"):
        assert col in row, col
    # result_path is POSIX (no backslashes) even if aggregated elsewhere
    assert "\\" not in row["result_path"]
