#!/usr/bin/env python
"""Make "installed == intended" machine-checkable for the benchmark venvs.

`pins.toml` (next to this file) is the single source of truth for the exact git
commit of every AMICA implementation, grouped by venv. Run this tool WITH THE
TARGET VENV'S PYTHON:

  check_env.py specs  --venv NAME   print `git+URL@COMMIT` lines for pip install
  check_env.py verify --venv NAME   assert each pinned dist is installed at the
                                    pinned commit; exit nonzero on any mismatch
                                    or missing dist. Call before the first fit.
  check_env.py lock   --venv NAME   write locks/NAME.lock.json — the as-built
                                    record: each dist's version + commit plus the
                                    resolved torch/jax/jaxlib/numpy/scipy/mne.

`verify` matches installed distributions by their PEP 610 `direct_url.json`
SOURCE URL, not by distribution name — so it is robust to name-canonicalization
collisions (e.g. `pyamica` vs `pyAMICA`) and reports the true installed commit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib import metadata as im
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_DEFAULT = HERE / "pins.toml"
LOCKS_DIR = HERE / "locks"
PUBLISHED_LOCKS_DIR = LOCKS_DIR / "published"  # committed publication reference locks

# Recorded so time/memory deltas can be separated from stack-version deltas.
STACK = ("torch", "jax", "jaxlib", "numpy", "scipy", "mne", "mne-bids")


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - all benchmark venvs are >=3.11
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_venv(pins_path: Path, venv_name: str) -> dict:
    doc = _load_toml(pins_path)
    for venv in doc.get("venv", []):
        if venv.get("name") == venv_name:
            return venv
    names = ", ".join(v.get("name", "?") for v in doc.get("venv", []))
    raise SystemExit(f"check_env: unknown venv {venv_name!r}; pins.toml has: {names}")


def canon_url(url: str) -> str:
    """Normalize a git URL for comparison: drop pip's git+ prefix, @rev, .git, trailing /."""
    u = (url or "").strip()
    if u.startswith("git+"):
        u = u[4:]
    # Strip a trailing @rev only when the '@' is in the last path segment (after
    # the final '/'), so a `user@host` authority — e.g. ssh://git@github.com/o/r
    # or git@github.com:o/r — is left intact while `…/repo.git@abc123` is trimmed.
    at = u.rfind("@")
    if at > u.rfind("/"):
        u = u[:at]
    u = u.rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    return u.lower()


def _direct_url(dist) -> dict:
    try:
        raw = dist.read_text("direct_url.json")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def installed_by_url() -> dict:
    """Map canonical source URL -> {name, version, commit} for every git install."""
    out: dict = {}
    for dist in im.distributions():
        info = _direct_url(dist)
        url = info.get("url")
        if not url:
            continue
        commit = (info.get("vcs_info") or {}).get("commit_id")
        out[canon_url(url)] = {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "commit": commit,
        }
    return out


def _base_version(v: str) -> str:
    """Strip a PEP 440 local segment so 2.12.0+computecanada == 2.12.0."""
    return (v or "").split("+", 1)[0]


def _installed_vcs_commit(name: str) -> str | None:
    """The VCS commit of an installed dist if it was a git install, else None
    (a plain PyPI wheel has no direct_url vcs_info)."""
    try:
        info = _direct_url(im.distribution(name))
        return (info.get("vcs_info") or {}).get("commit_id")
    except Exception:
        return None


def cmd_specs(venv: dict) -> int:
    for pkg in venv.get("packages", []):
        if pkg.get("version"):  # PyPI version pin
            print(f"{pkg['name']}=={pkg['version']}")
        else:                   # git commit pin
            print(f"git+{pkg['url']}@{pkg['commit']}")
    return 0


def _published_lock_path(venv: dict) -> Path:
    return PUBLISHED_LOCKS_DIR / f"{venv['name']}.lock.json"


def _strict_stack_check(venv: dict) -> bool:
    """Paper mode: the installed numerical stack MUST match the committed
    publication lock. A drift (e.g. torch 2.13.0 where the paper used 2.12.0) is
    FATAL, not a warning. Requires a committed published lock — a paper run whose
    stack cannot be checked against a reference is not paper-grade.
    """
    path = _published_lock_path(venv)
    if not path.exists():
        try:
            shown = path.relative_to(HERE.parent)
        except ValueError:
            shown = path
        print(f"STRICT: no committed published lock at {shown} — cannot verify the "
              f"numerical stack in paper mode. Recover + commit it, or run without "
              f"strict.", file=sys.stderr)
        return False
    try:
        published = json.loads(path.read_text()).get("stack", {})
    except Exception as exc:
        print(f"STRICT: unreadable published lock {path}: {exc}", file=sys.stderr)
        return False
    now = _stack_versions()
    ok = True
    for name, want in published.items():
        got = now.get(name)
        if got is None:
            print(f"STRICT: {name} not installed (published {want})", file=sys.stderr)
            ok = False
        elif _base_version(got) != _base_version(want):
            print(f"STRICT: stack drift {name} {got} != published {want}", file=sys.stderr)
            ok = False
    if ok:
        print(f"STRICT stack OK vs published lock ({len(published)} pkgs)")
    return ok


def cmd_verify(venv: dict, strict: bool = False) -> int:
    strict = strict or os.environ.get("AMICA_STRICT_STACK") == "1"
    installed = installed_by_url()
    ok = True
    for pkg in venv.get("packages", []):
        # Version-pinned (PyPI wheel): the installed release version IS the
        # identity — these have no direct_url commit to match on.
        if pkg.get("version"):
            try:
                got_v = im.version(pkg["name"])
            except Exception:
                got_v = None
            if got_v is None:
                ok = False
                print(f"MISSING  {pkg['name']}: not installed (pinned =={pkg['version']})")
            elif _base_version(got_v) != _base_version(pkg["version"]):
                ok = False
                print(f"MISMATCH {pkg['name']}: installed {got_v} != pinned =={pkg['version']}")
            else:
                # A version pin means the PUBLISHED wheel. If instead the dist was
                # installed from git, its metadata version can lie (an unreleased
                # commit past the tag still declares the release version — the exact
                # e15e1588-declares-0.1.1 trap). Require the git commit to equal the
                # release commit we recorded; a bare wheel (no direct_url) is fine.
                inst_commit = _installed_vcs_commit(pkg["name"])
                want_commit = pkg.get("commit")
                if inst_commit and want_commit and inst_commit != want_commit:
                    ok = False
                    print(f"MISMATCH {pkg['name']}: version {got_v} OK but installed from "
                          f"git {inst_commit} != release commit {want_commit}")
                else:
                    print(f"OK       {pkg['name']}: =={pkg['version']} ({got_v})")
            continue
        # Commit-pinned (git): match by direct_url source URL + commit.
        want_url = canon_url(pkg["url"])
        want_commit = pkg["commit"]
        got = installed.get(want_url)
        if got is None:
            ok = False
            print(f"MISSING  {pkg['name']}: no install from {pkg['url']} "
                  f"(is it uninstalled or clobbered by a name collision?)")
            continue
        if got["commit"] != want_commit:
            ok = False
            print(f"MISMATCH {pkg['name']}: installed {got['commit']} "
                  f"!= pinned {want_commit}")
            continue
        print(f"OK       {pkg['name']}: {want_commit} ({got['version']})")
    # Stack: FATAL drift vs the committed publication lock in strict/paper mode
    # (AMICA_STRICT_STACK=1 or --strict); otherwise a soft warn vs the as-built lock.
    if strict:
        if not _strict_stack_check(venv):
            ok = False
    else:
        _report_stack_drift(venv)
    if not ok:
        print("check_env: installed != intended — refusing to proceed.", file=sys.stderr)
        return 1
    return 0


def _stack_versions() -> dict:
    out = {}
    for name in STACK:
        try:
            out[name] = im.version(name)
        except Exception:
            pass
    return out


def _lock_path(venv: dict) -> Path:
    return LOCKS_DIR / f"{venv['name']}.lock.json"


def _report_stack_drift(venv: dict) -> None:
    """Warn (do not fail) if the numerical stack drifted from the recorded lock."""
    lock_path = _lock_path(venv)
    if not lock_path.exists():
        return
    try:
        recorded = json.loads(lock_path.read_text()).get("stack", {})
    except Exception:
        return
    now = _stack_versions()
    for name, was in recorded.items():
        is_now = now.get(name)
        if is_now is not None and is_now != was:
            print(f"WARN     stack drift: {name} {was} (locked) -> {is_now} (installed)",
                  file=sys.stderr)


def cmd_lock(venv: dict) -> int:
    installed = installed_by_url()
    packages = {}
    for pkg in venv.get("packages", []):
        if pkg.get("version"):  # PyPI: record the installed release version
            try:
                packages[pkg["name"]] = {"version": im.version(pkg["name"]),
                                         "commit": pkg.get("commit"), "url": pkg.get("url")}
            except Exception:
                pass
            continue
        got = installed.get(canon_url(pkg["url"]))
        if got is not None:
            packages[pkg["name"]] = {"version": got["version"], "commit": got["commit"],
                                     "url": pkg["url"]}
    lock = {
        "venv": venv["name"],
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "stack": _stack_versions(),
    }
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    out = _lock_path(venv)
    out.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"check_env: wrote {out}")
    return 0


def cmd_pin(venv: dict, name: str, field: str) -> int:
    """Print a single pinned field (default: commit) for scripts — the ONE source
    of truth, so a submit script never hard-codes a SHA that can drift from pins.toml."""
    for pkg in venv.get("packages", []):
        if pkg["name"] == name:
            value = pkg.get(field)
            if value is None:
                print(f"check_env: {name} has no '{field}' pin", file=sys.stderr)
                return 1
            print(value)
            return 0
    print(f"check_env: no package {name!r} in venv {venv.get('name')!r}", file=sys.stderr)
    return 1


def cmd_stack_specs(venv: dict) -> int:
    """Print `name==version` for each pinned transitive-stack package, so setup
    scripts install the reproducible stack (`pip install`) rather than latest."""
    for name, ver in (venv.get("stack") or {}).items():
        print(f"{name}=={ver}")
    return 0


def cmd_fortran_sha(pins_path: Path) -> int:
    """Print the pinned Fortran amica17 binary sha256 (the single source of truth)."""
    doc = _load_toml(pins_path)
    sha = (doc.get("fortran") or {}).get("sha256")
    if not sha:
        print("check_env: no [fortran].sha256 in pins.toml", file=sys.stderr)
        return 1
    print(sha)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=("specs", "stack-specs", "verify", "lock", "pin", "fortran-sha"))
    parser.add_argument("--venv", help="venv name in pins.toml (required except for fortran-sha)")
    parser.add_argument("--name", help="package name (for `pin`)")
    parser.add_argument("--field", default="commit", help="pin field to print (for `pin`): commit|version")
    parser.add_argument("--strict", action="store_true",
                        help="verify: FATAL on numerical-stack drift vs the committed "
                             "published lock (paper mode; also via AMICA_STRICT_STACK=1).")
    parser.add_argument("--pins", type=Path, default=PINS_DEFAULT)
    args = parser.parse_args(argv)
    if args.command == "fortran-sha":
        return cmd_fortran_sha(args.pins)
    if not args.venv:
        parser.error(f"--venv is required for '{args.command}'")
    venv = load_venv(args.pins, args.venv)
    if args.command == "pin":
        if not args.name:
            parser.error("--name is required for 'pin'")
        return cmd_pin(venv, args.name, args.field)
    if args.command == "verify":
        return cmd_verify(venv, strict=args.strict)
    return {"specs": cmd_specs, "stack-specs": cmd_stack_specs, "lock": cmd_lock}[args.command](venv)


if __name__ == "__main__":
    raise SystemExit(main())
