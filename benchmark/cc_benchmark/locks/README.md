# As-built environment locks

Each `setup_*.sh` writes `<venv>.lock.json` here after a successful build
(via `check_env.py lock --venv <name>`). A lock records, for that venv:

- `python`, `executable`
- `packages`: each pinned AMICA implementation's installed `version` + git `commit`
  (the commit is also asserted equal to `pins.toml` by `check_env.py verify`)
- `stack`: the resolved `torch / jax / jaxlib / numpy / scipy / mne` versions

`pins.toml` pins the git commits (exactly recoverable, the implementation
identity). The transitive numerical stack is not pinned there — the Alliance
wheelhouse resolves it per site — so the lock is where those resolved versions
are captured.

## Two kinds of lock

- **Auto-generated as-built locks** — `locks/<venv>.lock.json`, written by each
  `setup_*.sh` (via `check_env.py lock`). Machine/site-specific; **gitignored**.
  `check_env.py verify` WARNs (non-fatal) if the installed stack drifts from these.

- **Committed publication reference locks** — `locks/published/<venv>.lock.json`,
  version-controlled. These record the exact numerical stack the **published**
  numbers used, so the stack lives in the repo, not just in prose. Currently
  committed: `competitors.lock.json` (recovered 2026-08-11 from the surviving
  publication venv — torch 2.12.0+cc / numpy 2.4.2+cc / scipy 1.17.0+cc / mne
  1.12.1; see `../provenance/README.md`). Recover + commit `pamica`/`fir`/
  `neuromechanist` here too when re-running a paper campaign.

## Paper mode: fail on stack drift

`check_env.py verify --strict` (or `export AMICA_STRICT_STACK=1`, which every
`verify` call honors) makes a numerical-stack drift vs the committed published
lock **FATAL** instead of a warning — and requires a committed published lock to
exist. Enable it for paper campaigns (e.g. in `env.local`) so a rebuilt env that
resolved a different `torch` cannot silently produce "reproduced" numbers.
