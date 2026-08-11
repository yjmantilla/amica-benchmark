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
are captured, and `check_env.py verify` warns if the installed stack later drifts
from the recorded lock.

The `*.lock.json` files are gitignored: they are machine/site-specific build
records. Commit one deliberately (e.g. alongside a paper's result set) when you
want that exact environment archived with the numbers it produced.
