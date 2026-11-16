# Contributing

How the four of us work on this repository.

## Branches

| Branch | Purpose | Who writes to it |
|---|---|---|
| `main` | Stable, demo/submission-ready code | Only merges from `dev` via PR |
| `dev` | Integration branch: everything that's finished and reviewed | Only merges from feature branches via PR |
| `feature/*`, `fix/*`, `docs/*`, `chore/*` | One short-lived branch per task | The person doing the task |

No one pushes directly to `main` or `dev`.

## Workflow

1. **Start from an up-to-date `dev`:**
   ```bash
   git checkout dev
   git pull
   git checkout -b feature/<short-description>   # e.g. feature/c1-outlier-check
   ```
2. **Work in small commits** with clear messages (e.g. `Add range validation to C1`).
3. **Check locally before pushing:**
   ```bash
   uv sync --all-packages
   uv run ruff check .
   uv run ruff format .
   uv run pytest
   ```
4. **Push and open a Pull Request into `dev`:**
   ```bash
   git push -u origin feature/<short-description>
   ```
   In the PR description, say what changed and how you tested it.
5. **Review:** at least one teammate approves, and CI must pass.
6. **Merge** (squash merge preferred) and delete the branch.
7. **Release:** when `dev` is stable (for example before a milestone or demo),
   open a PR from `dev` into `main`.

## Keeping your branch up to date

If `dev` moved ahead while you were working:

```bash
git checkout dev && git pull
git checkout feature/<short-description>
git merge dev        # resolve any conflicts, run the tests, push
```

Keep branches short-lived (a few days at most) to avoid big conflicts.

## Branch naming

- `feature/...`: new functionality
- `fix/...`: bug fixes
- `docs/...`: documentation and journals
- `chore/...`: tooling, CI, dependencies

## Ground rules

- Do not commit data files or trained models. `data/` and `models/` are kept
  out of git except for `.gitkeep`.
- Respect the chronological data split described in
  [docs/data-split.md](docs/data-split.md). Never use test-year data for
  training or tuning.
- Add or update tests in `tests/` for any new behavior.
- Add new dependencies with `uv add` so that `uv.lock` stays in sync. CI fails
  if the lockfile is out of date.
