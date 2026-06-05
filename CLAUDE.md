# Project conventions

## Git workflow
- **Work directly on `master`.** Commit straight to `master` and push to `origin/master`.
  Do **not** create feature branches or PRs unless explicitly asked.
- `master` tracks **`origin`** = the fork `supertweety/Bidirectional-With-Ranking` (this is "our master").
- `upstream` = the original repo `andyjaramillo/Bidirectional-With-Ranking`. **Never push to upstream.**
  Local `master` may have once tracked `upstream/master`; it now tracks `origin/master`.
- `gh`'s default repo is set to the fork. Always pass `-R supertweety/Bidirectional-With-Ranking`
  to `gh` if in doubt — a missing default previously made `gh pr` commands hit upstream by mistake.

## Research goal (overarching constraint)
- The goal is **not** the best Sokoban bot. We want methods that generalize to planning
  problems broadly. **Avoid adding Sokoban-specific ingredients;** keep changes domain-agnostic.

## Data
- Default benchmark is the **solvable-only** dataset `data/solvable10_3box.txt`
  (load via `game.getData.get_solvable_data(limit=...)`). It excludes the player-goal-pinned
  unmeetable artifacts. The unfiltered source `data/states10_3box.txt` is left intact.
- Rebuild/extend the solvable set with `analysis/build_solvable_benchmark.py`
  (env knobs: `BENCH_N`, `BENCH_CAP`, `BENCH_KEEP`, `BENCH_WORKERS`).
- The online experiment `learning/online_run.py` reads the solvable dataset; `MAX_ITERS` is an env knob.
