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

## Method defaults (reference configuration)
- Bidirectional TTBS, `temporal` anchor, learned pairwise NN heuristic trained on-policy
  with `LOSS=both` (0.5·MSE + 0.5·margin) **+ `PATH_RANK`** (within-path pairs-of-pairs
  margin — default on since its 3-seed validation; `PATH_RANK=no` recovers the old baseline).
- **Meeting: `meet_on_generate=True` + `seam_repair=True`** (Wave 1a, default since its
  3-seed validation) — earliest detection + BFS-shortest path over the explored subgraph;
  `MEET_ON_GENERATE=no SEAM_REPAIR=no` recovers the legacy reference.
- **Scoring: `bhffa_g=True`** (Wave 1c, default since its 3-seed validation) —
  BHFFA-complete `f(n) = g(n) + h(n,d*) + g_opp(d*)`; `BHFFA_G=no` recovers pre-1c.
- **Direction-correct labels & queries: `dir_correct=True` / `DIRECTED`** (default since
  its 3-seed validation; the quasimetric fix) — direction-correct pair labels, and the
  backward frontier queries `h(anchor_fwd, target_fwd, flip(node))` in the forward frame;
  `DIRECTED=no` recovers the Wave-1 reference.
- Stored 3-seed reference numbers live in `rank_forward/experiments/README.md`
  (current reference: avg 198.7 solved / 155.3 median / 513 mean on the held-out 200).

## Data
- Default benchmark is the **solvable-only** dataset `data/solvable10_3box.txt`
  (load via `game.getData.get_solvable_data(limit=...)`). It excludes the player-goal-pinned
  unmeetable artifacts. The unfiltered source `data/states10_3box.txt` is left intact.
- Rebuild/extend the solvable set with `analysis/build_solvable_benchmark.py`
  (env knobs: `BENCH_N`, `BENCH_CAP`, `BENCH_KEEP`, `BENCH_WORKERS`).
- The online experiment `learning/online_run.py` reads the solvable dataset; `MAX_ITERS` is an env knob.
