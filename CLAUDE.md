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
- **Cost-generality (2026-07-03): do not bake unit-cost assumptions into new methods.**
  We want to venture beyond unit-cost problems later. New loss terms / graph machinery must
  read edge costs from a single hook (default 1.0 on this testbed) rather than hardwiring
  "+1"; shortest-path helpers should be Dijkstra-ready (BFS only as the uniform-cost fast
  path). See APPROACH.md "Cost-generality" for the inventory of existing unit-cost touchpoints.

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
- **Hindsight query supervision: `HINDSIGHT`** (Wave 2e, default since its 5-seed
  validation) — label the search's own `(node, anchor)` h-queries with exact union-graph
  distances and add them to the buffer; `HINDSIGHT=no` recovers the pre-2e reference.
  (Adopted despite narrowly missing the strict per-seed-regression bar — a mean/tail win;
  median is a wash. Caveats in experiments/README.md.)
- **Wave 2d `CONSIST` was tested and NOT adopted (wash).** Available as a flag, default off.
- **Wave 3 factorized (quasi)metric embedding (`MODEL=embed`) was tested and REJECTED.**
  Head-selection gate (seed 0): best head (`mlp`) is 3.3× worse on expansions and 1.5×
  slower than the `smallcnn` cross-encoder baseline; ℓ1/quasi worse still. Root cause is a
  bi-encoder representational ceiling (see APPROACH.md Wave 3). `EmbedCNN` + the embedding
  search branch are preserved off by default (`MODEL=smallcnn`); the deferred soft-min
  full-F2F arm was abandoned with it. Do not re-attempt on a bi-encoder substrate.
- Stored reference numbers live in `rank_forward/experiments/README.md`
  (current reference: avg 199.2 solved / 148.5 median / 453.5 mean on the held-out 200, 5 seeds).

## Data
- Default benchmark is the **solvable-only** dataset `data/solvable10_3box.txt`
  (load via `game.getData.get_solvable_data(limit=...)`). It excludes the player-goal-pinned
  unmeetable artifacts. The unfiltered source `data/states10_3box.txt` is left intact.
- Rebuild/extend the solvable set with `analysis/build_solvable_benchmark.py`
  (env knobs: `BENCH_N`, `BENCH_CAP`, `BENCH_KEEP`, `BENCH_WORKERS`).
- The online experiment `learning/online_run.py` reads the solvable dataset; `MAX_ITERS` is an env knob.
