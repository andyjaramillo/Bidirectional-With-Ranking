# rank_forward/experiments

One-off **experiment drivers** (not library code) that wire the `rank_forward`
baseline — and, for the head-to-head, our bidirectional `learning/online_run` —
into full train + evaluate comparisons. Run from the repo root with
`PYTHONPATH=.`.

All drivers default to the **fair full goal**: boxes-on-targets **and** the
player back at its start cell — the goal the bidirectional method actually
targets (its backward search is seeded at the start cell; `goal_map`'s `[7][5]`
is an unused attribute, *not* the goal). Set `FULL_GOAL=0` (or `H2H_FULL_GOAL=0`)
to use the easier classical goal (player anywhere) for reference — but note that
is **not** an apples-to-apples comparison against the bidirectional method.

Optimal training trajectories are solved once and cached under
`rank_forward/cache/` (gitignored); reruns over different losses reuse them.

| script | what it does | run |
|---|---|---|
| `head_to_head.py` | forward (ranking heuristic) vs bidirectional F2F on the same held-out split, same goal, same node-expansion unit; absolute-expansion table | `PYTHONPATH=. python rank_forward/experiments/head_to_head.py` |
| `loss_sweep.py` | trains all five losses (L\*, L_gbfs, L2, L_rt, Bellman), evaluates each in A\* and GBFS vs the blind baseline | `PYTHONPATH=. python rank_forward/experiments/loss_sweep.py` |
| `forward_fullgoal.py` | forward-only (L_gbfs + L\*) under the full goal — the quick way to the corrected forward numbers without the bidirectional run | `PYTHONPATH=. python rank_forward/experiments/forward_fullgoal.py` |
| `anchor_strategy_run.py` | one anchor-selection strategy (`temporal`/`top_of_open`/`closest_anchor`) trained on-policy + evaluated on the held-out tail — the anchor-search study | `N_TOTAL=1800 ANCHOR_STRATEGY=top_of_open NEVAL=200 PYTHONPATH=. python rank_forward/experiments/anchor_strategy_run.py` |

### Common env knobs
`N_TOTAL`/`H2H_NTRAIN`+`H2H_NEVAL` (split sizes), `STEPS`/`H2H_FWD_STEPS`
(forward SGD steps), `SOLVE_CAP` (per-instance budget for optimal label
generation), `MAX_ITERS` (eval expansion budget, default 10000 to match the
bidirectional run), `FULL_GOAL`/`H2H_FULL_GOAL`, `SEED`, `LOSSES` (sweep only).

These were run detached (`nohup … &`) because a full run is long (the bidirectional
side trains online over the train split). The drivers print incrementally
(`flush=True`) so progress is visible in the log.

### Headline result (full 2000 subset of the 10k solvable set, fair full goal)
Forward ranking is competitive with the bidirectional method: forward `L*`/A\*
solves ~196/200 (≈ bidirectional 198/200) with a **lower median** expansion
count, while bidirectional keeps a **lower mean** (lighter hard-instance tail).
See the git history / session notes for the full table.

## Anchor-selection (anchor search) results

Anchor search (Lavasani 2024) frames bidirectional F2F as: each side scores its
nodes against the opponent's single representative **anchor**. *How* that anchor
is chosen is the "anchor selection" axis. Our TTBS uses `temporal` (anchor = the
opponent's most-recently-expanded node). We compared three strategies through
the **identical on-policy pipeline** (train the NN on 1800 on-policy, eval frozen
on the held-out 200, `MAX_ITERS=10000`, same `SmallCNN`/metric):

**Seed 0 (single seed)** suggested `top_of_open` wins big — learned median 167 vs
`temporal` 255 (−35%) at equal solve rate, with `closest_anchor` at 239. But this
**did not replicate.**

**Replication across 3 seeds (learned median / mean / solved):**

| seed | `temporal` | `top_of_open` |
|---|---|---|
| 0 | 255 / 700 / 199 | 167 / 683 / 199 |
| 1 | 155 / 491 / 197 | 183 / 645 / 199 |
| 2 | 183 / 624 / 197 | 212 / 581 / 196 |
| **avg** | **198** / 605 / 197.7 | **188** / 636 / 198 |

**Verdict: the strategies are comparable; the single-seed 35% gap was seed noise.**
The median ranges overlap heavily (`temporal` 155–255 vs `top_of_open` 167–212);
`temporal`'s mean is actually slightly lower (605 vs 636) and solve rates are tied
(~197–198/200). `top_of_open` is only modestly more *consistent* (spread ±23 vs
±50) — not enough to outweigh `temporal`'s lower mean, simplicity, and speed. So
**`temporal` stays the default**; the paper-faithful anchor is not meaningfully
better under the learned heuristic at this scale. (Note: the *blind/analytic*
heuristic is seed-independent — `temporal` 545 vs `top_of_open` 564 vs
`closest_anchor` 506.)

Run via `anchor_strategy_run.py` once per (seed, strategy), under `caffeinate`.
Lesson: replicate across seeds before acting on a single-seed gap.

## Fairness notes (read before trusting the numbers)

**The shared goal includes the player position.** The bidirectional method's goal
is boxes-on-targets **and the player back at its START cell** — its backward search
is seeded at `initializeBackwardPuzzle(start)`, which keeps the player at the start,
and every reconstructed solution ends there (verified empirically). NOTE:
`SokobanGame.goal_map` pins the player at `[7][5]`, but that attribute is **unused**
by the search — it is *not* the goal (an easy red herring). The forward competitor
must therefore solve the SAME full goal (`FULL_GOAL=1`: boxes-on-targets AND
player@start), with full-board (player-aware) state hashing (`encodeMap`). Using the
classical boxes-only goal (player anywhere) for the forward side lets it stop as soon
as the boxes land and is **not** a fair comparison.

**Relaxation invariant (verified — no bug).** For a *fixed* heuristic and *fixed*
goal context, the full goal can only be harder-or-equal than the classical goal: the
full-goal search is the classical search *continued past* the boxes-on-targets state
until the player also reaches start (same f-ordering, cannot stop earlier). Controlled
test (one heuristic, vary only the stop condition, 120 boards): classical solved ==
full solved, classical median **152** ≤ full median **172**, **zero** violations
(full never solved anything classical didn't; classical never used more expansions).
So requiring player@start does **not** make search easier. An earlier observation
that the full goal "solved more" (≈196 vs ≈172) was an artifact of comparing **two
different heuristics**: the classical-goal run was handicapped by feeding the
heuristic the inconsistent `[7][5]` goal context, whereas the full-goal run feeds the
correct player@start goal — so the full-goal heuristic is simply better-specified,
independent of which goal it is scored against.

**Expanded-node counting is consistent across methods (audited).** The metric is
"number of nodes whose successors were generated." Verified against an independent
counter (`availableStates` calls, one per expansion): forward
`first_solved_iter == len(closed) == #expansions` exactly (the goal is detected on
pop and never expanded, correctly uncounted); bidirectional true expansions
`== len(closed_f)+len(closed_b) == self.iteration`, summed over BOTH frontiers (the
standard uni-vs-bi total). Caveat: `first_meeting_iter` equals (expansions − 1) — it
is read mid-step before the meeting-finding expansion is counted — so `head_to_head.py`
uses `len(closed_f)+len(closed_b)` for the bidirectional count to be exactly
consistent with the forward `len(closed)`. (The earlier reported bidirectional medians
used `first_meeting_iter`, i.e. were ~1 lower; immaterial vs medians of 86/249, and it
had slightly favored the bidirectional side.)

**Amount of learning — comparable in budget, asymmetric in supervision.**
- Forward: 12,000 SGD steps over 1,794 optimal-trajectory instances (~6.7 epochs),
  ~71 states/step → ~0.85M state-gradient evaluations.
- Bidirectional: 14,392 updates (8 per puzzle after warmup over 1,800 puzzles, +342
  re-mined re-solves), batch 64 → ~0.92M sample-gradients, sampled from a 20k replay
  buffer.

Gradient budget is close (~12k vs ~14k updates; ~0.85M vs ~0.92M sample-gradients),
same `SmallCNN` (23,585 params), same split, same node-expansion metric. The real
asymmetry is the **supervision source**: the *offline* forward method trains on
**optimal** trajectories from a separate admissible-A\* solver (expert imitation),
while the bidirectional method **bootstraps from its own satisficing** F2F solutions.

**Same-supervision result (`forward_bootstrap.py`).** We removed the asymmetry by
training the forward method in the SAME bootstrap manner — self-supervising on the
paths its own search finds (13,568 updates, matched to the bidirectional's 14,392).
On the held-out 200 (full goal, same `SmallCNN`):

| method (matched supervision) | solved | median exp | mean exp |
|---|---:|---:|---:|
| forward L\* / A\* — offline OPTIMAL labels | 196/200 | 148 | 957 |
| forward L\* / A\* — BOOTSTRAP (own paths) | 195/200 | **86** | **546** |
| bidirectional learned (bootstrap) | **198**/200 | 249 | 607 |

Two findings: (1) for the forward method, **bootstrap (on-policy) beats optimal
labels (off-policy)** — median 86 vs 148 — because training on the states the
heuristic's own search visits removes the distribution mismatch of imitating an
external solver (the optimal-label "advantage" was actually a handicap). (2) With
supervision matched (both bootstrap), the forward ranking method is **more
node-efficient** than the bidirectional method (median 86 vs 249, mean 546 vs 607);
the bidirectional keeps only a small solve-rate edge (198 vs 195/200). So the earlier
"bidirectional has the lighter tail" conclusion does NOT survive supervision matching.

Caveats: single seed / architecture / split; forward counts forward expansions vs
bidirectional total (both-frontier) expansions (the standard uni-vs-bi comparison);
the ~50-puzzle Manhattan warmup is mildly optimal for A\*. Replicate across seeds
before treating as a strong claim.
