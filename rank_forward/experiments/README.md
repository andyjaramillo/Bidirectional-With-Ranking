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

**Amount of learning — comparable in budget, asymmetric in supervision.**
- Forward: 12,000 SGD steps over 1,794 optimal-trajectory instances (~6.7 epochs),
  ~71 states/step → ~0.85M state-gradient evaluations.
- Bidirectional: 14,392 updates (8 per puzzle after warmup over 1,800 puzzles, +342
  re-mined re-solves), batch 64 → ~0.92M sample-gradients, sampled from a 20k replay
  buffer.

Gradient budget is close (~12k vs ~14k updates; ~0.85M vs ~0.92M sample-gradients),
same `SmallCNN` (23,585 params), same split, same node-expansion metric. The real
asymmetry is the **supervision source**: the forward method trains on **optimal**
(provably shortest) trajectories from a separate admissible-A\* solver (expert
imitation), while the bidirectional method **bootstraps from its own satisficing**
F2F solutions. This favors the forward side (better labels, plus an external solver's
search work that is not counted in its 12k steps). A same-supervision study — train
the bidirectional on optimal trajectories too, or the forward on bootstrapped paths —
is the way to remove that asymmetry, and is future work.
