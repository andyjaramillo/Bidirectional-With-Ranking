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
