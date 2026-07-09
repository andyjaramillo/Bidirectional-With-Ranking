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
| `path_rank_run.py` | default MSE+margin **plus the within-path pairs-of-pairs margin** (`PATH_RANK=yes`): rank h over pairs of same-path node pairs by subpath length — the within-INSTANCE ordering the search's open list actually uses (buffer margin pairs are almost always cross-puzzle) | `PATH_RANK=yes N_TOTAL=1800 NEVAL=200 PYTHONPATH=. python rank_forward/experiments/path_rank_run.py` |

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

| strategy | seed 0 | seed 1 | seed 2 | avg median | avg mean |
|---|---|---|---|---:|---:|
| `temporal` (ours) | 255/700/199 | 155/491/197 | 183/624/197 | **198** | 605 |
| `top_of_open` (paper TTBS) | 167/683/199 | 183/645/199 | 212/581/196 | **188** | 636 |
| `hybrid_af` (AST_AF) | 238/625/197 | 244/680/199 | 215/686/199 | **232** | 664 |
| `closest_anchor` (policy A) | 239/669/194 | — | — | (239, 1 seed) | 669 |

**Verdict: anchor selection is a wash under the learned heuristic.** All strategies
land within the seed-noise band — `temporal` alone spans 155–255. By 3-seed
average learned median: `top_of_open` 188 ≤ `temporal` 198 < `hybrid_af` 232
(≈ `closest_anchor` 239, 1 seed); but `temporal` has the lowest *mean* (605), all
have tied solve rates (~197–198/200), and the gaps are inside the noise. The
seed-0 "`top_of_open` −35%" was noise. `hybrid_af` (the thesis's most-robust
variant) is the most *consistent* (median 215–244) yet slightly *worse* than
`temporal` on median and mean — its robustness on hard classical-planning domains
does not translate to a win on this single Sokoban domain with the learned NN.
**`temporal` stays the default** (simplest, fastest, lowest mean). (Blind/analytic,
seed-independent: `temporal` 545, `top_of_open` 564, `hybrid_af` 583,
`closest_anchor` 506.)

**Conclusion of the anchor-search investigation:** having tried every instantiation
with a thesis-backed reason to differ — `temporal`, `top_of_open`, `closest_anchor`
(policy A), `hybrid_af` (AST_AF), plus full-front-to-front via the separate
`full_f2f` flag — **none meaningfully beats our simple temporal anchor under the
learned heuristic.** Anchor selection is not a lever for this method/domain.

Run via `anchor_strategy_run.py` once per (seed, strategy), under `caffeinate`.
Lesson: replicate across seeds before acting on a single-seed gap.

## Within-path pairs-of-pairs margin (PATH_RANK) — a small, consistent win

The buffer margin term already ranks pairs-of-pairs by their distance labels,
but its random batch pairs are almost always **cross-puzzle**. The search's open
list only ever compares nodes of the **same instance**, so within-instance
ordering is the operationally relevant constraint. `PATH_RANK` adds it: for two
pairs of nodes on the same fresh solution path, (p_i,p_j) and (p_i',p_j'),
enforce `h(p_i,p_j) + margin <= h(p_i',p_j')` whenever `|i-j| < |i'-j'|`
(32 sampled pairs per update → one h-batch, ~500 hinge comparisons; weight 0.5
**added on top of** the untouched 0.5·MSE + 0.5·margin — scale stays pinned,
unlike the failed pure-ranking losses; see git history `c197bae..75005cc`).

**3 seeds (learned solved / median / mean), held-out 200, blind = 194/545/1380:**

| seed | MSE+margin baseline | + PATH_RANK(w=0.5, pairs=32) |
|---|---|---|
| 0 | 197 / 247 / 726 | 199 / 237 / 669 |
| 1 | 198 / 160 / 607 | 196 / 168 / 540 |
| 2 | 197 / 225 / 710 | 197 / 184 / 615 |
| **avg** | 197.3 / **211** / **681** | 197.3 / **196** / **608** |

**Verdict: a modest but consistent improvement, clearest in the tail.** The
**mean improves on all 3 of 3 seeds** (−8%, −11%, −13%; avg −11%) — the
hard-instance tail is where within-instance ordering pays. Median improves on
2/3 (avg −7%, inside the seed-noise band on its own); solve rate unchanged.
No metric materially degrades on any seed. Cost: extra training-time compute
only (~32 extra h-queries per update); zero search-time cost.
**ADOPTED AS DEFAULT** (`PATH_RANK=yes`): from here on, the reference
bidirectional configuration is MSE+margin+PATH_RANK; the table above is the
adoption evidence, and `PATH_RANK=no` recovers the old baseline. Knobs
`PATH_RANK_W` (0.5) and `PATH_RANK_PAIRS` (32) are untuned first guesses.

## Wave 1a: meet-on-generate + seam repair — ADOPTED AS DEFAULT

`refine_path()` returns the BFS-shortest start→goal plan over the FULL recorded
transition graph (`edges_f` + flipped `edges_b`) instead of the parent-pointer
splice — provably the shortest plan using only explored transitions. This makes
all meeting-detection rules quality-equivalent, so the earliest detection
(`meet_on_generate`) becomes safe: earliness and path quality are decoupled.
(This also resolves/subsumes the 2026-06-22 "principled meeting" seam-rule
idea, whose goal was exactly this trade-off.) Driver: `seam_repair_run.py`
(4 arms: {blind, learned} × {new, legacy}, expansions AND plan lengths).

**3 seeds, trained meetgen+repair, held-out 200 (solved/median/mean):**

| seed | legacy reference (PATH_RANK) | new config | plan len new vs legacy* |
|---|---|---|---|
| 0 | 199 / 237 / 669 | 198 / 207.5 / 683 | 35.91 vs 36.34 |
| 1 | 196 / 168 / 540 | **200** / 159 / 539 | 36.56 vs 37.18 |
| 2 | 197 / 184 / 615 | 199 / 187 / 550 | 36.16 vs 36.64 |
| **avg** | 197.3 / 196 / 608 | **199.0 / 184.5 / 591** | shorter 3/3 |

*same trained model evaluated under both configs.

Verdict: **+1.7 solved (seed 1 = 200/200, first perfect score), −6% median,
−3% mean — and repaired plans are SHORTER than legacy's on all seeds** despite
stopping earlier (the quality objection that originally forced the
meet-on-generate revert is reversed, not just neutralized). Within-model,
new detection improves the median on 3/3 seeds; the one apparent mean
regression (seed 0) is a solved-set composition effect (+2 hard instances
solved). Blind: +2 solved, −15% median (deterministic).
**Defaults flipped** (`meet_on_generate=True`, `seam_repair=True` in the search
class; `MEET_ON_GENERATE`/`SEAM_REPAIR` env default yes); the pre-Wave-1a
reference is recovered with `MEET_ON_GENERATE=no SEAM_REPAIR=no`. NOTE: the
blind training-baseline pass now reads 1367/561 (was 1520/658).

## Wave 1c: BHFFA-complete scoring — ADOPTED AS DEFAULT; Wave 1b closed

**Wave 1b (exact on-path relabeling) is closed as SUBSUMED by Wave 1a**: a
subpath of a shortest path is a shortest path, so the repaired path's `|i-j|`
labels already ARE exact union-graph distances. Diagnostic over 118 puzzles:
62,266 on-path pairs, **zero slack**; legacy paths had slack on only 3.1% of
pairs. The residual label problem is **directional**: reverse pairs (half of
training data via `symmetric=True`, labeled `j-i`) have a finite reverse route
in the explored subgraph only 15.3% of the time, and 35.4% of those are
mislabeled (mean err +2.9, p90 8) — Sokoban's quasimetric asymmetry is real.
This opens the direction-correct-labels gate (APPROACH.md Wave 3 prerequisite).

**Wave 1c** restores the classical BHFFA front-to-front term our TTBS score
dropped: `f(n) = g(n) + h(n,d*) + g_opp(d*)` (`bhffa_g`). Cross-round heap
comparability is the only thing it changes (constant within a round). Driver:
`bhffa_run.py`. **3 seeds, trained+evaled with the term, held-out 200:**

| seed | pre-1c reference | + BHFFA | within-model on-vs-off |
|---|---|---|---|
| 0 | 198 / 207.5 / 683 | 199 / 190 / 684 | mean 684<691, solved 199>198 |
| 1 | 200 / 159 / 539 | 199 / 151 / 508 | mean 508<577, solved 199>197 |
| 2 | 199 / 187 / 550 | 200 / 178 / 561 | mean 561<771, solved 200>199 |
| **avg** | 199.0 / 184.5 / 591 | **199.3 / 173.0 / 584** | |

Verdict: **median better on 3/3 seeds (avg −6.2%)**; within-model the eval-time
term improves **mean and solve rate on 3/3 seeds** (it fixes the stale-entry
mis-ordering on the hard tail). Blind/analytic h gets *worse* under the term
(597 vs 466 median) — a weak h cannot exploit the corrected score; the
on-policy net adapts to it. **Defaults flipped** (`bhffa_g=True`, `BHFFA_G`
default yes); `BHFFA_G=no` recovers the pre-1c reference. Current reference:
**avg 199.3 / 173.0 / 584**. Wave 1 of APPROACH.md is complete.

## Direction-correct labels & queries (DIRECTED) — ADOPTED AS DEFAULT

State-space distance under irreversible moves is a **quasimetric**; the
pipeline symmetrized it by fiat (reverse pairs — half of all training data —
labeled with the forward `j-i`; the 1b diagnostic measured 35.4% of verifiable
reverse labels wrong, mean +2.9). `DIRECTED` fixes both halves: labels
(direction-correct pairs only, random budget 2L→4L, measured throughput ~2/3
of legacy — ordered-pair space is half; mirrored off-path insertion dropped)
and queries (the backward frontier asks for the forward-dynamics distance it
actually needs: `h(anchor_fwd, forward_target, flip(node))`, all forward-frame
— legacy queries passed backward-frame boards the buffer never contains).
Driver: `directed_run.py`.

**3 seeds, trained+evaled DIRECTED, held-out 200 (solved/median/mean):**

| seed | Wave-1 reference | DIRECTED | same net, legacy queries |
|---|---|---|---|
| 0 | 199 / 190 / 684 | 199 / 172 / 562 | 200 / 265.5 / 647 |
| 1 | 199 / 151 / 508 | 198 / 138 / 481 | 200 / 218.5 / 563 |
| 2 | 200 / 178 / 561 | 199 / 156 / 496 | 199 / 235.0 / 487 |
| **avg** | 199.3 / 173.0 / 584 | **198.7 / 155.3 / 513** | — |

Verdict: **median −10.2% and mean −12.2%, better on 3/3 seeds each** — the
largest single-change improvement of the roadmap; solve rate within noise
(596 vs 598/600); plans shorter (~33.3 vs ~34.4 mean moves); training loss
floor drops from ~12–20 to ~5–8 (the corrupted reverse labels were a large
part of the irreducible regression error). The legacy-query decomposition arms
(~220–265 median with the same nets) confirm the mechanism: the backward
frontier had been served by an out-of-distribution, direction-confused
heuristic. This empirically validates the quasimetric premise underlying the
Wave-3 asymmetric-embedding proposal. **Defaults flipped** (`dir_correct=True`,
`DIRECTED` env default yes); `DIRECTED=no` recovers the Wave-1 reference.
Current reference: **avg 198.7 / 155.3 / 513**.

## Wave 2d: local metric grounding — WASH (not adopted, kept as flags)

One-step consistency hinges on verified path edges (cost-general via
`path_edge_costs`) + `h(x,x)=0` zero-anchor pairs (`CONSIST`, `N_ZERO_PAIRS`;
driver `consist_run.py`). 3 seeds vs reference 198.7/155.3/513:
200/150/518 | 198/216/495 | 197/155/466 → avg 198.3/173.7/493. Mean −4% (2/3
seeds) but median mixed with a large seed-1 regression — inside the noise
band, far from the 3/3 adoption bar. **The judges' redundancy prediction was
right, and verification showed why**: with exact-in-subgraph labels (post
seam-repair) and DIRECTED semantics, models near the regression optimum
already satisfy the consistency constraints (~0 violations on fitted paths),
so the hinge mostly contributes gradient noise. Informative wash: the current
pipeline's h is already locally consistent where it is well-fit. Flags remain
available (default off).

## Wave 2e: hindsight query supervision (HINDSIGHT) — ADOPTED

The open list is ordered by `h(frontier-node, moving-anchor)` queries, but the
buffer contained no pairs of that type — a covariate shift stage-0 measured at
**1.91× |h−d| error** on query pairs vs training pairs. `HINDSIGHT` reservoir-
samples the queries each solve issues, labels their canonical DIRECTED
`(source→dest)` forward-frame pairs with exact union-graph distances, and adds
up to 32 finite ones/puzzle to the buffer (pairwise Hindsight Experience
Replay). Driver: `hindsight_run.py`.

**5 seeds, held-out 200 (learned solved/median/mean; seeds 3–4 run matched):**

| seed | reference (DIRECTED) | + HINDSIGHT |
|---|---|---|
| 0 | 199 / 172 / 562 | 199 / 155 / 512 |
| 1 | 198 / 138 / 481 | 199 / 118 / 372 |
| 2 | 199 / 156 / 496 | 199 / 162 / 501 |
| 3 | 200 / 150 / 471 | 199 / 170 / 455 |
| 4 | 199 / 149 / 471 | 200 / 137 / 427 |
| **avg** | 199.0 / 153.0 / 496.3 | **199.2 / 148.5 / 453.5** |

Verdict: **mean −8.6% (better on 4/5 seeds)** — the tail metric it targets;
**median a wash** (−2.9%, better 3/5); solve rate tied (996 vs 995).
**Adopted (lead call).** It cleared the pre-registered mean (≥4/5), median
(≥3/5), and solve-rate criteria but **narrowly missed the strict "no >10%
regression on any seed" clause** — seed-3 median 150→170 (+13%), which sits
inside the method's own historical median spread (138–172), i.e. reads as
noise. Standing caveats: labels are explored-subgraph upper bounds; ~32
samples/puzzle dilute the fixed 20k FIFO buffer; ~21% query finite-fraction.
`HINDSIGHT=no` recovers the pre-2e reference. **New reference: avg
199.2 / 148.5 / 453.5.**

## Wave 3: factorized (quasi)metric embedding (MODEL=embed) — REJECTED

`h(x,y) = d(φ(x), φ(y))` — score each board to ℝ⁶⁴ independently, then a
metric/quasimetric distance. Meant to buy `h(x,x)=0`, triangle inequality, and
O(k) per-anchor rescoring (→ feasible principled full-F2F). `EmbedCNN`
(`learning/nn.py`) + embedding branch in `AI_Bidirectional`; guarded to require
`DIRECTED`. Driver: `embed_run.py` (reports expansions AND wall-time). Axioms
verified exact by construction; O(k) cache verified.

**Head-selection gate — seed 0, N_train=1800, eval=200, matched default config
(meet-on-generate + seam-repair + bhffa_g + DIRECTED + HINDSIGHT). Blind
baseline identical across rows (194/200, median 597), confirming same eval set:**

| model | head | solved | median exp | mean exp | len_mean | ms/solve |
|---|---|---:|---:|---:|---:|---:|
| **smallcnn** (cross-encoder) | — | **200/200** | **118** | **452.7** | 33.71 | **257** |
| embed | quasi | 195/200 | 511 | 1171.7 | 33.39 | 492 |
| embed | ℓ1    | 193/200 | 567 | 1220.8 | 33.17 | 592 |
| embed | mlp   | 197/200 | 386 |  807.7 | 33.99 | 385 |

**Verdict: REJECTED.** Best head (`mlp`) is 3.3× worse on median expansions and
1.5× slower; ℓ1/quasi worse still. Adoption bar (expansions non-inferior AND
wall drops, OR metric head beats mean 3/3) missed on both clauses, so no 3-seed
confirmation (same-seed spread ~118↔155 median cannot close a 118↔386 gap).
Mechanism: head ordering `quasi < ℓ1 < mlp` tracks expressiveness → **bi-encoder
representational ceiling**; `smallcnn` is a cross-encoder (convolves both boards
jointly), worth ~3× for pairwise Sokoban distance. The O(k) speed premise never
materialized (all heads slower — expansion blowup dominates wall time). The
deferred soft-min full-F2F arm (Stage D) was **abandoned** with it: full-F2F
changes which anchor pair is scored, not the representation quality. Code kept
off by default (`MODEL=smallcnn`); do not re-attempt on a bi-encoder substrate.

## REVWALK: reverse-walk long-range pair generation — REJECTED (2026-07-09)

Hypothesis: every finite buffer label is short-range (bounded by solved-path /
explored-subgraph distances on instances the solver already cracks), while the
open list is ordered by h at long range early in each search — so self-generated
long-range labels should improve early ordering. Mechanism: random walks in the
backward game from its root (any backward trajectory reversed is a valid
forward path, so cumulative reversed edge costs — via the `path_edge_costs`
hook — are sound upper-bound labels); first-visit dedup; linear length
curriculum 8→64; ~64 pairs/puzzle; runs on failed solves too. Soundness was
machine-verified (all reversed walk edges valid forward moves; exact-BFS check
label ≥ true distance on all sampled pairs). Implementation: `REVWALK*` knobs +
`collect_reverse_walk_pairs` in `learning/online_run.py`; driver
`revwalk_run.py` (saves checkpoints to `MODEL_OUT` — on-policy training is not
run-to-run reproducible, so downstream evals must load the gated model).

**Seed-0 gate, matched arms, standard held-out 200 AND the hard200 set:**

| arm | standard: solved/median/mean | hard200 (cap 50k): solved/median/mean |
|---|---|---|
| blind | 194 / 597 / 1313 | 200 / 12469.5 / 13932.8 |
| **base** | **200 / 146 / 465.2** | **200 / 2887 / 3730.0** |
| revwalk | 200 / 189.5 / 599.4 | 200 / 3695 / 4658.1 |

**Verdict: REJECTED.** Worse on BOTH eval sets (+30%/+29% median/mean standard,
+28%/+25% hard). The long-range hypothesis specifically predicted the hard set
(plans 40–90) would favor it; it did not — a same-direction 25–30% loss on two
disjoint sets needs no 3-seed confirmation. Root cause (post-hoc): raw
walk-length labels are LOOSE upper bounds — a 64-pull random walk wanders and
typically ends far fewer true moves from the root than its cost, so long-range
labels carry large state-dependent inflation. HINDSIGHT's upper bounds are
subgraph-SHORTEST distances (tight); walk labels are single-trajectory costs.
DeepCubeA-style generation works via bootstrapped Bellman targets, not raw walk
lengths — a future bootstrapping arm (TD targets over generated states /
recorded graphs) is the principled fix and remains open. Flag default is
`REVWALK=no`; the reference configuration is unchanged.

## Hard held-out benchmark (hard200) — ADOPTED as secondary eval (2026-07-09)

The standard held-out 200 is near saturation (200/200 solved, median ~118–146,
same-seed noise band ~118↔155), hiding modest real effects.
`analysis/build_hard_eval.py` ranks the untouched solvable-pool tail (indices
2000–9999, disjoint from train/eval) by a domain-agnostic difficulty measure —
deterministic blind-search expansions — and keeps the hardest 200
(`data/solvable10_3box_hard200.txt`, loader `get_hard_eval_data()`; blind
expansions 8.7k–41k, median 12.5k; blind plans 40–90, median 60). Evaluate
saved checkpoints with `rank_forward/experiments/hard_eval.py`
(`EVAL_MAX_ITERS=50000` so blind solve rates stay meaningful).

**Generalization result (seed 0): the reference model, trained only on the
easy pool, solves 200/200 hard instances at 4.3× fewer expansions than blind**
(2887 vs 12469.5 median) — the learned stack extrapolates well beyond its
training difficulty. Hard-set reference (seed 0, cap 50k): blind
200/12469.5/13932.8; base 200/2887/3730.

### Forward (Chrestien) baseline on hard200 — head-to-head (2026-07-09)

`forward_hard.py`: canonical forward baselines (L\*, L_gbfs; SmallCNN; optimal
labels from the full-goal cache; same 1800-board train prefix; seed 0),
evaluated under the fair FULL GOAL on both eval sets. Standard-set rows
replicate the stored headline (forward L\*/A\* 196/200, median 114, mean 740
vs base 200/200 / 146 / 465.2). Hard200 (cap 50k; medians/means over solved;
"budget-inclusive" counts failures at their spent 50k):

| method | solved | median | mean (solved) | budget-incl. mean |
|---|---:|---:|---:|---:|
| fwd blind A\* | 5/200 | (37686) | (35085) | ~49.6k |
| fwd L\*/A\* | 193/200 | **2100** | 5435.5 | ~6995 |
| fwd L_gbfs/GBFS | 183/200 | 4573 | 9368.0 | ~12822 |
| bidir blind | 200/200 | 12469.5 | 13932.8 | 13932.8 |
| **base (bidir)** | **200/200** | 2887 | **3730.0** | **3730.0** |

**Reading:** the standard-set pattern amplifies on hard instances. Forward
L\*/A\* is ~27% faster on the median instance; the bidirectional reference is
decisively more robust — perfect solve rate (forward fails 7/200 at cap) and
~1.9× lower budget-inclusive mean. Bidirectionality itself carries most of the
blind-search win (bidir blind 200/200 vs forward blind 5/200 under the full
goal). Forward checkpoints saved via `MODEL_OUT` (default `/tmp/fwd_models`).

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

## Addendum — local session results (2026-07-04 → 07, merged 2026-07-07)

Run before the supervisor snapshot was merged in; controls are the
PATH_OFF_RANK `off` arm (3 seeds), so numbers are comparable within each
block, not to the 5-seed reference above. Full details in each results dir.

- **PATH_OFF_RANK** (path-vs-off-path A*-ranking margin, Chrestien Def. 1
  operationalized) — **wash**, not adopted (`path_off_rank_ablation_results/`).
- **OFF_PATH_PER_PUZZLE=24** (volume hypothesis) — **rejected**, mean worse 3/3
  (`followup_ablations/`).
- **PER** (prioritized replay) — **rejected**, worse 3/3 on both metrics
  (`per_ablation_results/`).
- **HINDSIGHT dose sweep** (`hindsight_tuning_results/`): at
  `PER_PUZZLE=64, LABEL_CAP=256` median better **3/3** (avg −8.8%) and mean
  −5.5% vs control — stronger than the adopted 32/128 dose on those seeds;
  candidate for a 5-seed validation before flipping the dose defaults.
- **Wave-3g independent replication** (`wave3g_quasi_results/`,
  `wave3g_quasi_hindsight_results/`, `wave3g_iqe_results/`): a parallel
  factorized implementation (`QuasiCNN` asym-L1 head; `IQECNN` interval-union
  IQE head, axioms unit-tested) lands at **blind level 3/3 seeds** for both
  heads, and HINDSIGHT on top does not move it — independently confirming the
  bi-encoder ceiling documented in the Wave 3 section above with two more head
  types. The 1-D potential-collapse diagnosis explains the asym-L1 arm but NOT
  the IQE arm (collapse-resistant by construction, still blind) — consistent
  with the ceiling being the pooled single-board embedding, not the head.
