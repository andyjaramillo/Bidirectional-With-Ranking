# The bidirectional approach: map & principled roadmap

*Status: written 2026-07-02, after adopting PATH_RANK as default. Sources: code
audit + experiment history (`rank_forward/experiments/README.md`) + a
multi-agent design/judging pass. The research constraint governing everything
here: methods must be domain-agnostic — Sokoban is only the testbed.*

## 1. What the method is (the map)

**Search** (`search/AI_Bidirectional.py`): Top-to-Top Bidirectional Search.
Two frontiers — forward from the start, backward from the goal seed (the
flipped puzzle; boxes are "pulled"). Each side scores nodes
`f(n) = g(n) + max(0, h(n, d*))` where `h` is a learned **pairwise** heuristic
and `d*` is the opponent's **temporal anchor** (its most recently expanded
node, flipped into the active frame). Anchor moves ⇒ stale f-scores are fixed
by TTBS **lazy re-evaluation** at pop time. Balancing: expand the side with
the smaller open list. **Meeting**: a freshly generated successor matches the
opposite side's CLOSED set under the full-state key (agent + boxes); the
search returns at the **first** meeting (satisficing, no quality bound), path
from parent pointers spliced at the seam. The full induced transition graph
(`edges_f/edges_b`) is recorded as a byproduct.

**Learning** (`learning/online_run.py`): fully on-policy loop — solve puzzle
*n* with the current net, mine training pairs, take K=8 SGD steps, occasionally
re-mine an old puzzle with the improved net. Data: on-path pairs
`(p_i, target, p_j)` labeled `|i−j|` (walk length along the satisficing path,
added symmetrically), plus ~8 off-path states per puzzle labeled with **exact**
BFS distances over the union graph of both frontiers, in a 20k FIFO buffer.
Loss: `0.5·MSE + 0.5·margin-ranking (cross-buffer pairs) + 0.5·PATH_RANK`
(within-path pairs-of-pairs margin — the within-instance ordering the open
list actually uses). Model: SmallCNN, 23.6k params, consumes (state, target,
other) as stacked one-hot planes; `h(x,y)` is a monolithic pairwise forward
pass with **no** structural constraints.

**Domain-specific parts** (flagged; keep contained): `hasDeadlock` pruning
(forward frontier only — an asymmetry), the one-hot board encoding, the
forward/backward frame flip. Everything else consumes only states,
transitions, and `h`.

**Established results** (3-seed methodology, held-out 200, expansions
`len(closed_f)+len(closed_b)`, blind = 194/545/1380):
- Reference config (MSE+margin+PATH_RANK, temporal anchor): ~197 / 196 / 608.
- **Washes**: anchor selection (4 strategies), brute-force full front-to-front
  (too slow), meeting-on-generate default (saved ~11% expansions but seam
  quality objection — see §2.1).
- **Dead ends**: pure ranking losses (perfect-ranking, path-order) — any
  ranking-only objective without scale calibration collapses below blind.
- **Levers that worked**: on-policy bootstrap (beats optimal off-policy
  labels), PATH_RANK (mean −11%, 3/3 seeds).

## 2. Principled gaps (why there is headroom)

1. **`h` has no distance structure.** Nothing enforces `h(x,x)=0`, the
   triangle inequality, one-step consistency `h(x,t) ≤ 1 + h(x′,t)`, or a
   coherent treatment of asymmetry — the true state-space distance of a
   planning problem with irreversible moves is a *quasimetric*, yet labels are
   symmetrized by fiat (`(x,y)` and `(y,x)` get the same label).
2. **Labels are walk lengths, not distances.** `|i−j|` along a satisficing
   path is an upper bound whose slack is the path's local suboptimality — even
   though the search already records the very graph (`edges_f/edges_b`) on
   which exact distances are computable.
3. **Train/test query mismatch.** The search orders its open list by
   `h(frontier-node, moving-anchor)` queries; the buffer contains almost no
   pairs of that type (path×path + 8 state→goal per puzzle) — a textbook
   covariate shift.
4. **The score drops a term.** Classical front-to-front scoring (BHFFA) is
   `f(n) = g_f(n) + h(n,d) + g_b(d)`; we omit `g_b(d*)`, so heap entries
   scored against anchors of different depths sit on incommensurable scales.
5. **No quality certificate.** First-meeting termination gives no bound; the
   balancing rule ("smaller open list") is folklore.
6. **Computational shape blocks principle.** A monolithic pairwise `h` makes
   full front-to-front `O(|frontier|)` forwards per node — which is *why* the
   principled objective (min over the frontier) had to be abandoned for a
   point-anchor approximation.

## 3. Roadmap (waves; each item domain-agnostic and 3-seed testable)

**Keystone refactor** (enables most of Wave 1–2): factor
`off_path_distance_to_goal` into a reusable `build_union_graph(s)` — forward +
flipped-backward adjacency over full-state keys. Four proposals consume it.

**Stage 0 — diagnostics (~1 h compute, run before committing to anything):**
- *Slack histogram*: on solved puzzles, `|i−j|` vs exact union-graph distance
  for mined pairs → quantifies label noise that Wave 1b would remove.
- *Asymmetry rate*: `d(x,y)` vs `d(y,x)` on the union graph → is the
  quasimetric issue material, gating Wave 3's asymmetric head.
- *Query finite-fraction*: how many logged `(node, anchor)` search queries
  have finite explored-subgraph distance → gates Wave 2e.

**Wave 1 — bank the certain wins (all small):**
- **a. Seam repair + meet-on-generate. ✅ DONE, ADOPTED (2026-07-02).**
  Detect the meeting at *generation* (earliest) but return the **BFS-shortest
  start→goal path in the recorded union graph** instead of the parent-pointer
  splice — provably the best plan in the explored subgraph; detection speed
  and path quality decoupled; tightens the labels Wave 1b needs. (Resolved
  the open "principled meeting" idea from 2026-06-22.)
  *3-seed result: held-out avg 199.0/184.5/591 vs legacy reference
  197.3/196/608 (+1.7 solved incl. a first-ever 200/200; −6% median), with
  repaired plans SHORTER than legacy's on all seeds. Defaults flipped.*
- **b. Exact relabeling of on-path pairs. ✅ CLOSED AS SUBSUMED BY 1a
  (2026-07-02).** A subpath of a shortest path is a shortest path, so the
  repaired path's `|i−j|` labels already ARE the exact union-graph distances.
  Stage-0 diagnostic (118 puzzles): 62,266 on-path pairs, **zero slack**
  (theorem confirmed end-to-end); legacy paths had slack on 3.1% of pairs
  (mean 2.3 when present) — the noise 1a removed. *The residual label problem
  is DIRECTIONAL*: reverse pairs (half of training data, labeled `j−i` by
  symmetric duplication) have a finite explored-subgraph reverse distance only
  15.3% of the time, and 35.4% of those are mislabeled (mean err +2.9, p90 8)
  — the quasimetric asymmetry is real, gating OPEN the direction-correct-labels
  proposal (see Wave 3 prerequisite).
- **c. BHFFA-correct scoring. ✅ DONE, ADOPTED (2026-07-03).**
  Add `+ g_b(d*)` to `f` (bhffa_g flag). *3-seed result: learned median better
  on 3/3 seeds (avg 184.5 → 173.0, −6.2%); within-model the eval-time term
  improves mean AND solve rate on 3/3 seeds (fixes stale-entry mis-ordering on
  the hard tail). Blind h gets worse under the term — only the on-policy net
  can exploit the corrected score. Defaults flipped. Current reference:
  avg 199.3 / 173.0 / 584. **Wave 1 complete.***

**Wave 2 — give `h` local distance structure (small, additive to the loss,
following the proven pattern "add structure on top of the calibrated loss"):**
- **d. Local metric grounding. ✅ TESTED — WASH, not adopted (2026-07-03).**
  `h(x,x)=0` anchor pairs plus one-step consistency hinges (implemented
  cost-generally via `path_edge_costs`). *3-seed result: avg 198.3/173.7/493
  vs 198.7/155.3/513 — mean −4% (2/3) but median mixed with a big seed-1
  regression; inside noise. Root cause confirmed live: near the regression
  optimum with exact labels, the constraints are already satisfied (~0
  violations on fitted paths) — the hinge adds mostly noise. Positive
  structural corollary: the pipeline's h is already locally consistent where
  well-fit. Flags kept, default off.*
- **e. Hindsight query supervision. ✅ DONE, ADOPTED (2026-07-03).**
  Reservoir-sample the search's *actual* `(node, anchor)` h-queries; after the
  solve, label the canonical DIRECTED `(source→dest)` pairs with exact
  union-graph distances and add them to the buffer — pairwise HER, training `h`
  on the query distribution it is evaluated on. *5-seed result: mean −8.6%
  (better 4/5, the tail metric it targets), median a wash (−2.9%), solve tied.
  Adopted as a lead call — cleared the mean/median-majority/solve criteria but
  narrowly missed the strict "no >10% regression on any seed" clause (seed-3
  median +13%, within the method's historical noise). Caveats: upper-bound
  labels, buffer dilution, ~21% query finite-fraction. New reference: avg
  199.2 / 148.5 / 453.5.*
- *(f. conditional: pinball/quantile loss for upper-bound labels — only if
  Stage 0 shows residual slack that 1b cannot remove.)*

**Interlude — direction-correct labels & queries. ✅ DONE, ADOPTED
(2026-07-03).** Pulled forward from the Wave-3 prerequisite after the 1b
diagnostic opened its gate. Labels: direction-correct pairs only; queries: the
backward frontier asks `h(anchor_fwd, target_fwd, flip(node))` — the
forward-dynamics distance it actually needs, all forward-frame. *3-seed
result: median −10.2% and mean −12.2% vs the Wave-1 reference, better on 3/3
seeds each — the largest single-change win of the roadmap; legacy-query
decomposition arms (~220–265 median, same nets) prove the backward frontier
had been served by an out-of-distribution, direction-confused heuristic.
Defaults flipped. Current reference: avg 198.7 / 155.3 / 513. The quasimetric
premise of Wave 3g is now empirically validated twice over.*

**Wave 3 — the architectural bet (medium/large):**
- **g. Factorized (quasi)metric embedding.** `h(x,y) = d(φ(x), φ(y))` with a
  metric or quasimetric head (e.g. asymmetric/interval embeddings if the
  Stage-0 asymmetry rate is material). Buys, by construction: `h(x,x)=0`,
  triangle inequality, and **O(1) per-anchor rescoring** — one embedding per
  node, cheap distances — which finally makes *principled full
  front-to-front* (min or soft-min over the whole opposing frontier)
  computationally feasible. Ablation lattice: {ℓ1 / quasimetric / unconstrained
  MLP head} × {point-anchor / full-frontier}. Run only after Waves 1–2, and
  only with the direction-correctness prerequisite from Stage 0.

**Explicitly not pursuing** (dead ends / unprincipled): more anchor-selection
variants; brute-force full-F2F with the monolithic net; ranking-only losses;
capacity-only architecture changes ("more attention" without a structural
argument); anything Sokoban-specific.

## 4. Cost-generality (design principle, 2026-07-03)

We intend to move beyond unit-cost problems, so **new methods must not bake in
unit costs**: loss terms take edge costs from a hook (`path_edge_costs`,
default 1.0 here); the consistency hinge is `relu(h(s_i,s_j) − c(s_i,s_{i+1})
− h(s_{i+1},s_j))`, not `−1−`; the telescoping-admissibility argument holds
verbatim with `Σc` in place of `j−i`. Inventory of EXISTING unit-cost
touchpoints to port when a weighted domain arrives (all mechanically
Dijkstra-/cost-generalizable, none conceptually unit-bound):
- `_expand`'s `new_g = g + 1` (→ `g + c(u,v)`);
- `refine_path` and `off_path_distance_to_goal` BFS (→ Dijkstra; the
  subpath-of-shortest-path theorem behind Wave 1b's closure holds for any
  nonnegative costs);
- path labels `|i−j|` (→ cumulative path cost differences), and PATH_RANK's
  index-based ordering (→ order by cumulative cost);
- fixed margins `RANK_MARGIN=1.0` etc. are *scale hyperparameters*, to be
  re-tuned (or cost-scaled) on weighted domains rather than assumed.

## 5. Why this is the elegant path

Every item either (i) replaces a biased surrogate with the exact quantity the
search semantically needs (1a, 1b, 2e), (ii) restores a term or constraint the
theory says should be there (1c, 2d, 3g), or (iii) converts a previously
infeasible principled computation into a feasible one via factorization (3g).
Nothing exploits Sokoban structure; every mechanism consumes only states,
transitions, recorded search graphs, and `h`.
