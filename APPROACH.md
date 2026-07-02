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
- **a. Seam repair + meet-on-generate.** Detect the meeting at *generation*
  (earliest, ~11% fewer expansions) but return the **BFS-shortest start→goal
  path in the recorded union graph** instead of the parent-pointer splice.
  The returned plan is then provably the best plan in the explored subgraph —
  the seam-quality objection that forced the revert disappears, since *any*
  detection rule now yields the same (repaired) path quality; detection speed
  and path quality are decoupled. This also *tightens the labels* Wave 1b
  needs. (Resolves the open "principled meeting" idea from 2026-06-22: repair
  subsumes the middle-ground seam rule — earliness becomes safe.)
- **b. Exact relabeling of on-path pairs.** Keep the pair *distribution*
  bit-identical (history: distribution, not label optimality, is what made
  bootstrap win) but label each mined pair with the exact directed
  union-graph distance instead of `|i−j|`. Strictly tighter targets, zero
  distribution shift.
- **c. BHFFA-correct scoring.** Add `+ g_b(d*)` to `f`. One line, zero cost,
  retires a real theoretical objection; possibly a wash (lazy re-eval may
  absorb it) — either outcome is knowledge.

**Wave 2 — give `h` local distance structure (small, additive to the loss,
following the proven pattern "add structure on top of the calibrated loss"):**
- **d. Local metric grounding.** `h(x,x)=0` anchor pairs plus a one-step
  consistency hinge `relu(h(s_i,t) − 1 − h(s_{i+1},t))` along verified path
  edges — Bellman self-consistency as a one-sided constraint; together they
  telescope to path-admissibility. Targets the tail (mean), like PATH_RANK.
- **e. Hindsight query supervision.** Reservoir-sample the search's *actual*
  `(node, anchor)` h-queries; after the solve, label the sampled pairs with
  exact union-graph distances and add them to the buffer — training `h`
  exactly on the query distribution it is evaluated on (pairwise HER).
- *(f. conditional: pinball/quantile loss for upper-bound labels — only if
  Stage 0 shows residual slack that 1b cannot remove.)*

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

## 4. Why this is the elegant path

Every item either (i) replaces a biased surrogate with the exact quantity the
search semantically needs (1a, 1b, 2e), (ii) restores a term or constraint the
theory says should be there (1c, 2d, 3g), or (iii) converts a previously
infeasible principled computation into a feasible one via factorization (3g).
Nothing exploits Sokoban structure; every mechanism consumes only states,
transitions, recorded search graphs, and `h`.
