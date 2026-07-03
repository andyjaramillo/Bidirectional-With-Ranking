"""Online learning curve for a learned TTBS heuristic.

For each puzzle n = 1, 2, ..., N (in order):
  1. Solve puzzle n with the current NN as the front-to-front heuristic.
     This is a held-out evaluation point — the model has never seen it.
  2. Mine (state_i, state_j, |i-j|) training pairs from the solved path
     and add them to the replay buffer (tagged with puzzle_id n).
  3. Run K minibatch gradient updates on the buffer (MSE + margin-ranking).
  4. Once the buffer is saturated, periodically re-mine an old puzzle with
     the now-better model to refresh stale labels.

Training can run on CPU (default) or MPS; search-time inference always runs
on a CPU twin (batch-1 inference is GPU-hostile). Compared against a no-NN
baseline pass over the same puzzles.

Key env knobs (all optional):
  N_TOTAL, MODEL_CHANNELS, BATCH_SIZE, UPDATES_PER_SOLVE, WARMUP, BUFFER_CAP,
  LOSS={mse,rank,both}, REG_LOSS={mae,mse}, RANK_MARGIN, USE_G={yes,no},
  TRAIN_DEVICE={cpu,mps}, K_REMINE, REMINE_RANDOM_FRAC,
  COLLECT_OFF_PATH={yes,no}, OFF_PATH_PER_PUZZLE
"""
import os
import random
import time
from collections import deque

import numpy as np
import torch

from game.getData import get_solvable_data
from learning.nn import build_model
from search.AI_Bidirectional import BidirectionalF2FSearch
from learning.replay_buffer import ReplayBuffer

SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
np.random.seed(SEED)
_rng = random.Random(SEED)

# ── Configuration ────────────────────────────────────────────────────────
N_TOTAL = int(os.environ.get("N_TOTAL", "1000"))
# Per-puzzle node-expansion budget. A puzzle that exceeds it yields no path
# and therefore contributes NO training signal, so the cap silently trims the
# hard tail of the dataset out of the learner's diet. Set it high to keep that
# tail in.
MAX_ITERS = int(os.environ.get("MAX_ITERS", "10000"))
MODEL_CHANNELS = int(os.environ.get("MODEL_CHANNELS", "32"))
# Heuristic architecture: "smallcnn" (conv tower + global avg-pool) or
# "smallcnn_attn" (adds one positional self-attention block between the
# conv tower and the MLP head — global receptive field in a single layer).
MODEL = os.environ.get("MODEL", "smallcnn").lower()
# Factorized (quasi)metric embedding model (APPROACH.md Wave 3): MODEL=embed.
# HEAD picks the combiner: 'quasi' (asymmetric quasimetric, triangle + h(x,x)=0
# by construction — the principled target), 'l1' (symmetric pseudometric),
# 'mlp' (unconstrained; isolates the pure-factorization effect). EMBED_K is the
# embedding dim. Intended to run with DIRECTED (default): h(x,y)=dist(phi(x),
# phi(y)) estimates the forward-dynamics distance the buffer labels match.
HEAD = os.environ.get("HEAD", "quasi").lower()
EMBED_K = int(os.environ.get("EMBED_K", "64"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
UPDATES_PER_SOLVE = int(os.environ.get("UPDATES_PER_SOLVE", "8"))
WARMUP = int(os.environ.get("WARMUP", "200"))

# Harvest off-path closed states as extra (state, goal, dist-to-goal) samples,
# labeled by a weight-1 graph search from the goal over the union of forward +
# flipped-backward transitions (valid because full-state meeting shares nodes).
COLLECT_OFF_PATH = os.environ.get("COLLECT_OFF_PATH", "yes").lower() == "yes"
OFF_PATH_PER_PUZZLE = int(os.environ.get("OFF_PATH_PER_PUZZLE", "8"))
BUFFER_CAP = int(os.environ.get("BUFFER_CAP", "20000"))
WINDOW = 25
LOG_EVERY = 25
MILESTONE = 100

# Loss: "mse" (regression only), "rank" (margin-ranking only), or "both".
LOSS = os.environ.get("LOSS", "both").lower()
# Regression-term flavor for "mse"/"both": "mae" (L1) or "mse" (L2).
REG_LOSS = os.environ.get("REG_LOSS", "mse").lower()
RANK_MARGIN = float(os.environ.get("RANK_MARGIN", "1.0"))
# Within-path pairs-of-pairs margin (DEFAULT since 3-seed validation): for two
# pairs of nodes on the SAME fresh solution path, (x,y)=(p_i,p_j) and
# (x',y')=(p_i',p_j'), enforce the margin-ranking constraint
#     h(x,y) + RANK_MARGIN <= h(x',y')   whenever |i-j| < |i'-j'|.
# The buffer margin term above already ranks pairs-of-pairs, but its random
# batch pairs are almost always CROSS-puzzle; the search's open list only ever
# compares nodes of the SAME instance, so within-instance ordering is the
# operationally relevant constraint. Added ON TOP of the (untouched) MSE+margin
# loss, which keeps h's scale pinned. Per update, PATH_RANK_PAIRS on-path pairs
# are sampled and ALL comparable combinations among them are penalised (dense:
# one h-batch of size PATH_RANK_PAIRS yields ~PAIRS^2/2 comparisons).
# 3-seed held-out result vs plain MSE+margin: mean expansions -11% (better on
# 3/3 seeds), median -7% (2/3), solve rate unchanged. PATH_RANK=no disables
# (recovers the pre-adoption reference configuration).
PATH_RANK = os.environ.get("PATH_RANK", "yes").lower() == "yes"
PATH_RANK_PAIRS = int(os.environ.get("PATH_RANK_PAIRS", "32"))
PATH_RANK_W = float(os.environ.get("PATH_RANK_W", "0.5"))
# If "no", drop g from the f-score (GBFS instead of A*-style).
USE_G = os.environ.get("USE_G", "yes").lower() == "yes"
# Detect the frontier meeting as soon as both sides have GENERATED the shared
# state (DEFAULT since Wave 1a validation) instead of only when one side has
# CLOSED it — earliest detection, −15% median blind expansions. "no" restores
# legacy meet-on-closed.
MEET_ON_GENERATE = os.environ.get("MEET_ON_GENERATE", "yes").lower() == "yes"
# Post-hoc seam/path repair (APPROACH.md Wave 1a; DEFAULT on): return the
# BFS-shortest start->goal path over the recorded union graph instead of the
# parent-pointer splice — never longer, optimal within the explored subgraph.
# Decouples meeting-detection earliness from path quality (the principled
# companion of MEET_ON_GENERATE) and tightens |i-j| path labels for training.
# 3-seed validated: held-out avg 199.0/184.5/591 vs legacy 197.3/196/608, with
# SHORTER plans than legacy on all seeds.
SEAM_REPAIR = os.environ.get("SEAM_REPAIR", "yes").lower() == "yes"
# BHFFA-complete scoring (APPROACH.md Wave 1c): f(n) = g(n) + h(n,d*) + g_opp(d*)
# — restores the classical front-to-front term our TTBS score drops, putting
# heap entries scored against anchors of different depths on one scale.
# DEFAULT on since 3-seed validation (learned median better 3/3 seeds, avg
# 184.5 -> 173.0; eval-time term improves mean + solve rate 3/3 within-model).
BHFFA_G = os.environ.get("BHFFA_G", "yes").lower() == "yes"
# Direction-correct labels & queries (quasimetric fix, APPROACH.md).
# State-space distance under irreversible moves is asymmetric; symmetric=True
# pair mining labels (s_j, s_i) with the FORWARD distance j-i — measured label
# corruption: reverse routes exist in the explored subgraph for only ~15% of
# pairs and ~35% of those exceed j-i (mean +2.9). DIRECTED=yes: (1) mine only
# direction-correct on-path pairs (symmetric=False) with num_random_pairs
# doubled 2L->4L to compensate; measured per-puzzle throughput is ~2/3 of the
# legacy symmetric setting (set-dedup + the ordered-pair space being half) —
# a known, documented secondary confound; (2) drop the mirrored (goal, state)
# off-path insertion (that direction's label is unknown); (3) the backward
# frontier queries h in the direction it actually needs (s.dir_correct; see
# search class docs). DEFAULT on since 3-seed validation: median -10.2%,
# mean -12.2% vs the Wave-1 reference, better on 3/3 seeds each.
DIRECTED = os.environ.get("DIRECTED", "yes").lower() == "yes"
# Local metric grounding (APPROACH.md Wave 2d): give h the two local axioms of
# a true cost-to-go that pointwise regression never enforces.
#   CONSIST: one-step consistency hinges along VERIFIED path edges —
#     h(s_i,s_j) <= c(s_i,s_{i+1}) + h(s_{i+1},s_j)   (step the source)
#     h(s_i,s_j) <= h(s_i,s_{j-1}) + c(s_{j-1},s_j)   (step the destination)
#   both are theorems of the true quasimetric (triangle inequality through a
#   verified edge; Pearl's consistency / one-sided Bellman on observed edges).
#   N_ZERO_PAIRS: (x, x, 0.0) anchor pairs through the existing MSE term.
#   Together they telescope to cost-admissibility along every observed path:
#   h(s_i,s_j) <= sum of edge costs. COST-GENERAL by construction: the hinge
#   reads edge costs from path_edge_costs() (1.0 on this unit-cost testbed),
#   never a hardwired "+1". Additive on top of the untouched loss (the proven
#   pattern). Default off pending 3-seed validation.
CONSIST = os.environ.get("CONSIST", "no").lower() == "yes"
CONSIST_W = float(os.environ.get("CONSIST_W", "0.5"))
CONSIST_TRIPLES = int(os.environ.get("CONSIST_TRIPLES", "16"))
N_ZERO_PAIRS = int(os.environ.get("N_ZERO_PAIRS", "0"))
# Hindsight supervision of the search's own h-queries (APPROACH.md Wave 2e).
# The open list is ordered by h(frontier-node, moving-anchor) queries, yet the
# buffer contains no pairs of that type (path x path + a few state->goal) — a
# covariate shift measured at 1.91x |h - d| error on query pairs vs training
# pairs (stage-0, 2026-07-03; finite fraction 20.9%). HINDSIGHT=yes: reservoir-
# sample the queries each solve actually issues (s.log_queries), then label
# the canonical (source -> dest) forward-frame pairs with exact DIRECTED
# distances over the recorded union graph and add the finite ones to the
# buffer — the pairwise analogue of hindsight experience replay: train on the
# test distribution with ground-truth-in-hindsight labels. DEFAULT on since
# 5-seed validation: mean -8.6% (better 4/5 seeds), the tail metric it targets;
# median a wash (-2.9%, better 3/5); solve rate tied. HINDSIGHT=no recovers the
# pre-2e reference (198.7/155.3/513). Caveats on record: labels are
# explored-subgraph upper bounds; ~32 samples/puzzle dilute the fixed FIFO
# buffer; only ~21% of logged queries are label-able. Adopted as a lead call --
# it narrowly missed the pre-registered "no >10% regression on any seed" bar
# (seed-3 median +13%, inside the method's historical median spread 138-172).
HINDSIGHT = os.environ.get("HINDSIGHT", "yes").lower() == "yes"
HINDSIGHT_PER_PUZZLE = int(os.environ.get("HINDSIGHT_PER_PUZZLE", "32"))
HINDSIGHT_LABEL_CAP = int(os.environ.get("HINDSIGHT_LABEL_CAP", "128"))
QUERY_LOG_K = int(os.environ.get("QUERY_LOG_K", "256"))
# Full front-to-front: score nodes against the WHOLE opponent open frontier
# (min over all live opponent open nodes) instead of the single temporal anchor.
# In principle slow; applies to BOTH the training solves and the CPU twin so the
# NN learns under the same search the eval uses.
FULL_F2F = os.environ.get("FULL_F2F", "no").lower() == "yes"
# Anchor-selection strategy: "temporal" (our TTBS, default), "top_of_open"
# (paper-faithful TTBS d-node), or "closest_anchor" (policy A). Applies to BOTH
# the on-policy training solves and the CPU twin so the NN trains under the same
# search it is evaluated in.
ANCHOR_STRATEGY = os.environ.get("ANCHOR_STRATEGY", "temporal").lower()
# Training device. Search always runs on a CPU twin regardless.
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "cpu").lower()
# Periodic re-mining: re-solve & re-mine 1 old puzzle every K_REMINE solves
# once the buffer is saturated. 0 disables.
K_REMINE = int(os.environ.get("K_REMINE", "5"))
REMINE_RANDOM_FRAC = float(os.environ.get("REMINE_RANDOM_FRAC", "0.1"))


# ── Search + training helpers ─────────────────────────────────────────────
def run_search(puzzle, nn_model=None):
    s = BidirectionalF2FSearch(puzzle, nn_model)
    s.use_g_in_f = USE_G
    s.meet_on_generate = MEET_ON_GENERATE
    s.seam_repair = SEAM_REPAIR
    s.bhffa_g = BHFFA_G
    s.dir_correct = DIRECTED
    s.log_queries = HINDSIGHT
    s.query_log_k = QUERY_LOG_K
    s.full_f2f = FULL_F2F
    s.anchor_strategy = ANCHOR_STRATEGY
    t0 = time.time()
    path = s.search(max_iterations=MAX_ITERS)
    return path, s, time.time() - t0


def union_forward_adj(s):
    """FORWARD-direction adjacency of the explored union graph over full-state
    keys (edges_f as-is; edges_b flipped: backward u->v is forward
    flip(v)->flip(u)). Mirror of the reverse adjacency built by
    off_path_distance_to_goal. Unit-cost BFS consumers are the uniform-cost
    fast path; a weighted domain would attach costs here and use Dijkstra
    (cost-generality principle, CLAUDE.md)."""
    fg, bg = s.forward_game, s.backward_game
    adj, kf, kb = {}, {}, {}

    def key_f(h):
        if h not in kf:
            kf[h] = s._full_key(fg.decodeMap(h))
        return kf[h]

    def key_b(h):
        if h not in kb:
            kb[h] = s._full_key(fg.flipGame(bg.decodeMap(h)))
        return kb[h]

    for u, vs in s.edges_f.items():
        ku = key_f(u)
        dst = adj.setdefault(ku, set())
        for v in vs:
            dst.add(key_f(v))
    for ub, vsb in s.edges_b.items():
        ku = key_b(ub)
        for vb in vsb:
            adj.setdefault(key_b(vb), set()).add(ku)
    return adj


def _bfs_forward(adj, src):
    """Unit-cost distances from src over a forward adjacency dict."""
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        d = dist[u]
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = d + 1
                dq.append(v)
    return dist


def collect_query_hindsight(s, buffer, rng, puzzle_id):
    """Wave 2e: label a sample of the solve's actual h-queries with exact
    DIRECTED distances over the explored union graph and add the finite ones
    to the buffer. Query entries are (node_board, anchor_hash, is_backward);
    the canonical forward-frame (source -> dest) pair follows the dir_correct
    convention (forward side: node -> anchor; backward side: anchor -> node).
    Returns the number of samples added."""
    if not s.query_log:
        return 0
    fg, bg = s.forward_game, s.backward_game
    root_b = next((v for v, p in s.parent_b.items() if p is None), None)
    goal_board = fg.flipGame(bg.decodeMap(root_b)) if root_b is not None else None

    canon = []
    for nm, ah, is_b in s.query_log:
        if is_b:
            src_b = s.puzzle if ah is None else fg.decodeMap(ah)
            canon.append((src_b, fg.flipGame(nm)))
        else:
            if ah is None and goal_board is None:
                continue
            dst_b = goal_board if ah is None else fg.flipGame(bg.decodeMap(ah))
            canon.append((nm, dst_b))
    if len(canon) > HINDSIGHT_LABEL_CAP:
        canon = rng.sample(canon, HINDSIGHT_LABEL_CAP)

    adj = union_forward_adj(s)
    by_src = {}
    for a, b in canon:
        by_src.setdefault(s._full_key(a), []).append((a, b))
    finite = []
    for src_key, items in by_src.items():
        dmap = _bfs_forward(adj, src_key)
        for a, b in items:
            dv = dmap.get(s._full_key(b))
            if dv is not None:
                finite.append((a, b, float(dv)))
    if len(finite) > HINDSIGHT_PER_PUZZLE:
        finite = rng.sample(finite, HINDSIGHT_PER_PUZZLE)
    for a, b, dv in finite:
        buffer.add(a, fg.target, b, dv, puzzle_id)
    return len(finite)


def path_edge_costs(path_boards):
    """Edge costs along a mined path — the SINGLE hook new loss terms read
    costs from (cost-generality principle, CLAUDE.md). This testbed is
    unit-cost; on a weighted domain, replace with the real cost function.
    Returns a list of length len(path_boards) - 1."""
    return [1.0] * (len(path_boards) - 1)


def path_consistency_loss(model, path_boards, target, rng, n_triples,
                          edge_costs):
    """One-step consistency hinges on verified path edges (Wave 2d).

    Samples ``n_triples`` index pairs (i, j) with i+1 < j and penalises
        relu(h(s_i,s_j) - c_i     - h(s_{i+1},s_j))    # step the source
      + relu(h(s_i,s_j) - h(s_i,s_{j-1}) - c_{j-1})    # step the destination
    where c_k = edge_costs[k] is the cost of the verified edge s_k -> s_{k+1}.
    One batched forward pass over the 3n involved pairs. Both hinges are
    one-sided theorems of the true quasimetric, so they only fire on genuine
    violations (with exact-in-subgraph labels the constraints are tight
    equalities at the regression optimum — the hinge adds signal exactly
    where regression is loose). Returns None if the path is too short.
    """
    L = len(path_boards)
    if L < 3:
        return None
    idx = [(i, j) for i in range(L - 2) for j in range(i + 2, L)]
    sel = rng.sample(idx, n_triples) if len(idx) > n_triples else idx
    n = len(sel)
    states, others = [], []
    for i, j in sel:
        states += [path_boards[i], path_boards[i + 1], path_boards[i]]
        others += [path_boards[j], path_boards[j], path_boards[j - 1]]
    h = model.forward_batch(states, [target] * (3 * n), others)
    if h.ndim == 0:
        h = h.unsqueeze(0)
    h = h.view(n, 3)                       # columns: (i,j), (i+1,j), (i,j-1)
    c_src = torch.tensor([edge_costs[i] for i, _ in sel], dtype=h.dtype,
                         device=h.device)
    c_dst = torch.tensor([edge_costs[j - 1] for _, j in sel], dtype=h.dtype,
                         device=h.device)
    viol = (torch.relu(h[:, 0] - c_src - h[:, 1])
            + torch.relu(h[:, 0] - h[:, 2] - c_dst))
    return viol.mean()


def path_pair_rank_loss(model, path_boards, target, rng, n_pairs, margin):
    """Within-path pairs-of-pairs margin term (see PATH_RANK above).

    Samples ``n_pairs`` on-path pairs (p_i, p_j), i<j, scores each with ONE
    batched h-query h(p_i, target, p_j) — the same query convention as the
    buffer pairs from ``add_pairs_from_path`` — and penalises every comparable
    combination: mean over {(p,q): d_p < d_q} of relu(margin + h_p - h_q),
    i.e. the closer pair must score at least ``margin`` below the farther one
    (identical hinge form to the buffer margin_ranking_loss). |i-j| labels are
    exact distances ALONG the satisficing path, the same label semantics the
    buffer already trains on. Returns None if the path is too short or no
    comparable combination exists in the sample.
    """
    L = len(path_boards)
    if L < 3:
        return None
    all_pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    sel = rng.sample(all_pairs, n_pairs) if len(all_pairs) > n_pairs else all_pairs
    h = model.forward_batch([path_boards[i] for i, _ in sel],
                            [target] * len(sel),
                            [path_boards[j] for _, j in sel])
    if h.ndim == 0:
        h = h.unsqueeze(0)
    d = torch.tensor([float(j - i) for i, j in sel], dtype=h.dtype, device=h.device)
    closer = d.unsqueeze(1) < d.unsqueeze(0)      # [p][q] = d_p < d_q
    if not bool(closer.any()):
        return None
    viol = torch.relu(margin + h.unsqueeze(1) - h.unsqueeze(0))  # [p][q] = m+h_p-h_q
    return viol[closer].mean()


def off_path_distance_to_goal(s):
    """Weight-1 graph distance from the goal to every closed state, keyed by
    full state (agent + boxes), over the union of forward transitions and
    flipped-backward transitions.

    With full-state meeting the forward graph and the (flipped) backward graph
    share *real* nodes wherever the agent+box configuration coincides — so the
    union is a single connected graph and a plain BFS suffices. No bridges, no
    domain-specific seam repair. Distances are valid forward path lengths to a
    goal state, hence admissible (never under-count) distance-to-goal labels.

    Returns (dist_by_fullkey, goal_board) or (None, None). ``dist`` maps a
    full-state key → #moves to the goal; ``goal_board`` is the forward-frame
    goal state the distances are measured to (agent at its backward-seed cell).
    """
    fg, bg = s.forward_game, s.backward_game
    fk_f, fk_b = {}, {}

    def key_fwd(h):                       # forward hash → full-state key
        k = fk_f.get(h)
        if k is None:
            k = s._full_key(fg.decodeMap(h)); fk_f[h] = k
        return k

    def key_bwd(h_b):                     # backward hash → full-state key (fwd frame)
        k = fk_b.get(h_b)
        if k is None:
            k = s._full_key(fg.flipGame(bg.decodeMap(h_b))); fk_b[h_b] = k
        return k

    # Reverse adjacency for a goal-source BFS: rev[to] lists the forward
    # predecessors `frm` (edge frm→to), so BFS distance from the goal equals
    # the forward move-distance node→goal.
    rev = {}

    def add_fwd_edge(frm, to):
        rev.setdefault(to, []).append(frm)

    for u, vs in s.edges_f.items():
        ku = key_fwd(u)
        for v in vs:
            add_fwd_edge(ku, key_fwd(v))
    for u_b, vs_b in s.edges_b.items():
        ku = key_bwd(u_b)
        for v_b in vs_b:
            # backward move u_b→v_b  ⇔  forward move flip(v_b)→flip(u_b)
            add_fwd_edge(key_bwd(v_b), ku)

    bwd_root = next((v for v, p in s.parent_b.items() if p is None), None)
    if bwd_root is None:
        return None, None
    goal_board = fg.flipGame(bg.decodeMap(bwd_root))
    goal_fk = s._full_key(goal_board)

    dist = {goal_fk: 0}
    dq = deque([goal_fk])
    while dq:
        u = dq.popleft()
        d = dist[u]
        for v in rev.get(u, ()):
            if v not in dist:
                dist[v] = d + 1
                dq.append(v)
    return dist, goal_board


def solve_and_collect(puzzle, puzzle_id, nn_model, buffer, rng):
    """Solve a puzzle and mine training tuples from the result.

    On-path: the reconstructed plan is a valid step-by-step path (full-state
    meeting), so ``add_pairs_from_path``'s |i-j| labels are exact move counts.
    Off-path (optional): closed states not on the path are added as
    (state, goal, dist-to-goal) tuples via a weight-1 graph search from the
    goal — extra supervision in regions the single path never visits.

    Returns (path, search, dt, n_pairs_added, n_offpath_added).
    """
    path, s, dt = run_search(puzzle, nn_model)
    n_pairs = n_off = 0
    if not path:
        return path, s, dt, n_pairs, n_off

    decoded = [s.forward_game.decodeMap(h) for h in path]
    # DIRECTED: direction-correct pairs only (no reverse duplicates whose
    # labels would be wrong under asymmetric dynamics), random-pair budget
    # doubled so buffer throughput matches the symmetric legacy setting.
    n_pairs = buffer.add_pairs_from_path(
        decoded, s.forward_game.target,
        num_random_pairs=(4 if DIRECTED else 2) * len(decoded),
        symmetric=not DIRECTED, include_endpoints=True,
        puzzle_id=puzzle_id, rng=rng,
    )

    # Wave 2d zero anchoring: (x, x, 0.0) pairs through the plain MSE term —
    # the base case of the telescoping cost-admissibility argument (and
    # consistent with the search-time clamp max(0, h)).
    for _ in range(min(N_ZERO_PAIRS, len(decoded))):
        b = decoded[rng.randrange(len(decoded))]
        buffer.add(b, s.forward_game.target, b, 0.0, puzzle_id)
        n_pairs += 1

    # Wave 2e: hindsight-labeled samples of the solve's own h-queries.
    if HINDSIGHT:
        n_pairs += collect_query_hindsight(s, buffer, rng, puzzle_id)

    if COLLECT_OFF_PATH:
        dist, goal_board = off_path_distance_to_goal(s)
        if dist is not None:
            on_keys = {s._full_key(b) for b in decoded}
            tgt = s.forward_game.target
            cand = []
            for v in s.closed_f:
                vb = s.forward_game.decodeMap(v)
                fk = s._full_key(vb)
                if fk in on_keys:
                    continue
                d = dist.get(fk)
                if d is not None:
                    cand.append((vb, d))
            for vb, d in rng.sample(cand, min(OFF_PATH_PER_PUZZLE, len(cand))):
                buffer.add(vb, tgt, goal_board, d, puzzle_id)
                n_off += 1
                if not DIRECTED:
                    # Mirrored insertion labels d(goal->state) with the
                    # state->goal distance — wrong under asymmetric dynamics.
                    buffer.add(goal_board, tgt, vb, d, puzzle_id)
                    n_off += 1
    return path, s, dt, n_pairs, n_off


def rolling_mean(xs, w):
    return float(np.mean(xs[-w:])) if xs else float("nan")


# ── Load puzzles ───────────────────────────────────────────────────────────
# Solvable-only benchmark (data/solvable10_3box.txt): the player-goal-pinned
# unmeetable artifacts are filtered out, so every puzzle here contributes a
# real training signal. Built by analysis/build_solvable_benchmark.py.
puzzles = get_solvable_data(limit=N_TOTAL)
print(f"Using {len(puzzles)} solvable puzzles from the filtered dataset.")


# ── Phase A: baseline (no NN) ──────────────────────────────────────────────
print("\n[Baseline] Solving all puzzles without NN for reference …")
base_iters, base_times = [], []
t0 = time.time()
for p in puzzles:
    _, s, dt = run_search(p, None)
    base_iters.append(s.iteration)
    base_times.append(dt)
print(f"  done in {time.time()-t0:.1f}s  mean iters={np.mean(base_iters):.1f}  "
      f"median iters={np.median(base_iters):.1f}")


# ── Model(s): training model on TRAIN_DEVICE, CPU twin for search ──────────
_pr_tag = (f" +PATH_RANK(w={PATH_RANK_W},pairs={PATH_RANK_PAIRS})"
           if PATH_RANK else "")
_pr_tag += " DIRECTED" if DIRECTED else ""
_pr_tag += (f" +CONSIST(w={CONSIST_W},triples={CONSIST_TRIPLES})"
            if CONSIST else "")
_pr_tag += f" zero_pairs={N_ZERO_PAIRS}" if N_ZERO_PAIRS else ""
_pr_tag += (f" +HINDSIGHT(per={HINDSIGHT_PER_PUZZLE},cap={HINDSIGHT_LABEL_CAP})"
            if HINDSIGHT else "")
print(f"\n[Online] batch={BATCH_SIZE} K={UPDATES_PER_SOLVE} warmup={WARMUP} "
      f"buffer={BUFFER_CAP} loss={LOSS} reg_loss={REG_LOSS}{_pr_tag} "
      f"use_g={USE_G} train_device={TRAIN_DEVICE} K_remine={K_REMINE}")
_model_kw = (dict(k=EMBED_K, head=HEAD)
             if MODEL in ("embed", "embedcnn", "quasinet") else {})
torch.manual_seed(SEED + 100)
train_model = build_model(MODEL, MODEL_CHANNELS, **_model_kw)
if TRAIN_DEVICE != "cpu":
    train_model = train_model.to(TRAIN_DEVICE)
criterion, optimizer = train_model.initialize_cr_opt(loss_type=REG_LOSS)
torch.manual_seed(SEED)
_mtag = f"{MODEL}({HEAD},k={EMBED_K})" if _model_kw else MODEL
print(f"  model={_mtag}  params={sum(p.numel() for p in train_model.parameters()):,}")

# Search runs on a CPU twin (same object if training is already CPU).
if TRAIN_DEVICE == "cpu":
    search_model = train_model
else:
    search_model = build_model(MODEL, MODEL_CHANNELS, **_model_kw)


def sync_search_model():
    if TRAIN_DEVICE != "cpu":
        search_model.load_state_dict(
            {k: v.cpu() for k, v in train_model.state_dict().items()})


sync_search_model()
buffer = ReplayBuffer(capacity=BUFFER_CAP)


# ── Phase B: online solve-then-train ───────────────────────────────────────
online_iters, online_times = [], []
loss_hist = []
offpath_added = []  # per-puzzle count of off-path samples harvested
total_updates = 0
nn_active_from = None
first_saturated_n = None
seen = deque()
remine_count = 0

t_online0 = time.time()
for n, p in enumerate(puzzles):
    use_nn = len(buffer) >= WARMUP
    if use_nn and nn_active_from is None:
        nn_active_from = n
        print(f"  >>> buffer warm at puzzle {n}; NN goes live <<<")

    path, s, dt, n_pairs, n_off = solve_and_collect(
        p, n, search_model if use_nn else None, buffer, _rng)
    if path:
        offpath_added.append(n_off)
    solve_iters = s.first_meeting_iter if s.first_meeting_iter is not None else s.iteration
    online_iters.append(solve_iters)
    online_times.append(dt)
    if n_pairs > 0:
        seen.append(n)
    if first_saturated_n is None and len(buffer) == buffer.buffer.maxlen:
        first_saturated_n = n

    # Training.
    if len(buffer) >= WARMUP:
        # Within-path terms (PATH_RANK / CONSIST): decode the fresh path once
        # per solve; each update below samples its own subset from it.
        path_boards = None
        if (PATH_RANK or CONSIST) and path and len(path) >= 3:
            path_boards = [s.forward_game.decodeMap(h) for h in path]
            pr_target = s.forward_game.target
            pr_costs = path_edge_costs(path_boards)
        for _ in range(UPDATES_PER_SOLVE):
            samples = buffer.sample(BATCH_SIZE)
            optimizer.zero_grad()
            states, targets, others, tars = [], [], [], []
            for st, tgt, gm, ctg in samples:
                states.append(st); targets.append(tgt); others.append(gm)
                tars.append(ctg)
            pt = train_model.forward_batch(states, targets, others)
            tt = torch.tensor(tars, dtype=torch.float32, device=pt.device)
            if pt.ndim == 0:
                pt = pt.unsqueeze(0); tt = tt.unsqueeze(0)

            mse = criterion(pt, tt)
            rank_loss = torch.tensor(0.0, device=pt.device)
            half = pt.shape[0] // 2
            if half >= 1 and LOSS in ("rank", "both"):
                a, b = pt[:half], pt[half:2 * half]
                y = torch.sign(tt[:half] - tt[half:2 * half])
                mask = y != 0
                if mask.any():
                    rank_loss = torch.nn.functional.margin_ranking_loss(
                        a[mask], b[mask], y[mask], margin=RANK_MARGIN)
            if LOSS == "mse":
                loss = mse
            elif LOSS == "rank":
                loss = rank_loss
            else:
                loss = 0.5 * mse + 0.5 * rank_loss

            # Additive within-path pairs-of-pairs margin (PATH_RANK).
            if path_boards is not None and PATH_RANK:
                pr = path_pair_rank_loss(train_model, path_boards, pr_target,
                                         _rng, PATH_RANK_PAIRS, RANK_MARGIN)
                if pr is not None:
                    loss = loss + PATH_RANK_W * pr

            # Additive one-step consistency hinge (CONSIST, Wave 2d).
            if path_boards is not None and CONSIST:
                lc = path_consistency_loss(train_model, path_boards, pr_target,
                                           _rng, CONSIST_TRIPLES, pr_costs)
                if lc is not None:
                    loss = loss + CONSIST_W * lc

            loss.backward()
            optimizer.step()
            loss_hist.append(float(loss.item()))
            total_updates += 1
        sync_search_model()

    # Periodic re-mining of an old puzzle with the current model.
    if (K_REMINE > 0 and use_nn and first_saturated_n is not None
            and (n - first_saturated_n) > 0
            and (n - first_saturated_n) % K_REMINE == 0 and seen):
        if _rng.random() < REMINE_RANDOM_FRAC and len(seen) > 1:
            idx = _rng.randrange(len(seen)); old_idx = seen[idx]; del seen[idx]
        else:
            old_idx = seen.popleft()
        buffer.remove_by_tag(old_idx)
        solve_and_collect(puzzles[old_idx], old_idx, search_model, buffer, _rng)
        seen.append(old_idx)
        remine_count += 1

    if (n + 1) % LOG_EVERY == 0:
        b_it = rolling_mean(base_iters[:n + 1], WINDOW)
        o_it = rolling_mean(online_iters, WINDOW)
        speed = b_it / o_it if o_it > 0 else float("nan")
        oldest = buffer.oldest_tag()
        age = (n - oldest) if oldest is not None else 0
        print(f"  n={n+1:>4d}  buf={len(buffer):>6d}  upd={total_updates:>5d}  "
              f"loss={rolling_mean(loss_hist, 50):>6.2f}  "
              f"solve_iters={solve_iters:>5d}  win{WINDOW}=x{speed:.2f}  "
              f"age={age:>4d} remined={remine_count:>4d}  "
              f"dt={dt:.2f}s {'NN' if use_nn else '--'}")

    if (n + 1) % MILESTONE == 0:
        sl = slice(n + 1 - MILESTONE, n + 1)
        bw, ow = base_iters[sl], online_iters[sl]
        bm, bmd = np.mean(bw), np.median(bw)
        om, omd = np.mean(ow), np.median(ow)
        if nn_active_from is not None and nn_active_from <= n:
            cb, co = base_iters[nn_active_from:n + 1], online_iters[nn_active_from:n + 1]
            cum_m = np.mean(cb) / max(np.mean(co), 1)
            cum_md = np.median(cb) / max(np.median(co), 1)
        else:
            cum_m = cum_md = float("nan")
        print(f"\n    ── milestone n={n+1} (last {MILESTONE}) ──")
        print(f"    iters   baseline mean={bm:>7.1f} median={bmd:>7.1f}")
        print(f"    iters   online   mean={om:>7.1f} median={omd:>7.1f}")
        print(f"    speedup x_mean={bm/max(om,1):>5.2f}  x_median={bmd/max(omd,1):>5.2f}")
        print(f"    cumulative since NN live: x_mean={cum_m:.2f}  x_median={cum_md:.2f}")
        if COLLECT_OFF_PATH and offpath_added:
            oa = offpath_added[sl]
            print(f"    off-path samples: mean {np.mean(oa):.1f}/puzzle\n")

t_online_total = time.time() - t_online0


# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n[Done] online wall time: {t_online_total:.1f}s  "
      f"updates={total_updates}  remined={remine_count}")
if COLLECT_OFF_PATH and offpath_added:
    oa = np.array(offpath_added)
    print(f"  off-path harvest: {oa.sum()} samples total "
          f"(mean {oa.mean():.1f}/puzzle over {len(oa)} solved)")
if nn_active_from is not None:
    b = np.array(base_iters[nn_active_from:])
    o = np.array(online_iters[nn_active_from:])
    print(f"\nFrom puzzle {nn_active_from} onward ({len(b)} puzzles):")
    print(f"  mean   solve_iters baseline={b.mean():.1f} online={o.mean():.1f} "
          f"speedup x{b.mean()/max(o.mean(),1):.2f}")
    print(f"  median solve_iters baseline={np.median(b):.1f} online={np.median(o):.1f} "
          f"speedup x{np.median(b)/max(np.median(o),1):.2f}")

print(f"\nLearning curve (window={MILESTONE}, mean and median):")
print(f"  {'puzzle':>6}  {'base_mn':>8} {'base_md':>8}  {'on_mn':>7} {'on_md':>7}"
      f"  {'x_mn':>5}  {'x_md':>5}")
for end in range(MILESTONE, N_TOTAL + 1, MILESTONE):
    bw = base_iters[end - MILESTONE:end]
    ow = online_iters[end - MILESTONE:end]
    bm, bmd = np.mean(bw), np.median(bw)
    om, omd = np.mean(ow), np.median(ow)
    mark = "  <- NN live" if (nn_active_from is not None
                              and end - MILESTONE < nn_active_from <= end) else ""
    print(f"  {end:>6d}  {bm:>8.1f} {bmd:>8.1f}  {om:>7.1f} {omd:>7.1f}  "
          f"x{bm/max(om,1):>4.2f}  x{bmd/max(omd,1):>4.2f}{mark}")
