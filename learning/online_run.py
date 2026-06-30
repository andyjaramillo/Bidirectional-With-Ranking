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
from learning.ranking_loss import bidir_ranking_loss

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
# Genuine ranking loss (NeurIPS-2023 perfect-ranking condition, ported to the
# bidirectional F2F search in learning/ranking_loss.py): train PURELY on the
# per-step condition "the on-path child has the lowest merit among its
# siblings", scored against the exact per-step anchor the search used. When
# "yes" it REPLACES the MSE+margin buffer loss (the buffer still fills, to gate
# warmup); default off, so the legacy loss above is untouched.
RANK_LOSS = os.environ.get("RANK_LOSS", "no").lower() == "yes"
# Under RANK_LOSS, rank on h only (L_gbfs) instead of g+h (L*).
RANK_GBFS = os.environ.get("RANK_GBFS", "no").lower() == "yes"
# If "no", drop g from the f-score (GBFS instead of A*-style).
USE_G = os.environ.get("USE_G", "yes").lower() == "yes"
# Detect the frontier meeting as soon as both sides have GENERATED the shared
# state (default), instead of only when one side has CLOSED it — avoids ~11%
# of node expansions wasted past a valid seam. "no" restores legacy behavior.
MEET_ON_GENERATE = os.environ.get("MEET_ON_GENERATE", "no").lower() == "yes"
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
# RANK_LOSS is incompatible with FULL_F2F: under full-F2F the search scores a
# node against the min over the WHOLE opponent frontier, but the per-step anchor
# log records only a single anchor, so the ranking loss would train against the
# wrong target. Fail fast rather than train a silently-mistargeted model.
if RANK_LOSS and FULL_F2F:
    raise SystemExit("RANK_LOSS=yes is incompatible with FULL_F2F=yes "
                     "(single-anchor log vs full-frontier scoring). Disable one.")
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
    s.full_f2f = FULL_F2F
    s.anchor_strategy = ANCHOR_STRATEGY
    # Record per-step anchors for the ranking loss — only on NN solves (the
    # blind baseline pass discards its search, so logging there is wasted work).
    s.log_expansions = RANK_LOSS and (nn_model is not None)
    t0 = time.time()
    path = s.search(max_iterations=MAX_ITERS)
    return path, s, time.time() - t0


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
    n_pairs = buffer.add_pairs_from_path(
        decoded, s.forward_game.target,
        num_random_pairs=2 * len(decoded),
        symmetric=True, include_endpoints=True,
        puzzle_id=puzzle_id, rng=rng,
    )

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
                buffer.add(goal_board, tgt, vb, d, puzzle_id)
                n_off += 2
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
_loss_desc = (f"RANKING({'L_gbfs' if RANK_GBFS else 'L*'}, per-step anchor)"
              if RANK_LOSS else f"{LOSS}/{REG_LOSS}")
print(f"\n[Online] batch={BATCH_SIZE} K={UPDATES_PER_SOLVE} warmup={WARMUP} "
      f"buffer={BUFFER_CAP} loss={_loss_desc} use_g={USE_G} "
      f"train_device={TRAIN_DEVICE} K_remine={K_REMINE}")
torch.manual_seed(SEED + 100)
train_model = build_model(MODEL, MODEL_CHANNELS)
if TRAIN_DEVICE != "cpu":
    train_model = train_model.to(TRAIN_DEVICE)
criterion, optimizer = train_model.initialize_cr_opt(loss_type=REG_LOSS)
torch.manual_seed(SEED)
print(f"  model={MODEL}  params={sum(p.numel() for p in train_model.parameters()):,}")

# Search runs on a CPU twin (same object if training is already CPU).
if TRAIN_DEVICE == "cpu":
    search_model = train_model
else:
    search_model = build_model(MODEL, MODEL_CHANNELS)


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
    if len(buffer) >= WARMUP and RANK_LOSS:
        # Pure genuine ranking loss, per instance. Only solved instances yield
        # a rankable path; unsolved ones contribute no gradient this round.
        if path:
            for _ in range(UPDATES_PER_SOLVE):
                optimizer.zero_grad()
                loss = bidir_ranking_loss(train_model, s, use_g=USE_G, gbfs=RANK_GBFS)
                if loss is None:
                    break
                loss.backward()
                optimizer.step()
                loss_hist.append(float(loss.item()))
                total_updates += 1
            sync_search_model()
    elif len(buffer) >= WARMUP:
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

            loss.backward()
            optimizer.step()
            loss_hist.append(float(loss.item()))
            total_updates += 1
        sync_search_model()

    # Periodic re-mining of an old puzzle with the current model. (Skipped
    # under RANK_LOSS: the buffer isn't used for gradients there, so re-mining
    # it would only waste a solve.)
    if (K_REMINE > 0 and use_nn and not RANK_LOSS and first_saturated_n is not None
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
