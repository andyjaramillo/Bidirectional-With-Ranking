"""Bootstrap forward method — the forward (paper) ranking heuristic trained in
the SAME self-supervised manner as our bidirectional online_run, instead of
imitating optimal trajectories from a separate solver.

Why: removes the optimal-label advantage the offline forward method enjoys (see
experiments/README.md "Amount of learning"). Here the forward A*/GBFS planner
solves each training puzzle with its OWN current heuristic and mines a
ranking-loss instance from the path IT FOUND (satisficing, possibly suboptimal)
plus that path's distance-1 off-path siblings. This matches the bidirectional
method's supervision (bootstrap from own solutions), so a head-to-head becomes
fair on both the amount AND the kind of learning. It also directly tests the
paper's claim under our conditions: does ranking-loss imitation still work
without optimal labels?

Mirrors online_run.py:
- analytic (Manhattan) warmup until the instance buffer reaches WARMUP, then the
  learned heuristic (the forward analogue of online_run's nn=None Manhattan-F2F
  warmup). NOTE: for ALG=astar the Manhattan warmup paths are optimal (Manhattan
  is admissible); this is a small bounded effect over ~WARMUP puzzles — the bulk
  of training (post-warmup) is self-generated/satisficing. Set WARMUP small.
- K updates per solve, Adam@LR, same SmallCNN, budget MAX_ITERS, skip unsolved.

Reports an online learning curve and a frozen held-out eval (A* and GBFS),
comparable to online_run and head_to_head.

Run: PYTHONPATH=. python rank_forward/experiments/forward_bootstrap.py
Env: NTRAIN(1800) NEVAL(200) ALG(astar) LOSS(lstar) K(8) WARMUP(50) LR(1e-3)
     MAX_ITERS(10000) BUFFER(300) FULL_GOAL(1) SEED(0) MILESTONE(200)
"""
import os
import random
import statistics
import time
from collections import deque

import numpy as np
import torch

from game.getData import get_solvable_data
from rank_forward.forward_search import ForwardSearch, manhattan_heuristic, model_heuristic
from rank_forward.ranking_net import build_forward_model
from rank_forward.trajectory import instance_from_path
from rank_forward.losses import instance_loss
from rank_forward.forward_run import evaluate

NTRAIN = int(os.environ.get("NTRAIN", "1800"))
NEVAL = int(os.environ.get("NEVAL", "200"))
ALG = os.environ.get("ALG", "astar").lower()
LOSS = os.environ.get("LOSS", "lstar").lower()
K = int(os.environ.get("K", "8"))
WARMUP = int(os.environ.get("WARMUP", "50"))
LR = float(os.environ.get("LR", "1e-3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "10000"))
BUFFER = int(os.environ.get("BUFFER", "300"))
FULL_GOAL = os.environ.get("FULL_GOAL", "1").lower() in ("1", "true", "yes", "y", "on")
SEED = int(os.environ.get("SEED", "0"))
MILESTONE = int(os.environ.get("MILESTONE", "200"))
USE_G = (ALG == "astar")

torch.manual_seed(SEED)
np.random.seed(SEED)
rng = random.Random(SEED)

boards = get_solvable_data(limit=NTRAIN + NEVAL)
train_boards = boards[:NTRAIN]
eval_boards = boards[NTRAIN:NTRAIN + NEVAL]
print(f"[BOOT] forward bootstrap: train={len(train_boards)} eval={len(eval_boards)} "
      f"ALG={ALG} LOSS={LOSS} K={K} WARMUP={WARMUP} BUFFER={BUFFER} full_goal={FULL_GOAL}",
      flush=True)

model = build_forward_model("smallcnn")
opt = torch.optim.Adam(model.parameters(), lr=LR)


def solve_with(puzzle, learned):
    """Solve with the current model (learned) or the analytic Manhattan
    heuristic (warmup). Returns (path_or_None, expansions_to_solve_or_spent)."""
    s = ForwardSearch(puzzle, heuristic=None, use_g_in_f=USE_G, full_goal=FULL_GOAL)
    if learned:
        s.heuristic = model_heuristic(model, s.target, s.goal_ctx)
    else:
        s.heuristic = manhattan_heuristic(s.game)
    p = s.search(max_iterations=MAX_ITERS)
    return p, (s.first_solved_iter if p is not None else s.iteration)


# ── online bootstrap loop ───────────────────────────────────────────────────
buf = deque(maxlen=BUFFER)
online_iters = []
loss_hist = deque(maxlen=50)
updates = 0
nn_from = None
solved_count = 0
t0 = time.time()

for n, pz in enumerate(train_boards):
    learned = len(buf) >= WARMUP
    if learned and nn_from is None:
        nn_from = n
        print(f"[BOOT] buffer warm at puzzle {n}; learned heuristic now drives search",
              flush=True)
    model.eval()
    p, si = solve_with(pz, learned)
    online_iters.append(si)
    if p is not None:
        solved_count += 1
        inst = instance_from_path(pz, p, full_goal=FULL_GOAL)
        if inst is not None:
            buf.append(inst)

    if len(buf) >= WARMUP:
        model.train()
        for _ in range(K):
            inst = rng.choice(buf)
            opt.zero_grad()
            loss = instance_loss(model, inst, LOSS, reduction="sum")
            loss.backward()
            opt.step()
            updates += 1
            loss_hist.append(float(loss.detach()))

    if (n + 1) % MILESTONE == 0:
        w = online_iters[-MILESTONE:]
        sw = [x for x in w]  # online_iters already records solve-or-budget
        lh = statistics.mean(loss_hist) if loss_hist else float("nan")
        print(f"[BOOT] n={n+1:>4d} buf={len(buf):>3d} upd={updates:>5d} "
              f"loss={lh:7.2f} win{MILESTONE} median_iters={statistics.median(w):.1f} "
              f"mean={statistics.mean(w):.1f} solved_so_far={solved_count} "
              f"dt={time.time()-t0:.0f}s {'NN' if learned else 'MAN'}", flush=True)

print(f"\n[BOOT] online done: {updates} updates, {solved_count}/{len(train_boards)} solved "
      f"during training, nn_live_from={nn_from} ({time.time()-t0:.0f}s)", flush=True)

# ── held-out eval (frozen) ──────────────────────────────────────────────────
model.eval()
print("[BOOT] held-out eval (full goal) ...", flush=True)
for alg in ("astar", "gbfs"):
    r = evaluate(model, eval_boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
    print(f"[BOOT] eval {LOSS}(bootstrap)/{alg}: solved={r['solved']}/{r['n']} "
          f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f} wall={r['wall']:.0f}s",
          flush=True)
print("[BOOT] DONE", flush=True)
