"""Forward (Chrestien-style ranking) baseline on the HARD held-out set.

Trains the canonical forward baselines (L*, L_gbfs; SmallCNN; optimal labels
from the full-goal cache) on the SAME train prefix the bidirectional reference
uses (first NTRAIN solvable boards), then evaluates:
  1. standard held-out 200 (cap MAX_ITERS)      — sanity vs stored numbers
  2. hard200 set (cap HARD_MAX_ITERS)           — the comparison of interest
under the fair FULL GOAL (boxes-on-targets AND player@start — the goal the
bidirectional method actually targets). Blind A* rows included for reference
(hard blind last: it is the slowest row). Trained models are saved to
MODEL_OUT for reuse.

Run from repo root:
    PYTHONPATH=. nohup caffeinate -i python -u \
        rank_forward/experiments/forward_hard.py > /tmp/forward_hard.log &
Env: NTRAIN(1800) NEVAL(200) STEPS(12000) SOLVE_CAP(300000)
     MAX_ITERS(10000) HARD_MAX_ITERS(50000) FULL_GOAL(1)
     MODEL_OUT(/tmp/fwd_models) HARD_KEEP(200)
"""
import os
import time

import torch

from game.getData import get_hard_eval_data
from rank_forward.dataset import load_split, build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate
import rank_forward.config as C

N_TRAIN = int(os.environ.get("NTRAIN", "1800"))
N_EVAL = int(os.environ.get("NEVAL", "200"))
STEPS = int(os.environ.get("STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("SOLVE_CAP", "300000"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "10000"))
HARD_MAX_ITERS = int(os.environ.get("HARD_MAX_ITERS", "50000"))
HARD_KEEP = int(os.environ.get("HARD_KEEP", "200"))
FULL_GOAL = os.environ.get("FULL_GOAL", "1").lower() in ("1", "true", "yes", "y", "on")
MODEL_OUT = os.environ.get("MODEL_OUT", "/tmp/fwd_models")

train_boards, eval_boards = load_split(N_TRAIN + N_EVAL, N_EVAL)
hard_boards = get_hard_eval_data(keep=HARD_KEEP)
print(f"[FWDH] train={len(train_boards)} eval={len(eval_boards)} "
      f"hard={len(hard_boards)} full_goal={FULL_GOAL}", flush=True)

ck = cache_key(C.CACHE_DIR, N_TRAIN + N_EVAL, N_EVAL, SOLVE_CAP, False,
               full_goal=FULL_GOAL)
t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[FWDH] {len(instances)}/{len(train_boards)} optimal instances "
      f"({time.time()-t0:.0f}s)", flush=True)


def report(label, r):
    print(f"[FWDH] {label:22s} solved={r['solved']}/{r['n']} "
          f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f} "
          f"wall={r['wall']:.0f}s", flush=True)


# ── standard-set sanity rows (cheap) ───────────────────────────────────────
report("std blind/astar", evaluate(None, eval_boards, alg="astar",
                                   max_iters=MAX_ITERS, full_goal=FULL_GOAL))

models = {}
for loss in ("lstar", "lgbfs"):
    t1 = time.time()
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=STEPS, lr=1e-3, reduction="sum", seed=0)
    m.eval()
    models[loss] = m
    if MODEL_OUT:
        os.makedirs(MODEL_OUT, exist_ok=True)
        p = os.path.join(MODEL_OUT, f"fwd_{loss}_seed0.pt")
        torch.save(m.state_dict(), p)
        print(f"[FWDH] trained {loss} ({time.time()-t1:.0f}s) -> {p}", flush=True)
    for alg in ("astar", "gbfs"):
        report(f"std {loss}/{alg}",
               evaluate(m, eval_boards, alg=alg, max_iters=MAX_ITERS,
                        full_goal=FULL_GOAL))

# ── hard200: canonical pairings first, blind last (slowest) ────────────────
report("hard lstar/astar", evaluate(models["lstar"], hard_boards, alg="astar",
                                    max_iters=HARD_MAX_ITERS, full_goal=FULL_GOAL))
report("hard lgbfs/gbfs", evaluate(models["lgbfs"], hard_boards, alg="gbfs",
                                   max_iters=HARD_MAX_ITERS, full_goal=FULL_GOAL))
report("hard blind/astar", evaluate(None, hard_boards, alg="astar",
                                    max_iters=HARD_MAX_ITERS, full_goal=FULL_GOAL))
print("[FWDH] DONE", flush=True)
