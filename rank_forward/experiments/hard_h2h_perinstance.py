"""Per-instance rerun of the hard-instance head-to-head (seed 0) + the
BOTH-SOLVED INTERSECTION metric.

The first driver run only logged summary rows, so the median/mean comparison
at high box counts is confounded by survivorship (each method's "solved" pool
differs). This rerun retrains the same seed-0 models (forward optimal-label
cache is on disk; training is seeded, so models reproduce), re-evaluates the
three learned rows, and this time PERSISTS:
  hard_h2h_results/perinstance_seed0.json   per-instance iters/solved
  hard_h2h_results/model_{fwd_lstar,fwd_lgbfs,bidir}.pt  state dicts
then prints, per box count and method pair, the median/mean expansions over
the instances BOTH methods solved.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/hard_h2h_perinstance.py
Env: same knobs as hard_head_to_head.py.
"""
import json
import os
import statistics
import time

import numpy as np
import torch

from game.getData import get_solvable_data, DATA_DIR

MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "1800"))
N_EVAL = int(os.environ.get("HH_NEVAL", "200"))
FWD_STEPS = int(os.environ.get("HH_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("HH_SOLVE_CAP", "300000"))
BOXES = [int(b) for b in os.environ.get("HH_BOXES", "3,4,5,6,7").split(",")]
SEED = int(os.environ.get("SEED", "0"))
FULL_GOAL = True
OUT_DIR = "hard_h2h_results"

_boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
train_boards = _boards[:N_TRAIN]


def eval_set(nb):
    if nb == 3:
        return _boards[N_TRAIN:N_TRAIN + N_EVAL]
    with open(os.path.join(DATA_DIR, f"hard10_{nb}box.txt")) as f:
        return [np.array([int(v) for v in line.split()]).reshape(10, 10)
                for line in f if line.strip()]


per = {}   # method -> {nb: {"iters": [...], "solved": [...]}}

# ── forward side (models reproduce from the cached optimal instances) ───────
from rank_forward.dataset import build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate
import rank_forward.config as C

ck = cache_key(C.CACHE_DIR, N_TRAIN + N_EVAL, N_EVAL, SOLVE_CAP, False,
               full_goal=FULL_GOAL)
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[PI] {len(instances)} optimal instances (from cache)", flush=True)

for loss, alg in (("lstar", "astar"), ("lgbfs", "gbfs")):
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
    m.eval()
    torch.save(m.state_dict(), os.path.join(OUT_DIR, f"model_fwd_{loss}.pt"))
    name = f"fwd {loss}/{alg}"
    per[name] = {}
    for nb in BOXES:
        r = evaluate(m, eval_set(nb), alg=alg, max_iters=MAX_ITERS,
                     full_goal=FULL_GOAL)
        per[name][nb] = {"iters": list(map(int, r["iters"])),
                         "solved": list(map(bool, r["solved_flags"]))}
        print(f"[PI] {name} {nb}box: solved={r['solved']}/{r['n']} "
              f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f}",
              flush=True)

# ── bidirectional side (reproduces the main driver's seed-0 training) ───────
os.environ["N_TOTAL"] = str(N_TRAIN)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()
torch.save(model_bd.state_dict(), os.path.join(OUT_DIR, "model_bidir.pt"))

per["bidir learned"] = {}
for nb in BOXES:
    iters, solved = [], []
    t1 = time.time()
    for p in eval_set(nb):
        s = BidirectionalF2FSearch(p, model_bd)
        s.use_g_in_f = True
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = orun.DIRECTED
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    per["bidir learned"][nb] = {"iters": iters, "solved": solved}
    sv = [it for it, ok in zip(iters, solved) if ok]
    print(f"[PI] bidir learned {nb}box: solved={sum(solved)}/{len(iters)} "
          f"median={statistics.median(sv) if sv else float('nan'):.1f} "
          f"mean={statistics.mean(sv) if sv else float('nan'):.1f} "
          f"({time.time()-t1:.0f}s)", flush=True)

with open(os.path.join(OUT_DIR, "perinstance_seed0.json"), "w") as f:
    json.dump(per, f)
print(f"[PI] per-instance data saved to {OUT_DIR}/perinstance_seed0.json",
      flush=True)

# ── both-solved intersection metric ─────────────────────────────────────────
print("\n[PI] ========== BOTH-SOLVED INTERSECTION ==========", flush=True)
for fwd_name in ("fwd lstar/astar", "fwd lgbfs/gbfs"):
    print(f"\n[PI] --- {fwd_name} vs bidir learned ---", flush=True)
    for nb in BOXES:
        a, b = per[fwd_name][nb], per["bidir learned"][nb]
        both = [(ai, bi) for ai, ao, bi, bo in
                zip(a["iters"], a["solved"], b["iters"], b["solved"])
                if ao and bo]
        if not both:
            print(f"[PI] {nb}box: no common solves", flush=True)
            continue
        fa = [x for x, _ in both]
        fb = [y for _, y in both]
        ratios = [y / x for x, y in both if x > 0]
        print(f"[PI] {nb}box  n_both={len(both):3d}  "
              f"fwd med/mean={statistics.median(fa):6.1f}/{statistics.mean(fa):7.1f}  "
              f"bidir med/mean={statistics.median(fb):6.1f}/{statistics.mean(fb):7.1f}  "
              f"bidir/fwd ratio med={statistics.median(ratios):.2f} "
              f"mean={statistics.mean(ratios):.2f}", flush=True)
print("[PI] DONE", flush=True)
