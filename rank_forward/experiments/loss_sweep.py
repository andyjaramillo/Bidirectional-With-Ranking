"""Loss-comparison sweep for rank_forward: trains a fresh SmallCNN per loss
(L*, L_gbfs, L2, L_rt, Bellman) and evaluates each in forward A* and GBFS on a
held-out split, vs the blind forward-A* baseline. Prints solved%, mean/median
expanded nodes, and speedup vs blind. Optimal training trajectories are built
once and cached.

By default both training and eval use the FULL goal (boxes-on-targets AND
player@start, FULL_GOAL=1) — the goal the bidirectional method targets, so these
numbers are comparable to the head-to-head. Set FULL_GOAL=0 for the classical
boxes-only goal.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/loss_sweep.py
Env: N_TOTAL(2000) N_EVAL(200) STEPS(12000) SOLVE_CAP(300000) FULL_GOAL(1)
     MAX_ITERS(10000) SEED(0)
"""
import os
import time

from rank_forward.dataset import load_split, build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate, _speedup
import rank_forward.config as C

N_TOTAL = int(os.environ.get("N_TOTAL", "2000"))
N_EVAL = int(os.environ.get("N_EVAL", "200"))
STEPS = int(os.environ.get("STEPS", "12000"))
LR = float(os.environ.get("LR", "1e-3"))
SOLVE_CAP = int(os.environ.get("SOLVE_CAP", "300000"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "10000"))
SEED = int(os.environ.get("SEED", "0"))
FULL_GOAL = os.environ.get("FULL_GOAL", "1").lower() in ("1", "true", "yes", "y", "on")
LOSSES = os.environ.get("LOSSES", "lstar,lgbfs,l2,lrt,bellman").split(",")

print(f"=== rank_forward loss sweep: N_TOTAL={N_TOTAL} N_EVAL={N_EVAL} STEPS={STEPS} "
      f"MAX_ITERS={MAX_ITERS} full_goal={FULL_GOAL} model=smallcnn ===", flush=True)

train_boards, eval_boards = load_split(N_TOTAL, N_EVAL)
print(f"[data] train={len(train_boards)} eval={len(eval_boards)}", flush=True)

ck = cache_key(C.CACHE_DIR, N_TOTAL, N_EVAL, SOLVE_CAP, False, full_goal=FULL_GOAL)
t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[data] {len(instances)} optimal instances ({time.time()-t0:.0f}s)", flush=True)

print("[eval] blind forward A* (h=0) baseline ...", flush=True)
blind = evaluate(None, eval_boards, alg="astar", max_iters=MAX_ITERS, full_goal=FULL_GOAL)
print(f"  blind A*: solved={blind['solved']}/{blind['n']} mean={blind['mean_iters']:.1f} "
      f"median={blind['median_iters']:.1f} wall={blind['wall']:.0f}s", flush=True)

rows = []
for loss in LOSSES:
    print(f"\n[train] loss={loss} steps={STEPS} ...", flush=True)
    t1 = time.time()
    model = build_forward_model("smallcnn")
    train(instances, model, loss, steps=STEPS, lr=LR, reduction="sum", seed=SEED)
    model.eval()
    print(f"[train] {loss} done ({time.time()-t1:.0f}s)", flush=True)
    for alg in ("astar", "gbfs"):
        r = evaluate(model, eval_boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
        xm, xmd, nc = _speedup(blind, r)
        rows.append((loss, alg, r["solved"], r["n"], r["mean_iters"],
                     r["median_iters"], xm, xmd, nc))
        print(f"  {loss:8s}/{alg:5s} solved={r['solved']:3d}/{r['n']} "
              f"mean={r['mean_iters']:8.1f} median={r['median_iters']:7.1f} "
              f"x_mean={xm:5.2f} x_median={xmd:5.2f} (n={nc}) wall={r['wall']:.0f}s", flush=True)

print(f"\n================ SUMMARY (full_goal={FULL_GOAL}) ================", flush=True)
print(f"blind A*: solved={blind['solved']}/{blind['n']} mean={blind['mean_iters']:.1f} "
      f"median={blind['median_iters']:.1f}", flush=True)
print(f"{'loss':8s} {'alg':5s} {'solved':>8s} {'mean':>9s} {'median':>8s} "
      f"{'x_mean':>7s} {'x_med':>7s}", flush=True)
for loss, alg, sv, n, mn, md, xm, xmd, nc in rows:
    print(f"{loss:8s} {alg:5s} {sv:3d}/{n:<4d} {mn:9.1f} {md:8.1f} {xm:7.2f} {xmd:7.2f}",
          flush=True)
print("SWEEP DONE", flush=True)
