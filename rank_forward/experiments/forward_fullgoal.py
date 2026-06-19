"""Forward-only driver: train L_gbfs and L* and evaluate them (plus the blind
baseline) in forward A* and GBFS on a held-out split, under the full goal
(boxes-on-targets AND player@start) by default. This is the forward side of the
head-to-head in isolation (no bidirectional training), so it's the quick way to
get the corrected forward numbers without the ~20-min bidirectional run.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/forward_fullgoal.py
Env: NTRAIN(1800) NEVAL(200) STEPS(12000) SOLVE_CAP(300000) FULL_GOAL(1) MAX_ITERS(10000)
"""
import os
import time

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
FULL_GOAL = os.environ.get("FULL_GOAL", "1").lower() in ("1", "true", "yes", "y", "on")

train_boards, eval_boards = load_split(N_TRAIN + N_EVAL, N_EVAL)
print(f"[FG] train={len(train_boards)} eval={len(eval_boards)} full_goal={FULL_GOAL}", flush=True)

ck = cache_key(C.CACHE_DIR, N_TRAIN + N_EVAL, N_EVAL, SOLVE_CAP, False, full_goal=FULL_GOAL)
t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[FG] {len(instances)}/{len(train_boards)} optimal instances ({time.time()-t0:.0f}s)",
      flush=True)

print("[FG] blind forward A* (h=0) ...", flush=True)
blind = evaluate(None, eval_boards, alg="astar", max_iters=MAX_ITERS, full_goal=FULL_GOAL)
print(f"[FG] blind A*: solved={blind['solved']}/{blind['n']} median={blind['median_iters']:.1f} "
      f"mean={blind['mean_iters']:.1f} wall={blind['wall']:.0f}s", flush=True)

rows = []
for loss in ("lgbfs", "lstar"):
    t1 = time.time()
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=STEPS, lr=1e-3, reduction="sum", seed=0)
    m.eval()
    print(f"[FG] trained {loss} ({time.time()-t1:.0f}s)", flush=True)
    for alg in ("astar", "gbfs"):
        r = evaluate(m, eval_boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
        rows.append((loss, alg, r))
        print(f"[FG] {loss}/{alg}: solved={r['solved']}/{r['n']} median={r['median_iters']:.1f} "
              f"mean={r['mean_iters']:.1f} wall={r['wall']:.0f}s", flush=True)

print(f"\n[FG] ====== FORWARD (full_goal={FULL_GOAL}) ======", flush=True)
print(f"{'method':26s} {'solved':>9s} {'median':>9s} {'mean':>9s}", flush=True)
print(f"{'forward blind A*':26s} {blind['solved']:3d}/{blind['n']:<5d} "
      f"{blind['median_iters']:9.1f} {blind['mean_iters']:9.1f}", flush=True)
for loss, alg, r in rows:
    print(f"{'forward '+loss+' / '+alg:26s} {r['solved']:3d}/{r['n']:<5d} "
          f"{r['median_iters']:9.1f} {r['mean_iters']:9.1f}", flush=True)
print("[FG] DONE", flush=True)
