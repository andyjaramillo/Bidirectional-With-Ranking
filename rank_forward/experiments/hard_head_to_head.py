"""Generalization head-to-head on the HARDER Sokoban sets (4-7 boxes):
the apples-to-apples version of Chrestien et al.'s Table 1 axis — train on
3-box instances, evaluate the frozen heuristics on 200 instances each of
3, 4, 5, 6, 7 boxes at the same 10x10 size, same expansion budget, same
fair full goal (boxes-on-targets AND player back at start; see
head_to_head.py for why that is the apples-to-apples setting here).

Train (both on the same first N_TRAIN 3-box solvable boards):
  forward  — offline optimal labels (paper protocol), L* and Lgbfs losses,
             evaluated in the search each loss targets (L*->A*, Lgbfs->GBFS);
  bidirect — on-policy online_run at the CURRENT reference defaults
             (MSE+margin+PATH_RANK, DIRECTED, HINDSIGHT, meet-on-generate,
             seam repair, BHFFA-g).

Eval sets: the held-out 3-box split (boards[N_TRAIN:N_TRAIN+N_EVAL]) plus
data/hard10_{4,5,6,7}box.txt (analysis/build_hard_benchmark.py — gym-sokoban
reverse-play, the paper's generator, filtered solvable under the full goal).

Metrics per (method, box count): solved/N within MAX_ITERS expansions,
median & mean expansions over the SOLVED instances (repo convention), same
expansion unit both sides (forward closed-set size vs closed_f+closed_b).

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/hard_head_to_head.py
Env: HH_BOXES("3,4,5,6,7") HH_NTRAIN(1800) HH_NEVAL(200) HH_FWD_STEPS(12000)
     HH_SOLVE_CAP(300000) HH_MAX_ITERS(10000) SEED(0)
"""
import os
import statistics
import time

import numpy as np

from game.getData import get_solvable_data, DATA_DIR

MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "1800"))
N_EVAL = int(os.environ.get("HH_NEVAL", "200"))
FWD_STEPS = int(os.environ.get("HH_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("HH_SOLVE_CAP", "300000"))
BOXES = [int(b) for b in os.environ.get("HH_BOXES", "3,4,5,6,7").split(",")]
SEED = int(os.environ.get("SEED", "0"))
FULL_GOAL = True   # the only apples-to-apples setting (see module docstring)


def _read_boards(path):
    with open(path) as f:
        return [np.array([int(v) for v in line.split()]).reshape(10, 10)
                for line in f if line.strip()]


def summ(iters, solved):
    sv = [it for it, ok in zip(iters, solved) if ok]
    return {"solved": sum(solved), "n": len(iters),
            "median": statistics.median(sv) if sv else float("nan"),
            "mean": statistics.mean(sv) if sv else float("nan")}


_boards_all = get_solvable_data(limit=N_TRAIN + N_EVAL)
train_boards = _boards_all[:N_TRAIN]
_eval_cache = {3: _boards_all[N_TRAIN:N_TRAIN + N_EVAL]}


def get_eval_set(nb):
    """3-box: the held-out split. 4-7 box: data/hard10_{nb}box.txt, possibly
    still being produced by a concurrent build_hard_benchmark job — WAIT for
    the file to be complete (200 lines). ALL TRAINING runs before the first
    call, so the wait overlaps generation with training, never blocks it."""
    if nb in _eval_cache:
        return _eval_cache[nb]
    path = os.path.join(DATA_DIR, f"hard10_{nb}box.txt")
    waited = 0
    while True:
        bs = _read_boards(path) if os.path.exists(path) else []
        if len(bs) >= 200:
            _eval_cache[nb] = bs
            return bs
        if waited % 600 == 0:
            print(f"[HARD-H2H] waiting for {path} ({len(bs)}/200 so far)...",
                  flush=True)
        time.sleep(60)
        waited += 60


print(f"[HARD-H2H] train on {len(train_boards)} 3-box boards; eval boxes: "
      f"{BOXES}; budget={MAX_ITERS}", flush=True)

results = {}   # (method, nb) -> summ dict

# ═══════════════════ PHASE 1 — ALL TRAINING (3-box only) ═══════════════════
print("\n[HARD-H2H] ===== FORWARD training (the paper) =====", flush=True)
from rank_forward.dataset import build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate
import rank_forward.config as C

ck = cache_key(C.CACHE_DIR, N_TRAIN + N_EVAL, N_EVAL, SOLVE_CAP, False,
               full_goal=FULL_GOAL)
t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[HARD-H2H] {len(instances)} optimal instances ({time.time()-t0:.0f}s)",
      flush=True)

fwd_models = {}
for loss, alg in (("lstar", "astar"), ("lgbfs", "gbfs")):
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
    m.eval()
    fwd_models[(loss, alg)] = m
    print(f"[HARD-H2H] forward {loss} trained", flush=True)

print(f"\n[HARD-H2H] ===== BIDIRECTIONAL training (online_run on first "
      f"{N_TRAIN}, current reference defaults) =====", flush=True)
os.environ["N_TOTAL"] = str(N_TRAIN)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()

# ═══════════════ PHASE 2 — FROZEN EVALS on 3..7 boxes ══════════════════════
for nb in BOXES:
    bs = get_eval_set(nb)
    r = evaluate(None, bs, alg="astar", max_iters=MAX_ITERS, full_goal=FULL_GOAL)
    results[("fwd blind A*", nb)] = {"solved": r["solved"], "n": r["n"],
                                     "median": r["median_iters"],
                                     "mean": r["mean_iters"]}
    print(f"[HARD-H2H] fwd blind A* {nb}box: solved={r['solved']}/{r['n']} "
          f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f}", flush=True)

for (loss, alg), m in fwd_models.items():
    for nb in BOXES:
        bs = get_eval_set(nb)
        r = evaluate(m, bs, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
        results[(f"fwd {loss}/{alg}", nb)] = {"solved": r["solved"], "n": r["n"],
                                              "median": r["median_iters"],
                                              "mean": r["mean_iters"]}
        print(f"[HARD-H2H] fwd {loss}/{alg} {nb}box: solved={r['solved']}/{r['n']} "
              f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f}",
              flush=True)


def eval_bd(nn, boards):
    iters, solved = [], []
    for p in boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = True
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = orun.DIRECTED
        path = s.search(max_iterations=MAX_ITERS)
        # Same expansion unit as the forward side (see head_to_head.py).
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return summ(iters, solved)


for name, nn in (("bidir Manhattan", None), ("bidir learned", model_bd)):
    for nb in BOXES:
        bs = get_eval_set(nb)
        t1 = time.time()
        r = eval_bd(nn, bs)
        results[(name, nb)] = r
        print(f"[HARD-H2H] {name} {nb}box: solved={r['solved']}/{r['n']} "
              f"median={r['median']:.1f} mean={r['mean']:.1f} "
              f"({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────── TABLE ─────────────────────────────────────
methods = ["fwd blind A*", "fwd lstar/astar", "fwd lgbfs/gbfs",
           "bidir Manhattan", "bidir learned"]
print(f"\n[HARD-H2H] ========== GENERALIZATION TABLE (trained on 3-box, "
      f"budget={MAX_ITERS}, full goal) ==========", flush=True)
print(f"{'method':18s} " + " ".join(f"{f'{nb} boxes':>21s}" for nb in BOXES),
      flush=True)
print(f"{'':18s} " + " ".join(f"{'solved  med   mean':>21s}" for _ in BOXES),
      flush=True)
for name in methods:
    cells = []
    for nb in BOXES:
        r = results.get((name, nb))
        cells.append(f"{r['solved']:3d}/{r['n']:<3d} {r['median']:5.0f} "
                     f"{r['mean']:7.1f}" if r else " " * 21)
    print(f"{name:18s} " + " ".join(f"{c:>21s}" for c in cells), flush=True)
print("[HARD-H2H] DONE", flush=True)
