"""hs64 row (HINDSIGHT_PER_PUZZLE=64, LABEL_CAP=256) rerun WITH per-instance
logging, so it joins the self-consistent seed-0 set of hard_h2h_perinstance.py
(same train split, same test sets, same budget; results merge into
hard_h2h_results/perinstance_seed0.json). Also saves the model weights and
prints the both-solved intersection of hs64 against the rows already in the
JSON (fwd lstar/astar, fwd lgbfs/gbfs, bidir learned).

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/hard_h2h_hs64_perinstance.py
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
BOXES = [int(b) for b in os.environ.get("HH_BOXES", "3,4,5,6,7").split(",")]
OUT_DIR = "hard_h2h_results"
PI_PATH = os.path.join(OUT_DIR, "perinstance_seed0.json")

os.environ["HINDSIGHT_PER_PUZZLE"] = "64"
os.environ["HINDSIGHT_LABEL_CAP"] = "256"
os.environ["N_TOTAL"] = str(N_TRAIN)

_boards = get_solvable_data(limit=N_TRAIN + N_EVAL)


def eval_set(nb):
    if nb == 3:
        return _boards[N_TRAIN:N_TRAIN + N_EVAL]
    with open(os.path.join(DATA_DIR, f"hard10_{nb}box.txt")) as f:
        return [np.array([int(v) for v in line.split()]).reshape(10, 10)
                for line in f if line.strip()]


print(f"[HS64-PI] training bidirectional on first {N_TRAIN} 3-box boards "
      f"(hindsight dose 64/256)", flush=True)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model = orun.search_model
model.eval()
torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_bidir_hs64.pt"))

mine = {}
for nb in BOXES:
    iters, solved = [], []
    t1 = time.time()
    for p in eval_set(nb):
        s = BidirectionalF2FSearch(p, model)
        s.use_g_in_f = True
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = orun.DIRECTED
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    mine[nb] = {"iters": iters, "solved": solved}
    sv = [it for it, ok in zip(iters, solved) if ok]
    print(f"[HS64-PI] bidir hs64 {nb}box: solved={sum(solved)}/{len(iters)} "
          f"median={statistics.median(sv) if sv else float('nan'):.1f} "
          f"mean={statistics.mean(sv) if sv else float('nan'):.1f} "
          f"({time.time()-t1:.0f}s)", flush=True)

# merge into the self-consistent per-instance store
with open(PI_PATH) as f:
    per = json.load(f)
per["bidir hs64"] = {str(nb): mine[nb] for nb in BOXES}
with open(PI_PATH, "w") as f:
    json.dump(per, f)
print(f"[HS64-PI] merged into {PI_PATH}", flush=True)

# intersections vs the existing rows
print("\n[HS64-PI] ===== BOTH-SOLVED INTERSECTION (hs64 vs others) =====",
      flush=True)
for other in ("fwd lstar/astar", "fwd lgbfs/gbfs", "bidir learned"):
    print(f"\n[HS64-PI] --- {other} vs bidir hs64 ---", flush=True)
    for nb in BOXES:
        a = per[other][str(nb)] if str(nb) in per[other] else per[other][nb]
        b = mine[nb]
        both = [(ai, bi) for ai, ao, bi, bo in
                zip(a["iters"], a["solved"], b["iters"], b["solved"])
                if ao and bo]
        if not both:
            print(f"[HS64-PI] {nb}box: no common solves", flush=True)
            continue
        fa = [x for x, _ in both]
        fb = [y for _, y in both]
        ratios = [y / x for x, y in both if x > 0]
        print(f"[HS64-PI] {nb}box n_both={len(both):3d}  "
              f"other med/mean={statistics.median(fa):7.1f}/{statistics.mean(fa):7.1f}  "
              f"hs64 med/mean={statistics.median(fb):7.1f}/{statistics.mean(fb):7.1f}  "
              f"hs64/other ratio med={statistics.median(ratios):.2f}",
              flush=True)
print("[HS64-PI] DONE", flush=True)
