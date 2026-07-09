"""Extra row for the hard-instance generalization table: the bidirectional
method at the hs64 hindsight dose (HINDSIGHT_PER_PUZZLE=64, LABEL_CAP=256 —
hindsight_tuning_results/: median better 3/3 seeds vs control, the best local
configuration; not yet 5-seed validated, hence a separate row rather than the
default in hard_head_to_head.py).

Trains online_run on the same first HH_NTRAIN 3-box boards, evaluates frozen
on the same 3..7-box sets, same budget and expansion unit as the main driver.

Run from repo root (after the hard sets exist):
    PYTHONPATH=. python rank_forward/experiments/hard_h2h_hs64.py
Env: HH_BOXES("3,4,5,6,7") HH_NTRAIN(1800) HH_NEVAL(200) HH_MAX_ITERS(10000)
"""
import os
import statistics

import numpy as np

from game.getData import get_solvable_data, DATA_DIR

MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "1800"))
N_EVAL = int(os.environ.get("HH_NEVAL", "200"))
BOXES = [int(b) for b in os.environ.get("HH_BOXES", "3,4,5,6,7").split(",")]

# hs64 dose + same train split, BEFORE the training import.
os.environ["HINDSIGHT_PER_PUZZLE"] = "64"
os.environ["HINDSIGHT_LABEL_CAP"] = "256"
os.environ["N_TOTAL"] = str(N_TRAIN)

_boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
print(f"[HS64] training bidirectional on first {N_TRAIN} 3-box boards "
      f"(hindsight dose 64/256)", flush=True)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model = orun.search_model
model.eval()


def eval_set(nb):
    if nb == 3:
        return _boards[N_TRAIN:N_TRAIN + N_EVAL]
    with open(os.path.join(DATA_DIR, f"hard10_{nb}box.txt")) as f:
        return [np.array([int(v) for v in line.split()]).reshape(10, 10)
                for line in f if line.strip()]


print(f"\n[HS64] ===== bidir learned (hs64) rows, budget={MAX_ITERS} =====",
      flush=True)
for nb in BOXES:
    iters, solved = [], []
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
    sv = [it for it, ok in zip(iters, solved) if ok]
    print(f"[HS64] bidir hs64 {nb}box: solved={sum(solved)}/{len(iters)} "
          f"median={statistics.median(sv) if sv else float('nan'):.1f} "
          f"mean={statistics.mean(sv) if sv else float('nan'):.1f}", flush=True)
print("[HS64] DONE", flush=True)
