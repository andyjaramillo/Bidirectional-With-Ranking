"""REVWALK: reverse-walk long-range pair generation (with length curriculum).

Trains through the reference pipeline (online_run trains at import) with
REVWALK on/off and evaluates the frozen model + blind baseline on the held-out
tail. The hypothesis: the buffer's finite labels are all short-range (bounded
by solved-path / explored-subgraph distances), while the open list is ordered
by h at long range early in each search; reverse walks supply sound
upper-bound labels at ranges no solve reaches, so early-search ordering — and
with it expansions — should improve.

Run one (seed, arm) per process; baseline is REVWALK=no:

    REVWALK=no  SEED=0 N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/revwalk_run.py
    REVWALK=yes SEED=0 N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/revwalk_run.py
"""
import os, statistics, time
import torch
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

TAG = (f"revwalk(w={orun.REVWALK_WALKS},len={orun.REVWALK_LEN_MIN}->"
       f"{orun.REVWALK_LEN},p={orun.REVWALK_PAIRS})"
       if orun.REVWALK else "base")

# Persist the trained net: on-policy training has known run-to-run
# nondeterminism (buffer.sample), so downstream evals (e.g. the hard set)
# must load THIS model, not a retrain. MODEL_OUT="" disables.
MODEL_OUT = os.environ.get("MODEL_OUT", "/tmp/revwalk_models")
if MODEL_OUT:
    os.makedirs(MODEL_OUT, exist_ok=True)
    _arm = "revwalk" if orun.REVWALK else "base"
    _ckpt = os.path.join(MODEL_OUT, f"{_arm}_seed{orun.SEED}.pt")
    torch.save(orun.search_model.state_dict(), _ckpt)
    print(f"[REVWALK] model saved -> {_ckpt}", flush=True)
N_EVAL = int(os.environ.get("NEVAL", "200"))
MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
N_TRAIN = orun.N_TOTAL
model = orun.search_model
model.eval()

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]


def eval_bd(nn):
    its, lens, solved = [], [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)   # current defaults throughout
        s.use_g_in_f = orun.USE_G
        s.anchor_strategy = orun.ANCHOR_STRATEGY
        path = s.search(max_iterations=MAX_ITERS)
        its.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
        if path:
            lens.append(len(path) - 1)
    sv = [i for i, ok in zip(its, solved) if ok]
    return dict(solved=sum(solved), n=len(eval_boards),
                median=statistics.median(sv) if sv else float('nan'),
                mean=statistics.mean(sv) if sv else float('nan'),
                len_mean=statistics.mean(lens) if lens else float('nan'),
                wall=time.time() - t0)


def report(label, r):
    print(f"[REVWALK] train:{TAG} seed={orun.SEED} {label:8s} "
          f"solved={r['solved']}/{r['n']} median={r['median']:.1f} "
          f"mean={r['mean']:.1f} len_mean={r['len_mean']:.2f} "
          f"wall={r['wall']:.0f}s ({r['wall']*1000/r['n']:.0f}ms/solve)", flush=True)


print(f"[REVWALK] === train:{TAG} seed={orun.SEED} "
      f"N_TRAIN={N_TRAIN} eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===",
      flush=True)
report("blind", eval_bd(None))
report("learned", eval_bd(model))
print(f"[REVWALK] DONE train:{TAG} seed={orun.SEED}", flush=True)
