"""Wave 3 (APPROACH.md): factorized (quasi)metric embedding heuristic.

Trains MODEL=embed (HEAD in {quasi,l1,mlp}) through the reference pipeline
(online_run trains at import) and evaluates the frozen model + blind baseline
on the held-out tail. Reports expansions AND eval wall-time (the factorization
speed claim: O(k) cached rescoring vs a conv pass per query).

Run one (seed, head) per process; baseline is MODEL=smallcnn:

    for h in quasi l1 mlp; do
      MODEL=embed HEAD=$h SEED=0 N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/embed_run.py
    done
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

_mk = orun.MODEL in ("embed", "embedcnn", "quasinet")
TAG = f"{orun.MODEL}({orun.HEAD},k={orun.EMBED_K})" if _mk else orun.MODEL
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
    print(f"[EMBED] train:{TAG} seed={orun.SEED} {label:8s} "
          f"solved={r['solved']}/{r['n']} median={r['median']:.1f} "
          f"mean={r['mean']:.1f} len_mean={r['len_mean']:.2f} "
          f"wall={r['wall']:.0f}s ({r['wall']*1000/r['n']:.0f}ms/solve)", flush=True)


print(f"[EMBED] === train:{TAG} seed={orun.SEED} params="
      f"{sum(p.numel() for p in model.parameters()):,} "
      f"N_TRAIN={N_TRAIN} eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
report("blind", eval_bd(None))
report("learned", eval_bd(model))
print(f"[EMBED] DONE train:{TAG} seed={orun.SEED}", flush=True)
