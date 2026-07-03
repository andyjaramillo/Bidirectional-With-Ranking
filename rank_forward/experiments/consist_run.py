"""Wave 2d (APPROACH.md): local metric grounding — one-step consistency
hinges on verified path edges (cost-general: reads path_edge_costs) plus
h(x,x)=0 zero-anchor pairs, additive on the reference loss.

Trains with CONSIST=yes N_ZERO_PAIRS=4 (set by the caller; online_run trains
at import), evaluates the frozen model and the blind baseline on the held-out
tail under the current default search (meetgen+repair+bhffa+dir_correct).

Run one seed per process:

    for sd in 0 1 2; do
      CONSIST=yes N_ZERO_PAIRS=4 SEED=$sd N_TOTAL=1800 NEVAL=200 \
        EVAL_MAX_ITERS=10000 PYTHONPATH=. \
        python -u rank_forward/experiments/consist_run.py
    done
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

TAG = (f"consist={'y' if orun.CONSIST else 'n'}(w={orun.CONSIST_W},"
       f"tri={orun.CONSIST_TRIPLES}),zeros={orun.N_ZERO_PAIRS}")
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
    print(f"[CONSIST] train:{TAG} seed={orun.SEED} {label:8s} "
          f"solved={r['solved']}/{r['n']} median={r['median']:.1f} "
          f"mean={r['mean']:.1f} len_mean={r['len_mean']:.2f} "
          f"({r['wall']:.0f}s)", flush=True)


print(f"[CONSIST] === train:{TAG} seed={orun.SEED} N_TRAIN={N_TRAIN} "
      f"eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
report("blind", eval_bd(None))
report("learned", eval_bd(model))
print(f"[CONSIST] DONE train:{TAG} seed={orun.SEED}", flush=True)
