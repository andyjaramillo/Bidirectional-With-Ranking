"""Train the bidirectional NN with the within-path pairs-of-pairs margin term
added to the default MSE+margin loss (PATH_RANK=yes) and evaluate the frozen
model on the held-out tail. Compares against the plain default (PATH_RANK=no —
whose 3-seed numbers are already on record; see experiments/README.md).

The idea: the buffer margin term ranks random pairs-of-pairs that are almost
always CROSS-puzzle, but the search's open list only ever compares nodes of the
SAME instance. PATH_RANK adds the within-instance version: for two pairs of
nodes on the same fresh solution path, rank their pairwise h by the respective
subpath lengths (see path_pair_rank_loss in learning/online_run.py).

`online_run` trains at import; run ONE config per process:

    for sd in 0 1 2; do
      PATH_RANK=yes SEED=$sd N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/path_rank_run.py
    done

Env: PATH_RANK (yes|no), PATH_RANK_W, PATH_RANK_PAIRS, SEED, N_TOTAL, NEVAL,
     EVAL_MAX_ITERS, plus all online_run knobs.
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

TAG = (f"pathrank=yes(w={orun.PATH_RANK_W},pairs={orun.PATH_RANK_PAIRS})"
       if orun.PATH_RANK else "pathrank=no")
N_EVAL = int(os.environ.get("NEVAL", "200"))
MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
N_TRAIN = orun.N_TOTAL
model = orun.search_model
model.eval()

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]


def eval_bd(nn):
    its, solved = [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = orun.USE_G
        s.anchor_strategy = orun.ANCHOR_STRATEGY
        path = s.search(max_iterations=MAX_ITERS)
        its.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    sv = [i for i, ok in zip(its, solved) if ok]
    return dict(solved=sum(solved), n=len(eval_boards),
                median=statistics.median(sv) if sv else float('nan'),
                mean=statistics.mean(sv) if sv else float('nan'),
                wall=time.time() - t0)


print(f"[PATHRANK] === {TAG} seed={orun.SEED} N_TRAIN={N_TRAIN} "
      f"eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
bl = eval_bd(None)
print(f"[PATHRANK] {TAG} seed={orun.SEED} blind   solved={bl['solved']}/{bl['n']} "
      f"median={bl['median']:.1f} mean={bl['mean']:.1f} ({bl['wall']:.0f}s)", flush=True)
le = eval_bd(model)
print(f"[PATHRANK] {TAG} seed={orun.SEED} learned solved={le['solved']}/{le['n']} "
      f"median={le['median']:.1f} mean={le['mean']:.1f} ({le['wall']:.0f}s)", flush=True)
print(f"[PATHRANK] DONE {TAG} seed={orun.SEED}", flush=True)
