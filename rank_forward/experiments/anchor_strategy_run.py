"""Run ONE anchor-selection strategy through our exact on-policy pipeline and
evaluate it on the held-out tail. Studies the anchor-search "anchor selection"
axis (Lavasani 2024) with the learned NN F2F heuristic.

`online_run` trains the bidirectional NN on the first N_TOTAL puzzles on-policy
using ANCHOR_STRATEGY; we then evaluate the frozen model (and the blind
baseline) on boards[N_TOTAL:N_TOTAL+NEVAL] with the same strategy, in the same
node-expansion metric (len(closed_f)+len(closed_b)) as our TTBS experiments.

Because online_run trains at import, run ONE strategy per process. The full
comparison is a shell loop (run under caffeinate so sleep can't kill it):

    for s in temporal top_of_open closest_anchor; do
      N_TOTAL=1800 ANCHOR_STRATEGY=$s NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/anchor_strategy_run.py
    done

Env: ANCHOR_STRATEGY (temporal|top_of_open|closest_anchor), N_TOTAL (train),
     NEVAL (held-out), EVAL_MAX_ITERS. See experiments/README.md for results.
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

STRAT = os.environ.get("ANCHOR_STRATEGY", "temporal")
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
        s.use_g_in_f = True
        s.anchor_strategy = STRAT
        path = s.search(max_iterations=MAX_ITERS)
        its.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    sv = [i for i, ok in zip(its, solved) if ok]
    return dict(solved=sum(solved), n=len(eval_boards),
                median=statistics.median(sv) if sv else float('nan'),
                mean=statistics.mean(sv) if sv else float('nan'),
                wall=time.time() - t0)


print(f"[ANCHOR] === strategy={STRAT} N_TRAIN={N_TRAIN} eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
bl = eval_bd(None)
print(f"[ANCHOR] {STRAT} blind   solved={bl['solved']}/{bl['n']} median={bl['median']:.1f} "
      f"mean={bl['mean']:.1f} ({bl['wall']:.0f}s)", flush=True)
le = eval_bd(model)
print(f"[ANCHOR] {STRAT} learned solved={le['solved']}/{le['n']} median={le['median']:.1f} "
      f"mean={le['mean']:.1f} ({le['wall']:.0f}s)", flush=True)
print(f"[ANCHOR] DONE {STRAT}", flush=True)
