"""Train the bidirectional NN through our exact on-policy pipeline under a
chosen LOSS, then evaluate the frozen model on the held-out tail. Used to
compare the GENUINE ranking loss (RANK_LOSS=yes — the NeurIPS-2023 perfect-
ranking condition ported to bidirectional F2F, scored against the per-step
anchor; see learning/ranking_loss.py) against our default MSE+margin buffer
loss (RANK_LOSS=no), all else equal.

`online_run` trains the NN on the first N_TOTAL puzzles on-policy at import; we
then evaluate the frozen model (and the blind baseline) on
boards[N_TOTAL:N_TOTAL+NEVAL] with the same anchor strategy and the same node-
expansion metric (len(closed_f)+len(closed_b)) as all our TTBS experiments.

Run ONE loss per process (online_run trains at import). The comparison loop
(under caffeinate so sleep can't kill it):

    for rl in no yes; do
      RANK_LOSS=$rl N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 SEED=0 \
        PYTHONPATH=. python -u rank_forward/experiments/ranking_loss_run.py
    done

Env: RANK_LOSS (yes|no), RANK_GBFS (yes|no), ANCHOR_STRATEGY (default temporal),
     N_TOTAL (train), NEVAL (held-out), EVAL_MAX_ITERS, SEED.
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

STRAT = orun.ANCHOR_STRATEGY
N_EVAL = int(os.environ.get("NEVAL", "200"))
MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
N_TRAIN = orun.N_TOTAL
model = orun.search_model
model.eval()

if orun.RANK_LOSS:
    LOSS_DESC = ("RANKING(pathorder)" if orun.RANK_MODE == "pathorder"
                 else f"RANKING(perfect,{'L_gbfs' if orun.RANK_GBFS else 'L*'})")
else:
    LOSS_DESC = f"{orun.LOSS}/{orun.REG_LOSS}"

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]


def eval_bd(nn):
    its, solved = [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = orun.USE_G
        s.anchor_strategy = STRAT
        path = s.search(max_iterations=MAX_ITERS)
        its.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    sv = [i for i, ok in zip(its, solved) if ok]
    return dict(solved=sum(solved), n=len(eval_boards),
                median=statistics.median(sv) if sv else float('nan'),
                mean=statistics.mean(sv) if sv else float('nan'),
                wall=time.time() - t0)


print(f"[RANKLOSS] === loss={LOSS_DESC} anchor={STRAT} seed={orun.SEED} "
      f"N_TRAIN={N_TRAIN} eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
bl = eval_bd(None)
print(f"[RANKLOSS] loss={LOSS_DESC} seed={orun.SEED} blind   "
      f"solved={bl['solved']}/{bl['n']} median={bl['median']:.1f} "
      f"mean={bl['mean']:.1f} ({bl['wall']:.0f}s)", flush=True)
le = eval_bd(model)
print(f"[RANKLOSS] loss={LOSS_DESC} seed={orun.SEED} learned "
      f"solved={le['solved']}/{le['n']} median={le['median']:.1f} "
      f"mean={le['mean']:.1f} ({le['wall']:.0f}s)", flush=True)
print(f"[RANKLOSS] DONE loss={LOSS_DESC} seed={orun.SEED}", flush=True)
