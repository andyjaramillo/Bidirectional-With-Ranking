"""Direction-correct labels & queries (quasimetric fix; APPROACH.md).

Trains the reference pipeline with DIRECTED=yes (set by the caller; online_run
trains at import): direction-correct pair labels (no symmetric duplication,
random-pair budget doubled to match throughput) and backward-frontier queries
h(anchor_fwd, flip(node)) in the forward frame. Evaluates the frozen model on
the held-out tail with directed queries ON (headline) and OFF (decomposition:
same net, legacy backward-frame queries), plus the blind baseline (unaffected
by the flag — logged once for the record).

Run one seed per process:

    for sd in 0 1 2; do
      DIRECTED=yes SEED=$sd N_TOTAL=1800 NEVAL=200 EVAL_MAX_ITERS=10000 \
        PYTHONPATH=. python -u rank_forward/experiments/directed_run.py
    done
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

TAG = f"directed={'y' if orun.DIRECTED else 'n'}"
N_EVAL = int(os.environ.get("NEVAL", "200"))
MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
N_TRAIN = orun.N_TOTAL
model = orun.search_model
model.eval()

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]


def eval_bd(nn, dir_correct):
    its, lens, solved = [], [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = orun.USE_G
        s.anchor_strategy = orun.ANCHOR_STRATEGY
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = dir_correct
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
    print(f"[DIRECTED] train:{TAG} seed={orun.SEED} {label:14s} "
          f"solved={r['solved']}/{r['n']} median={r['median']:.1f} "
          f"mean={r['mean']:.1f} len_mean={r['len_mean']:.2f} "
          f"({r['wall']:.0f}s)", flush=True)


print(f"[DIRECTED] === train:{TAG} seed={orun.SEED} N_TRAIN={N_TRAIN} "
      f"eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
report("blind", eval_bd(None, False))
report("learned/dirq", eval_bd(model, True))
report("learned/legq", eval_bd(model, False))
print(f"[DIRECTED] DONE train:{TAG} seed={orun.SEED}", flush=True)
