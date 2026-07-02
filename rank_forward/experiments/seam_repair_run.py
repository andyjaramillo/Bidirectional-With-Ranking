"""Wave 1a (APPROACH.md): meet-on-generate + post-hoc seam repair.

Trains the reference pipeline (MSE+margin+PATH_RANK) with MEET_ON_GENERATE=yes
SEAM_REPAIR=yes (set by the caller; online_run trains at import), then
evaluates the frozen model — and the blind baseline — on the held-out tail
under BOTH search configs:

  new    = meet_on_generate + seam_repair   (earliest detection, repaired path)
  legacy = meet-on-closed, parent-pointer path (the stored-reference protocol)

reporting node expansions (len(closed_f)+len(closed_b)) AND plan length
(moves = len(path)-1) per arm. Expansions measure the detection saving;
plan lengths measure the quality claim (repair should make the new config's
plans no worse — typically better — than legacy's).

Run one seed per process:

    for sd in 0 1 2; do
      MEET_ON_GENERATE=yes SEAM_REPAIR=yes SEED=$sd N_TOTAL=1800 NEVAL=200 \
        EVAL_MAX_ITERS=10000 PYTHONPATH=. \
        python -u rank_forward/experiments/seam_repair_run.py
    done
"""
import os, statistics, time
from game.getData import get_solvable_data
import learning.online_run as orun           # trains on-policy at import
from search.AI_Bidirectional import BidirectionalF2FSearch

TAG = f"meetgen={'y' if orun.MEET_ON_GENERATE else 'n'},repair={'y' if orun.SEAM_REPAIR else 'n'}"
N_EVAL = int(os.environ.get("NEVAL", "200"))
MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
N_TRAIN = orun.N_TOTAL
model = orun.search_model
model.eval()

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]


def eval_bd(nn, meet_on_gen, repair):
    its, lens, solved = [], [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = orun.USE_G
        s.anchor_strategy = orun.ANCHOR_STRATEGY
        s.meet_on_generate = meet_on_gen
        s.seam_repair = repair
        path = s.search(max_iterations=MAX_ITERS)
        its.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
        if path:
            lens.append(len(path) - 1)
    sv = [i for i, ok in zip(its, solved) if ok]
    return dict(solved=sum(solved), n=len(eval_boards),
                median=statistics.median(sv) if sv else float('nan'),
                mean=statistics.mean(sv) if sv else float('nan'),
                len_med=statistics.median(lens) if lens else float('nan'),
                len_mean=statistics.mean(lens) if lens else float('nan'),
                wall=time.time() - t0)


def report(label, r):
    print(f"[SEAM] train:{TAG} seed={orun.SEED} {label:16s} "
          f"solved={r['solved']}/{r['n']} median={r['median']:.1f} "
          f"mean={r['mean']:.1f} len_med={r['len_med']:.1f} "
          f"len_mean={r['len_mean']:.2f} ({r['wall']:.0f}s)", flush=True)


print(f"[SEAM] === train:{TAG} seed={orun.SEED} N_TRAIN={N_TRAIN} "
      f"eval={len(eval_boards)} MAX_ITERS={MAX_ITERS} ===", flush=True)
report("blind/new", eval_bd(None, True, True))
report("blind/legacy", eval_bd(None, False, False))
report("learned/new", eval_bd(model, True, True))
report("learned/legacy", eval_bd(model, False, False))
print(f"[SEAM] DONE train:{TAG} seed={orun.SEED}", flush=True)
