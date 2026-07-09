"""Evaluate saved checkpoints on the HARD held-out set (build_hard_eval.py).

Loads frozen models saved by revwalk_run.py (MODEL_OUT) — does NOT retrain
(on-policy training is not run-to-run reproducible, so the gated model must be
the evaluated model). Reports blind + each checkpoint on the hard set.

    CKPTS="/tmp/revwalk_models/base_seed0.pt:/tmp/revwalk_models/revwalk_seed0.pt" \
        PYTHONPATH=. python -u rank_forward/experiments/hard_eval.py
"""
import os, statistics, time
import torch
from game.getData import get_hard_eval_data
from learning.nn import build_model
from search.AI_Bidirectional import BidirectionalF2FSearch

MAX_ITERS = int(os.environ.get("EVAL_MAX_ITERS", "10000"))
HARD_KEEP = int(os.environ.get("HARD_KEEP", "200"))
MODEL = os.environ.get("MODEL", "smallcnn")
MODEL_CHANNELS = int(os.environ.get("MODEL_CHANNELS", "32"))
CKPTS = [c for c in os.environ.get(
    "CKPTS",
    "/tmp/revwalk_models/base_seed0.pt:/tmp/revwalk_models/revwalk_seed0.pt"
).split(":") if c]

eval_boards = get_hard_eval_data(keep=HARD_KEEP)


def eval_bd(nn):
    its, lens, solved = [], [], []
    t0 = time.time()
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)   # adopted defaults
        s.use_g_in_f = True
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
    print(f"[HARD] {label:24s} solved={r['solved']}/{r['n']} "
          f"median={r['median']:.1f} mean={r['mean']:.1f} "
          f"len_mean={r['len_mean']:.2f} wall={r['wall']:.0f}s "
          f"({r['wall']*1000/r['n']:.0f}ms/solve)", flush=True)


print(f"[HARD] === hard{HARD_KEEP} eval  MAX_ITERS={MAX_ITERS} "
      f"ckpts={len(CKPTS)} ===", flush=True)
report("blind", eval_bd(None))
for ck in CKPTS:
    model = build_model(MODEL, MODEL_CHANNELS)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    report(os.path.basename(ck), eval_bd(model))
print("[HARD] DONE", flush=True)
