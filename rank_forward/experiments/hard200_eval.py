"""Evaluate our EXISTING saved checkpoints on the hard200 held-out set
(the supervisor's adopted secondary eval: the 200 hardest-by-blind-search
3-box instances). No retraining — loads the models saved during the
per-instance hard-box runs, so forward vs. bidirectional on genuinely hard
3-box instances is a load-and-eval, exactly what the checkpoint discipline is
for. Per-instance results are logged via runlog for survivorship metrics.

Checkpoints (trained on the same 1800-board prefix, full goal):
    hard_h2h_results/model_fwd_lstar.pt   forward L*   (eval in A*)
    hard_h2h_results/model_fwd_lgbfs.pt   forward Lgbfs(eval in GBFS)
    hard_h2h_results/model_bidir.pt       bidir reference (hindsight 32/128)
    hard_h2h_results/model_bidir_hs64.pt  bidir tuned hindsight (64/256)

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/hard200_eval.py
Env: HARD_MAX_ITERS(50000) HARD_KEEP(200)
"""
import os
import statistics
import time

import torch

from analysis import runlog
from game.getData import get_hard_eval_data
from game.domain import SokobanDomain
from learning.nn import build_model
from rank_forward.ranking_net import build_forward_model
from rank_forward.forward_run import evaluate
from search.AI_Bidirectional import BidirectionalF2FSearch

MAX_ITERS = int(os.environ.get("HARD_MAX_ITERS", "50000"))
HARD_KEEP = int(os.environ.get("HARD_KEEP", "200"))
CKPT_DIR = "hard_h2h_results"
FULL_GOAL = True   # the goal the bidirectional method targets; hard200 was
                   # ranked by blind bidirectional cost under this goal.

boards = get_hard_eval_data(keep=HARD_KEEP)
print(f"[HARD200] {len(boards)} hard instances, budget={MAX_ITERS}, "
      f"full_goal={FULL_GOAL}", flush=True)

BASE_CFG = {"domain": "sokoban", "eval_set": f"hard{HARD_KEEP}",
            "max_iters": MAX_ITERS, "full_goal": FULL_GOAL, "seed": 0}
results = {}


def log(method, iters, solved, run):
    s = runlog.record_eval(run, f"hard{HARD_KEEP}", iters, solved,
                           domain="sokoban", method=method, seed=0)
    results[method] = s
    print(f"[HARD200] {method:22s} solved={s['solved']}/{s['n']} "
          f"median={s['median'] if s['median'] is not None else float('nan'):.1f} "
          f"mean={s['mean'] if s['mean'] is not None else float('nan'):.1f}",
          flush=True)


def eval_bd(nn, boards):
    iters, solved = [], []
    for p in boards:
        s = BidirectionalF2FSearch(p, nn, domain=SokobanDomain())
        s.use_g_in_f = True
        s.meet_on_generate = True
        s.seam_repair = True
        s.bhffa_g = True
        s.dir_correct = True
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return iters, solved


# ── forward checkpoints ─────────────────────────────────────────────────────
for tag, ckpt, alg in (("fwd lstar/astar", "model_fwd_lstar.pt", "astar"),
                       ("fwd lgbfs/gbfs", "model_fwd_lgbfs.pt", "gbfs")):
    m = build_forward_model("smallcnn")
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt), map_location="cpu"))
    m.eval()
    run = runlog.run_dir("sokoban", tag.split()[1].replace("/", "_"), 0,
                         dict(BASE_CFG, method=tag))
    runlog.save_config(run, dict(BASE_CFG, method=tag, ckpt=ckpt))
    t0 = time.time()
    r = evaluate(m, boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
    log(tag, r["iters"], r["solved_flags"], run)
    print(f"           ({time.time()-t0:.0f}s)", flush=True)

# ── blind bidirectional (Manhattan) + our two bidir checkpoints ─────────────
run_bb = runlog.run_dir("sokoban", "bidir_blind", 0, dict(BASE_CFG, method="bidir_blind"))
runlog.save_config(run_bb, dict(BASE_CFG, method="bidir_blind"))
t0 = time.time()
it, sv = eval_bd(None, boards)
log("bidir blind", it, sv, run_bb)
print(f"           ({time.time()-t0:.0f}s)", flush=True)

for tag, ckpt in (("bidir learned", "model_bidir.pt"),
                  ("bidir hs64", "model_bidir_hs64.pt")):
    m = build_model("smallcnn", 32)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt), map_location="cpu"))
    m.eval()
    run = runlog.run_dir("sokoban", tag.replace(" ", "_"), 0,
                         dict(BASE_CFG, method=tag))
    runlog.save_config(run, dict(BASE_CFG, method=tag, ckpt=ckpt))
    t0 = time.time()
    it, sv = eval_bd(m, boards)
    log(tag, it, sv, run)
    print(f"           ({time.time()-t0:.0f}s)", flush=True)

# ── table ───────────────────────────────────────────────────────────────────
print(f"\n[HARD200] ===== hard{HARD_KEEP} (budget={MAX_ITERS}, full goal) =====",
      flush=True)
print(f"{'method':22s} {'solved':>9s} {'median':>9s} {'mean':>9s}", flush=True)
for tag in ("fwd lstar/astar", "fwd lgbfs/gbfs", "bidir blind",
            "bidir learned", "bidir hs64"):
    r = results.get(tag)
    if r:
        md = r["median"] if r["median"] is not None else float("nan")
        mn = r["mean"] if r["mean"] is not None else float("nan")
        print(f"{tag:22s} {r['solved']:3d}/{r['n']:<4d} {md:9.1f} {mn:9.1f}",
              flush=True)
print("[HARD200] DONE", flush=True)
