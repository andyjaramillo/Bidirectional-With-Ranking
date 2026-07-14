"""One G value of the goal-variability sweep: all arms trained + evaluated on a
goal-pool tilesRG dataset (pool size G). Run one per subprocess (online_run
trains at import). Results go to runlog under domain 'tilesG{n}', testset
'G{G}'. The orchestrator tiles_gsweep.py loops G and assembles the table.

Arms (reproducing the supervisor's G-sweep; 'bidir h=0' omitted — our F2F has
no h=0 mode, nn=None is Manhattan):
  fwd h=0            forward A*, no heuristic (uninformed)
  fwd Manhattan      forward A*, analytic Manhattan
  fwd L* offline     forward L*/A*, optimal offline labels (Chrestien)
  fwd L* bootstrap   forward L*/A*, self-supervised on own solves (fair)
  bidir Manhattan    our F2F, analytic Manhattan (null)
  bidir learned      our F2F, on-policy learned (ours)

Env: RG_N(4) GSWEEP_G(1) GSWEEP_NTRAIN(1000) GSWEEP_NEVAL(200)
     GSWEEP_MAXITERS(10000) FWD_STEPS(8000) SOLVE_CAP(100000)
     FWD_K(8) FWD_WARMUP(50) FWD_BUFFER(300) SEED(0)
"""
import os
import random
import statistics
from collections import deque

import numpy as np
import torch

from analysis import runlog
from analysis.build_tiles_goalpool import make_goalpool_pairs, write_pairs
from game.getData import DATA_DIR
from game.domain import get_domain

N = int(os.environ.get("RG_N", "4"))
G = int(os.environ.get("GSWEEP_G", "1"))
NTRAIN = int(os.environ.get("GSWEEP_NTRAIN", "1000"))
NEVAL = int(os.environ.get("GSWEEP_NEVAL", "200"))
MAX_ITERS = int(os.environ.get("GSWEEP_MAXITERS", "10000"))
FWD_STEPS = int(os.environ.get("FWD_STEPS", "8000"))
SOLVE_CAP = int(os.environ.get("SOLVE_CAP", "100000"))
SEED = int(os.environ.get("SEED", "0"))
DOM = get_domain(f"tilesRG{N}")
GTAG = "inf" if G >= NTRAIN else str(G)
LOGDOM = f"tilesG{N}"
TESTSET = f"G{GTAG}"
CFG = {"domain": LOGDOM, "G": GTAG, "n_train": NTRAIN, "max_iters": MAX_ITERS,
       "seed": SEED}
print(f"[GSWEEP] N={N} G={GTAG} ntrain={NTRAIN} neval={NEVAL} budget={MAX_ITERS}",
      flush=True)

# ── data: goal pool of size G ────────────────────────────────────────────────
train_pairs = make_goalpool_pairs(N, G, NTRAIN, seed=SEED)
eval_pairs = make_goalpool_pairs(N, G, NEVAL, seed=SEED * 100 + 1)
# online_run (bidir learned) consumes the tilesRG train file
write_pairs(train_pairs, os.path.join(DATA_DIR, f"tilesRG{N}_train.txt"), N)


def logrow(method, iters, solved):
    run = runlog.run_dir(LOGDOM, method, SEED, dict(CFG, method=method))
    runlog.save_config(run, dict(CFG, method=method))
    s = runlog.record_eval(run, TESTSET, iters, solved,
                           domain=LOGDOM, method=method, seed=SEED)
    md = s["median"] if s["median"] is not None else float("nan")
    print(f"[GSWEEP] G={GTAG} {method:18s} solved={s['solved']}/{s['n']} "
          f"median={md:.1f}", flush=True)


from rank_forward.forward_search import (ForwardSearch, manhattan_heuristic,
                                         model_heuristic)
from rank_forward.ranking_net import build_forward_model
from rank_forward.trajectory import instance_from_path
from rank_forward.dataset import build_train_instances
from rank_forward.trainer import train
from rank_forward.losses import instance_loss
from rank_forward.forward_run import evaluate
from search.AI_Bidirectional import BidirectionalF2FSearch


def eval_fwd_manhattan(pairs):
    iters, solved = [], []
    for p in pairs:
        s = ForwardSearch(p, heuristic=None, use_g_in_f=True,
                          full_goal=False, domain=DOM)
        s.heuristic = manhattan_heuristic(s.game)
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(s.first_solved_iter if path is not None else s.iteration)
        solved.append(path is not None)
    return iters, solved


def eval_bd(nn, pairs):
    iters, solved = [], []
    for p in pairs:
        s = BidirectionalF2FSearch(p, nn, domain=DOM)
        s.use_g_in_f = True
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return iters, solved


# ── analytic / uninformed arms (no training) ────────────────────────────────
r = evaluate(None, eval_pairs, alg="astar", max_iters=MAX_ITERS,
             full_goal=False, domain=DOM)
logrow("fwd h=0", r["iters"], r["solved_flags"])
it, sv = eval_fwd_manhattan(eval_pairs); logrow("fwd Manhattan", it, sv)
it, sv = eval_bd(None, eval_pairs); logrow("bidir Manhattan", it, sv)

# ── forward L* offline (optimal labels) ─────────────────────────────────────
insts = build_train_instances(train_pairs, SOLVE_CAP, False, cache_path=None,
                              full_goal=False, domain=DOM)
m_off = build_forward_model("smallcnn", **DOM.model_kwargs())
train(insts, m_off, "lstar", steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
m_off.eval()
r = evaluate(m_off, eval_pairs, alg="astar", max_iters=MAX_ITERS,
             full_goal=False, domain=DOM)
logrow("fwd L* offline", r["iters"], r["solved_flags"])

# ── forward L* bootstrap (own satisficing solves) ───────────────────────────
FWD_K = int(os.environ.get("FWD_K", "8"))
FWD_WARMUP = int(os.environ.get("FWD_WARMUP", "50"))
FWD_BUFFER = int(os.environ.get("FWD_BUFFER", "300"))
torch.manual_seed(SEED); np.random.seed(SEED); frng = random.Random(SEED)
m_boot = build_forward_model("smallcnn", **DOM.model_kwargs())
bopt = torch.optim.Adam(m_boot.parameters(), lr=1e-3)
buf = deque(maxlen=FWD_BUFFER)
for pz in train_pairs:
    learned = len(buf) >= FWD_WARMUP
    s = ForwardSearch(pz, heuristic=None, use_g_in_f=True, full_goal=False, domain=DOM)
    m_boot.eval()
    s.heuristic = (model_heuristic(m_boot, s.target, s.goal_ctx) if learned
                   else manhattan_heuristic(s.game))
    path = s.search(max_iterations=MAX_ITERS)
    if path is not None:
        bi = instance_from_path(pz, path, full_goal=False, domain=DOM)
        if bi is not None:
            buf.append(bi)
    if len(buf) >= FWD_WARMUP:
        m_boot.train()
        for _ in range(FWD_K):
            bi = frng.choice(buf); bopt.zero_grad()
            loss = instance_loss(m_boot, bi, "lstar", reduction="sum")
            loss.backward(); bopt.step()
m_boot.eval()
r = evaluate(m_boot, eval_pairs, alg="astar", max_iters=MAX_ITERS,
             full_goal=False, domain=DOM)
logrow("fwd L* bootstrap", r["iters"], r["solved_flags"])

# ── bidir learned (ours, online_run on the goal-pool stream) ─────────────────
os.environ["DOMAIN"] = f"tilesRG{N}"
os.environ["N_TOTAL"] = str(NTRAIN)
os.environ["MAX_ITERS"] = str(MAX_ITERS)
os.environ["SKIP_BASELINE"] = "yes"
import learning.online_run as orun
model_bd = orun.search_model
model_bd.eval()
it, sv = eval_bd(model_bd, eval_pairs); logrow("bidir learned", it, sv)
print(f"[GSWEEP] G={GTAG} DONE", flush=True)
