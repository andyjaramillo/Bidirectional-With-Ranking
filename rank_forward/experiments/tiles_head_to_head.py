"""Sliding-tile head-to-head (domain #2): forward ranking (Chrestien et al.)
vs bidirectional F2F, trained on 5x5 scrambles and evaluated frozen on the
5x5 / 6x6 / 7x7 test sets — the tile analog of hard_head_to_head.py.

Both methods train on the SAME first HH_NTRAIN boards of tiles5_train, use
the SAME SmallCNN tower with the domain's size-invariant 3-channel
displacement encoding (one net evaluates all board sizes), the same 10k-node
budget, the same expansion unit. Goal = the exact solved board (for tiles the
classical goal IS the full goal — no player-pinning subtlety), so
full_goal=False on the forward side matches the bidirectional target.

EVERYTHING IS PERSISTED via analysis/runlog.py: per-method run dirs with
config + model weights, and per-instance eval JSONs (runs/tiles5/...), so any
later metric — including the both-solved intersection — reads JSON, never
retrains.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/tiles_head_to_head.py
Env: HH_NTRAIN(1800) HH_FWD_STEPS(12000) HH_SOLVE_CAP(100000)
     HH_MAX_ITERS(10000) HH_SIZES("5,6,7") SEED(0)
"""
import os
import statistics
import time

import numpy as np

from analysis import runlog
from game.getData import DATA_DIR
from game.domain import get_domain

MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "1800"))
FWD_STEPS = int(os.environ.get("HH_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("HH_SOLVE_CAP", "100000"))
SIZES = [int(s) for s in os.environ.get("HH_SIZES", "5,6,7").split(",")]
# Test suites: comma-separated file suffixes; "test" = the original
# L~U[10,60] sets, "testhard" = L~U[60,140] (closer to the difficulty regime
# of the paper's Orseau&Lelis-derived instances).
SUITES = os.environ.get("HH_TESTSETS", "test,testhard").split(",")
SEED = int(os.environ.get("SEED", "0"))

DOM5 = get_domain("tiles5")
BASE_CFG = {"domain": "tiles5", "n_train": N_TRAIN, "max_iters": MAX_ITERS,
            "solve_cap": SOLVE_CAP, "fwd_steps": FWD_STEPS, "seed": SEED,
            "encoding": "displacement3ch_fixedscale10"}


def load_boards(path, n):
    with open(path) as f:
        return [np.array([int(v) for v in line.split()]).reshape(n, n)
                for line in f if line.strip()]


train_boards = load_boards(os.path.join(DATA_DIR, "tiles5_train.txt"), 5)[:N_TRAIN]
test_sets = {(suite, n): load_boards(
                  os.path.join(DATA_DIR, f"tiles{n}_{suite}.txt"), n)
             for suite in SUITES for n in SIZES}
print(f"[TILES-H2H] train on {len(train_boards)} 5x5 boards; suites: "
      + ", ".join(f"{s}:{n}x{n}×{len(b)}" for (s, n), b in test_sets.items())
      + f"; budget={MAX_ITERS}", flush=True)

results = {}


def record(method, suite, nb, iters, solved, run):
    s = runlog.record_eval(run, f"tiles{nb}_{suite}", iters, solved,
                           domain="tiles5", method=method, seed=SEED)
    results[(method, suite, nb)] = s
    print(f"[TILES-H2H] {method} {suite} {nb}x{nb}: solved={s['solved']}/{s['n']} "
          f"median={s['median'] if s['median'] is not None else float('nan'):.1f} "
          f"mean={s['mean'] if s['mean'] is not None else float('nan'):.1f}",
          flush=True)


# ═══════════════════ PHASE 1 — TRAINING (5x5 only) ═════════════════════════
print("\n[TILES-H2H] ===== FORWARD training (the paper) =====", flush=True)
from rank_forward.dataset import build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate
import rank_forward.config as C

ck = cache_key(C.CACHE_DIR, N_TRAIN, 0, SOLVE_CAP, False, full_goal=False,
               domain_name="tiles5")
t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=ck, full_goal=False, domain=DOM5)
print(f"[TILES-H2H] {len(instances)} optimal instances ({time.time()-t0:.0f}s)",
      flush=True)

fwd_models = {}
for loss, alg in (("lstar", "astar"), ("lgbfs", "gbfs")):
    m = build_forward_model("smallcnn", **DOM5.model_kwargs())
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
    m.eval()
    fwd_models[(loss, alg)] = m
    cfg = dict(BASE_CFG, method=f"fwd_{loss}", search=alg,
               labels="optimal_astar_manhattan")
    run = runlog.run_dir("tiles5", f"fwd_{loss}", SEED, cfg)
    runlog.save_config(run, cfg)
    runlog.save_model(run, m)
    fwd_models[(loss, alg)] = (m, run)
    print(f"[TILES-H2H] forward {loss} trained -> {run}", flush=True)

print(f"\n[TILES-H2H] ===== BIDIRECTIONAL training (online_run, "
      f"DOMAIN=tiles5, current reference defaults) =====", flush=True)
os.environ["DOMAIN"] = "tiles5"
os.environ["N_TOTAL"] = str(N_TRAIN)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()
cfg_bd = dict(BASE_CFG, method="bidir_learned", config_defaults="reference",
              hindsight=True)
run_bd = runlog.run_dir("tiles5", "bidir_learned", SEED, cfg_bd)
runlog.save_config(run_bd, cfg_bd)
runlog.save_model(run_bd, model_bd)
print(f"[TILES-H2H] bidirectional trained -> {run_bd}", flush=True)

# ═══════════════ PHASE 2 — FROZEN EVALS on 5/6/7 ═══════════════════════════
run_fb = runlog.run_dir("tiles5", "fwd_blind", SEED, dict(BASE_CFG, method="fwd_blind"))
runlog.save_config(run_fb, dict(BASE_CFG, method="fwd_blind"))
for suite in SUITES:
    for nb in SIZES:
        dom_nb = get_domain(f"tiles{nb}")
        r = evaluate(None, test_sets[(suite, nb)], alg="astar",
                     max_iters=MAX_ITERS, full_goal=False, domain=dom_nb)
        record("fwd blind A*", suite, nb, r["iters"], r["solved_flags"], run_fb)

for (loss, alg), (m, run) in fwd_models.items():
    for suite in SUITES:
        for nb in SIZES:
            dom_nb = get_domain(f"tiles{nb}")
            r = evaluate(m, test_sets[(suite, nb)], alg=alg, max_iters=MAX_ITERS,
                         full_goal=False, domain=dom_nb)
            record(f"fwd {loss}/{alg}", suite, nb, r["iters"], r["solved_flags"], run)


def eval_bd(nn, boards, dom):
    iters, solved = [], []
    for p in boards:
        s = BidirectionalF2FSearch(p, nn, domain=dom)
        s.use_g_in_f = True
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = orun.DIRECTED
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return iters, solved


run_bb = runlog.run_dir("tiles5", "bidir_blind", SEED, dict(BASE_CFG, method="bidir_blind"))
runlog.save_config(run_bb, dict(BASE_CFG, method="bidir_blind"))
for name, nn, run in (("bidir blind", None, run_bb),
                      ("bidir learned", model_bd, run_bd)):
    for suite in SUITES:
        for nb in SIZES:
            t1 = time.time()
            iters, solved = eval_bd(nn, test_sets[(suite, nb)],
                                    get_domain(f"tiles{nb}"))
            record(name, suite, nb, iters, solved, run)

# ═══════════════════════════════ TABLE ═════════════════════════════════════
methods = ["fwd blind A*", "fwd lstar/astar", "fwd lgbfs/gbfs",
           "bidir blind", "bidir learned"]
for suite in SUITES:
    print(f"\n[TILES-H2H] ===== TABLE ({suite}; trained on 5x5, "
          f"budget={MAX_ITERS}) =====", flush=True)
    print(f"{'method':18s} " + " ".join(f"{f'{n}x{n}':>21s}" for n in SIZES),
          flush=True)
    for name in methods:
        cells = []
        for nb in SIZES:
            r = results.get((name, suite, nb))
            if r and r["median"] is not None:
                cells.append(f"{r['solved']:3d}/{r['n']:<3d} {r['median']:5.0f} "
                             f"{r['mean']:7.1f}")
            elif r:
                cells.append(f"{r['solved']:3d}/{r['n']:<3d}   nan     nan")
            else:
                cells.append(" " * 21)
        print(f"{name:18s} " + " ".join(f"{c:>21s}" for c in cells), flush=True)
print("[TILES-H2H] DONE", flush=True)
