"""Random-goal sliding-tile head-to-head: forward (Chrestien) vs bidirectional,
on (start, goal) pairs with an ARBITRARY goal per instance. Independent check of
the supervisor hypothesis (2026-07-10): the forward method's sliding-tile win is
an artifact of the single canonical goal; with random goals the bidirectional
method should regain its edge.

Both methods train on the SAME first HH_NTRAIN pairs of tilesRG{n}_train, use
the SAME SmallCNN tower + displacement encoding, same budget, same expansion
unit, same goal (each instance's own goal). Persisted via runlog.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/tiles_rg_head_to_head.py
Env: RG_N(4) HH_NTRAIN(1400) HH_NEVAL(200) HH_FWD_STEPS(12000)
     HH_SOLVE_CAP(100000) HH_MAX_ITERS(10000) SEED(0)
"""
import os
import statistics
import time

import numpy as np

from analysis import runlog
from game.getData import DATA_DIR
from game.domain import get_domain

N = int(os.environ.get("RG_N", "4"))
MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "1400"))
N_EVAL = int(os.environ.get("HH_NEVAL", "200"))
FWD_STEPS = int(os.environ.get("HH_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("HH_SOLVE_CAP", "100000"))
SEED = int(os.environ.get("SEED", "0"))
DOM = get_domain(f"tilesRG{N}")


def load_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            v = [int(x) for x in line.split()]
            if len(v) != 2 * N * N:
                continue
            out.append((np.array(v[:N * N]).reshape(N, N),
                        np.array(v[N * N:]).reshape(N, N)))
    return out


all_train = load_pairs(os.path.join(DATA_DIR, f"tilesRG{N}_train.txt"))
train_pairs = all_train[:N_TRAIN]
eval_pairs = load_pairs(os.path.join(DATA_DIR, f"tilesRG{N}_test.txt"))[:N_EVAL]
print(f"[RG-H2H] n={N}: train {len(train_pairs)} pairs, eval {len(eval_pairs)} "
      f"(RANDOM goal per instance), budget={MAX_ITERS}", flush=True)

BASE_CFG = {"domain": f"tilesRG{N}", "n_train": N_TRAIN, "max_iters": MAX_ITERS,
            "seed": SEED, "goal": "random_per_instance"}
results = {}


def log(method, iters, solved, run):
    s = runlog.record_eval(run, f"tilesRG{N}_test", iters, solved,
                           domain=f"tilesRG{N}", method=method, seed=SEED)
    results[method] = s
    md = s["median"] if s["median"] is not None else float("nan")
    mn = s["mean"] if s["mean"] is not None else float("nan")
    print(f"[RG-H2H] {method:20s} solved={s['solved']}/{s['n']} "
          f"median={md:.1f} mean={mn:.1f}", flush=True)


# ── FORWARD (optimal labels) ────────────────────────────────────────────────
print("\n[RG-H2H] ===== FORWARD training =====", flush=True)
from rank_forward.dataset import build_train_instances
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate

t0 = time.time()
instances = build_train_instances(train_pairs, SOLVE_CAP, False,
                                  cache_path=None, full_goal=False, domain=DOM)
print(f"[RG-H2H] {len(instances)} optimal instances ({time.time()-t0:.0f}s)",
      flush=True)

fwd_models = {}
for loss, alg in (("lstar", "astar"), ("lgbfs", "gbfs")):
    m = build_forward_model("smallcnn", **DOM.model_kwargs())
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
    m.eval()
    run = runlog.run_dir(f"tilesRG{N}", f"fwd_{loss}", SEED,
                         dict(BASE_CFG, method=f"fwd_{loss}"))
    runlog.save_config(run, dict(BASE_CFG, method=f"fwd_{loss}"))
    runlog.save_model(run, m)
    fwd_models[(loss, alg)] = (m, run)
    print(f"[RG-H2H] forward {loss} trained", flush=True)

# ── BIDIRECTIONAL (online, reference defaults) ──────────────────────────────
print(f"\n[RG-H2H] ===== BIDIRECTIONAL training (online, DOMAIN=tilesRG{N}) =====",
      flush=True)
os.environ["DOMAIN"] = f"tilesRG{N}"
os.environ["N_TOTAL"] = str(N_TRAIN)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()
run_bd = runlog.run_dir(f"tilesRG{N}", "bidir_learned", SEED,
                        dict(BASE_CFG, method="bidir_learned"))
runlog.save_config(run_bd, dict(BASE_CFG, method="bidir_learned"))
runlog.save_model(run_bd, model_bd)


def eval_bd(nn, pairs):
    iters, solved = [], []
    for p in pairs:
        s = BidirectionalF2FSearch(p, nn, domain=DOM)
        s.use_g_in_f = True
        s.meet_on_generate = orun.MEET_ON_GENERATE
        s.seam_repair = orun.SEAM_REPAIR
        s.bhffa_g = orun.BHFFA_G
        s.dir_correct = orun.DIRECTED
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return iters, solved


# ── FROZEN EVALS ────────────────────────────────────────────────────────────
run_fb = runlog.run_dir(f"tilesRG{N}", "fwd_blind", SEED, dict(BASE_CFG, method="fwd_blind"))
runlog.save_config(run_fb, dict(BASE_CFG, method="fwd_blind"))
r = evaluate(None, eval_pairs, alg="astar", max_iters=MAX_ITERS, full_goal=False, domain=DOM)
log("fwd blind A*", r["iters"], r["solved_flags"], run_fb)
for (loss, alg), (m, run) in fwd_models.items():
    r = evaluate(m, eval_pairs, alg=alg, max_iters=MAX_ITERS, full_goal=False, domain=DOM)
    log(f"fwd {loss}/{alg}", r["iters"], r["solved_flags"], run)

run_bb = runlog.run_dir(f"tilesRG{N}", "bidir_blind", SEED, dict(BASE_CFG, method="bidir_blind"))
runlog.save_config(run_bb, dict(BASE_CFG, method="bidir_blind"))
it, sv = eval_bd(None, eval_pairs); log("bidir Manhattan", it, sv, run_bb)
it, sv = eval_bd(model_bd, eval_pairs); log("bidir learned", it, sv, run_bd)

# ── TABLE ───────────────────────────────────────────────────────────────────
print(f"\n[RG-H2H] ===== RANDOM-GOAL TABLE (n={N}, budget={MAX_ITERS}) =====",
      flush=True)
print(f"{'method':20s} {'solved':>9s} {'median':>9s} {'mean':>9s}", flush=True)
for tag in ("fwd blind A*", "fwd lstar/astar", "fwd lgbfs/gbfs",
            "bidir Manhattan", "bidir learned"):
    r = results.get(tag)
    if r:
        md = r["median"] if r["median"] is not None else float("nan")
        mn = r["mean"] if r["mean"] is not None else float("nan")
        print(f"{tag:20s} {r['solved']:3d}/{r['n']:<4d} {md:9.1f} {mn:9.1f}",
              flush=True)
print("[RG-H2H] DONE", flush=True)
