"""Same-goal Sokoban head-to-head: forward (Chrestien) vs bidirectional, on a
dataset where every instance shares ONE fixed goal (walls + targets + player
home from analysis/build_sokoban_samegoal.py) and differs only in the start.

Mirror of the random-goal tile experiment. Prediction (supervisor): fixing the
goal lets the forward heuristic specialize to it (as in fixed-goal tiles), so
forward should CATCH UP to or beat bidirectional — completing the story that
GOAL DIVERSITY, not the domain, is the axis: bidirectional+learning wins with
diverse/hard-to-reach goals (standard Sokoban, random-goal tiles), forward wins
with a single fixed goal (fixed-goal tiles, and — tested here — same-goal Sokoban).

Both train on the same first HH_NTRAIN starts (optimal labels for forward,
on-policy for bidir), full goal (boxes on targets AND player home — fixed across
all instances), evaluated frozen on the held-out same-goal test starts. Persisted
via runlog.

Run: PYTHONPATH=. python rank_forward/experiments/sokoban_samegoal_head_to_head.py
Env: HH_NTRAIN(700) HH_FWD_STEPS(12000) HH_SOLVE_CAP(300000) HH_MAX_ITERS(10000)
"""
import os
import time

import numpy as np

from analysis import runlog
from game.getData import DATA_DIR

MAX_ITERS = int(os.environ.get("HH_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("HH_NTRAIN", "700"))
FWD_STEPS = int(os.environ.get("HH_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("HH_SOLVE_CAP", "300000"))
SEED = int(os.environ.get("SEED", "0"))
FULL_GOAL = True
TRAIN_FILE = os.path.join(DATA_DIR, "sokoban_samegoal_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "sokoban_samegoal_test.txt")


def load(path):
    with open(path) as f:
        return [np.array([int(v) for v in line.split()]).reshape(10, 10)
                for line in f if line.strip()]


train_boards = load(TRAIN_FILE)[:N_TRAIN]
eval_boards = load(TEST_FILE)
print(f"[SG-H2H] same-goal Sokoban: train {len(train_boards)}, eval "
      f"{len(eval_boards)} (ONE fixed goal), budget={MAX_ITERS}", flush=True)
BASE_CFG = {"domain": "sokoban_samegoal", "n_train": N_TRAIN,
            "max_iters": MAX_ITERS, "seed": SEED, "goal": "single_fixed"}
results = {}


def log(method, iters, solved, run):
    s = runlog.record_eval(run, "samegoal_test", iters, solved,
                           domain="sokoban_samegoal", method=method, seed=SEED)
    results[method] = s
    md = s["median"] if s["median"] is not None else float("nan")
    mn = s["mean"] if s["mean"] is not None else float("nan")
    print(f"[SG-H2H] {method:20s} solved={s['solved']}/{s['n']} "
          f"median={md:.1f} mean={mn:.1f}", flush=True)


# ── FORWARD (optimal labels, single fixed goal) ─────────────────────────────
print("\n[SG-H2H] ===== FORWARD training =====", flush=True)
from rank_forward.dataset import build_train_instances
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate

t0 = time.time()
instances = build_train_instances(train_boards, SOLVE_CAP, False,
                                  cache_path=None, full_goal=FULL_GOAL)
print(f"[SG-H2H] {len(instances)} optimal instances ({time.time()-t0:.0f}s)",
      flush=True)
fwd_models = {}
for loss, alg in (("lstar", "astar"), ("lgbfs", "gbfs")):
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=SEED)
    m.eval()
    run = runlog.run_dir("sokoban_samegoal", f"fwd_{loss}", SEED,
                         dict(BASE_CFG, method=f"fwd_{loss}"))
    runlog.save_config(run, dict(BASE_CFG, method=f"fwd_{loss}"))
    runlog.save_model(run, m)
    fwd_models[(loss, alg)] = (m, run)
    print(f"[SG-H2H] forward {loss} trained", flush=True)

# ── BIDIRECTIONAL (online, same-goal train file) ────────────────────────────
print("\n[SG-H2H] ===== BIDIRECTIONAL training (online, same-goal) =====",
      flush=True)
os.environ["SOKOBAN_TRAIN_FILE"] = TRAIN_FILE
os.environ["N_TOTAL"] = str(N_TRAIN)
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()
run_bd = runlog.run_dir("sokoban_samegoal", "bidir_learned", SEED,
                        dict(BASE_CFG, method="bidir_learned"))
runlog.save_config(run_bd, dict(BASE_CFG, method="bidir_learned"))
runlog.save_model(run_bd, model_bd)


def eval_bd(nn):
    iters, solved = [], []
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
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
run_fb = runlog.run_dir("sokoban_samegoal", "fwd_blind", SEED, dict(BASE_CFG, method="fwd_blind"))
runlog.save_config(run_fb, dict(BASE_CFG, method="fwd_blind"))
r = evaluate(None, eval_boards, alg="astar", max_iters=MAX_ITERS, full_goal=FULL_GOAL)
log("fwd blind A*", r["iters"], r["solved_flags"], run_fb)
for (loss, alg), (m, run) in fwd_models.items():
    r = evaluate(m, eval_boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
    log(f"fwd {loss}/{alg}", r["iters"], r["solved_flags"], run)
run_bb = runlog.run_dir("sokoban_samegoal", "bidir_blind", SEED, dict(BASE_CFG, method="bidir_blind"))
runlog.save_config(run_bb, dict(BASE_CFG, method="bidir_blind"))
it, sv = eval_bd(None); log("bidir Manhattan", it, sv, run_bb)
it, sv = eval_bd(model_bd); log("bidir learned", it, sv, run_bd)

# ── TABLE ───────────────────────────────────────────────────────────────────
print(f"\n[SG-H2H] ===== SAME-GOAL SOKOBAN TABLE (budget={MAX_ITERS}) =====",
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
print("[SG-H2H] DONE", flush=True)
