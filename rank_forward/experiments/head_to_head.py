"""Direct head-to-head: forward unidirectional ranking heuristic (the paper)
vs our bidirectional F2F, on the SAME held-out puzzles, the SAME goal, and the
SAME node-expansion unit. Reports ABSOLUTE expansions — the two baselines
differ, so speedup-vs-own-baseline is not comparable across methods; absolute
expansions on identical instances is.

FAIRNESS: by default (H2H_FULL_GOAL=1) BOTH methods solve the identical full
goal — boxes-on-targets AND the player back at its start cell. That is the goal
the bidirectional method always targets (its backward search is seeded at the
start cell), so this is the apples-to-apples setting. Set H2H_FULL_GOAL=0 to let
the forward side use the easier classical goal (player anywhere) for reference;
that is NOT a fair comparison (the forward search may stop as soon as the boxes
land, while the bidirectional method also returns the player home).

Metric: forward = first_solved_iter; bidirectional = first_meeting_iter (both =
productive node expansions until solved). State hashing is full-board
(player-aware) on both sides.

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/head_to_head.py
Env: H2H_NTRAIN(1800) H2H_NEVAL(200) H2H_FWD_STEPS(12000) H2H_SOLVE_CAP(300000)
     H2H_FULL_GOAL(1) H2H_MAX_ITERS(10000)
"""
import os
import statistics
import time

from game.getData import get_solvable_data

MAX_ITERS = int(os.environ.get("H2H_MAX_ITERS", "10000"))
N_TRAIN = int(os.environ.get("H2H_NTRAIN", "1800"))
N_EVAL = int(os.environ.get("H2H_NEVAL", "200"))
FWD_STEPS = int(os.environ.get("H2H_FWD_STEPS", "12000"))
SOLVE_CAP = int(os.environ.get("H2H_SOLVE_CAP", "300000"))
FULL_GOAL = os.environ.get("H2H_FULL_GOAL", "1").lower() in ("1", "true", "yes", "y", "on")
# Bidirectional meet-on-generation (default on): the bidirectional method's best
# config — detects the seam as soon as both frontiers generate the shared state,
# cutting ~11% of expansions wasted past a valid seam. Applies to BOTH the
# bidirectional training (via online_run) and its eval here.
MEET_ON_GEN = os.environ.get("H2H_MEET_ON_GEN", "1").lower() in ("1", "true", "yes", "y", "on")

boards = get_solvable_data(limit=N_TRAIN + N_EVAL)
eval_boards = boards[N_TRAIN:N_TRAIN + N_EVAL]
print(f"[H2H] eval on {len(eval_boards)} held-out puzzles "
      f"(boards[{N_TRAIN}:{N_TRAIN+N_EVAL}]), full_goal={FULL_GOAL} "
      f"bidir_meet_on_generate={MEET_ON_GEN}", flush=True)


def summ(iters, solved):
    sv = [it for it, ok in zip(iters, solved) if ok]
    return {"iters": iters, "solved_flags": solved, "solved": sum(solved),
            "n": len(iters), "mean": statistics.mean(sv) if sv else float("nan"),
            "median": statistics.median(sv) if sv else float("nan")}


def common_speedup(a, b):
    """median/mean of a/b over puzzles solved by both."""
    xs = [ai / bi for ai, aok, bi, bok in
          zip(a["iters"], a["solved_flags"], b["iters"], b["solved_flags"])
          if aok and bok and bi > 0]
    return (statistics.mean(xs) if xs else float("nan"),
            statistics.median(xs) if xs else float("nan"), len(xs))


# ─────────────────────────── FORWARD (the paper) ───────────────────────────
print("\n[H2H] ===== FORWARD side =====", flush=True)
from rank_forward.dataset import build_train_instances, cache_key
from rank_forward.ranking_net import build_forward_model
from rank_forward.trainer import train
from rank_forward.forward_run import evaluate
import rank_forward.config as C

ck = cache_key(C.CACHE_DIR, N_TRAIN + N_EVAL, N_EVAL, SOLVE_CAP, False, full_goal=FULL_GOAL)
t0 = time.time()
instances = build_train_instances(boards[:N_TRAIN], SOLVE_CAP, False,
                                  cache_path=ck, full_goal=FULL_GOAL)
print(f"[H2H] {len(instances)} optimal instances ({time.time()-t0:.0f}s)", flush=True)

fwd_blind = evaluate(None, eval_boards, alg="astar", max_iters=MAX_ITERS, full_goal=FULL_GOAL)
print(f"[H2H] forward blind A*: solved={fwd_blind['solved']}/{fwd_blind['n']} "
      f"median={fwd_blind['median_iters']:.1f} mean={fwd_blind['mean_iters']:.1f}", flush=True)

fwd_results = {}
for loss in ("lgbfs", "lstar"):
    m = build_forward_model("smallcnn")
    train(instances, m, loss, steps=FWD_STEPS, lr=1e-3, reduction="sum", seed=0)
    m.eval()
    for alg in ("astar", "gbfs"):
        r = evaluate(m, eval_boards, alg=alg, max_iters=MAX_ITERS, full_goal=FULL_GOAL)
        fwd_results[(loss, alg)] = r
        print(f"[H2H] forward {loss}/{alg}: solved={r['solved']}/{r['n']} "
              f"median={r['median_iters']:.1f} mean={r['mean_iters']:.1f}", flush=True)


# ─────────────────────── BIDIRECTIONAL (ours) ──────────────────────────────
# Train on the first N_TRAIN by importing online_run (it trains at module level),
# then freeze orun.search_model and evaluate on the held-out split. The
# bidirectional method always targets boxes-on-targets + player@start.
print(f"\n[H2H] ===== BIDIRECTIONAL side (training on first {N_TRAIN} via online_run) =====",
      flush=True)
os.environ["N_TOTAL"] = str(N_TRAIN)
os.environ["MEET_ON_GENERATE"] = "yes" if MEET_ON_GEN else "no"
import learning.online_run as orun
from search.AI_Bidirectional import BidirectionalF2FSearch

model_bd = orun.search_model
model_bd.eval()


def eval_bd(nn):
    iters, solved = [], []
    for p in eval_boards:
        s = BidirectionalF2FSearch(p, nn)
        s.use_g_in_f = True
        s.meet_on_generate = MEET_ON_GEN
        path = s.search(max_iterations=MAX_ITERS)
        # Expansion count = total nodes whose successors were generated, summed
        # over BOTH frontiers = len(closed_f)+len(closed_b). This is exactly the
        # same unit as the forward searcher's len(closed)/first_solved_iter.
        # NB: we deliberately do NOT use first_meeting_iter — verified to equal
        # (expansions - 1) because it is read mid-step before the meeting-finding
        # expansion is counted — so the two methods' counts are exactly consistent.
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    return summ(iters, solved)


t1 = time.time()
bd_blind = eval_bd(None)
print(f"[H2H] bidir blind: solved={bd_blind['solved']}/{bd_blind['n']} "
      f"median={bd_blind['median']:.1f} mean={bd_blind['mean']:.1f} ({time.time()-t1:.0f}s)",
      flush=True)
t1 = time.time()
bd_learn = eval_bd(model_bd)
print(f"[H2H] bidir learned: solved={bd_learn['solved']}/{bd_learn['n']} "
      f"median={bd_learn['median']:.1f} mean={bd_learn['mean']:.1f} ({time.time()-t1:.0f}s)",
      flush=True)


# ─────────────────────────────── TABLE ─────────────────────────────────────
print(f"\n[H2H] ============= HEAD-TO-HEAD (same {N_EVAL} instances, "
      f"full_goal={FULL_GOAL}) =============", flush=True)
print(f"{'method':28s} {'solved':>9s} {'median_exp':>11s} {'mean_exp':>10s}", flush=True)


def fr(name, r, med_key, mean_key):
    print(f"{name:28s} {r['solved']:3d}/{r['n']:<5d} {r[med_key]:11.1f} {r[mean_key]:10.1f}",
          flush=True)


fr("forward blind A* (h=0)", fwd_blind, "median_iters", "mean_iters")
for loss in ("lgbfs", "lstar"):
    for alg in ("astar", "gbfs"):
        fr(f"forward {loss} / {alg}", fwd_results[(loss, alg)], "median_iters", "mean_iters")
print("", flush=True)
fr("bidirectional blind (F2F)", bd_blind, "median", "mean")
fr("bidirectional learned", bd_learn, "median", "mean")

best_fwd = min((fwd_results[(l, a)] for l in ("lgbfs", "lstar") for a in ("astar", "gbfs")),
               key=lambda r: r["median_iters"])
xm, xmd, nc = common_speedup(
    {"iters": fwd_results[("lstar", "astar")]["iters"],
     "solved_flags": fwd_results[("lstar", "astar")]["solved_flags"]},
    {"iters": bd_learn["iters"], "solved_flags": bd_learn["solved_flags"]})
print(f"\n[H2H] best forward learned: median={best_fwd['median_iters']:.1f} "
      f"solved={best_fwd['solved']}/{N_EVAL}", flush=True)
print(f"[H2H] forward-lstar/astar vs bidir-learned over both-solved (n={nc}): "
      f"forward/bidir median ratio={xmd:.2f} mean={xm:.2f}", flush=True)
print("[H2H] DONE", flush=True)
