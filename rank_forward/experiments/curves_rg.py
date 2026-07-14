"""Online learning-curves benchmark on random-goal sliding-tile — a port of the
supervisor's SlidingPuzzleLocal protocol onto the Domain abstraction, for
independent reproduction.

Protocol (NO train/test split): three arms consume the IDENTICAL sequence of
N random-goal (start, goal) instances; each instance is solved with the arm's
CURRENT state (so it is held-out at that moment), logged (solved, expansions,
plan_len), then mined for training. Comparison = learning curves over
WINDOW-instance windows (solve% and budget-inclusive median expansions, with
failures counted at the MAX_ITERS cap).

Arms:
  bidir   — our bidirectional F2F, on-policy (learning.online_run, adopted
            defaults) — the learned method.
  fwd     — Chrestien forward ranking, BOOTSTRAP (its own satisficing A*/L*
            solves; forward_bootstrap.py's loop, generalized to the domain) —
            the fair on-policy baseline (no optimal labels).
  manhattan — bidirectional with the analytic Manhattan heuristic, no learning
            — the NULL MODEL (the analytic incumbent). NOTE: this is NOT h=0
            "blind"; Manhattan is a strong admissible heuristic, which is
            precisely why it is a hard null to beat.

Writes per-instance CSVs (curves_<arm>_seed{S}.csv) + a windowed table.

Run: PYTHONPATH=. python rank_forward/experiments/curves_rg.py
Env: RG_N(4) CURVES_N(4000) WINDOW(500) MAX_ITERS(10000) SEED(0)
     FWD_K(8) FWD_WARMUP(50) FWD_BUFFER(300) FWD_LR(1e-3)
"""
import csv
import os
import random
import statistics
import time
from collections import deque

import numpy as np
import torch

RG_N = int(os.environ.get("RG_N", "4"))
CURVES_N = int(os.environ.get("CURVES_N", "4000"))
WINDOW = int(os.environ.get("WINDOW", "500"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "10000"))
SEED = int(os.environ.get("SEED", "0"))
OUT = "curves_rg_results"
os.makedirs(OUT, exist_ok=True)

from game.getData import DATA_DIR
from game.domain import get_domain
from game.SlidingTileGame import SlidingTileGame

DOM = get_domain(f"tilesRG{RG_N}")
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def gen_stream(n, seed):
    """N random-goal (start, goal) pairs — the shared instance sequence."""
    rng = random.Random(seed)
    solved = SlidingTileGame.solved_board(RG_N)

    def scr(board, L):
        b = board.copy(); r, c = [int(x) for x in np.argwhere(b == 0)[0]]; prev = None
        for _ in range(L):
            opts = [d for d in _DIRS if 0 <= r + d[0] < RG_N and 0 <= c + d[1] < RG_N
                    and (prev is None or d != (-prev[0], -prev[1]))]
            d = rng.choice(opts); nr, nc = r + d[0], c + d[1]
            b[r, c] = b[nr, nc]; b[nr, nc] = 0; r, c, prev = nr, nc, d
        return b
    out = []
    while len(out) < n:
        goal = scr(solved, rng.randint(30, 60))
        start = scr(goal, rng.randint(10, 50))
        if not np.array_equal(start, goal):
            out.append((start, goal))
    return out


def windows(records):
    """records: list of (n, solved, expansions, plan). Returns rows of
    (window_end, solve_pct, budget_incl_median)."""
    rows = []
    for i in range(0, len(records), WINDOW):
        w = records[i:i + WINDOW]
        if not w:
            continue
        solv = sum(1 for _n, ok, _e, _p in w if ok)
        # budget-inclusive expansions: cap for failures
        exp = [e if ok else MAX_ITERS for _n, ok, e, _p in w]
        rows.append((i + len(w), 100.0 * solv / len(w), statistics.median(exp)))
    return rows


def write_csv(arm, records):
    with open(os.path.join(OUT, f"curves_{arm}_seed{SEED}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["n", "solved", "expansions", "plan_len"])
        wr.writerows(records)


print(f"[CURVES] n={RG_N} instances={CURVES_N} window={WINDOW} budget={MAX_ITERS} "
      f"seed={SEED}", flush=True)
stream = gen_stream(CURVES_N, SEED)
# write the stream so online_run (bidir arm) consumes the SAME sequence
stream_path = os.path.join(DATA_DIR, f"tilesRG{RG_N}_train.txt")
with open(stream_path, "w") as f:
    for start, goal in stream:
        f.write(" ".join(str(int(v)) for v in start.reshape(-1)) + " "
                + " ".join(str(int(v)) for v in goal.reshape(-1)) + "\n")

all_records = {}

# ── BLIND (null model): Manhattan bidirectional, no learning ────────────────
from search.AI_Bidirectional import BidirectionalF2FSearch
print("\n[CURVES] arm=manhattan (Manhattan bidirectional, null model) ...", flush=True)
t0 = time.time(); rec = []
for n, inst in enumerate(stream):
    s = BidirectionalF2FSearch(inst, None, domain=DOM); s.use_g_in_f = True
    path = s.search(max_iterations=MAX_ITERS)
    rec.append((n, path is not None, len(s.closed_f) + len(s.closed_b),
                (len(path) - 1) if path else -1))
all_records["manhattan"] = rec; write_csv("manhattan", rec)
print(f"[CURVES] manhattan done ({time.time()-t0:.0f}s)", flush=True)

# ── FORWARD bootstrap (fair on-policy baseline) ─────────────────────────────
from rank_forward.forward_search import ForwardSearch, manhattan_heuristic, model_heuristic
from rank_forward.ranking_net import build_forward_model
from rank_forward.trajectory import instance_from_path
from rank_forward.losses import instance_loss
print("\n[CURVES] arm=fwd (forward bootstrap, A*/L*) ...", flush=True)
FWD_K = int(os.environ.get("FWD_K", "8"))
FWD_WARMUP = int(os.environ.get("FWD_WARMUP", "50"))
FWD_BUFFER = int(os.environ.get("FWD_BUFFER", "300"))
FWD_LR = float(os.environ.get("FWD_LR", "1e-3"))
torch.manual_seed(SEED); np.random.seed(SEED); frng = random.Random(SEED)
fmodel = build_forward_model("smallcnn", **DOM.model_kwargs())
fopt = torch.optim.Adam(fmodel.parameters(), lr=FWD_LR)
buf = deque(maxlen=FWD_BUFFER); rec = []; t0 = time.time()
for n, inst in enumerate(stream):
    learned = len(buf) >= FWD_WARMUP
    s = ForwardSearch(inst, heuristic=None, use_g_in_f=True, full_goal=False, domain=DOM)
    fmodel.eval()
    s.heuristic = (model_heuristic(fmodel, s.target, s.goal_ctx) if learned
                   else manhattan_heuristic(s.game))
    path = s.search(max_iterations=MAX_ITERS)
    rec.append((n, path is not None,
                s.first_solved_iter if path is not None else s.iteration,
                (len(path) - 1) if path else -1))
    if path is not None:
        bi = instance_from_path(inst, path, full_goal=False, domain=DOM)
        if bi is not None:
            buf.append(bi)
    if len(buf) >= FWD_WARMUP:
        fmodel.train()
        for _ in range(FWD_K):
            bi = frng.choice(buf); fopt.zero_grad()
            loss = instance_loss(fmodel, bi, "lstar", reduction="sum")
            loss.backward(); fopt.step()
    if (n + 1) % 1000 == 0:
        print(f"[CURVES]   fwd n={n+1} solved_so_far="
              f"{sum(1 for r in rec if r[1])} ({time.time()-t0:.0f}s)", flush=True)
all_records["fwd"] = rec; write_csv("fwd", rec)
print(f"[CURVES] fwd done ({time.time()-t0:.0f}s)", flush=True)

# ── BIDIR learned (our method) via online_run ───────────────────────────────
print("\n[CURVES] arm=bidir (our bidirectional, online_run) ...", flush=True)
os.environ["DOMAIN"] = f"tilesRG{RG_N}"
os.environ["N_TOTAL"] = str(CURVES_N)
os.environ["MAX_ITERS"] = str(MAX_ITERS)
os.environ["SEED"] = str(SEED)
os.environ["SKIP_BASELINE"] = "yes"   # curves has its own blind arm
import learning.online_run as orun
all_records["bidir"] = orun.curve_records
write_csv("bidir", orun.curve_records)

# ── TABLE ───────────────────────────────────────────────────────────────────
print(f"\n[CURVES] ===== LEARNING CURVES (solve% / budget-incl median), "
      f"window={WINDOW} =====", flush=True)
wr = {arm: windows(rec) for arm, rec in all_records.items()}
ends = [r[0] for r in wr["manhattan"]]
hdr = f"{'n':>6s} " + " ".join(f"{a:>18s}" for a in ("bidir", "fwd", "manhattan"))
print(hdr, flush=True)
for i, end in enumerate(ends):
    cells = []
    for a in ("bidir", "fwd", "manhattan"):
        if i < len(wr[a]):
            _e, pct, med = wr[a][i]
            cells.append(f"{pct:5.0f}% / {med:8.0f}")
        else:
            cells.append(" " * 18)
    print(f"{end:>6d} " + " ".join(f"{c:>18s}" for c in cells), flush=True)
print("[CURVES] DONE", flush=True)
