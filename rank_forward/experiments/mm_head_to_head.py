"""MM (Holte et al. 2016, meet-in-the-middle) head-to-head: adds the MM
baseline arms alongside the existing forward / bidirectional arms, on the
goal-diversity axis (fixed-goal tiles, random-goal tiles, hard200 Sokoban).

Arms (named by search direction + heuristic, per protocol):
    MM, blind (h=0)         — the true MM0 brute-force baseline
    MM, Manhattan           — front-to-end analytic (pairwise evaluateBoard)
    MM, learned (F2E)       — optional: a FROZEN pairwise bidir checkpoint
                              queried front-to-end (h_F(n)=nn(n->goal),
                              h_B(n)=nn(start->flip(n)), directed convention)
    Forward, blind (h=0) A* — in-run deterministic reference
    Bidirectional, Manhattan— in-run deterministic reference
The learned forward / bidirectional arms are NOT retrained here — read them
from runlog (they were persisted by the existing head-to-head drivers); this
driver only adds MM rows measured under the identical protocol (same eval
instances, same budget, same expansion unit, failures at spent budget).

MM's "solved" = the optimality PROOF fires (MM is an optimal algorithm; the
proof cost is the comparison point). Instances where MM found a meeting but
could not prove it within budget are counted UNSOLVED and reported separately
(met_unproven).

Run from repo root:
    PYTHONPATH=. python rank_forward/experiments/mm_head_to_head.py
Env: MM_DOMAIN(tilesRG4)   tilesRG4|tilesRG5|tiles5|tiles6|tiles7|
                           sokoban (= hard200)|sokoban_samegoal
     MM_TESTSET(test)      tiles suites: test | testhard (L~U[60,140])
     MM_ARMS(blind,manhattan,learned)   which MM arms to run
     MM_MAX_ITERS(10000; 50000 when MM_DOMAIN=sokoban)
     MM_NEVAL(200) SEED(0) MM_REFS(yes)
     MM_BIDIR_CKPT           path to a state_dict for the learned arm.
                             Defaults: sokoban -> hard_h2h_results/
                             model_bidir.pt; otherwise the newest
                             runs/<domain>/bidir_learned/seed<SEED>_*/model.pt
                             (tiles6/7 fall back to the tiles5 model — the
                             fixed-goal tiles net, evaluated across sizes).
"""
import glob
import os
import time

import numpy as np

from analysis import runlog
from game.getData import DATA_DIR, get_hard_eval_data
from game.domain import get_domain, SokobanDomain
from rank_forward.forward_run import evaluate
from search.AI_Bidirectional import BidirectionalF2FSearch
from search.mm_search import MMSearch, mm_analytic_heuristics, mm_learned_heuristics

DOMAIN_NAME = os.environ.get("MM_DOMAIN", "tilesRG4")
IS_SOKOBAN = DOMAIN_NAME.lower() in ("sokoban", "sok")
IS_SAMEGOAL = DOMAIN_NAME.lower() == "sokoban_samegoal"
TILES_SUITE = os.environ.get("MM_TESTSET", "test")
MAX_ITERS = int(os.environ.get("MM_MAX_ITERS", "50000" if IS_SOKOBAN else "10000"))
N_EVAL = int(os.environ.get("MM_NEVAL", "200"))
SEED = int(os.environ.get("SEED", "0"))
RUN_REFS = os.environ.get("MM_REFS", "yes").lower() in ("yes", "true", "1")
ARMS = [a.strip() for a in
        os.environ.get("MM_ARMS", "blind,manhattan,learned").split(",") if a.strip()]


def default_ckpt(log_domain):
    if IS_SOKOBAN:
        return os.path.join("hard_h2h_results", "model_bidir.pt")
    # Newest saved bidir checkpoint for this domain; fixed-goal tiles6/7 have
    # no own checkpoint — they evaluate the tiles5-trained net (size axis).
    for d in (log_domain, "tiles5" if log_domain.startswith("tiles") else None):
        if d is None:
            continue
        hits = sorted(glob.glob(os.path.join(runlog.RUNS_DIR, d, "bidir_learned",
                                             f"seed{SEED}_*", "model.pt")),
                      key=os.path.getmtime)
        if hits:
            return hits[-1]
    return ""


def load_instances():
    """(instances, domain, log_domain, testset_name, full_goal-for-fwd-ref)."""
    if IS_SOKOBAN:
        boards = get_hard_eval_data(keep=200)[:N_EVAL]
        return boards, SokobanDomain(), "sokoban", "hard200", True
    if IS_SAMEGOAL:
        path = os.path.join(DATA_DIR, "sokoban_samegoal_test.txt")
        with open(path) as f:
            boards = [np.array([int(v) for v in line.split()]).reshape(10, 10)
                      for line in f if line.strip()][:N_EVAL]
        return boards, SokobanDomain(), "sokoban_samegoal", "samegoal_test", True
    dom = get_domain(DOMAIN_NAME)
    n = dom.n
    path = os.path.join(DATA_DIR, f"{dom.name}_{TILES_SUITE}.txt")
    per_line = 2 * n * n if DOMAIN_NAME.lower().startswith("tilesrg") else n * n
    out = []
    with open(path) as f:
        for line in f:
            v = [int(x) for x in line.split()]
            if len(v) != per_line:
                continue
            if per_line == n * n:
                out.append(np.array(v).reshape(n, n))
            else:
                out.append((np.array(v[:n * n]).reshape(n, n),
                            np.array(v[n * n:]).reshape(n, n)))
            if len(out) == N_EVAL:
                break
    return out, dom, dom.name, f"{dom.name}_{TILES_SUITE}", False


instances, DOM, LOG_DOMAIN, TESTSET, FULL_GOAL = load_instances()
BIDIR_CKPT = os.environ.get("MM_BIDIR_CKPT") or default_ckpt(LOG_DOMAIN)
print(f"[MM-H2H] domain={LOG_DOMAIN} eval={len(instances)} budget={MAX_ITERS} "
      f"testset={TESTSET} arms={ARMS} seed={SEED}", flush=True)

BASE_CFG = {"domain": LOG_DOMAIN, "eval_set": TESTSET, "max_iters": MAX_ITERS,
            "seed": SEED, "full_goal": FULL_GOAL}
results = {}


def log(label, method, iters, solved, extra=None):
    run = runlog.run_dir(LOG_DOMAIN, method, SEED, dict(BASE_CFG, method=method))
    runlog.save_config(run, dict(BASE_CFG, method=method))
    s = runlog.record_eval(run, TESTSET, iters, solved,
                           domain=LOG_DOMAIN, method=method, seed=SEED,
                           extra=extra)
    results[label] = s
    md = s["median"] if s["median"] is not None else float("nan")
    mn = s["mean"] if s["mean"] is not None else float("nan")
    note = f"  met_unproven={extra['met_unproven']}" if extra else ""
    print(f"[MM-H2H] {label:26s} solved={s['solved']}/{s['n']} "
          f"median={md:.1f} mean={mn:.1f}{note}", flush=True)


def eval_mm(make_h, insts):
    """make_h: None (blind) or callable(search) -> (heuristic_f, heuristic_b).
    Solved iters = expansions at the optimality proof; unsolved = spent budget
    (failures counted at cap downstream, like every other arm)."""
    iters, solved, met_unproven = [], [], 0
    for inst in insts:
        s = MMSearch(inst, domain=DOM)
        if make_h is not None:
            s.heuristic_f, s.heuristic_b = make_h(s)
        path = s.search(max_iterations=MAX_ITERS)
        if path is not None:
            iters.append(s.first_solved_iter)
            solved.append(True)
        else:
            iters.append(s.expansions)
            solved.append(False)
            if s.U < float("inf"):
                met_unproven += 1
    return iters, solved, met_unproven


# ── MM arms ─────────────────────────────────────────────────────────────────
for label, method, make_h in (
        ("MM, blind (h=0)", "mm_blind", None),
        ("MM, Manhattan", "mm_manhattan", mm_analytic_heuristics)):
    if method.split("_", 1)[1] not in ARMS:
        continue
    t0 = time.time()
    it, sv, met = eval_mm(make_h, instances)
    log(label, method, it, sv, extra={"met_unproven": met})
    print(f"           ({time.time()-t0:.0f}s)", flush=True)

if "learned" in ARMS and BIDIR_CKPT and os.path.exists(BIDIR_CKPT):
    import torch
    from learning.nn import build_model
    m = build_model("smallcnn", 32, **DOM.model_kwargs())
    payload = torch.load(BIDIR_CKPT, map_location="cpu")
    m.load_state_dict(payload.get("state_dict", payload))
    m.eval()
    print(f"[MM-H2H] learned arm ckpt: {BIDIR_CKPT}", flush=True)
    t0 = time.time()
    it, sv, met = eval_mm(lambda s: mm_learned_heuristics(m, s), instances)
    log("MM, learned (F2E)", "mm_learned", it, sv,
        extra={"met_unproven": met, "ckpt": BIDIR_CKPT})
    print(f"           ({time.time()-t0:.0f}s)", flush=True)
elif "learned" in ARMS:
    print(f"[MM-H2H] no checkpoint found (MM_BIDIR_CKPT={BIDIR_CKPT!r}) — "
          f"skipping 'MM, learned (F2E)'", flush=True)

# ── deterministic references (same protocol, for an in-run table) ───────────
if RUN_REFS:
    t0 = time.time()
    r = evaluate(None, instances, alg="astar", max_iters=MAX_ITERS,
                 full_goal=FULL_GOAL, domain=DOM)
    log("Forward, blind (h=0) A*", "fwd_blind_mmref", r["iters"], r["solved_flags"])
    print(f"           ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    iters, solved = [], []
    for inst in instances:
        s = BidirectionalF2FSearch(inst, None, domain=DOM)
        s.use_g_in_f = True
        path = s.search(max_iterations=MAX_ITERS)
        iters.append(len(s.closed_f) + len(s.closed_b))
        solved.append(path is not None)
    log("Bidirectional, Manhattan", "bidir_manhattan_mmref", iters, solved)
    print(f"           ({time.time()-t0:.0f}s)", flush=True)

# ── table ───────────────────────────────────────────────────────────────────
print(f"\n[MM-H2H] ===== {LOG_DOMAIN} / {TESTSET} (budget={MAX_ITERS}) =====",
      flush=True)
print(f"{'arm':26s} {'solved':>9s} {'median':>9s} {'mean':>9s}", flush=True)
for label, r in results.items():
    md = r["median"] if r["median"] is not None else float("nan")
    mn = r["mean"] if r["mean"] is not None else float("nan")
    print(f"{label:26s} {r['solved']:3d}/{r['n']:<4d} {md:9.1f} {mn:9.1f}",
          flush=True)
print("[MM-H2H] learned forward/bidirectional rows: read them from runlog "
      "(runs/index.jsonl) — they are persisted by the existing head-to-head "
      "drivers; never retrained to remeasure.", flush=True)
print("[MM-H2H] DONE", flush=True)
