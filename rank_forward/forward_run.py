"""End-to-end driver: build data, train a ranking heuristic, evaluate forward
A*/GBFS on a held-out split, and report node-expansion metrics comparable to
learning/online_run.py.

Run from the repo root:
    PYTHONPATH=. python rank_forward/forward_run.py
Knobs are env vars (see rank_forward/config.py), e.g.
    LOSS=lgbfs ALG=gbfs N_TOTAL=2000 STEPS=20000 PYTHONPATH=. python rank_forward/forward_run.py
"""
import os
import statistics
import time
from typing import List, Optional

import numpy as np

from game.SokobanGame import SokobanGame
from rank_forward.forward_search import ForwardSearch, model_heuristic
from rank_forward.ranking_net import build_forward_model
from rank_forward import config as C


def evaluate(model: Optional[object], boards: List[np.ndarray], alg: str = "astar",
             max_iters: int = 10000, use_deadlock: bool = False,
             full_goal: bool = False, domain=None) -> dict:
    """Run forward search over ``boards``. ``model=None`` is the blind (h=0)
    baseline. Returns per-puzzle expansion counts, solved flags, and summaries.
    For solved puzzles the recorded count is ``first_solved_iter`` (expansions
    until the goal is popped); for unsolved it is the budget spent."""
    use_g = (alg.lower() == "astar")
    iters: List[int] = []
    solved: List[bool] = []
    plan_len: List[Optional[int]] = []
    t0 = time.time()
    for b in boards:
        # Build the searcher first so its goal_ctx (the real goal — player@start
        # when full_goal) is known, then point the heuristic at that SAME goal.
        s = ForwardSearch(b, heuristic=None, use_g_in_f=use_g,
                          use_deadlock=use_deadlock, full_goal=full_goal,
                          domain=domain)
        if model is not None:
            s.heuristic = model_heuristic(model, s.target, s.goal_ctx)
        path = s.search(max_iterations=max_iters)
        if path is not None:
            iters.append(s.first_solved_iter)
            solved.append(True)
            plan_len.append(len(path) - 1)
        else:
            iters.append(s.iteration)
            solved.append(False)
            plan_len.append(None)
    n_solved = sum(solved)
    solved_iters = [it for it, ok in zip(iters, solved) if ok]
    return {
        "iters": iters,
        "solved_flags": solved,
        "plan_len": plan_len,
        "solved": n_solved,
        "n": len(boards),
        "mean_iters": statistics.mean(solved_iters) if solved_iters else float("nan"),
        "median_iters": statistics.median(solved_iters) if solved_iters else float("nan"),
        "wall": time.time() - t0,
    }


def _speedup(base: dict, learned: dict):
    """Mean/median speedup over puzzles solved by BOTH base and learned."""
    xs = []
    for bi, bok, li, lok in zip(base["iters"], base["solved_flags"],
                                learned["iters"], learned["solved_flags"]):
        if bok and lok and li > 0:
            xs.append(bi / li)
    if not xs:
        return float("nan"), float("nan"), 0
    return statistics.mean(xs), statistics.median(xs), len(xs)


def main():
    print("=== rank_forward: unidirectional ranking-heuristic baseline ===")
    print(f"config: N_TOTAL={C.N_TOTAL} N_EVAL={C.N_EVAL} MODEL={C.MODEL} "
          f"LOSS={C.LOSS} ALG={C.ALG} STEPS={C.STEPS} LR={C.LR} "
          f"MAX_ITERS={C.MAX_ITERS} USE_DEADLOCK={C.USE_DEADLOCK} "
          f"REDUCTION={C.REDUCTION} LOCAL_PAIRS={C.LOCAL_PAIRS}")

    from rank_forward.dataset import load_split, build_train_instances, cache_key
    from rank_forward.trainer import train

    train_boards, eval_boards = load_split(C.N_TOTAL, C.N_EVAL)
    print(f"\n[data] train={len(train_boards)} eval={len(eval_boards)}")

    print("[data] building/loading optimal training trajectories …")
    ck = cache_key(C.CACHE_DIR, C.N_TOTAL, C.N_EVAL, C.SOLVE_CAP, C.USE_DEADLOCK,
                   C.FULL_GOAL)
    t0 = time.time()
    instances = build_train_instances(train_boards, C.SOLVE_CAP, C.USE_DEADLOCK,
                                      cache_path=ck, full_goal=C.FULL_GOAL)
    print(f"[data] {len(instances)} solved instances ready "
          f"({time.time()-t0:.0f}s), cache={ck}")

    model = build_forward_model(C.MODEL, channels=C.MODEL_CHANNELS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] {C.MODEL}  params={n_params:,}")

    ckpt = os.path.join(C.CKPT_DIR, f"{C.MODEL}_{C.LOSS}.pt")
    print(f"[train] loss={C.LOSS} steps={C.STEPS} lr={C.LR} -> {ckpt}")
    train(instances, model, C.LOSS, steps=C.STEPS, lr=C.LR,
          weight_decay=C.WEIGHT_DECAY, reduction=C.REDUCTION,
          local_pairs=C.LOCAL_PAIRS, seed=C.SEED, eval_every=C.EVAL_EVERY,
          ckpt_every=C.CKPT_EVERY, ckpt_path=ckpt, eval_boards=eval_boards,
          eval_alg=C.ALG, eval_max_iters=C.MAX_ITERS, eval_subset=C.EVAL_SUBSET)
    model.eval()

    # ── Evaluation: blind baseline + learned heuristic in A* and GBFS ──────
    print("\n[eval] blind baseline (forward A*, h=0) …")
    blind = evaluate(None, eval_boards, alg="astar", max_iters=C.MAX_ITERS,
                     use_deadlock=C.USE_DEADLOCK, full_goal=C.FULL_GOAL)
    _report("blind A*", blind)

    results = {}
    for alg in ("astar", "gbfs"):
        print(f"\n[eval] learned heuristic ({C.LOSS}) in forward {alg} …")
        r = evaluate(model, eval_boards, alg=alg, max_iters=C.MAX_ITERS,
                     use_deadlock=C.USE_DEADLOCK, full_goal=C.FULL_GOAL)
        results[alg] = r
        _report(f"{alg} / {C.LOSS}", r)
        xm, xmd, ncommon = _speedup(blind, r)
        print(f"    speedup vs blind A* (n={ncommon} solved-by-both): "
              f"x_mean={xm:.2f} x_median={xmd:.2f}")

    print("\n=== summary ===")
    print(f"  loss={C.LOSS} model={C.MODEL} params={n_params:,} "
          f"train_instances={len(instances)} eval={len(eval_boards)}")
    print(f"  blind A*         solved={blind['solved']}/{blind['n']} "
          f"mean={blind['mean_iters']:.1f} median={blind['median_iters']:.1f}")
    for alg in ("astar", "gbfs"):
        r = results[alg]
        print(f"  {C.LOSS:>7s}/{alg:<5s} solved={r['solved']}/{r['n']} "
              f"mean={r['mean_iters']:.1f} median={r['median_iters']:.1f}")


def _report(name: str, r: dict):
    print(f"    {name}: solved={r['solved']}/{r['n']}  "
          f"mean_iters={r['mean_iters']:.1f}  median_iters={r['median_iters']:.1f}  "
          f"wall={r['wall']:.1f}s")


if __name__ == "__main__":
    main()
