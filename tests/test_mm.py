"""Smoke test for the MM baseline (search/mm_search.py).

Checks, on a handful of tilesRG4 and Sokoban instances:
  1. MM0 (blind) and MM-Manhattan solve and PROVE within a small budget;
  2. the returned path is step-valid (every consecutive pair is one legal
     forward move) and runs start -> goal;
  3. the solution cost equals Manhattan-A*'s optimal cost on the SAME problem
     (admissible h => optimal; Sokoban: full goal = the backward seed state,
     i.e. the exact problem MM/bidirectional solve) — MM with admissible h is
     optimal too, so the costs must match.

Tiles instances are SHALLOW scrambles of goals from the test file (blind MM
must stay cheap for a smoke test); Sokoban uses the standard solvable set.

Run: PYTHONPATH=. python tests/test_mm.py   (or pytest tests/test_mm.py)
"""
import os
import random

import numpy as np

from game.domain import get_domain, SokobanDomain
from game.getData import DATA_DIR, get_solvable_data
from rank_forward.forward_search import ForwardSearch, manhattan_heuristic
from search.mm_search import MMSearch, mm_analytic_heuristics


def _assert_valid_path(search: MMSearch, path):
    """Path is forward-frame encoded maps; verify endpoints + step validity."""
    fg = search.forward_game
    boards = [fg.decodeMap(k) for k in path]
    assert fg.fullStateKey(boards[0]) == fg.fullStateKey(search.puzzle), "path must start at start"
    assert fg.fullStateKey(boards[-1]) == fg.fullStateKey(search.goal_fwd), "path must end at goal"
    for a, b in zip(boards, boards[1:]):
        fg.puzzle = a
        loc = fg.getPlayerLocation(a)
        succs = []
        for _dir, action in fg.availableStates(loc):
            child = action.moveAndUpdateBoard(loc, a)
            if child is not None:
                succs.append(fg.fullStateKey(child))
        assert fg.fullStateKey(b) in succs, "consecutive path states must be one legal move apart"


def _run_instance(inst, domain, optimal_cost, budget=50000):
    for tag, use_h in (("MM, blind (h=0)", False), ("MM, Manhattan", True)):
        s = MMSearch(inst, domain=domain)
        if use_h:
            s.heuristic_f, s.heuristic_b = mm_analytic_heuristics(s)
        path = s.search(max_iterations=budget)
        assert path is not None, f"{tag}: unsolved within {budget}"
        assert s.first_solved_iter is not None and s.first_meeting_iter is not None
        assert s.first_meeting_iter <= s.first_solved_iter
        _assert_valid_path(s, path)
        assert len(path) - 1 == s.U, f"{tag}: path length != U"
        assert s.U == optimal_cost, (f"{tag}: cost {s.U} != optimal {optimal_cost}")
        print(f"    {tag:18s} cost={s.U} met@{s.first_meeting_iter} "
              f"proved@{s.first_solved_iter} exp(f/b)={s.fwd.expansions}/{s.bwd.expansions}")


def _scramble(board, moves, rng):
    b = board.copy()
    r, c = [int(x) for x in np.argwhere(b == 0)[0]]
    prev = None
    for _ in range(moves):
        opts = [d for d in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + d[0] < b.shape[0] and 0 <= c + d[1] < b.shape[1]
                and (prev is None or d != (-prev[0], -prev[1]))]
        d = rng.choice(opts)
        nr, nc = r + d[0], c + d[1]
        b[r, c] = b[nr, nc]
        b[nr, nc] = 0
        r, c, prev = nr, nc, d
    return b


def test_mm_tiles_rg():
    dom = get_domain("tilesRG4")
    rng = random.Random(0)
    goals = []
    with open(os.path.join(DATA_DIR, "tilesRG4_test.txt")) as f:
        for line in f:
            v = [int(x) for x in line.split()]
            if len(v) == 32:
                goals.append(np.array(v[16:]).reshape(4, 4))
            if len(goals) == 5:
                break
    pairs = [(_scramble(g, rng.randint(6, 12), rng), g) for g in goals]
    print(f"[MM smoke] tilesRG4: {len(pairs)} instances (shallow scrambles)")
    for i, inst in enumerate(pairs):
        ref = ForwardSearch(inst, heuristic=None, use_g_in_f=True, domain=dom)
        ref.heuristic = manhattan_heuristic(ref.game)
        ref_path = ref.search(max_iterations=200000)
        assert ref_path is not None
        optimal = len(ref_path) - 1
        print(f"  instance {i}: optimal={optimal}")
        _run_instance(inst, dom, optimal)


def test_mm_sokoban():
    dom = SokobanDomain()
    boards = get_solvable_data(limit=3)
    print(f"[MM smoke] sokoban: {len(boards)} instances (full goal)")
    for i, b in enumerate(boards):
        # Same problem as MM/bidirectional: reach the exact backward seed.
        ref = ForwardSearch(b, heuristic=None, use_g_in_f=True,
                            full_goal=True, domain=dom)
        ref.heuristic = manhattan_heuristic(ref.game)
        ref_path = ref.search(max_iterations=200000)
        assert ref_path is not None
        optimal = len(ref_path) - 1
        print(f"  instance {i}: optimal={optimal}")
        _run_instance(b, dom, optimal)


if __name__ == "__main__":
    test_mm_tiles_rg()
    test_mm_sokoban()
    print("[MM smoke] ALL OK")
