"""Same-goal Sokoban: ONE fixed goal (wall + target layout), many different
start states. Tests the mirror of the random-goal tile experiment — does fixing
the Sokoban goal (so the forward heuristic can specialize to it, as in
fixed-goal tiles) let forward search catch up to bidirectional?

A start = a state backward-reachable from the goal with all boxes OFF the
targets (so the standard board encoding — targets marked 2, boxes 4 — is exact).
Systematic backward BFS from the goal enumerates these; each is solvable to the
goal by construction (reverse the backward path). All instances share the base's
walls + targets and differ only in box/player start positions.

Outputs (data/): sokoban_samegoal_train.txt, sokoban_samegoal_test.txt (normal
Sokoban board format, 100 ints/line) + a .meta with the base index and depths.

Env: SG_BASE (150, index into get_solvable_data)  SG_TRAIN (800)  SG_TEST (200)
     SG_MIN_DEPTH (6, skip near-goal trivial starts)  SG_MAX_EXPAND (60000)
     SG_SEED (0)
"""
import os
import random
from collections import deque

import numpy as np

from game.getData import get_solvable_data, DATA_DIR
from game.SokobanGame import SokobanGame

BASE = int(os.environ.get("SG_BASE", "150"))
TRAIN = int(os.environ.get("SG_TRAIN", "800"))
TEST = int(os.environ.get("SG_TEST", "200"))
MIN_DEPTH = int(os.environ.get("SG_MIN_DEPTH", "6"))
MAX_EXPAND = int(os.environ.get("SG_MAX_EXPAND", "60000"))
SEED = int(os.environ.get("SG_SEED", "0"))


def harvest(base):
    fwd = SokobanGame(base, isBackward=False)
    bwd = SokobanGame(SokobanGame.initializeBackwardPuzzle(base), isBackward=True)
    T = list(map(tuple, np.argwhere(base == 2)))
    base_targets = set(T); NT = len(T)

    def succ(board):
        bwd.puzzle = board
        loc = bwd.getPlayerLocation(board)
        return [nb for _d, act in bwd.availableStates(loc)
                if (nb := act.moveAndUpdateBoard(loc, board)) is not None]

    def as_start(board):
        cand = fwd.flipGame(board)
        if int((cand == 2).sum()) != NT: return None
        if set(map(tuple, np.argwhere(cand == 2))) != base_targets: return None
        if int((cand == 4).sum()) != NT or int((cand == 3).sum()) != 1: return None
        if not np.array_equal((cand == 0), (base == 0)): return None
        return cand

    start = np.array(bwd.puzzle, copy=True)
    seen = {start.tobytes()}
    q = deque([(start, 0)])
    out = {}   # forward-start bytes -> (board, depth)
    ex = 0
    while q and ex < MAX_EXPAND:
        b, d = q.popleft(); ex += 1
        st = as_start(b)
        if st is not None and d >= MIN_DEPTH:
            out.setdefault(st.tobytes(), (st, d))
        for nb in succ(b):
            k = nb.tobytes()
            if k not in seen:
                seen.add(k); q.append((nb, d + 1))
    return list(out.values())


def write(rows, split):
    path = os.path.join(DATA_DIR, f"sokoban_samegoal_{split}.txt")
    with open(path, "w") as f:
        for board, _d in rows:
            f.write(" ".join(str(int(v)) for v in board.reshape(-1)) + "\n")
    print(f"[samegoal] {split}: {len(rows)} starts -> {path} "
          f"(depths {min(d for _,d in rows)}-{max(d for _,d in rows)})", flush=True)


if __name__ == "__main__":
    base = get_solvable_data(limit=BASE + 1)[BASE]
    print(f"[samegoal] base #{BASE}: targets={int((base==2).sum())} "
          f"walls fixed; harvesting backward-reachable starts...", flush=True)
    rows = harvest(base)
    print(f"[samegoal] harvested {len(rows)} unique valid starts "
          f"(min_depth={MIN_DEPTH})", flush=True)
    random.Random(SEED).shuffle(rows)
    need = TRAIN + TEST
    if len(rows) < need:
        print(f"[samegoal] WARNING: only {len(rows)} starts (< {need}); "
              f"using all, splitting proportionally", flush=True)
    rows = rows[:need]
    ntest = min(TEST, len(rows) // 5)
    write(rows[ntest:], "train")
    write(rows[:ntest], "test")
    with open(os.path.join(DATA_DIR, "sokoban_samegoal.meta.txt"), "w") as f:
        f.write(f"# base={BASE} min_depth={MIN_DEPTH} harvested={len(rows)} "
                f"train={len(rows)-ntest} test={ntest} seed={SEED}\n")
    print("[samegoal] DONE", flush=True)
