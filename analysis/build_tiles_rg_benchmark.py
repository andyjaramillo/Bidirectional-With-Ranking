"""Random-goal sliding-tile datasets: each instance is a (start, goal) PAIR of
permutations in the same solvability class. Goal is an arbitrary scramble of the
solved board; start is a further scramble FROM the goal (so start->goal is
solvable, of controlled length). Stored one instance per line as
    <n*n start ints> <n*n goal ints>
(row-major each). Supervisor-hypothesis substrate (2026-07-10): does randomizing
the goal restore the bidirectional method's edge over forward search?

Outputs (data/): tilesRG{n}_train.txt, tilesRG{n}_test.txt (+ .meta).
Env: RG_N (4)  RG_TRAIN_N (2000)  RG_TEST_N (200)
     RG_GOAL_L (30,60 scramble range for the goal)  RG_START_L (8,30 start<-goal)
     RG_SEED (0)
"""
import os
import random

import numpy as np

from game.getData import DATA_DIR
from game.SlidingTileGame import SlidingTileGame

N = int(os.environ.get("RG_N", "4"))
TRAIN_N = int(os.environ.get("RG_TRAIN_N", "2000"))
TEST_N = int(os.environ.get("RG_TEST_N", "200"))
GOAL_LMIN, GOAL_LMAX = [int(x) for x in os.environ.get("RG_GOAL_L", "30,60").split(",")]
START_LMIN, START_LMAX = [int(x) for x in os.environ.get("RG_START_L", "8,30").split(",")]
SEED = int(os.environ.get("RG_SEED", "0"))
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def scramble_from(board, length, rng):
    b = board.copy()
    r, c = [int(x) for x in np.argwhere(b == 0)[0]]
    prev = None
    for _ in range(length):
        opts = [d for d in _DIRS if 0 <= r + d[0] < N and 0 <= c + d[1] < N
                and (prev is None or d != (-prev[0], -prev[1]))]
        d = rng.choice(opts)
        nr, nc = r + d[0], c + d[1]
        b[r, c] = b[nr, nc]; b[nr, nc] = 0; r, c, prev = nr, nc, d
    return b


def make_pairs(count, rng):
    solved = SlidingTileGame.solved_board(N)
    pairs, seen = [], set()
    while len(pairs) < count:
        goal = scramble_from(solved, rng.randint(GOAL_LMIN, GOAL_LMAX), rng)
        start = scramble_from(goal, rng.randint(START_LMIN, START_LMAX), rng)
        key = start.tobytes() + goal.tobytes()
        if np.array_equal(start, goal) or key in seen:
            continue
        seen.add(key)
        pairs.append((start, goal))
    return pairs


def write(pairs, split):
    path = os.path.join(DATA_DIR, f"tilesRG{N}_{split}.txt")
    with open(path, "w") as f:
        for start, goal in pairs:
            f.write(" ".join(str(int(v)) for v in start.reshape(-1)) + " "
                    + " ".join(str(int(v)) for v in goal.reshape(-1)) + "\n")
    print(f"[tilesRG] {split}: {len(pairs)} pairs -> {path}", flush=True)


if __name__ == "__main__":
    print(f"[tilesRG] n={N} train={TRAIN_N} test={TEST_N} "
          f"goalL~[{GOAL_LMIN},{GOAL_LMAX}] startL~[{START_LMIN},{START_LMAX}] "
          f"seed={SEED}", flush=True)
    write(make_pairs(TRAIN_N, random.Random(SEED)), "train")
    write(make_pairs(TEST_N, random.Random(SEED * 100 + 1)), "test")
    print("[tilesRG] DONE", flush=True)
