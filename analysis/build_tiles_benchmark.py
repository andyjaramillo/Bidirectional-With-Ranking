"""Generate sliding-tile datasets by SCRAMBLING FROM THE GOAL: random walks
of the blank from the solved board, rejecting the immediate inverse move so
walks don't trivially undo themselves. Solvable by construction (moves are
self-inverse, so the walk reversed is a witness solution of length <= L) and
parity-safe (never leaves the goal's solvability class).

Mirrors the paper's sliding-puzzle protocol shape: a 5x5 TRAINING set and
200-instance TEST sets at 5x5, 6x6, 7x7 (difficulty axis = board size; the
per-instance scramble length is the depth axis, recorded in the meta file).

Outputs (under data/):
    tiles{n}_train.txt      one board per line, n*n space-separated ints
    tiles{n}_test.txt       (row-major; 0 = blank)
    tiles{n}_{split}.meta.txt   generation bookkeeping incl. per-line L

Env knobs:
    TILES_TRAIN_N (2200)   TILES_TRAIN_SIZE (5)
    TILES_TEST_N  (200)    TILES_TEST_SIZES ("5,6,7")
    TILES_LMIN (10)  TILES_LMAX (60)   scramble-length range, uniform per
                                       instance (both splits)
    TILES_SEED (0)
Boards are deduplicated within a split and never equal to the goal.
"""
import os
import random

import numpy as np

from game.getData import DATA_DIR
from game.SlidingTileGame import SlidingTileGame

TRAIN_N = int(os.environ.get("TILES_TRAIN_N", "2200"))
TRAIN_SIZE = int(os.environ.get("TILES_TRAIN_SIZE", "5"))
TEST_N = int(os.environ.get("TILES_TEST_N", "200"))
TEST_SIZES = [int(s) for s in os.environ.get("TILES_TEST_SIZES", "5,6,7").split(",")]
LMIN = int(os.environ.get("TILES_LMIN", "10"))
LMAX = int(os.environ.get("TILES_LMAX", "60"))
SEED = int(os.environ.get("TILES_SEED", "0"))

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def scramble(n: int, length: int, rng: random.Random) -> np.ndarray:
    """Random walk of the blank, ``length`` moves, no immediate backtrack."""
    board = SlidingTileGame.solved_board(n)
    r, c = n - 1, n - 1                      # blank starts in the last cell
    prev = None
    for _ in range(length):
        options = [d for d in _DIRS
                   if 0 <= r + d[0] < n and 0 <= c + d[1] < n
                   and (prev is None or d != (-prev[0], -prev[1]))]
        d = rng.choice(options)
        nr, nc = r + d[0], c + d[1]
        board[r, c] = board[nr, nc]
        board[nr, nc] = 0
        r, c, prev = nr, nc, d
    return board


def build_split(n: int, count: int, split: str, rng: random.Random):
    goal_key = SlidingTileGame.solved_board(n).tobytes()
    boards, lengths, seen = [], [], set()
    attempts = 0
    while len(boards) < count:
        attempts += 1
        L = rng.randint(LMIN, LMAX)
        b = scramble(n, L, rng)
        key = b.tobytes()
        if key == goal_key or key in seen:
            continue
        seen.add(key)
        boards.append(b)
        lengths.append(L)

    boards_path = os.path.join(DATA_DIR, f"tiles{n}_{split}.txt")
    with open(boards_path, "w") as f:
        for b in boards:
            f.write(" ".join(str(int(v)) for v in b.reshape(-1)) + "\n")
    with open(os.path.join(DATA_DIR, f"tiles{n}_{split}.meta.txt"), "w") as f:
        f.write(f"# scramble-from-goal, n={n} count={count} "
                f"L~U[{LMIN},{LMAX}] seed={SEED} attempts={attempts}\n")
        f.write("lengths " + " ".join(map(str, lengths)) + "\n")
    print(f"[tiles] {split} n={n}: {count} boards -> {boards_path} "
          f"(mean L={sum(lengths)/len(lengths):.1f})", flush=True)
    return boards


if __name__ == "__main__":
    print(f"[tiles] train: {TRAIN_N}@{TRAIN_SIZE}x{TRAIN_SIZE}  "
          f"test: {TEST_N}@{TEST_SIZES}  L~U[{LMIN},{LMAX}] seed={SEED}",
          flush=True)
    rng = random.Random(SEED)
    build_split(TRAIN_SIZE, TRAIN_N, "train", rng)
    for n in TEST_SIZES:
        build_split(n, TEST_N, "test", random.Random(SEED * 100 + n))
    print("[tiles] ALL DONE", flush=True)
