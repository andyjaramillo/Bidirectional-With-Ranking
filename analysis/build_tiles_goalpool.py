"""Goal-POOL sliding-tile instances — the substrate for the G-sweep
(dependence on goal variability). A pool of G distinct goal permutations is
drawn once; each instance is (start, goal) with goal sampled uniformly from the
pool and start scrambled START_L moves from it. G=1 is a single fixed goal
(same-goal), G=inf (any G >= count) gives every instance its own goal (the
random-goal case). Instances are (start, goal) pairs, consumed by the tilesRG
domain unchanged.

Reusable function: make_goalpool_pairs(n, G, count, seed) -> list[(start, goal)].
"""
import random

import numpy as np

from game.SlidingTileGame import SlidingTileGame

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _scramble(board, length, n, rng):
    b = board.copy()
    r, c = [int(x) for x in np.argwhere(b == 0)[0]]
    prev = None
    for _ in range(length):
        opts = [d for d in _DIRS if 0 <= r + d[0] < n and 0 <= c + d[1] < n
                and (prev is None or d != (-prev[0], -prev[1]))]
        d = rng.choice(opts)
        nr, nc = r + d[0], c + d[1]
        b[r, c] = b[nr, nc]; b[nr, nc] = 0; r, c, prev = nr, nc, d
    return b


def make_goalpool_pairs(n, G, count, seed,
                        goal_lmin=30, goal_lmax=60,
                        start_lmin=10, start_lmax=50):
    """`count` (start, goal) pairs whose goals come from a pool of size G
    (G>=count ⇒ effectively unbounded, one goal per instance)."""
    rng = random.Random(seed)
    solved = SlidingTileGame.solved_board(n)
    pool_size = count if G >= count else G
    pool = []
    seen = set()
    while len(pool) < pool_size:
        g = _scramble(solved, rng.randint(goal_lmin, goal_lmax), n, rng)
        k = g.tobytes()
        if k == solved.tobytes() or k in seen:
            continue
        seen.add(k); pool.append(g)

    pairs, pseen = [], set()
    while len(pairs) < count:
        goal = pool[rng.randrange(len(pool))]
        start = _scramble(goal, rng.randint(start_lmin, start_lmax), n, rng)
        key = start.tobytes() + goal.tobytes()
        if np.array_equal(start, goal) or key in pseen:
            continue
        pseen.add(key); pairs.append((start, goal))
    return pairs


def write_pairs(pairs, path, n):
    with open(path, "w") as f:
        for start, goal in pairs:
            f.write(" ".join(str(int(v)) for v in start.reshape(-1)) + " "
                    + " ".join(str(int(v)) for v in goal.reshape(-1)) + "\n")
