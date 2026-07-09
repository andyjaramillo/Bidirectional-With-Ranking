"""Build the HARDER Sokoban test sets (4-7 boxes, 10x10) for the
apples-to-apples generalization comparison with Chrestien et al. (NeurIPS
2023), whose Table 1 tests 3-box-trained heuristics on 200 instances each of
3/4/5/6/7 boxes at the same grid size.

Instances come from the paper's own generator (gym-sokoban reverse-play,
vendored in analysis/gym_sokoban_room_utils.py), converted to this repo's
board encoding (0 wall, 1 floor, 2 target, 3 player, 4 box), then filtered to
instances solvable under the FAIR FULL GOAL (boxes-on-targets AND player back
at its start cell) within a blind-search cap — the same filter and rationale
as analysis/build_solvable_benchmark.py: the bidirectional method always
targets the full goal, so instances unmeetable under it are artifacts for
this testbed, not difficulty.

Rejection at conversion time (before the solvability filter), matching the
observable properties of the paper's published 3-box sets (every line has
#boxes == #targets == n and a bare player cell):
  - any box already on a target (gym code 3; our format cannot represent it)
  - player standing on a target  (our format cannot represent it)

Outputs (under data/): hard10_{n}box.txt        (KEEP solvable boards/line)
                       hard10_{n}box.meta.txt   (generation bookkeeping)

Env knobs: HARD_BOXES ("4,5,6,7"), HARD_KEEP (200 per box count),
           HARD_CAP (blind-search expansion budget, default 300000),
           HARD_WORKERS (default cpu_count), HARD_SEED (default 0),
           HARD_GEN_STEPS (reverse-play depth, default 34 = gym-sokoban's
           int(1.7*(10+10)) for a 10x10 room).
"""
import os
import random
import time
import multiprocessing as mp

import numpy as np

from analysis.gym_sokoban_room_utils import generate_room
from game.getData import DATA_DIR
from search.AI_Bidirectional import BidirectionalF2FSearch

BOXES = [int(b) for b in os.environ.get("HARD_BOXES", "4,5,6,7").split(",")]
KEEP = int(os.environ.get("HARD_KEEP", "200"))
CAP = int(os.environ.get("HARD_CAP", "300000"))
WORKERS = int(os.environ.get("HARD_WORKERS", str(os.cpu_count() or 1)))
SEED = int(os.environ.get("HARD_SEED", "0"))
GEN_STEPS = int(os.environ.get("HARD_GEN_STEPS", "34"))


def gen_candidate(num_boxes, rng_tag):
    """One gym-sokoban room -> our encoding, or None if not representable.
    Also returns the raw reverse-play score (0 = degenerate, rejected)."""
    try:
        _, room_state, box_mapping = generate_room(
            dim=(10, 10), num_boxes=num_boxes, num_steps=GEN_STEPS)
    except (RuntimeWarning, RuntimeError, IndexError):
        return None
    state = room_state.astype(int)
    if (state == 3).any():          # box already on target: unrepresentable
        return None
    board = state.copy()
    board[state == 5] = 3           # player: gym 5 -> ours 3
    # gym room_fixed target under the player would be hidden by our encoding.
    # generate_room returns room_state with the player OVERWRITING a target
    # cell (fixed==2, state==5); detect via the count invariant instead of
    # room_fixed (not returned here): #targets must equal #boxes.
    if (board == 2).sum() != num_boxes or (board == 4).sum() != num_boxes:
        return None
    return board


_CAP = None


def _init(cap):
    global _CAP
    _CAP = cap


def _classify_one(puzzle):
    """(solvable, exhausted, iters) under the fair full goal, blind search."""
    s = BidirectionalF2FSearch(puzzle, nn=None)
    s.use_g_in_f = True
    path = s.search(max_iterations=_CAP)
    exhausted = (not s.open_f and not s.open_b)
    return path is not None, exhausted, s.iteration


def build_for_boxes(num_boxes):
    random.seed(SEED * 1000 + num_boxes)
    np.random.seed(SEED * 1000 + num_boxes)
    kept, generated, rejected_repr, unsolvable, budget = [], 0, 0, 0, 0
    t0 = time.time()
    batch = max(WORKERS * 4, 32)
    with mp.Pool(WORKERS, initializer=_init, initargs=(CAP,)) as pool:
        while len(kept) < KEEP:
            cands = []
            while len(cands) < batch:
                generated += 1
                b = gen_candidate(num_boxes, generated)
                if b is None:
                    rejected_repr += 1
                    continue
                cands.append(b)
            for b, (ok, exhausted, _iters) in zip(
                    cands, pool.imap(_classify_one, cands, chunksize=4)):
                if ok and len(kept) < KEEP:
                    kept.append(b)
                elif exhausted:
                    unsolvable += 1
                elif not ok:
                    budget += 1
            print(f"  [{num_boxes} boxes] kept={len(kept)}/{KEEP} "
                  f"generated={generated} unrepresentable={rejected_repr} "
                  f"unsolvable={unsolvable} budget={budget} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    boards_path = os.path.join(DATA_DIR, f"hard10_{num_boxes}box.txt")
    with open(boards_path, "w") as f:
        for b in kept:
            f.write(" ".join(str(int(v)) for v in b.reshape(-1)) + "\n")
    meta_path = os.path.join(DATA_DIR, f"hard10_{num_boxes}box.meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"# gym-sokoban reverse-play, dim=(10,10) num_steps={GEN_STEPS} "
                f"seed={SEED * 1000 + num_boxes}\n")
        f.write(f"# kept={len(kept)} generated={generated} "
                f"unrepresentable={rejected_repr} unsolvable_fullgoal={unsolvable} "
                f"budget_capped={budget} cap={CAP}\n")
    print(f"[{num_boxes} boxes] DONE -> {boards_path} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    print(f"[hard-benchmark] boxes={BOXES} keep={KEEP}/count cap={CAP} "
          f"workers={WORKERS} seed={SEED} gen_steps={GEN_STEPS}", flush=True)
    for nb in BOXES:
        build_for_boxes(nb)
    print("[hard-benchmark] ALL DONE", flush=True)
