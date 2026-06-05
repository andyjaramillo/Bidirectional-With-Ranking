"""Build a "solvable-only" benchmark by filtering out instances the
bidirectional search can never meet on.

Why some instances are unsolvable (an artifact, not genuine difficulty):
``initializeBackwardPuzzle`` swaps boxes(4)↔targets(2) but leaves the player
pinned at its original cell, so the backward search is seeded from ONE specific
goal state — boxes-on-targets AND the player at that exact cell. Classical
Sokoban only requires boxes-on-targets (player anywhere). Fixing the player's
goal cell makes some instances unmeetable: the forward search may reach
boxes-on-targets with the player elsewhere, but never that exact full state.
Those instances exhaust both frontiers without meeting and pollute any
solve-rate / speedup metric, so we exclude them.

The test is domain-general — "did the planner solve it within budget?" — not a
Sokoban-specific deadlock check, so the same builder works for any TTBS domain:

    solvable  := search returns a path
    unsolvable:= search returns None with BOTH open lists empty (frontiers
                 fully exhausted ⇒ no meeting exists — see the early-exit in
                 BidirectionalF2FSearch.search)
    budget    := search returns None with a non-empty frontier (hit the cap;
                 should not happen for this domain at a high CAP, flagged if so)

Outputs (under data/):
  solvable10_3box.txt      one solvable board per line, 100 space-separated
                           ints (row-major) — same per-line format as
                           states10_3box.txt; loadable via
                           game.getData.get_solvable_data(). If BENCH_KEEP is
                           set, only the FIRST BENCH_KEEP solvable boards (in
                           get_data() order) are written here, while the idx
                           file still records the full classification.
  solvable10_3box.idx.txt  bookkeeping: the get_data() indices kept/dropped.

Env knobs:  BENCH_N      (#puzzles to scan, default 2000),
            BENCH_CAP    (per-puzzle expansion budget, default 1_000_000),
            BENCH_KEEP   (cap the #solvable boards written, default = all),
            BENCH_WORKERS(parallel worker processes, default os.cpu_count()).
"""
import os
import time
import multiprocessing as mp

from game.getData import get_data, DATA_DIR
from search.AI_Bidirectional import BidirectionalF2FSearch


# --- worker plumbing (top-level so it pickles for multiprocessing) -----------
_CAP = None


def _init(cap):
    """Pool initializer: stash the per-puzzle expansion budget in each worker."""
    global _CAP
    _CAP = cap


def _classify_one(puzzle):
    """Return (solvable: bool, exhausted: bool, iters: int) for one board.

    Deterministic (analytic search, no NN, no randomness), so running this
    under Pool.imap preserves a 1:1 input→output correspondence and the result
    does not depend on which worker handled it.
    """
    s = BidirectionalF2FSearch(puzzle, nn=None)
    s.use_g_in_f = True
    path = s.search(max_iterations=_CAP)
    exhausted = (not s.open_f and not s.open_b)
    return path is not None, exhausted, s.iteration


def classify(puzzle, cap):
    """Single-process convenience wrapper (used by tests / smoke checks)."""
    _init(cap)
    return _classify_one(puzzle)


def build(n, cap, keep=None, workers=None):
    puzzles = get_data(False)[:n]
    n = len(puzzles)
    workers = workers or os.cpu_count() or 1
    solvable_idx, exhausted_idx, budget_idx = [], [], []
    t0 = time.time()

    # imap preserves input order, so `solvable_idx` comes out ascending and
    # "first KEEP solvable" is well defined regardless of worker scheduling.
    with mp.Pool(workers, initializer=_init, initargs=(cap,)) as pool:
        for i, (ok, exhausted, _iters) in enumerate(
                pool.imap(_classify_one, puzzles, chunksize=8)):
            if ok:
                solvable_idx.append(i)
            elif exhausted:
                exhausted_idx.append(i)
            else:
                budget_idx.append(i)
            if (i + 1) % 250 == 0:
                print(f"  scanned {i+1:>5d}/{n}  solvable={len(solvable_idx)}  "
                      f"unsolvable={len(exhausted_idx)}  budget={len(budget_idx)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0

    # Optionally keep only the first `keep` solvable boards for the dataset file
    # (the idx file still records the complete classification below).
    kept_idx = solvable_idx if keep is None else solvable_idx[:keep]

    boards_path = os.path.join(DATA_DIR, "solvable10_3box.txt")
    with open(boards_path, "w") as f:
        for i in kept_idx:
            f.write(" ".join(str(int(v)) for v in puzzles[i].reshape(-1)) + "\n")

    idx_path = os.path.join(DATA_DIR, "solvable10_3box.idx.txt")
    with open(idx_path, "w") as f:
        f.write(f"# scanned first {n} get_data() puzzles, cap={cap}\n")
        f.write(f"# solvable={len(solvable_idx)} unsolvable(exhausted)="
                f"{len(exhausted_idx)} budget_capped={len(budget_idx)}\n")
        f.write(f"# written_boards={len(kept_idx)} "
                f"(keep={'all' if keep is None else keep})\n")
        f.write("solvable_idx " + " ".join(map(str, solvable_idx)) + "\n")
        f.write("unsolvable_idx " + " ".join(map(str, exhausted_idx)) + "\n")
        f.write("budget_idx " + " ".join(map(str, budget_idx)) + "\n")

    print(f"\nscanned {n} puzzles in {dt:.0f}s ({dt/n*1000:.0f} ms/puzzle, "
          f"{workers} workers)")
    print(f"  solvable           : {len(solvable_idx):>5d} "
          f"({len(solvable_idx)/n*100:.1f}%)")
    print(f"  written to dataset : {len(kept_idx):>5d}  -> {boards_path}")
    print(f"  unsolvable(exhaust): {len(exhausted_idx):>5d} "
          f"({len(exhausted_idx)/n*100:.1f}%)  (player-goal artifact / deadlock)")
    print(f"  budget-capped      : {len(budget_idx):>5d} "
          f"({len(budget_idx)/n*100:.1f}%)  (raise BENCH_CAP if > 0)")
    if exhausted_idx:
        head = exhausted_idx[:20]
        print(f"  example unsolvable get_data() indices: {head}"
              f"{' ...' if len(exhausted_idx) > 20 else ''}")


if __name__ == "__main__":
    BENCH_N = int(os.environ.get("BENCH_N", "2000"))
    BENCH_CAP = int(os.environ.get("BENCH_CAP", "1000000"))
    BENCH_KEEP = os.environ.get("BENCH_KEEP")
    BENCH_KEEP = int(BENCH_KEEP) if BENCH_KEEP else None
    BENCH_WORKERS = os.environ.get("BENCH_WORKERS")
    BENCH_WORKERS = int(BENCH_WORKERS) if BENCH_WORKERS else None
    print(f"Building solvable benchmark: BENCH_N={BENCH_N} BENCH_CAP={BENCH_CAP} "
          f"BENCH_KEEP={BENCH_KEEP} BENCH_WORKERS={BENCH_WORKERS or os.cpu_count()}")
    build(BENCH_N, BENCH_CAP, keep=BENCH_KEEP, workers=BENCH_WORKERS)
