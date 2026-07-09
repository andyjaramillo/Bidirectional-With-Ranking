"""Build a HARD held-out eval set from the tail of the solvable benchmark.

Motivation (2026-07-07): the standard held-out 200 is near saturation for the
reference configuration (200/200 solved, median ~118 expansions, same-seed
noise band ~118-155), so real-but-modest method improvements are becoming
invisible. This builder ranks the UNTOUCHED tail of the solvable pool
(boards beyond HARD_FROM, disjoint from the train 0..N_TOTAL and the held-out
eval slice) by a domain-agnostic difficulty measure — the deterministic blind
bidirectional search's expansion count (len(closed_f)+len(closed_b)) — and
keeps the hardest HARD_KEEP boards. "Difficulty = reference-search cost" uses
no Sokoban-specific features, so the same builder works for any TTBS domain.

Outputs (under data/):
  solvable10_3box_hard{K}.txt      one board per line (same format as
                                   solvable10_3box.txt); load via
                                   game.getData.get_hard_eval_data()
  solvable10_3box_hard{K}.idx.txt  provenance: source indices + blind
                                   expansions + blind solution length

Env knobs:  HARD_FROM    (first solvable-pool index to scan, default 2000),
            HARD_N       (#boards to scan from there, default 8000),
            HARD_KEEP    (#hardest boards to keep, default 200),
            HARD_CAP     (per-puzzle expansion budget, default 100000),
            HARD_WORKERS (parallel workers, default 4 — deliberately low so
                          a concurrent training run keeps its cores).
"""
import os
import time
import multiprocessing as mp

from game.getData import get_solvable_data, DATA_DIR
from search.AI_Bidirectional import BidirectionalF2FSearch

_CAP = None


def _init(cap):
    global _CAP
    _CAP = cap


def _difficulty_one(puzzle):
    """(solved, expansions, plan_len) under the deterministic blind search."""
    s = BidirectionalF2FSearch(puzzle, nn=None)
    s.use_g_in_f = True
    path = s.search(max_iterations=_CAP)
    exp = len(s.closed_f) + len(s.closed_b)
    return path is not None, exp, (len(path) - 1 if path else -1)


def build(from_idx, n, keep, cap, workers):
    pool_boards = get_solvable_data(limit=from_idx + n)
    boards = pool_boards[from_idx:]
    print(f"scanning {len(boards)} solvable boards "
          f"(pool indices {from_idx}..{from_idx + len(boards) - 1})", flush=True)
    t0 = time.time()
    rows = []   # (pool_idx, exp, plan_len)
    with mp.Pool(workers, initializer=_init, initargs=(cap,)) as pool:
        for i, (ok, exp, plen) in enumerate(
                pool.imap(_difficulty_one, boards, chunksize=8)):
            if ok:                       # keep only genuinely solvable-in-cap
                rows.append((from_idx + i, exp, plen))
            if (i + 1) % 500 == 0:
                print(f"  scanned {i+1:>5d}/{len(boards)} solved={len(rows)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: -r[1])       # hardest (most expansions) first
    kept = rows[:keep]
    kept_idx = {i for i, _, _ in kept}

    out = os.path.join(DATA_DIR, f"solvable10_3box_hard{keep}.txt")
    with open(out, "w") as f:
        for i, b in enumerate(pool_boards):
            if i in kept_idx:
                f.write(" ".join(str(int(v)) for v in b.reshape(-1)) + "\n")

    idx_out = os.path.join(DATA_DIR, f"solvable10_3box_hard{keep}.idx.txt")
    with open(idx_out, "w") as f:
        f.write(f"# hardest {len(kept)} of solvable pool "
                f"[{from_idx}..{from_idx + len(boards) - 1}], cap={cap}\n")
        f.write("# columns: pool_idx blind_expansions blind_plan_len "
                "(sorted hardest first)\n")
        for i, exp, plen in kept:
            f.write(f"{i} {exp} {plen}\n")

    exps = [e for _, e, _ in kept]
    lens = [l for _, _, l in kept]
    print(f"\nwrote {len(kept)} boards -> {out}")
    print(f"  blind expansions: min={min(exps)} median={sorted(exps)[len(exps)//2]} "
          f"max={max(exps)}")
    print(f"  blind plan len  : min={min(lens)} median={sorted(lens)[len(lens)//2]} "
          f"max={max(lens)}")
    print(f"  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    HARD_FROM = int(os.environ.get("HARD_FROM", "2000"))
    HARD_N = int(os.environ.get("HARD_N", "8000"))
    HARD_KEEP = int(os.environ.get("HARD_KEEP", "200"))
    HARD_CAP = int(os.environ.get("HARD_CAP", "100000"))
    HARD_WORKERS = int(os.environ.get("HARD_WORKERS", "4"))
    print(f"Building hard eval set: FROM={HARD_FROM} N={HARD_N} "
          f"KEEP={HARD_KEEP} CAP={HARD_CAP} WORKERS={HARD_WORKERS}")
    build(HARD_FROM, HARD_N, HARD_KEEP, HARD_CAP, HARD_WORKERS)
