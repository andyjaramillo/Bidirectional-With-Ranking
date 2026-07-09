"""Metrics from saved eval JSONs (analysis/runlog.py) — no retraining.

Reads per-instance eval files and computes:
  - the generalization table (solved / median / mean per method per test set);
  - the BOTH-SOLVED INTERSECTION between two methods on a test set (median/mean
    over instances both solved, plus the per-instance expansion ratio) — the
    survivorship-free efficiency comparison.

CLI:
  python -m analysis.run_metrics table  <eval.json> [<eval.json> ...]
  python -m analysis.run_metrics inter  <eval_A.json> <eval_B.json>

Each eval.json is {"iters": [...], "solved": [...], "summary": {...}} as
written by runlog.record_eval. Instance order must match across files being
intersected (same test set, same order — guaranteed if produced by the same
driver / test-set file).
"""
import json
import statistics
import sys


def _load(path):
    with open(path) as f:
        return json.load(f)


def summarize(ev):
    it, sv = ev["iters"], ev["solved"]
    solved = [x for x, ok in zip(it, sv) if ok]
    return {"n": len(it), "solved": sum(sv),
            "median": statistics.median(solved) if solved else float("nan"),
            "mean": statistics.mean(solved) if solved else float("nan")}


def intersection(ev_a, ev_b):
    """median/mean over instances BOTH solved, and per-instance b/a ratio."""
    both = [(a, b) for a, ao, b, bo in
            zip(ev_a["iters"], ev_a["solved"], ev_b["iters"], ev_b["solved"])
            if ao and bo]
    if not both:
        return None
    fa = [a for a, _ in both]
    fb = [b for _, b in both]
    ratios = [b / a for a, b in both if a > 0]
    return {"n_both": len(both),
            "a_median": statistics.median(fa), "a_mean": statistics.mean(fa),
            "b_median": statistics.median(fb), "b_mean": statistics.mean(fb),
            "ratio_median": statistics.median(ratios),
            "ratio_mean": statistics.mean(ratios)}


def _cli(argv):
    if len(argv) < 2:
        print(__doc__)
        return
    cmd = argv[1]
    if cmd == "table":
        print(f"{'file':40s} {'solved':>8s} {'median':>8s} {'mean':>9s}")
        for p in argv[2:]:
            s = summarize(_load(p))
            print(f"{p:40s} {s['solved']:3d}/{s['n']:<4d} "
                  f"{s['median']:8.1f} {s['mean']:9.1f}")
    elif cmd == "inter":
        a, b = _load(argv[2]), _load(argv[3])
        r = intersection(a, b)
        if r is None:
            print("no common solves")
            return
        print(f"n_both={r['n_both']}")
        print(f"  A: median={r['a_median']:.1f} mean={r['a_mean']:.1f}")
        print(f"  B: median={r['b_median']:.1f} mean={r['b_mean']:.1f}")
        print(f"  B/A ratio: median={r['ratio_median']:.2f} "
              f"mean={r['ratio_mean']:.2f}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli(sys.argv)
