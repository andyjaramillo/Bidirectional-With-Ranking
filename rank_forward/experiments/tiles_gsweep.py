"""Goal-variability sweep orchestrator — reproduces the supervisor's G-table.
Runs tiles_gsweep_one.py once per G (subprocess, since online_run trains at
import), then reads runlog and prints the method x G table (solved (median)).

Run: PYTHONPATH=. python rank_forward/experiments/tiles_gsweep.py
Env: RG_N(4) GSWEEP_GS("1,4,16,256,inf") GSWEEP_NTRAIN(1000) GSWEEP_NEVAL(200)
     (inf is passed as a large G >= NTRAIN)  + the tiles_gsweep_one knobs.
"""
import os
import subprocess
import sys

from analysis import runlog

N = int(os.environ.get("RG_N", "4"))
NTRAIN = int(os.environ.get("GSWEEP_NTRAIN", "1000"))
GS = os.environ.get("GSWEEP_GS", "1,4,16,256,inf").split(",")
LOGDOM = f"tilesG{N}"
METHODS = ["fwd h=0", "fwd Manhattan", "bidir Manhattan",
           "fwd L* offline", "fwd L* bootstrap", "bidir learned"]


def g_value(tok):
    return NTRAIN * 1000 if tok.strip().lower() in ("inf", "infty", "∞") else int(tok)


if __name__ == "__main__":
    for tok in GS:
        g = g_value(tok)
        print(f"\n########## G-SWEEP: G={tok.strip()} (g={g}) ##########", flush=True)
        env = dict(os.environ, GSWEEP_G=str(g), RG_N=str(N),
                   GSWEEP_NTRAIN=str(NTRAIN), PYTHONPATH=".")
        subprocess.run([sys.executable, "-m",
                        "rank_forward.experiments.tiles_gsweep_one"],
                       env=env, check=False)

    # ── assemble the table from runlog ──────────────────────────────────────
    idx = runlog.read_index()
    tags = ["inf" if t.strip().lower() in ("inf", "infty", "∞") else t.strip()
            for t in GS]
    cell = {}   # (method, Gtag) -> (solved, median)
    for row in idx:
        if row.get("domain") != LOGDOM:
            continue
        ts = str(row.get("testset", ""))
        if not ts.startswith("G"):
            continue
        gt = ts[1:]
        md = row.get("median")
        cell[(row.get("method"), gt)] = (row.get("solved"), md)

    print(f"\n===== GOAL-VARIABILITY SWEEP (N={N}x{N}, solved / (median)) =====",
          flush=True)
    hdr = f"{'method':18s} " + " ".join(f"{'G='+t:>14s}" for t in tags)
    print(hdr, flush=True)
    for m in METHODS:
        cells = []
        for t in tags:
            v = cell.get((m, t))
            if v and v[0] is not None:
                md = v[1]
                cells.append(f"{v[0]:3d} ({md:.0f})" if md is not None
                             else f"{v[0]:3d} (--)")
            else:
                cells.append("--")
        print(f"{m:18s} " + " ".join(f"{c:>14s}" for c in cells), flush=True)
    print("[GSWEEP] TABLE DONE", flush=True)
