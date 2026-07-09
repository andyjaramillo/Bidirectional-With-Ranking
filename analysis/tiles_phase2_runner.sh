#!/bin/bash
# Self-contained Phase-2 data pipeline: waits for the calibration job's
# output, picks the scramble-length range, generates the tile datasets, and
# validates them. Runs to completion locally (no agent/API needed).
set -u
cd "$(dirname "$0")/.."
CAL_OUT="/tmp/claude-1000/-home-fadwa-Desktop-Bidirectional-With-Ranking-master/7a6af1e7-8267-4e8c-a2bb-4d8bcc25c1fc/tasks/bwdmcnerg.output"
LOG="data/tiles_build.log"

{
echo "[phase2-runner] started $(date)"

# ── 1. wait for calibration (max 45 min), else fall back to defaults ──────
LMAX=60
waited=0
while [ $waited -lt 2700 ]; do
  if [ -f "$CAL_OUT" ] && grep -q "n=7 L=80" "$CAL_OUT"; then
    break
  fi
  sleep 30; waited=$((waited+30))
done
if [ -f "$CAL_OUT" ]; then
  echo "[phase2-runner] calibration output:"
  cat "$CAL_OUT"
  # Pick LMAX = largest L whose 5x5 blind solve-rate is >= 10/12 (train set
  # must be mostly blind-solvable at budget so on-policy learning gets
  # labels). Falls back to 60 if nothing parses.
  PICK=$(awk '/^n=5 L=/ { split($2,a,"="); L=a[2];
                          split($4,b,"/"); s=b[1]+0;
                          if (s >= 10 && L+0 > best) best=L+0 }
              END { if (best>0) print best }' "$CAL_OUT")
  [ -n "${PICK:-}" ] && LMAX=$PICK
fi
echo "[phase2-runner] chosen scramble range: L ~ U[10, $LMAX]"

# ── 2. generate all datasets ──────────────────────────────────────────────
TILES_LMIN=10 TILES_LMAX=$LMAX TILES_TRAIN_N=2200 TILES_TEST_N=200 \
  TILES_TRAIN_SIZE=5 TILES_TEST_SIZES=5,6,7 TILES_SEED=0 \
  OMP_NUM_THREADS=2 python -m analysis.build_tiles_benchmark

# ── 3. validate: line counts, permutation integrity, blind sample solve ───
OMP_NUM_THREADS=4 python - <<'EOF'
import numpy as np, random
from game.domain import get_domain
from search.AI_Bidirectional import BidirectionalF2FSearch

def load(path, n):
    with open(path) as f:
        return [np.array([int(v) for v in l.split()]).reshape(n, n)
                for l in f if l.strip()]

for n, split, want in ((5,"train",2200),(5,"test",200),(6,"test",200),(7,"test",200)):
    bs = load(f"data/tiles{n}_{split}.txt", n)
    assert len(bs) == want, (n, split, len(bs))
    for b in bs[:50]:
        assert sorted(b.reshape(-1).tolist()) == list(range(n*n)), "bad permutation"
    keys = {b.tobytes() for b in bs}
    assert len(keys) == len(bs), "duplicates!"
    print(f"[validate] tiles{n}_{split}: {len(bs)} boards OK (unique, valid perms)")

# blind sample solve on each test set (10 instances, budget 10k)
for n in (5, 6, 7):
    dom = get_domain(f"tiles{n}")
    bs = load(f"data/tiles{n}_test.txt", n)
    rng = random.Random(0); sample = rng.sample(bs, 10)
    solved = 0
    for b in sample:
        s = BidirectionalF2FSearch(b, nn=None, domain=dom)
        s.use_g_in_f = True
        solved += s.search(max_iterations=10000) is not None
    print(f"[validate] tiles{n}_test blind sample: {solved}/10 solved @10k")
print("[validate] ALL OK")
EOF

echo "[phase2-runner] DONE $(date)"
} > "$LOG" 2>&1
