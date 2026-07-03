"""Instrumentation pass: estimate the label-noise rate for the
hard-negative ranking idea (use off-path closed nodes as "ranks above
the on-path node at the same depth").

For a sample of off-path closed nodes v from baseline TTBS:
  * compute true d(v, goal)        via exhaustive forward BFS
  * compute true d(s_k, goal)      same way for the on-path state at depth k
  * categorize:
      worse (d(v) > d(s_k))     — ranking constraint is correct
      equal (d(v) = d(s_k))     — ambiguous (alternate equal-cost path)
      better (d(v) < d(s_k))    — WRONG: v is genuinely closer than s_k
                                    (satisficing path was suboptimal)

If worse-rate is high (>>80%) the hard-negative idea is worth trying.
If better-rate is non-trivial (>10%) we'd be teaching the model wrong things.
"""
import random
import time
from collections import deque

from game.getData import get_data
from game.SokobanGame import SokobanGame
from search.AI_Bidirectional import BidirectionalF2FSearch

random.seed(0)
N_PUZZLES = 30
SAMPLES_PER_PUZZLE = 10
BFS_LIMIT = 50000  # max forward-game states explored per BFS


def bfs_dist_to_goal(start_state, base_puzzle, limit):
    """Exact BFS from start_state to any goal in the forward Sokoban game.
    Returns the shortest distance or None if the budget is exhausted."""
    g = SokobanGame(base_puzzle, isBackward=False)
    if g.isGoal(start_state):
        return 0
    visited = {g.encodeMap(start_state)}
    queue = deque([(start_state, 0)])
    while queue and len(visited) < limit:
        state, d = queue.popleft()
        g.puzzle = state
        player_loc = g.getPlayerLocation(state)
        for _dir, action in g.availableStates(player_loc):
            new_state = action.moveAndUpdateBoard(player_loc, state)
            if new_state is None:
                continue
            h = g.encodeMap(new_state)
            if h in visited:
                continue
            if g.isGoal(new_state):
                return d + 1
            visited.add(h)
            queue.append((new_state, d + 1))
    return None  # over budget


puzzles = get_data(False)[:N_PUZZLES]
worse = equal = better = timeout = 0
t0 = time.time()
for i, p in enumerate(puzzles):
    s = BidirectionalF2FSearch(p, None)
    path = s.search(max_iterations=10000)
    if not path:
        continue
    L = len(path)

    # on-path forward hashes + their g values (depth → hash)
    on_path_hashes = set()
    s_k_by_depth = {}
    cur = s.meeting_fwd
    while cur is not None:
        on_path_hashes.add(cur)
        s_k_by_depth[s.g_f[cur]] = cur
        cur = s.parent_f.get(cur)

    off_path = [h for h in s.closed_f if h not in on_path_hashes]
    if not off_path:
        continue
    sample = random.sample(off_path, min(SAMPLES_PER_PUZZLE, len(off_path)))

    # Cache d(s_k, goal) per (puzzle, depth) since multiple v's may share k.
    ds_cache = {}
    for v_hash in sample:
        k = s.g_f[v_hash]
        if k not in s_k_by_depth:
            continue
        d_v = bfs_dist_to_goal(
            s.forward_game.decodeMap(v_hash), p, BFS_LIMIT)
        if d_v is None:
            timeout += 1
            continue
        if k not in ds_cache:
            ds_cache[k] = bfs_dist_to_goal(
                s.forward_game.decodeMap(s_k_by_depth[k]), p, BFS_LIMIT)
        d_sk = ds_cache[k]
        if d_sk is None:
            timeout += 1
            continue
        if d_v > d_sk:
            worse += 1
        elif d_v == d_sk:
            equal += 1
        else:
            better += 1
    print(f"puzzle {i:>2d}: L={L:>3d} off={len(off_path):>4d} "
          f"sampled={len(sample):>2d}  cum: worse={worse} eq={equal} "
          f"better={better} timeout={timeout}  [{time.time()-t0:.0f}s]")

print(f"\n=== final ({worse+equal+better+timeout} samples) ===")
total = worse + equal + better
if total > 0:
    print(f"  worse  (d(v) >  d(s_k))   {worse:>4d}  {worse/total*100:>5.1f}%  ← correct signal")
    print(f"  equal  (d(v) =  d(s_k))   {equal:>4d}  {equal/total*100:>5.1f}%  ← ambiguous")
    print(f"  better (d(v) <  d(s_k))   {better:>4d}  {better/total*100:>5.1f}%  ← WRONG signal")
print(f"  BFS timeouts                {timeout}")
print(f"\ntotal wall: {time.time()-t0:.0f}s")
