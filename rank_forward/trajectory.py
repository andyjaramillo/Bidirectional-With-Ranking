"""Training-trajectory generation.

The ranking losses assume an OPTIMAL solution path: the on-path state must be
out-ranked by its off-path competitors. We therefore solve each training
instance with an admissible-heuristic A* (provably optimal shortest path) and
then replay the path to collect, for every non-goal path node, its distance-1
off-path siblings — the negative set of paper Definition 1 / Eq. (1).

Per instance we materialise:
  path        : [s_0, ..., s_L]                     optimal path (L+1 boards)
  g_path[i]   = i ,  hstar[i] = L - i               (unit action costs)
  off_states  : distinct off-path siblings (deduped, min parent index kept)
  off_parent  : parent path-index p for each off state (sibling depth = p+1)
  node_off[i] : off-path children of s_i (for the Bellman loss' min over children)

Off-path siblings are the children of path nodes that are not themselves on the
path. ``node_off`` keeps them per-parent (for L_be's per-state min over all
children), while ``off_states``/``off_parent`` is the deduped flat set used by
the ranking losses.
"""
from typing import List, Optional

import numpy as np

from game.SokobanGame import SokobanGame
from .forward_search import ForwardSearch, manhattan_heuristic


class Instance:
    __slots__ = ("path", "L", "off_states", "off_parent", "node_off",
                 "target", "goal_ctx", "opt_len")

    def __init__(self, path, off_states, off_parent, node_off, target, goal_ctx):
        self.path = path
        self.L = len(path) - 1
        self.off_states = off_states
        self.off_parent = off_parent
        self.node_off = node_off
        self.target = target
        self.goal_ctx = goal_ctx
        self.opt_len = self.L


def solve_optimal(puzzle: np.ndarray, max_iterations: int = 200000
                  ) -> Optional[List[np.ndarray]]:
    """Return a provably shortest path (list of boards) or None.

    A* with the admissible MWPM-Manhattan heuristic and reopening is optimal.
    Deadlock pruning is intentionally OFF here: SokobanGame.hasDeadlock does
    not exempt boxes already on targets, so it could prune the goal.
    """
    game = SokobanGame(puzzle)
    s = ForwardSearch(puzzle, heuristic=manhattan_heuristic(game),
                      use_g_in_f=True, use_deadlock=False, reopen=True)
    return s.search(max_iterations=max_iterations)


def build_instance(puzzle: np.ndarray, max_iterations: int = 200000,
                   use_deadlock: bool = False) -> Optional[Instance]:
    """Solve ``puzzle`` optimally and assemble its training Instance, or None
    if it cannot be solved within ``max_iterations``."""
    path = solve_optimal(puzzle, max_iterations=max_iterations)
    if path is None or len(path) < 2:
        return None

    game = SokobanGame(puzzle)
    path_keys = {game.encodeMap(b) for b in path}

    node_off: List[List[np.ndarray]] = []
    flat_states: List[np.ndarray] = []
    flat_parent: List[int] = []
    seen = {}  # off-state key -> index in flat_states (dedup, keep min parent)

    for i in range(len(path) - 1):           # non-goal path nodes
        s_i = path[i]
        game.puzzle = s_i
        loc = game.getPlayerLocation(s_i)
        offs = []
        for _dir, action in game.availableStates(loc):
            child = action.moveAndUpdateBoard(loc, s_i)
            if child is None:
                continue
            if use_deadlock and game.hasDeadlock(child):
                continue
            ck = game.encodeMap(child)
            if ck in path_keys:
                continue                      # on-path successor (or a later path node)
            offs.append(child)
            if ck not in seen:
                seen[ck] = len(flat_states)
                flat_states.append(child)
                flat_parent.append(i)
        node_off.append(offs)

    return Instance(path, flat_states, flat_parent, node_off,
                    game.target, game.goal_map)
