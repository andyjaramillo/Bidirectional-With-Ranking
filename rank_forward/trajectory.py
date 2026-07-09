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
from rank_forward.forward_search import ForwardSearch, manhattan_heuristic


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


def solve_optimal(puzzle: np.ndarray, max_iterations: int = 200000,
                  full_goal: bool = False, domain=None) -> Optional[List[np.ndarray]]:
    """Return a provably shortest path (list of boards) or None.

    A* with the admissible MWPM-Manhattan heuristic and reopening is optimal
    (the box-Manhattan bound stays admissible even for the full goal, which
    also routes the player to its pinned cell). Deadlock pruning is
    intentionally OFF here: SokobanGame.hasDeadlock does not exempt boxes
    already on targets, so it could prune the goal.
    """
    if domain is None:
        from game.domain import SokobanDomain
        domain = SokobanDomain()
    game = domain.make_forward(puzzle)
    s = ForwardSearch(puzzle, heuristic=manhattan_heuristic(game),
                      use_g_in_f=True, use_deadlock=False, reopen=True,
                      full_goal=full_goal, domain=domain)
    return s.search(max_iterations=max_iterations)


def instance_from_path(puzzle: np.ndarray, path, use_deadlock: bool = False,
                       full_goal: bool = False, domain=None) -> Optional[Instance]:
    """Assemble a training Instance from a GIVEN solution path (list of boards,
    start -> goal). Used both by build_instance (optimal path from a separate
    solver) and by the bootstrap experiment (the satisficing path the forward
    search found with its own current heuristic). Returns None for a degenerate
    path. The ranking loss treats ``path`` as the trajectory and the distance-1
    off-path siblings as negatives — exactly as for an optimal path; if ``path``
    is suboptimal the supervision is simply noisier (the bootstrap reality)."""
    if path is None or len(path) < 2:
        return None

    if domain is None:
        from game.domain import SokobanDomain
        domain = SokobanDomain()
    game = domain.make_forward(puzzle)
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

    # Heuristic goal context must match the search's goal: the player@start
    # full goal when full_goal, else the classical goal_map reference.
    goal_ctx = (game.flipGame(game.initializeBackwardPuzzle(puzzle))
                if full_goal else game.goal_map)
    return Instance(path, flat_states, flat_parent, node_off,
                    game.target, goal_ctx)


def build_instance(puzzle: np.ndarray, max_iterations: int = 200000,
                   use_deadlock: bool = False, full_goal: bool = False,
                   domain=None) -> Optional[Instance]:
    """Solve ``puzzle`` OPTIMALLY (admissible A*) and assemble its training
    Instance, or None if it cannot be solved within ``max_iterations``. This is
    the imitation-from-expert path; the bootstrap experiment instead calls
    instance_from_path on a self-found (satisficing) path. ``full_goal`` targets
    the player-pinned full goal (matching the bidirectional method)."""
    path = solve_optimal(puzzle, max_iterations=max_iterations,
                         full_goal=full_goal, domain=domain)
    return instance_from_path(puzzle, path, use_deadlock=use_deadlock,
                              full_goal=full_goal, domain=domain)
