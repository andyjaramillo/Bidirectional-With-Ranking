"""Unidirectional forward search (A* / GBFS) over Sokoban states.

This is the search side of the paper's method. It reuses ``SokobanGame`` for
successors / goal test / state hashing verbatim, so the state space is
identical to the forward side of our bidirectional searcher and the two are
comparable on the same expansion metric.

Merit:  f(s) = alpha * g(s) + beta * h(s)
    A*   : alpha = 1, beta = 1   ->  f = g + h      (``use_g_in_f=True``)
    GBFS : alpha = 0, beta = 1   ->  f = h          (``use_g_in_f=False``)

Counting contract (mirrors ``AI_Bidirectional``): ``self.iteration`` is
incremented exactly once per *productive* expansion — a node that is popped,
is not stale/closed/superseded, is not the goal, and gets its successors
generated. Stale/superseded pops are skipped without counting. The goal is
detected on *selection* (pop), as in the paper's Alg. 2.1 step 4c, and is NOT
counted as an expansion, so a perfectly efficient search on an (l+1)-state
optimal path performs exactly l expansions. ``first_solved_iter`` is the
expansion count at the moment the goal is popped — the forward analogue of the
bidirectional ``first_meeting_iter``.

Reopening: A* reopens a closed node when a strictly cheaper g is found (paper
Alg. 2.1 step 4e-ii), which keeps A* optimal even for inconsistent admissible
heuristics; GBFS neglects reopening (paper §2.1). Reopening is therefore tied
to ``use_g_in_f`` but also exposed as an explicit flag.
"""
from typing import Callable, List, Optional, Tuple
import heapq

import numpy as np

from game.SokobanGame import SokobanGame


# A heuristic is a batched callable: list[board] -> list[float] (h >= 0).
Heuristic = Optional[Callable[[List[np.ndarray]], List[float]]]


class ForwardSearch:
    def __init__(self, puzzle: np.ndarray, heuristic: Heuristic = None,
                 use_g_in_f: bool = True, use_deadlock: bool = False,
                 reopen: Optional[bool] = None):
        """
        Args:
            puzzle: initial 10x10 board (0 wall, 1 floor, 2 target, 3 player, 4 box).
            heuristic: batched h(boards)->floats, or None for the blind baseline
                (h == 0; with use_g_in_f this is uniform-cost / Dijkstra).
            use_g_in_f: True -> A* (f = g + h); False -> GBFS (f = h).
            use_deadlock: apply SokobanGame.hasDeadlock pruning to successors.
                Default False: the paper's method is domain-agnostic and our
                CLAUDE.md asks to avoid Sokoban-specific ingredients. Set True
                to match the bidirectional forward side's pruning exactly.
            reopen: reopen closed nodes on a strictly better g. Defaults to
                use_g_in_f (A* reopens, GBFS does not), per the paper.
        """
        self.game = SokobanGame(puzzle)
        self.start = np.asarray(puzzle)
        self.heuristic = heuristic
        self.use_g_in_f = use_g_in_f
        self.use_deadlock = use_deadlock
        self.reopen = use_g_in_f if reopen is None else reopen
        # Fixed goal context fed as the net's second ("other") input. We reuse
        # the repo's canonical goal_map (boxes on targets, player at a fixed
        # cell) so the convention is identical in training and search.
        self.goal_ctx = self.game.goal_map
        self.target = self.game.target

        self.open: List[Tuple] = []
        self.g = {}
        self.parent = {}
        self.closed = set()
        self.iteration = 0
        self.first_solved_iter: Optional[int] = None
        self._tie = 0

    # ── heuristic plumbing ────────────────────────────────────────────────
    def _h_batch(self, boards: List[np.ndarray]) -> List[float]:
        if self.heuristic is None:
            return [0.0] * len(boards)
        return self.heuristic(boards)

    def _push(self, f: float, g: int, key: bytes) -> None:
        heapq.heappush(self.open, (f, -g, self._tie, key))
        self._tie += 1

    # ── main loop ─────────────────────────────────────────────────────────
    def search(self, max_iterations: int = 10000) -> Optional[List[np.ndarray]]:
        """Run the search; return the solution path (list of boards, start ->
        goal) or None if unsolved within the expansion budget / exhausted."""
        game = self.game
        start_key = game.encodeMap(self.start)
        self.g[start_key] = 0
        self.parent[start_key] = None
        h0 = self._h_batch([self.start])[0]
        self._push(self._merit(0, h0), 0, start_key)

        while self.open and self.iteration < max_iterations:
            f, neg_g, _tie, key = heapq.heappop(self.open)
            g = -neg_g
            if key in self.closed:
                continue                       # stale duplicate — not counted
            if g > self.g.get(key, float("inf")):
                continue                       # superseded entry — not counted

            board = game.decodeMap(key)
            if game.isGoal(board):
                self.first_solved_iter = self.iteration
                return self._reconstruct(key)

            self.closed.add(key)
            self.iteration += 1                # one productive expansion

            self._expand(board, key, g)

        return None

    def _merit(self, g: int, h: float) -> float:
        return (g + h) if self.use_g_in_f else h

    def _expand(self, board: np.ndarray, key: bytes, g: int) -> None:
        game = self.game
        game.puzzle = board                    # availableStates checks walls here
        loc = game.getPlayerLocation(board)
        ng = g + 1

        survivors = []                          # (child_key, child_board)
        for _dir, action in game.availableStates(loc):
            child = action.moveAndUpdateBoard(loc, board)
            if child is None:
                continue
            if self.use_deadlock and game.hasDeadlock(child):
                continue
            ck = game.encodeMap(child)
            if ck in self.closed:
                if not (self.reopen and ng < self.g.get(ck, float("inf"))):
                    continue
                self.closed.discard(ck)         # reopen on a strictly better g
            if ng >= self.g.get(ck, float("inf")):
                continue
            self.g[ck] = ng
            self.parent[ck] = key
            survivors.append((ck, child))

        if not survivors:
            return
        hs = self._h_batch([b for _, b in survivors])
        for (ck, _b), h in zip(survivors, hs):
            self._push(self._merit(ng, h), ng, ck)

    def _reconstruct(self, key: bytes) -> List[np.ndarray]:
        path_keys = []
        cur = key
        while cur is not None:
            path_keys.append(cur)
            cur = self.parent.get(cur)
        path_keys.reverse()
        return [self.game.decodeMap(k) for k in path_keys]


# ── heuristic factories ───────────────────────────────────────────────────
def manhattan_heuristic(game: SokobanGame) -> Heuristic:
    """Admissible MWPM-Manhattan box->target heuristic (a lower bound on plan
    length). Used ONLY to generate provably-optimal training trajectories fast
    — it is not part of the learned method under comparison."""
    def h(boards: List[np.ndarray]) -> List[float]:
        return [float(game.evaluateBoard(b)) for b in boards]
    return h


def model_heuristic(model, target, goal_ctx) -> Heuristic:
    """Wrap a trained heuristic net as a batched, non-negative h(boards)."""
    import torch

    def h(boards: List[np.ndarray]) -> List[float]:
        n = len(boards)
        with torch.no_grad():
            out = model.forward_batch(boards, [target] * n, [goal_ctx] * n)
            return out.clamp_min(0.0).tolist()
    return h
