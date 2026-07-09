"""Sliding-tile puzzle (n x n, blank-swap moves) behind the GameLike contract
documented in game/domain.py — the second domain of the testbed (the paper's
"Sliding puzzle": train 5x5, test 5x5/6x6/7x7).

State: n x n numpy array holding a permutation of 0..n*n-1, 0 = the BLANK.
The blank plays the Sokoban player's role: ``getPlayerLocation`` returns the
blank's cell and a move swaps the blank with an adjacent tile. Moves are
self-inverse, so the state graph is UNDIRECTED — which collapses two pieces
of the bidirectional machinery:
  * ``flipGame`` is the identity (forward and backward frames coincide);
  * the backward game runs the same dynamics seeded at the goal board
    (``initializeBackwardPuzzle`` returns the solved board).
This also makes the domain a clean symmetric counterpoint to Sokoban's
quasimetric structure (DIRECTED still runs; its labels are simply symmetric).

Goal: tiles 1..n*n-1 in row-major order, blank last. One exact state, so the
classical and "full" goals coincide (no Sokoban player-pinning subtlety).

Blind heuristic (``evaluateBoard``): summed Manhattan distance of each tile
(blank excluded) between its cells in the two boards — admissible for the
pairwise distance d(x, y) since one move displaces exactly one tile by 1.
"""
from typing import Optional, Tuple

import numpy as np


class _TileMove:
    """Action object: swap the blank with the tile one step in ``direction``.
    Mirrors the Sokoban action interface: ``moveAndUpdateBoard(loc, board)``
    returns the successor board, or None if the move leaves the grid."""

    def __init__(self, direction: Tuple[int, int], n: int):
        self.direction = direction
        self.n = n

    def moveAndUpdateBoard(self, current_location, board):
        r, c = current_location
        dr, dc = self.direction
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            return None
        new_board = board.copy()
        new_board[r, c] = new_board[nr, nc]
        new_board[nr, nc] = 0
        return new_board


class SlidingTileGame:
    """GameLike implementation for the n x n sliding-tile puzzle."""

    def __init__(self, puzzle, isBackward: bool = False):
        self.puzzle = np.asarray(puzzle, dtype=np.uint8)
        self.n = self.puzzle.shape[0]
        self.isBackward = isBackward
        self.goal_map = self.solved_board(self.n)
        # NN-facing extra input (Sokoban passes target cells). The tile goal
        # is fixed given n, so there is nothing instance-specific to pass;
        # kept as an empty list for interface compatibility (Phase 3 wires
        # the tile NN encoder, which ignores it).
        self.target = []
        # One action object per direction, shared across expansions.
        self.action_map = {d: _TileMove(d, self.n)
                           for d in ((-1, 0), (1, 0), (0, -1), (0, 1))}

    # ── construction helpers ────────────────────────────────────────────
    @staticmethod
    def solved_board(n: int) -> np.ndarray:
        """Tiles 1..n*n-1 row-major, blank (0) in the last cell."""
        vals = list(range(1, n * n)) + [0]
        return np.array(vals, dtype=np.uint8).reshape(n, n)

    def initializeBackwardPuzzle(self, board) -> np.ndarray:
        """Backward search is seeded at the goal state (dynamics are the
        same in both directions — moves are self-inverse)."""
        b = np.asarray(board)
        return self.solved_board(b.shape[0])

    # ── GameLike contract ───────────────────────────────────────────────
    def getPlayerLocation(self, board) -> Tuple[int, int]:
        loc = np.where(np.asarray(board) == 0)
        return (loc[0][0], loc[1][0])

    def availableStates(self, current_location):
        r, c = current_location
        out = []
        for d, action in self.action_map.items():
            if 0 <= r + d[0] < self.n and 0 <= c + d[1] < self.n:
                out.append((d, action))
        return out

    def hasDeadlock(self, board) -> bool:
        """Every sliding-tile state is solvable within its parity class and
        scramble-generated instances stay in the goal's class — no deadlocks."""
        return False

    def encodeMap(self, map) -> bytes:
        return np.asarray(map, dtype=np.uint8).tobytes()

    def decodeMap(self, map_bytes) -> np.ndarray:
        return np.frombuffer(map_bytes, dtype=np.uint8).reshape(
            self.n, self.n).copy()

    def isGoal(self, board) -> bool:
        return bool(np.array_equal(np.asarray(board), self.goal_map))

    def fullStateKey(self, board) -> bytes:
        """Canonical FULL-state key for frontier meeting: every cell is a
        movable object in this domain, so the key is the whole board. (NOT
        Sokoban's (3,4)-positions key — tile values 3 and 4 are ordinary
        tiles here, and keying on them alone collides wildly.)"""
        return np.asarray(board, dtype=np.uint8).tobytes()

    def flipGame(self, board) -> np.ndarray:
        """Identity: forward and backward frames coincide (undirected moves).
        Returns a copy so callers can mutate without aliasing."""
        return np.asarray(board).copy()

    def evaluateBoard(self, board, open_set_state=None) -> float:
        """Summed per-tile Manhattan distance between ``board`` and
        ``open_set_state`` (default: the goal). Admissible pairwise."""
        other = self.goal_map if open_set_state is None else open_set_state
        a = np.asarray(board)
        b = np.asarray(other)
        total = 0
        pos_b = {int(v): (r, c) for (r, c), v in np.ndenumerate(b) if v != 0}
        for (r, c), v in np.ndenumerate(a):
            if v == 0:
                continue
            br, bc = pos_b[int(v)]
            total += abs(r - br) + abs(c - bc)
        return float(total)
