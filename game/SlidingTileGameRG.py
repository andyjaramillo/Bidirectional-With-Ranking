"""Random-goal sliding-tile puzzle: the goal is an ARBITRARY board, not the
canonical solved state. An instance is a (start, goal) pair of permutations in
the same solvability class.

Motivation (supervisor hypothesis, 2026-07-10): the forward (Chrestien) method
dominates fixed-goal sliding-tile largely because there is a SINGLE canonical
goal — the heuristic only has to learn distance-to-one-state, and unidirectional
forward search to that state is efficient. Randomizing the goal forces a genuine
pairwise heuristic h(x, y) and removes forward search's structural advantage,
where the bidirectional meet-in-the-middle should regain its edge. This class is
the substrate for that independent check.

Reuses SlidingTileGame's mechanics; the ONLY differences are that ``goal_map``
is the instance goal (passed in) rather than solved_board(n), and the backward
game is seeded at that goal. Moves are still self-inverse, so flipGame is the
identity and the two frames coincide.
"""
import numpy as np

from game.SlidingTileGame import SlidingTileGame, _TileMove


class SlidingTileGameRG(SlidingTileGame):
    """Sliding-tile game with an explicit (arbitrary) goal board."""

    def __init__(self, board, goal, isBackward: bool = False):
        # Deliberately bypass SlidingTileGame.__init__ (it hardcodes the solved
        # goal); set up the same attributes with an explicit goal instead.
        self.puzzle = np.asarray(board, dtype=np.uint8)
        self.n = self.puzzle.shape[0]
        self.isBackward = isBackward
        self.goal_map = np.asarray(goal, dtype=np.uint8)
        self.target = []
        self.action_map = {d: _TileMove(d, self.n)
                           for d in ((-1, 0), (1, 0), (0, -1), (0, 1))}

    def initializeBackwardPuzzle(self, board):
        """Not used by the RG domain (the backward game is built directly from
        the instance goal); kept for interface completeness — returns the
        instance goal, matching the fixed-goal contract's intent."""
        return self.goal_map.copy()
