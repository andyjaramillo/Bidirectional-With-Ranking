"""Domain abstraction for the bidirectional / forward search stack.

The search (search/AI_Bidirectional.py) and the forward baseline
(rank_forward/forward_search.py) were hardwired to SokobanGame. This module
factors the coupling into a small, documented contract so a new domain
(sliding-tile, maze, ...) plugs in without touching the search logic.

Two contracts:

1. ``Domain`` — a FACTORY + domain-level constants. It builds the forward and
   backward game objects for a puzzle and reports how many one-hot channels
   the NN input needs per board. The search takes a ``Domain`` (default
   ``SokobanDomain``) and never names a concrete game class.

2. ``GameLike`` (documented below, not enforced) — the per-state methods the
   search calls on the game objects it gets back. A new domain's game class
   must provide these with the same signatures; they stay duck-typed so the
   hot expand loop is unchanged:

     attributes:  puzzle, target, goal_map, isBackward
     getPlayerLocation(board) -> loc
     availableStates(loc)     -> iterable of (dir, action);
                                 action.moveAndUpdateBoard(loc, board) -> board|None
     hasDeadlock(board)       -> bool           (False for domains w/o deadlocks)
     encodeMap(board) -> hash ;  decodeMap(hash) -> board
     fullStateKey(board) -> bytes  (canonical key of the MOVABLE state for
                                 frontier meeting; Sokoban: agent+boxes,
                                 tiles: the whole board)
     isGoal(board)            -> bool
     evaluateBoard(board, other=None) -> float  (blind heuristic; pairwise if other)
     flipGame(board)          -> board          (identity for undirected domains)
     initializeBackwardPuzzle(board) -> board   (goal state for backward search)

Keeping construction behind the factory is the ONLY structural change for
Sokoban — every per-state call remains exactly as before, so Sokoban results
are byte-identical (regression-checked).
"""
from abc import ABC, abstractmethod

from game.SokobanGame import SokobanGame


class Domain(ABC):
    """Factory for a domain's forward/backward game objects + NN constants."""

    #: short lowercase identifier, used in run-log paths and game_name.
    name: str = "domain"
    #: one-hot channels per board for the NN input encoder.
    nn_channels: int = 5

    @abstractmethod
    def make_forward(self, puzzle):
        """Forward game object for the start state."""

    @abstractmethod
    def make_backward(self, puzzle):
        """Backward game object, seeded at the goal state."""

    def make_games(self, puzzle):
        """(forward_game, backward_game) — the pair the search holds."""
        return self.make_forward(puzzle), self.make_backward(puzzle)

    def encode_board(self, board, target):
        """Per-board NN input: (H, W, nn_channels) float array. ``target`` is
        the game's ``target`` attribute (may be ignored). Passed to SmallCNN
        as ``encode_fn`` for non-default domains; None means "use the model's
        built-in Sokoban encoding" and is what SokobanDomain relies on."""
        raise NotImplementedError

    def model_kwargs(self):
        """kwargs for build_model so the net matches this domain's encoding.
        Sokoban returns {} — the model's built-in default, byte-identical."""
        return {}


class SokobanDomain(Domain):
    """The original Sokoban wiring, verbatim behind the factory.

    Reproduces exactly:
        forward  = SokobanGame(puzzle, isBackward=False)
        backward = SokobanGame(initializeBackwardPuzzle(puzzle), isBackward=True)
    ``initializeBackwardPuzzle`` is a staticmethod, so building the backward
    puzzle here is identical to the old ``self.forward_game.initialize...``.
    """

    name = "sokoban"
    nn_channels = 5

    def make_forward(self, puzzle):
        return SokobanGame(puzzle, isBackward=False)

    def make_backward(self, puzzle):
        backward_puzzle = SokobanGame.initializeBackwardPuzzle(puzzle)
        return SokobanGame(backward_puzzle, isBackward=True)


class TileDomain(Domain):
    """n x n sliding-tile puzzle (game/SlidingTileGame.py). Undirected moves:
    the backward game runs the same dynamics from the solved board and
    flipGame is the identity.

    NN encoding (``encode_board``): SIZE-INVARIANT, 3 channels per cell —
    the normalized row/col displacement of the cell's tile from its goal
    cell, plus a blank flag. One-hot over tile ids would tie the channel
    count to n^2 (a 5x5-trained net could never see a 6x6 board); the
    displacement encoding keeps channels fixed, and the conv tower's global
    avg-pool handles the spatial size — so board size becomes a true
    generalization axis for a single net."""

    def __init__(self, n: int = 5):
        self.n = n
        self.name = f"tiles{n}"
        self.nn_channels = 3

    def make_forward(self, puzzle):
        from game.SlidingTileGame import SlidingTileGame
        return SlidingTileGame(puzzle, isBackward=False)

    def make_backward(self, puzzle):
        from game.SlidingTileGame import SlidingTileGame
        goal = SlidingTileGame.solved_board(len(puzzle))
        return SlidingTileGame(goal, isBackward=True)

    #: fixed feature scale for tile displacements. NOT the board size: dividing
    #: by n made the same physical displacement produce SMALLER inputs on
    #: bigger boards, so a 5x5-calibrated net under-predicted h at 7x7 by ~2x
    #: (ranking intact, magnitude collapsed — measured 2026-07-09). A fixed
    #: constant keeps the input scale size-consistent so the OUTPUT calibration
    #: transfers across board sizes.
    DISP_SCALE = 10.0

    @classmethod
    def encode_board(cls, board, target):
        """(n, n, 3): [dr/DISP_SCALE, dc/DISP_SCALE, is_blank] per cell,
        displacement of the cell's tile from its goal cell (blank's goal =
        bottom-right)."""
        import numpy as np
        b = np.asarray(board)
        n = b.shape[0]
        out = np.zeros((n, n, 3), dtype=np.float32)
        for (r, c), v in np.ndenumerate(b):
            v = int(v)
            gr, gc = ((n - 1, n - 1) if v == 0
                      else ((v - 1) // n, (v - 1) % n))
            out[r, c, 0] = (gr - r) / cls.DISP_SCALE
            out[r, c, 1] = (gc - c) / cls.DISP_SCALE
            out[r, c, 2] = 1.0 if v == 0 else 0.0
        return out

    def model_kwargs(self):
        return {"in_channels": self.nn_channels, "encode_fn": self.encode_board}


def get_domain(name: str = "sokoban") -> Domain:
    """Resolve a domain by name (used by env-knob-driven entrypoints).
    Tile domains parse their size from the name: "tiles5", "tiles7", ..."""
    name = (name or "sokoban").lower()
    if name in ("sokoban", "sok"):
        return SokobanDomain()
    if name.startswith("tiles"):
        return TileDomain(int(name[len("tiles"):] or "5"))
    raise ValueError(f"unknown domain: {name!r}")
