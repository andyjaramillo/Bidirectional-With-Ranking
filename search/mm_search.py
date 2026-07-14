"""Meet in the Middle (MM) bidirectional search — comparison baseline.

Holte, Felner, Sharon & Sturtevant, "Bidirectional Search That Is Guaranteed
to Meet in the Middle" (AAAI 2016). Front-to-END: each side scores nodes
against the FIXED opposite seed (forward: the goal state, backward: the
start), unlike our TTBS front-to-front arm. Priority of a node n on the
forward side is

    pr_F(n) = max(f_F(n), 2 * g_F(n)),   f_F(n) = g_F(n) + h_F(n),

(symmetrically backward); MM always expands the side attaining
C = min(prmin_F, prmin_B), which guarantees no node with g > C*/2 is ever
expanded — the "meet in the middle" property. Meetings are checked on
GENERATION against every state the opposite side has assigned a g-value
(open or closed), maintaining the incumbent U; MM terminates — i.e. the
solution is PROVEN optimal (for admissible h) — when

    U <= max(C, fmin_F, fmin_B, gmin_F + gmin_B + eps),

with eps the minimum edge cost. "Solved" for this arm therefore means the
proof fired, not the first meeting (MM is an optimal algorithm; the proof
cost IS the comparison point). ``first_meeting_iter`` records when U first
became finite, so drivers can report met-but-unproven-within-budget counts.

Integration mirrors rank_forward/forward_search.py:
  - takes ``domain=`` and works on any Domain (instance may be one board or a
    (start, goal) pair; the goal state is the backward game's seed, so MM
    solves the EXACT same problem as the bidirectional arm — Sokoban full
    goal, tiles exact board).
  - heuristics are batched callables list[boards] -> list[float]; ``None``
    means h = 0 (the true blind MM0). ``heuristic_f`` receives forward-frame
    boards, ``heuristic_b`` receives BACKWARD-frame boards (the factories
    below handle frame flips internally).
  - counting contract: ``expansions`` increments exactly once per productive
    expansion (a pop that is closed and generates successors); stale /
    superseded heap entries are skipped without counting. Reopened nodes
    count again on re-expansion (honest work). Budget = expansions.
  - cost-generality: edge costs come from the ``cost_fn`` hook (default 1)
    and ``eps`` is the minimum edge cost (default 1) — no hardwired "+1".
"""
from typing import Callable, Dict, List, Optional, Set, Tuple
import heapq

import numpy as np

# A heuristic is a batched callable: list[board] -> list[float].
Heuristic = Optional[Callable[[List[np.ndarray]], List[float]]]

INF = float("inf")


class _Side:
    """One direction's bookkeeping. OPEN membership = has a g-value and not
    closed; the three heaps expose MM's prmin / fmin / gmin lazily (entries
    are stale iff the node is closed or its stored g was superseded)."""
    __slots__ = ("g", "parent", "closed", "h", "pr_heap", "f_heap", "g_heap",
                 "fkey_to_hash", "tie", "expansions")

    def __init__(self):
        self.g: Dict[bytes, float] = {}
        self.parent: Dict[bytes, Optional[bytes]] = {}
        self.closed: Set[bytes] = set()
        self.h: Dict[bytes, float] = {}          # h(state): state-only, never stale
        self.pr_heap: List[Tuple] = []           # (pr, -g, tie, hash)
        self.f_heap: List[Tuple] = []            # (f, g, tie, hash)
        self.g_heap: List[Tuple] = []            # (g, tie, hash)
        self.fkey_to_hash: Dict[bytes, bytes] = {}   # fwd-frame full key -> hash
        self.tie = 0
        self.expansions = 0


class MMSearch:
    def __init__(self, instance, heuristic_f: Heuristic = None,
                 heuristic_b: Heuristic = None, domain=None,
                 use_deadlock: bool = False, eps: float = 1,
                 cost_fn: Optional[Callable] = None):
        """
        Args:
            instance: one board (Sokoban/fixed-goal tiles) or a (start, goal)
                pair (random-goal domains) — whatever the domain's factory takes.
            heuristic_f: batched h estimating d(node -> goal), forward-frame
                boards. None = 0 (blind).
            heuristic_b: batched h estimating remaining backward distance
                (= forward-dynamics d(start -> flip(node))), BACKWARD-frame
                boards. None = 0.
            use_deadlock: apply hasDeadlock pruning to FORWARD successors only
                (mirrors the bidirectional arm's convention). Default False,
                matching ForwardSearch's domain-agnostic default.
            eps: minimum edge cost (the termination bound's epsilon).
            cost_fn: edge-cost hook (game, parent_board, child_board) -> cost;
                default constant 1 (this testbed's uniform cost).
        """
        if domain is None:
            from game.domain import SokobanDomain
            domain = SokobanDomain()
        self.domain = domain
        self.forward_game, self.backward_game = domain.make_games(instance)
        # START board in the forward frame (see ForwardSearch: always taken
        # from the forward game, never the raw arg).
        self.puzzle = np.asarray(self.forward_game.puzzle)
        # GOAL state = the backward seed, flipped into the forward frame.
        self.goal_fwd = self.forward_game.flipGame(self.backward_game.puzzle)
        self.heuristic_f = heuristic_f
        self.heuristic_b = heuristic_b
        self.use_deadlock = use_deadlock
        self.eps = eps
        self.cost_fn = cost_fn if cost_fn is not None else (lambda game, u, v: 1)

        self.fwd = _Side()
        self.bwd = _Side()

        # ── Solution tracking ────────────────────────────────────────────
        self.U: float = INF
        self.meeting_fwd: Optional[bytes] = None
        self.meeting_bwd: Optional[bytes] = None
        self.expansions: int = 0
        self.first_meeting_iter: Optional[int] = None   # U first finite
        self.first_solved_iter: Optional[int] = None    # termination proof
        self._initialized = False

    # Driver parity with the bidirectional arm's expansion unit.
    @property
    def closed_f(self) -> Set[bytes]:
        return self.fwd.closed

    @property
    def closed_b(self) -> Set[bytes]:
        return self.bwd.closed

    # ── heuristic plumbing ────────────────────────────────────────────────
    def _h_values(self, side: _Side, heuristic: Heuristic,
                  keys: List[bytes], boards: List[np.ndarray]) -> List[float]:
        """h for each (key, board), batching only the uncached ones."""
        if heuristic is None:
            return [0.0] * len(keys)
        missing = [(i, b) for i, b in enumerate(boards) if keys[i] not in side.h]
        if missing:
            hs = heuristic([b for _, b in missing])
            for (i, _), h in zip(missing, hs):
                side.h[keys[i]] = float(h)
        return [side.h[k] for k in keys]

    def _push(self, side: _Side, key: bytes, g: float, h: float) -> None:
        f = g + h
        pr = max(f, 2 * g)
        side.tie += 1
        t = side.tie
        heapq.heappush(side.pr_heap, (pr, -g, t, key))
        heapq.heappush(side.f_heap, (f, g, t, key))
        heapq.heappush(side.g_heap, (g, t, key))

    # ── lazy heap minima (stale entries: closed or g-superseded) ──────────
    def _prmin(self, side: _Side) -> float:
        hp = side.pr_heap
        while hp:
            pr, ng, _t, k = hp[0]
            if k in side.closed or side.g.get(k) != -ng:
                heapq.heappop(hp)
                continue
            return pr
        return INF

    def _fmin(self, side: _Side) -> float:
        hp = side.f_heap
        while hp:
            f, g, _t, k = hp[0]
            if k in side.closed or side.g.get(k) != g:
                heapq.heappop(hp)
                continue
            return f
        return INF

    def _gmin(self, side: _Side) -> float:
        hp = side.g_heap
        while hp:
            g, _t, k = hp[0]
            if k in side.closed or side.g.get(k) != g:
                heapq.heappop(hp)
                continue
            return g
        return INF

    # ── meeting / incumbent maintenance ────────────────────────────────────
    def _check_meet(self, is_forward: bool, key: bytes, fkey: bytes,
                    g: float) -> None:
        """Called on every g-assignment: if the opposite side has a g for the
        same full state, update the incumbent U (and the meeting pair)."""
        opp = self.bwd if is_forward else self.fwd
        ok = opp.fkey_to_hash.get(fkey)
        if ok is None:
            return
        total = g + opp.g[ok]
        if total < self.U:
            self.U = total
            if is_forward:
                self.meeting_fwd, self.meeting_bwd = key, ok
            else:
                self.meeting_fwd, self.meeting_bwd = ok, key
            if self.first_meeting_iter is None:
                self.first_meeting_iter = self.expansions

    # ── initialisation ─────────────────────────────────────────────────────
    def init_search(self) -> None:
        for is_forward, side, game, board, heur in (
                (True, self.fwd, self.forward_game, self.puzzle,
                 self.heuristic_f),
                (False, self.bwd, self.backward_game,
                 np.asarray(self.backward_game.puzzle), self.heuristic_b)):
            k = game.encodeMap(board)
            side.g[k] = 0
            side.parent[k] = None
            board_fwd = board if is_forward else game.flipGame(board)
            fkey = self.forward_game.fullStateKey(board_fwd)
            side.fkey_to_hash[fkey] = k
            self._check_meet(is_forward, k, fkey, 0)   # start == goal case
            h0 = self._h_values(side, heur, [k], [board])[0]
            self._push(side, k, 0, h0)
        self._initialized = True

    # ── one MM expansion ────────────────────────────────────────────────────
    def _expand(self, is_forward: bool) -> None:
        side = self.fwd if is_forward else self.bwd
        game = self.forward_game if is_forward else self.backward_game
        heur = self.heuristic_f if is_forward else self.heuristic_b

        while True:                      # caller guarantees a live entry exists
            _pr, ng, _t, k = heapq.heappop(side.pr_heap)
            if k in side.closed or side.g.get(k) != -ng:
                continue
            break
        g = -ng
        side.closed.add(k)
        side.expansions += 1
        self.expansions += 1

        board = game.decodeMap(k)
        game.puzzle = board              # availableStates checks walls here
        loc = game.getPlayerLocation(board)

        survivors = []                   # (child_key, child_g, child_board)
        for _dir, action in game.availableStates(loc):
            child = action.moveAndUpdateBoard(loc, board)
            if child is None:
                continue
            if is_forward and self.use_deadlock and game.hasDeadlock(child):
                continue
            ck = game.encodeMap(child)
            cg = g + self.cost_fn(game, board, child)
            if cg >= side.g.get(ck, INF):
                continue
            side.closed.discard(ck)      # reopen on a strictly better g
            side.g[ck] = cg
            side.parent[ck] = k
            child_fwd = child if is_forward else game.flipGame(child)
            fkey = self.forward_game.fullStateKey(child_fwd)
            side.fkey_to_hash[fkey] = ck
            self._check_meet(is_forward, ck, fkey, cg)
            survivors.append((ck, cg, child))

        if not survivors:
            return
        hs = self._h_values(side, heur, [ck for ck, _, _ in survivors],
                            [b for _, _, b in survivors])
        for (ck, cg, _), h in zip(survivors, hs):
            self._push(side, ck, cg, h)

    # ── main loop ────────────────────────────────────────────────────────────
    def search(self, max_iterations: int = 10000) -> Optional[List[bytes]]:
        """Run MM; return the solution path (forward-frame encoded maps,
        start -> goal, same format as the bidirectional arm) once PROVEN, or
        None if the budget runs out first / no solution exists. On a budget
        failure the incumbent (U, first_meeting_iter) remains inspectable."""
        if not self._initialized:
            self.init_search()
        while True:
            prF, prB = self._prmin(self.fwd), self._prmin(self.bwd)
            C = min(prF, prB)
            bound = max(C, self._fmin(self.fwd), self._fmin(self.bwd),
                        self._gmin(self.fwd) + self._gmin(self.bwd) + self.eps)
            if self.U <= bound:
                # Proof fired (or both frontiers exhausted with U = inf:
                # inf <= inf, no solution exists — see the meeting-on-
                # generation argument in the module docstring).
                if self.U < INF:
                    self.first_solved_iter = self.expansions
                    return self._reconstruct()
                return None
            if self.expansions >= max_iterations:
                return None
            self._expand(prF <= prB)

    # ── path reconstruction ───────────────────────────────────────────────
    def _reconstruct(self) -> List[bytes]:
        fg, bg = self.forward_game, self.backward_game
        path_f: List[bytes] = []
        cur = self.meeting_fwd
        while cur is not None:
            path_f.append(cur)
            cur = self.fwd.parent.get(cur)
        path_f.reverse()                 # start -> meeting
        path_b: List[bytes] = []
        cur = self.bwd.parent.get(self.meeting_bwd)   # meeting already in path_f
        while cur is not None:
            path_b.append(fg.encodeMap(fg.flipGame(bg.decodeMap(cur))))
            cur = self.bwd.parent.get(cur)
        return path_f + path_b


# ── heuristic factories ──────────────────────────────────────────────────────
def mm_analytic_heuristics(search: MMSearch) -> Tuple[Heuristic, Heuristic]:
    """Front-to-end analytic (Manhattan) pair for a constructed MMSearch:
    h_F(n) = evaluateBoard(n, goal state), h_B(n) = evaluateBoard(flip(n),
    start) — both computed in the forward frame via the pairwise evaluateBoard
    (admissible: MWPM box Manhattan for Sokoban, per-tile Manhattan for tiles)."""
    fg = search.forward_game
    goal_fwd, start = search.goal_fwd, search.puzzle

    def hf(boards: List[np.ndarray]) -> List[float]:
        return [float(fg.evaluateBoard(b, goal_fwd)) for b in boards]

    def hb(boards: List[np.ndarray]) -> List[float]:
        return [float(fg.evaluateBoard(fg.flipGame(b), start)) for b in boards]

    return hf, hb


def mm_learned_heuristics(model, search: MMSearch) -> Tuple[Heuristic, Heuristic]:
    """Front-to-end pair from a FROZEN pairwise net h(a, target, b) trained
    under the DIRECTED convention (h estimates forward-dynamics d(a -> b)):
        h_F(n)   = h(n -> goal state)
        h_B(n_b) = h(start -> flip(n_b))
    Clamped >= 0 (the bidirectional searcher's convention — MM's 2g term and
    termination bound assume h >= 0). With a learned, possibly inadmissible h
    the termination proof is heuristic too: MM degrades to satisficing, which
    is the setting every arm here is compared in."""
    import torch
    fg = search.forward_game
    target = fg.target
    goal_fwd, start = search.goal_fwd, search.puzzle

    def hf(boards: List[np.ndarray]) -> List[float]:
        n = len(boards)
        with torch.no_grad():
            out = model.forward_batch(boards, [target] * n, [goal_fwd] * n)
        return out.clamp_min(0.0).tolist()

    def hb(boards: List[np.ndarray]) -> List[float]:
        n = len(boards)
        flipped = [fg.flipGame(b) for b in boards]
        with torch.no_grad():
            out = model.forward_batch([start] * n, [target] * n, flipped)
        return out.clamp_min(0.0).tolist()

    return hf, hb
