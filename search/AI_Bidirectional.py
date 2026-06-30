import numpy as np
import heapq
from typing import Tuple, Dict, List, Optional, Set
from game.SokobanGame import SokobanGame


class BidirectionalF2FSearch:
    """
    Implementation of Top-to-Top Bidirectional Search (TTBS) for Sokoban.
    Based on the IJCAI 2020 paper:
      "Front-to-Front Heuristic Search for Satisficing Classical Planning"
      by Ryo Kuroiwa and Alex Fukunaga.

    Algorithm Overview:
    -------------------
    TTBS is a satisficing bidirectional search with a front-to-front (F2F)
    heuristic. Each side maintains an "anchor", which is the most recently
    expanded node (Temporal Anchor strategy). Nodes are ranked by:

        f(s) = g(s) + h(s, d*)

    where d* is the opponent's current anchor and h is the F2F heuristic —
    either the analytic MWPM-Manhattan box distance or a learned NN.

    When the opponent's anchor changes, stored f-scores may be stale.
    TTBS handles this via lazy re-evaluation: when a node is popped from
    the heap, if its stored target differs from the current anchor, it is
    re-scored and re-inserted (the "TTBS re-evaluation" mechanism). This
    avoids the full open-list re-sort needed by d-node retargeting (DNR).

    Convergence is detected when a freshly generated successor matches a
    state already CLOSED on the opposite side, keyed by the full forward-
    frame state — agent position AND box positions (O(1) set lookup). The
    full-state key makes the seam a genuinely shared state, so the
    reconstructed plan is a valid step-by-step path.

    State Representation (SokobanGame tile values):
        0 = wall
        1 = floor
        2 = target/goal cell
        3 = player
        4 = box (or box-on-target in the backward puzzle)

    Forward vs Backward:
        - Forward game: boxes start scattered, goals are value-2 cells.
        - Backward game: initializeBackwardPuzzle swaps 2<->4, so boxes
          start on goal cells and the backward search "pulls" them off.
        - flipGame converts between the two orientations so their full-
          state keys can be compared for intersection detection.
    """

    def __init__(self, puzzle: np.ndarray, nn=None):
        self.game_name = "Sokoban"
        self.puzzle = puzzle
        self.nn = nn

        # ── Game instances ─────────────────────────────────────────────
        self.forward_game = SokobanGame(puzzle, isBackward=False)
        backward_puzzle = self.forward_game.initializeBackwardPuzzle(puzzle)
        self.backward_game = SokobanGame(backward_puzzle, isBackward=True)

        # ── Open lists: min-heap of (f, neg_g, hash) ──────────────────
        # neg_g breaks f-ties in favour of deeper (larger g) nodes.
        self.open_f: List[Tuple[float, int, str]] = []
        self.open_b: List[Tuple[float, int, str]] = []

        # ── g-values: encoded_hash → cost from start/goal ─────────────
        self.g_f: Dict[str, int] = {}
        self.g_b: Dict[str, int] = {}

        # ── Parent pointers for path reconstruction ────────────────────
        self.parent_f: Dict[str, Optional[str]] = {}
        self.parent_b: Dict[str, Optional[str]] = {}

        # ── Full transition graph induced over visited states ──────────
        # parent_f / parent_b only record one (best-g) predecessor per
        # state, so as a graph they form a tree. We additionally record
        # every successor edge generated during expansion — including
        # transitions into already-closed nodes — so post-hoc analyses
        # can run shortest-path queries over the *actual* graph induced
        # by the search, not just its spanning tree.
        self.edges_f: Dict[str, Set[str]] = {}
        self.edges_b: Dict[str, Set[str]] = {}

        # ── Closed sets: encoded map hash → True ──────────────────────
        self.closed_f: Set[str] = set()
        self.closed_b: Set[str] = set()

        # ── Frontier-meeting keys for O(1) intersection checks ─────────
        # Convergence fires when a freshly generated successor on one side
        # matches a CLOSED state on the other side. The *meeting key* is the
        # FULL forward-frame state — agent position AND box positions. This is
        # the standard, domain-agnostic bidirectional-search meeting condition:
        # the two frontiers meet at a genuinely identical state, so the
        # reconstructed plan is a valid step-by-step path (the seam is one
        # shared state, not an agent "teleport") and distance labels are
        # correct by construction.
        self.fkey_closed_f: Set[str] = set()
        self.fkey_closed_b: Set[str] = set()   # backward states flipped to fwd
        self.fkey_to_hash_f: Dict[str, str] = {}
        self.fkey_to_hash_b: Dict[str, str] = {}

        # ── Optional: meet on GENERATION instead of on closing ─────────
        # LEGACY (default): the seam fires only when a freshly generated
        # successor matches the opposite CLOSED set. meet_on_generate=True fires
        # as soon as both frontiers have GENERATED the shared state — ~11% fewer
        # expansions, but can pick a slightly suboptimal seam. Kept available;
        # default reverted to legacy to keep the focus on anchor selection
        # (full front-to-front) as the next thing to study.
        self.meet_on_generate: bool = False
        self.fkey_gen_f: Dict[str, str] = {}   # full-key -> fwd hash, all generated
        self.fkey_gen_b: Dict[str, str] = {}   # full-key -> bwd hash, all generated

        # ── TTBS anchor: most recently expanded node on each side ──────
        self.anchor_f: Optional[str] = None
        self.anchor_b: Optional[str] = None

        # ── Lazy re-evaluation: last anchor used to score each node ────
        self.last_target_f: Dict[str, Optional[str]] = {}
        self.last_target_b: Dict[str, Optional[str]] = {}

        # ── Optional: FULL front-to-front scoring ───────────────────────
        # When True, a node is scored against the WHOLE opponent OPEN frontier
        # (min over all live opponent open nodes) instead of the single temporal
        # anchor: f(s)=g(s)+min_{d in opp_open} h(s, flip(d)). This is the
        # "gold standard" the temporal-anchor TTBS approximates; comparing the
        # two measures how much the single-anchor choice costs. In principle
        # slow (per-node min over the frontier). Anchor mode (False) is unchanged.
        # Staleness of the moving-set target is tracked by a per-side frontier
        # version counter (bumped once per expansion) + a per-node version stamp.
        self.full_f2f: bool = False
        self.frontier_ver_f: int = 0
        self.frontier_ver_b: int = 0
        self.last_ver_f: Dict[str, int] = {}
        self.last_ver_b: Dict[str, int] = {}

        # ── Anchor-selection strategy (anchor search framework) ─────────
        # Each side scores its nodes front-to-front against the OPPONENT's
        # "anchor". How that anchor is chosen is the anchor-selection axis
        # (Lavasani, "Anchor Search", 2024):
        #   "temporal"       — anchor = opponent's most-recently-EXPANDED node
        #                      (our original TTBS; the DEFAULT, byte-identical).
        #   "top_of_open"    — anchor = lowest-f LIVE node in the opponent's OPEN
        #                      (paper-faithful TTBS d-node).
        #   "closest_anchor" — policy A: a side moves its anchor to a just-
        #                      expanded node only if that node is strictly closer
        #                      (by h) to the opponent's anchor (meet-in-middle).
        #   "hybrid_af"      — AST_AF: forward side uses policy A (moving anchor),
        #                      backward side uses policy F (anchor fixed at its
        #                      goal seed, never updated). So forward scores F2E
        #                      toward the goal and backward scores F2F toward
        #                      forward's moving anchor (heuristic diversified
        #                      across directions).
        # The pairwise heuristic h(a,b) (learned NN or analytic) is unchanged;
        # only which (a,b) pairs anchor the scoring differs.
        self.anchor_strategy: str = "temporal"

        # ── Per-expansion anchor log (for the genuine ranking loss) ─────
        # When True, every expansion records the opponent anchor that was
        # live when the node was closed — i.e. the exact F2F target its
        # children were scored against (the survivor-scoring pass uses this
        # same opp_anchor). Off by default so normal solves/evals pay nothing.
        # Lets a post-solve trainer replay each step and enforce the
        # perfect-ranking condition (the on-path child must have the lowest
        # f among its siblings) against the per-step anchor the search used.
        self.log_expansions: bool = False
        self.expand_anchor_f: Dict[str, Optional[str]] = {}
        self.expand_anchor_b: Dict[str, Optional[str]] = {}

        # ── Solution tracking ──────────────────────────────────────────
        self.U: float = float('inf')
        self.meeting_fwd: Optional[str] = None
        self.meeting_bwd: Optional[str] = None

        self.iteration: int = 0
        self._initialized: bool = False

        # Iteration index at which the two frontiers first met (the search's
        # "work to solve"; the search returns at this point).
        self.first_meeting_iter: Optional[int] = None

        # If False, the f-score collapses to pure h (GBFS-style) instead of
        # g + h. Combined with closed-set cycle prevention this is the
        # natural pairing for a ranking-only heuristic whose scale is
        # unconstrained.
        self.use_g_in_f: bool = True

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _full_key(self, board: np.ndarray) -> bytes:
        """Canonical FULL-state key: positions of every movable object —
        the agent (3) and all boxes (4).

        This is the domain-agnostic notion of "the same state": only the
        things that move are part of the state; walls and goal markers are
        static background. Crucially it ignores the 2-vs-1 (target-vs-floor)
        cell marking, which differs between the forward frame and a flipped
        backward state, so the two frontiers can be compared frame-to-frame.
        """
        r3, c3 = np.where(board == 3)
        rows4, cols4 = np.where(board == 4)
        return (r3.astype(np.uint8).tobytes() + c3.astype(np.uint8).tobytes()
                + b"|" + rows4.astype(np.uint8).tobytes()
                + cols4.astype(np.uint8).tobytes())

    def _f_score(self, node_hash: str, g: int, opp_anchor_hash: Optional[str],
                 active_game: SokobanGame, opp_game: SokobanGame) -> float:
        """
        f(s) = g(s) + h(s, d*)  — front-to-front cost estimate.

        h is the MWPM-Manhattan distance between the box positions of s and
        the opponent anchor d*, where d* is first flipped into the active
        game's coordinate system.
        """
        node_map = active_game.decodeMap(node_hash)
        if opp_anchor_hash is None:
            return float(g + active_game.evaluateBoard(node_map))
        opp_anchor_map = opp_game.decodeMap(opp_anchor_hash)
        anchor_in_active = active_game.flipGame(opp_anchor_map)
        h = active_game.evaluateBoard(node_map, anchor_in_active)
        return float(g + h)

    def _anchor_other(self, opp_anchor_hash: Optional[str],
                      active_game: SokobanGame, opp_game: SokobanGame):
        """The opposite-frontier anchor flipped into the active frame, cached
        per (active_game, anchor). The decode/flip is shared by every node
        scored against this anchor within a round — single-node lazy
        re-evaluation and batched successor scoring alike.
        """
        cache = getattr(self, "_other_cache", None)
        if cache is None:
            cache = {}
            self._other_cache = cache
        ck = (id(active_game), opp_anchor_hash)
        if ck in cache:
            return cache[ck]
        if opp_anchor_hash is None:
            other = active_game.goal_map
        else:
            other = active_game.flipGame(opp_game.decodeMap(opp_anchor_hash))
        cache[ck] = other
        return other

    def _f_score_nn(self, node_hash: str, g: int, opp_anchor_hash: Optional[str],
                 active_game: SokobanGame, opp_game: SokobanGame) -> float:
        """f(s) = g(s) + max(0, h_nn(s, anchor)).

        h_nn is the learned model's predicted distance from the node to the
        opposite frontier's anchor (flipped into the active frame).
        """
        import torch
        node_map = active_game.decodeMap(node_hash)
        other = self._anchor_other(opp_anchor_hash, active_game, opp_game)
        with torch.no_grad():
            h_nn = float(self.nn(node_map, active_game.target, other).item())
        h = max(0.0, h_nn)
        return float((g + h) if self.use_g_in_f else h)

    def _f_score_nn_batch(self, node_maps, g_list, opp_anchor_hash: Optional[str],
                          active_game: SokobanGame, opp_game: SokobanGame):
        """Batched f-score for several nodes scored against ONE anchor.

        Axis-1 within-node batching: the survivors of a single expansion are
        all scored in one ``forward_batch`` call, amortizing the per-call
        dispatch (and, on a GPU/MPS twin, kernel-launch) overhead that
        dominates batch-1 inference of this tiny model. Order-preserving:
        produces identical f-scores to per-node ``_f_score_nn``, just in one
        shot. All survivors share the round's (target, anchor), so the batch
        differs only in the board state.
        """
        import torch
        if not node_maps:
            return []
        other = self._anchor_other(opp_anchor_hash, active_game, opp_game)
        n = len(node_maps)
        with torch.no_grad():
            h = self.nn.forward_batch(
                node_maps, [active_game.target] * n, [other] * n).clamp_min(0.0)
        if self.use_g_in_f:
            h = h + torch.tensor(g_list, dtype=h.dtype, device=h.device)
        return h.tolist()

    # ── FULL front-to-front scorers (used iff self.full_f2f) ───────────────
    def _live_frontier_hashes(self, opp_heap, opp_closed, opp_g_map):
        """Live opponent-frontier hashes: heap entries that are neither closed
        nor g-superseded (mirrors the pop-loop guards). Deduped and SORTED for
        determinism — heap internal order is not reproducible and the row order
        feeds torch.min's tie index."""
        seen, out = set(), []
        for f, neg_g, h in opp_heap:
            if h in seen:
                continue
            if h in opp_closed:
                continue
            if opp_g_map.get(h, float('inf')) < (-neg_g):   # superseded g-value
                continue
            seen.add(h)
            out.append(h)
        out.sort()
        return out

    def _frontier_others(self, opp_hashes, active_game, opp_game, ver):
        """Decode+flip every frontier hash into the active frame, cached per
        (active_game, frontier_version) so the decode/flip pass is shared by all
        nodes scored in this expansion. Empty frontier falls back to the active
        goal_map (reproduces the opp_anchor is None degenerate case)."""
        cache = getattr(self, "_others_cache", None)
        if cache is None:
            cache = {}
            self._others_cache = cache
        ck = (id(active_game), ver)
        hit = cache.get(ck)
        if hit is not None:
            return hit
        if not opp_hashes:
            others = [active_game.goal_map]
        else:
            others = [active_game.flipGame(opp_game.decodeMap(h)) for h in opp_hashes]
        cache[ck] = others
        return others

    def _f_score_nn_f2f(self, node_hash: str, g: int, others, active_game):
        """Single-node full-F2F NN score: g + min over the opponent frontier of
        max(0, h_nn(node, other))."""
        import torch
        node_map = active_game.decodeMap(node_hash)
        m = len(others)
        with torch.no_grad():
            h = self.nn.forward_batch(
                [node_map] * m, [active_game.target] * m, others).clamp_min(0.0)
        h_min = float(h.min().item())
        return float((g + h_min) if self.use_g_in_f else h_min)

    def _f_score_nn_batch_f2f(self, node_maps, g_list, others, active_game,
                              row_cap: int = 8192):
        """Full-F2F NN score for several nodes: the N x M (survivors x frontier)
        cross-product, chunked to row_cap rows (pure chunking — min is
        associative, so identical to one shot), per-node min over the frontier."""
        import torch
        n = len(node_maps)
        if n == 0:
            return []
        m = len(others)
        tgt = active_game.target
        out = [0.0] * n
        nodes_per_chunk = max(1, row_cap // m)
        for c0 in range(0, n, nodes_per_chunk):
            chunk = node_maps[c0:c0 + nodes_per_chunk]
            k = len(chunk)
            states = [nm for nm in chunk for _ in range(m)]   # node-major order
            targets = [tgt] * (k * m)
            oth = others * k
            with torch.no_grad():
                h = self.nn.forward_batch(states, targets, oth).clamp_min(0.0)
            assert h.numel() == k * m
            hmin = h.view(k, m).min(dim=1).values
            for i in range(k):
                out[c0 + i] = float(hmin[i].item())
        if self.use_g_in_f:
            out = [hm + gg for hm, gg in zip(out, g_list)]
        return out

    def _f_score_f2f(self, node_hash: str, g: int, others, active_game):
        """Single-node full-F2F analytic score: g + min over the opponent
        frontier of the MWPM-Manhattan box distance."""
        node_map = active_game.decodeMap(node_hash)
        best = min(active_game.evaluateBoard(node_map, o) for o in others)
        return float(g + best)

    # ── Anchor-selection helpers (anchor_strategy != "temporal") ───────────
    def _pair_h(self, node_hash: str, opp_anchor_hash: Optional[str],
                active_game: SokobanGame, opp_game: SokobanGame) -> float:
        """Pairwise heuristic h(node, opp_anchor): the learned NN distance
        (clamped >=0) or the analytic MWPM-Manhattan fallback, between an
        active-side node and the opponent anchor flipped into the active frame.
        h-only (no g); mirrors the h term of _f_score_nn. Used by closest_anchor."""
        node_map = active_game.decodeMap(node_hash)
        other = self._anchor_other(opp_anchor_hash, active_game, opp_game)
        if self.nn is not None:
            import torch
            with torch.no_grad():
                return max(0.0, float(self.nn(node_map, active_game.target, other).item()))
        return float(active_game.evaluateBoard(node_map, other))

    def _get_opp_anchor(self, is_forward: bool):
        """The opponent side's anchor for this expansion, per anchor_strategy.
        'temporal'/'closest_anchor' read the stored anchor (maintained at
        expansion time); 'top_of_open' returns the lowest-f LIVE node in the
        opponent OPEN (the paper-faithful TTBS d-node)."""
        opp_anch = 'anchor_b' if is_forward else 'anchor_f'
        if self.anchor_strategy == "top_of_open":
            opp_heap = self.open_b if is_forward else self.open_f
            opp_closed = self.closed_b if is_forward else self.closed_f
            opp_g = self.g_b if is_forward else self.g_f
            best = None
            for f, neg_g, h in opp_heap:
                if h in opp_closed:
                    continue
                if opp_g.get(h, float('inf')) < (-neg_g):
                    continue
                if best is None or f < best[0]:
                    best = (f, h)
            if best is not None:
                return best[1]
        return getattr(self, opp_anch)   # temporal / closest_anchor / empty fallback

    def _push(self, heap: List, f: float, g: int, h: str) -> None:
        heapq.heappush(heap, (f, -g, h))   # neg_g: deeper = smaller = better tie

    def _pop(self, heap: List) -> Tuple[float, int, str]:
        f, neg_g, h = heapq.heappop(heap)
        return f, -neg_g, h

    # ──────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_search(self) -> None:
        """Initialise both search frontiers."""
        start_hash = self.forward_game.encodeMap(self.puzzle)
        goal_hash  = self.backward_game.encodeMap(self.backward_game.puzzle)

        self.g_f[start_hash] = 0
        self.g_b[goal_hash]  = 0
        self.parent_f[start_hash] = None
        self.parent_b[goal_hash]  = None

        # Anchors are initialised to start/goal
        self.anchor_f = start_hash
        self.anchor_b = goal_hash

        # Initial f-scores with each side's start as anchor for the other
        _f = self._f_score_nn if self.nn is not None else self._f_score
        f_start = _f(start_hash, 0, goal_hash,
                     self.forward_game, self.backward_game)
        f_goal  = _f(goal_hash, 0, start_hash,
                     self.backward_game, self.forward_game)

        self._push(self.open_f, f_start, 0, start_hash)
        self._push(self.open_b, f_goal,  0, goal_hash)

        if self.full_f2f:
            # At t=0 each OPEN list is a single seed, so the seed-anchor scores
            # above already equal the 1-element full-F2F min; only stamp the
            # frontier version (start re: backward seed, goal re: forward seed).
            self.frontier_ver_f = self.frontier_ver_b = 0
            self.last_ver_f[start_hash] = 0
            self.last_ver_b[goal_hash] = 0
        else:
            self.last_target_f[start_hash] = goal_hash
            self.last_target_b[goal_hash]  = start_hash

        # Register start/goal as GENERATED (full-key, forward frame) so a
        # generated-vs-generated meeting can fire against them.
        if self.meet_on_generate:
            self.fkey_gen_f[self._full_key(self.puzzle)] = start_hash
            goal_fwd = self.backward_game.flipGame(self.backward_game.puzzle)
            self.fkey_gen_b[self._full_key(goal_fwd)] = goal_hash

        self._initialized = True

    # ──────────────────────────────────────────────────────────────────
    # Core expansion step
    # ──────────────────────────────────────────────────────────────────

    def _expand(self, is_forward: bool) -> Tuple[bool, Optional[List[str]]]:
        """
        One TTBS expansion on the chosen side.
        Returns (solution_found, path_or_None).
        """
        if is_forward:
            heap         = self.open_f
            g_map        = self.g_f
            parent_map   = self.parent_f
            closed       = self.closed_f
            fkey_closed  = self.fkey_closed_f
            fkey_opp     = self.fkey_closed_b
            fkey_map_self= self.fkey_to_hash_f
            fkey_map_opp = self.fkey_to_hash_b
            fkey_gen_self= self.fkey_gen_f
            fkey_gen_opp = self.fkey_gen_b
            last_tgt     = self.last_target_f
            game         = self.forward_game
            opp_game     = self.backward_game
            opp_g_map    = self.g_b
            cur_anch     = 'anchor_f'
            opp_anch     = 'anchor_b'
        else:
            heap         = self.open_b
            g_map        = self.g_b
            parent_map   = self.parent_b
            closed       = self.closed_b
            fkey_closed  = self.fkey_closed_b
            fkey_opp     = self.fkey_closed_f
            fkey_map_self= self.fkey_to_hash_b
            fkey_map_opp = self.fkey_to_hash_f
            fkey_gen_self= self.fkey_gen_b
            fkey_gen_opp = self.fkey_gen_f
            last_tgt     = self.last_target_b
            game         = self.backward_game
            opp_game     = self.forward_game
            opp_g_map    = self.g_f
            cur_anch     = 'anchor_b'
            opp_anch     = 'anchor_f'

        # When meeting on generation, the opposite side's GENERATED map is the
        # set/lookup to test against (a superset of its CLOSED map).
        meet_set = fkey_gen_opp if self.meet_on_generate else fkey_opp
        meet_map = fkey_gen_opp if self.meet_on_generate else fkey_map_opp

        # Opponent anchor per anchor_strategy ("temporal" returns the stored
        # last-expanded value verbatim — byte-identical default).
        opp_anchor = self._get_opp_anchor(is_forward)

        # ── FULL front-to-front: snapshot the opponent OPEN frontier once per
        # expansion (it cannot change mid-expansion — only the active side runs)
        # and reuse it for both lazy re-eval and survivor scoring. ──────────
        if self.full_f2f:
            opp_heap   = self.open_b   if is_forward else self.open_f
            opp_closed = self.closed_b if is_forward else self.closed_f
            opp_ver    = self.frontier_ver_b if is_forward else self.frontier_ver_f
            last_ver   = self.last_ver_f if is_forward else self.last_ver_b
            f2f_hashes = self._live_frontier_hashes(opp_heap, opp_closed, opp_g_map)
            f2f_others = self._frontier_others(f2f_hashes, game, opp_game, opp_ver)

        if not heap:
            return False, None

        # ── TTBS Lazy Re-evaluation ────────────────────────────────────
        while True:
            if not heap:
                return False, None

            f, g, u_hash = self._pop(heap)

            # Skip already-closed nodes
            if u_hash in closed:
                continue

            # Skip superseded g-values (another path improved this node)
            if g_map.get(u_hash, float('inf')) < g:
                continue

            if self.full_f2f:
                # Re-score iff the opponent frontier changed since u was scored.
                if last_ver.get(u_hash) != opp_ver:
                    if self.nn is not None:
                        new_f = self._f_score_nn_f2f(u_hash, g, f2f_others, game)
                    else:
                        new_f = self._f_score_f2f(u_hash, g, f2f_others, game)
                    last_ver[u_hash] = opp_ver
                    self._push(heap, new_f, g, u_hash)
                    continue
            else:
                # Lazy re-evaluation: anchor changed since this f-score was set
                if last_tgt.get(u_hash) != opp_anchor:
                    if self.nn is not None:
                        new_f = self._f_score_nn(u_hash, g, opp_anchor, game, opp_game)
                    else:
                        new_f = self._f_score(u_hash, g, opp_anchor, game, opp_game)
                    last_tgt[u_hash] = opp_anchor
                    self._push(heap, new_f, g, u_hash)
                    continue

            break

        # ── Close node ────────────────────────────────────────────────
        closed.add(u_hash)

        # Record the anchor this node's children are about to be scored
        # against (opp_anchor is fixed for the whole expansion and is what
        # the survivor-scoring pass below uses). Logged at close so a later
        # ranking-loss trainer can rebuild the exact per-step F2F target.
        if self.log_expansions:
            (self.expand_anchor_f if is_forward
             else self.expand_anchor_b)[u_hash] = opp_anchor

        # Register meeting keys for this side (forward-frame).
        u_map    = game.decodeMap(u_hash)
        u_fwd    = u_map if is_forward else game.flipGame(u_map)
        fk       = self._full_key(u_fwd)        # full-state key (agent + boxes)
        fkey_closed.add(fk)
        fkey_map_self[fk] = u_hash

        # Update this side's anchor per the anchor-selection strategy.
        if self.anchor_strategy in ("closest_anchor", "hybrid_af"):
            # Policy A on the relevant side(s): adopt the just-expanded node as
            # the anchor only if it is strictly closer (by h) to the opponent
            # anchor than the current one (meet-in-middle). closest_anchor uses A
            # on BOTH sides; hybrid_af uses A on the forward side and F (fixed,
            # no update) on the backward side, so anchor_b stays at the goal seed.
            side_uses_A = (self.anchor_strategy == "closest_anchor") or is_forward
            if side_uses_A:
                opp_a = getattr(self, opp_anch)
                cur_a = getattr(self, cur_anch)
                if opp_a is None or cur_a is None:
                    setattr(self, cur_anch, u_hash)
                elif self._pair_h(u_hash, opp_a, game, opp_game) < \
                        self._pair_h(cur_a, opp_a, game, opp_game):
                    setattr(self, cur_anch, u_hash)
            # else policy F: leave this side's anchor fixed (no update).
        else:
            # "temporal" (default): last-expanded becomes the anchor.
            # "top_of_open": stored value is unused (anchor read from live OPEN),
            # so this write is harmless.
            setattr(self, cur_anch, u_hash)

        # ── Expand successors ─────────────────────────────────────────
        game.puzzle = u_map   # required by availableStates
        player_loc = game.getPlayerLocation(u_map)

        # Adjacency list for the actual transition graph (not the spanning
        # tree). We populate this for *every* generated successor, even
        # ones that are already closed or g-dominated, because the post-
        # hoc path-refinement BFS needs the true graph topology.
        edges_map = self.edges_f if is_forward else self.edges_b
        u_adj = edges_map.setdefault(u_hash, set())

        # ── Phase 1: generate, filter, intersection-check ──────────────
        # Collect the survivors (those that still need an f-score) so the
        # learned heuristic can score them all in ONE batched NN call below
        # (Axis-1 within-node batching). The intersection check can short-
        # circuit the whole expansion on a meeting, in which case zero
        # inference is spent. Order/node-count are unchanged vs. per-node
        # scoring — this only groups the inference.
        survivors = []   # list of (v_hash, new_g, v_map)
        for _dir, action in game.availableStates(player_loc):
            v_map = action.moveAndUpdateBoard(player_loc, u_map)
            if v_map is None:
                continue

            # Deadlock pruning for forward direction only
            if is_forward and game.hasDeadlock(v_map):
                continue

            v_hash = game.encodeMap(v_map)
            new_g  = g + 1

            # Record the transition u→v in the induced graph regardless
            # of whether v is already closed or g-dominated. This is what
            # distinguishes the transition graph from the parent-tree.
            u_adj.add(v_hash)

            if v_hash in closed:
                continue
            if new_g >= g_map.get(v_hash, float('inf')):
                continue

            # Record path
            g_map[v_hash]      = new_g
            parent_map[v_hash] = u_hash
            if self.full_f2f:
                last_ver[v_hash] = opp_ver
            else:
                last_tgt[v_hash] = opp_anchor

            # ── Intersection check (O(1)) ─────────────────────────────
            # Full-state meeting requires the agent position to match too, so
            # the seam is a genuinely shared state and the reconstructed plan
            # is a valid step-by-step path. Against the opposite side's CLOSED
            # set by default, or its GENERATED set when meet_on_generate (which
            # fires as soon as both frontiers have generated the shared state).
            v_fwd = v_map if is_forward else game.flipGame(v_map)
            key_v = self._full_key(v_fwd)

            if self.meet_on_generate:
                fkey_gen_self[key_v] = v_hash   # register this generation

            if key_v in meet_set:
                opp_hash = meet_map[key_v]
                opp_g    = opp_g_map.get(opp_hash, float('inf'))
                cost     = new_g + opp_g
                if cost < self.U:
                    self.U = cost
                    if is_forward:
                        self.meeting_fwd = v_hash
                        self.meeting_bwd = opp_hash
                    else:
                        self.meeting_bwd = v_hash
                        self.meeting_fwd = opp_hash
                    if self.first_meeting_iter is None:
                        self.first_meeting_iter = self.iteration
                # Satisficing: return immediately on first intersection.
                return True, self.reconstruct_path()

            survivors.append((v_hash, new_g, v_map))

        # ── Phase 2: score survivors and push ──────────────────────────
        if self.full_f2f:
            if self.nn is not None:
                f_scores = self._f_score_nn_batch_f2f(
                    [vm for _, _, vm in survivors],
                    [ng for _, ng, _ in survivors],
                    f2f_others, game)
                for (v_hash, new_g, _), v_f in zip(survivors, f_scores):
                    self._push(heap, v_f, new_g, v_hash)
            else:
                for v_hash, new_g, _ in survivors:
                    v_f = self._f_score_f2f(v_hash, new_g, f2f_others, game)
                    self._push(heap, v_f, new_g, v_hash)
        elif self.nn is not None:
            f_scores = self._f_score_nn_batch(
                [vm for _, _, vm in survivors],
                [ng for _, ng, _ in survivors],
                opp_anchor, game, opp_game)
            for (v_hash, new_g, _), v_f in zip(survivors, f_scores):
                self._push(heap, v_f, new_g, v_hash)
        else:
            for v_hash, new_g, _ in survivors:
                v_f = self._f_score(v_hash, new_g, opp_anchor, game, opp_game)
                self._push(heap, v_f, new_g, v_hash)

        # FULL-F2F: bump THIS side's frontier version once per expansion (covers
        # both the close and any pushes). After the meeting-return above, so it
        # never fires on the solving expansion.
        if self.full_f2f:
            if is_forward:
                self.frontier_ver_f += 1
            else:
                self.frontier_ver_b += 1

        return False, None

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self) -> Tuple[bool, Optional[List[str]]]:
        """
        One alternating expansion step.
        Picks the side with the smaller open list (balancing strategy).

        Returns (solution_found, path_or_None).
        """
        if not self._initialized:
            self.init_search()

        if not self.open_f and not self.open_b:
            return False, None

        if not self.open_b or (self.open_f and len(self.open_f) <= len(self.open_b)):
            found, path = self._expand(is_forward=True)
        else:
            found, path = self._expand(is_forward=False)

        self.iteration += 1
        return found, path

    # ──────────────────────────────────────────────────────────────────
    # Path reconstruction
    # ──────────────────────────────────────────────────────────────────

    def reconstruct_path(self) -> List[str]:
        """
        Reconstruct start → goal as a list of forward-game encoded maps.

        Forward half:  trace parent_f from meeting_fwd → start, then reverse.
        Backward half: trace parent_b from meeting_bwd → goal, flip each
                       state into forward coordinate system.
        """
        if self.meeting_fwd is None:
            return []

        # Forward segment: meeting_fwd → start (reversed)
        path_f: List[str] = []
        curr = self.meeting_fwd
        while curr is not None:
            path_f.append(curr)
            curr = self.parent_f.get(curr)
        path_f.reverse()   # start → meeting_fwd

        # Backward segment: parent of meeting_bwd → goal (already fwd-flipped)
        path_b: List[str] = []
        curr = self.parent_b.get(self.meeting_bwd)   # skip meeting (already in path_f)
        while curr is not None:
            bwd_map = self.backward_game.decodeMap(curr)
            fwd_map = self.forward_game.flipGame(bwd_map)
            path_b.append(self.forward_game.encodeMap(fwd_map))
            curr = self.parent_b.get(curr)

        return path_f + path_b

    def reconstruct_segments(self) -> Tuple[List[str], List[str]]:
        """Split the solution path by frontier, each in its OWN game frame,
        ordered seed → meeting:

            fwd_chain: [start, ..., meeting_fwd]   (forward-game hashes)
            bwd_chain: [goal,  ..., meeting_bwd]   (backward-game hashes)

        Consecutive entries are parent → child in that frontier's search
        tree, so ``chain[i]`` was expanded to generate ``chain[i+1]``. Unlike
        ``reconstruct_path`` (which flips everything into the forward frame and
        concatenates), this keeps each side's native hashes so the per-step
        anchor log and the induced-graph edges can be looked up directly.
        """
        if self.meeting_fwd is None:
            return [], []
        fwd: List[str] = []
        curr = self.meeting_fwd
        while curr is not None:
            fwd.append(curr)
            curr = self.parent_f.get(curr)
        fwd.reverse()
        bwd: List[str] = []
        curr = self.meeting_bwd
        while curr is not None:
            bwd.append(curr)
            curr = self.parent_b.get(curr)
        bwd.reverse()
        return fwd, bwd

    # ──────────────────────────────────────────────────────────────────
    # Public search interface
    # ──────────────────────────────────────────────────────────────────

    def search(self, max_iterations: int = 10000) -> Optional[List[str]]:
        """Execute TTBS bidirectional search, returning the reconstructed
        path (start → goal) at the first frontier intersection, or None if
        the search graph is exhausted (no meeting exists) or the iteration
        budget runs out first.
        """
        self.init_search()
        for _ in range(max_iterations):
            # Both frontiers exhausted ⇒ the reachable search graph is fully
            # explored and no full-state meeting exists. Expansion only ever
            # adds to the open lists, so once both are empty no later step can
            # change the outcome — stop now instead of spinning out the rest
            # of the (possibly huge) budget on no-op steps.
            if not self.open_f and not self.open_b:
                break
            found, path = self.step()
            if found:
                return path
        return None
