import numpy as np
import heapq
from typing import Tuple, Dict, List, Optional, Set
from src.environment.SokobanGame import SokobanGame
import torch
from src.algos.Algo import Algo
class TTBS(Algo):
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

    where d* is the opponent's current anchor and h is the F2F heuristic
    (here: MWPM Manhattan distance between box positions).

    When the opponent's anchor changes, stored f-scores may be stale.
    TTBS handles this via lazy re-evaluation: when a node is popped from
    the heap, if its stored target differs from the current anchor, it is
    re-scored and re-inserted (the "TTBS re-evaluation" mechanism). This
    avoids the full open-list re-sort needed by d-node retargeting (DNR).

    Convergence is detected when a successor's box configuration matches
    a state hash already in the opponent's closed set (O(1) lookup via a
    precomputed box-key set maintained alongside each closed set).

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
        - flipGame converts between the two orientations so their box
          hashes can be compared for intersection detection.
    """

    def __init__(self, puzzle: np.ndarray,  nn=None):
        self.game_name = "Sokoban"
        self.puzzle = puzzle
        self.nn = nn

        # ── Game instances ─────────────────────────────────────────────
        self.game = SokobanGame(puzzle, isBackward=False)
        backward_puzzle = self.game.initializeBackwardPuzzle(puzzle)
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

        # ── Closed sets: encoded map hash → True ──────────────────────
        self.closed_f: Set[str] = set()
        self.closed_b: Set[str] = set()

        # ── Full transition graph induced over visited states ──────────
        # parent_f / parent_b only record one (best-g) predecessor per
        # state, so as a graph they form a tree. We additionally record
        # every successor edge generated during expansion — including
        # transitions into already-closed nodes — so post-hoc analyses
        # can run shortest-path queries over the *actual* graph induced
        # by the search, not just its spanning tree.

        self.edges_f: Dict[str, Set[str]] = {}
        self.edges_b: Dict[str, Set[str]] = {}

                # Full-state keys: the encoded forward-frame board (agent + boxes).
        self.fkey_closed_f: Set[str] = set()
        self.fkey_closed_b: Set[str] = set()   # backward states flipped to fwd
        self.fkey_to_hash_f: Dict[str, str] = {}
        self.fkey_to_hash_b: Dict[str, str] = {}

        # Box-only keys (canonical box-position bytes in the forward frame).
        self.bkey_closed_f: Set[str] = set()
        self.bkey_closed_b: Set[str] = set()   # backward box keys (flipped to fwd)
        self.bkey_to_hash_f: Dict[str, str] = {}
        self.bkey_to_hash_b: Dict[str, str] = {}

        # ── TTBS anchor: most recently expanded node on each side ──────
        self.anchor_f: Optional[str] = None
        self.anchor_b: Optional[str] = None

        # ── Lazy re-evaluation: last anchor used to score each node ────
        self.last_target_f: Dict[str, Optional[str]] = {}
        self.last_target_b: Dict[str, Optional[str]] = {}

        # ── Solution tracking ──────────────────────────────────────────
        self.U: float = float('inf')
        self.meeting_fwd: Optional[str] = None
        self.meeting_bwd: Optional[str] = None

        self.iteration: int = 0
        self._initialized: bool = False

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _box_key(self, board: np.ndarray) -> str:
        """
        Canonical key based solely on sorted box positions.
        Board must be in the *forward* game's coordinate system
        (i.e., boxes are value 4, targets are value 2).
        """
        locs = sorted(zip(*np.where(board == 4)))
        return str(locs)
    
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

    def _f_score_nn(self, node_hash: str, g: int, opp_anchor_hash: Optional[str],
                 active_game: SokobanGame, opp_game: SokobanGame) -> float:
        """
        f(s) = g(s) + h_nn(s, anchor)  — pairwise-NN heuristic.

        The NN learns dist(s1, s2). For the front-to-front search we feed
        (active node, opponent anchor flipped into the active frame). Same
        call works for both directions because backward-Sokoban is the
        inverse problem and dist is symmetric under that involution.
        """
        node_map = active_game.decodeMap(node_hash)
        if opp_anchor_hash is None:
           other = active_game.goal_map
        else:
            opp_anchor_map = opp_game.decodeMap(opp_anchor_hash)
            other = active_game.flipGame(opp_anchor_map)
        with torch.no_grad():
            h = float(self.nn(node_map, active_game.target, other).item())
        h = max(0.0, h)
        return float(g + h)

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
        start_hash = self.game.encodeMap(self.puzzle)
        goal_hash  = self.backward_game.encodeMap(self.backward_game.puzzle)

        self.g_f[start_hash] = 0
        self.g_b[goal_hash]  = 0
        self.parent_f[start_hash] = None
        self.parent_b[goal_hash]  = None

        # Anchors are initialised to start/goal
        self.anchor_f = start_hash
        self.anchor_b = goal_hash

        _f = self._f_score_nn if self.nn is not None else self._f_score
        f_start = _f(start_hash, 0, goal_hash,
                     self.game, self.backward_game)
        f_goal  = _f(goal_hash, 0, start_hash,
                     self.backward_game, self.game)

        self._push(self.open_f, f_start, 0, start_hash)
        self._push(self.open_b, f_goal,  0, goal_hash)

        self.last_target_f[start_hash] = goal_hash
        self.last_target_b[goal_hash]  = start_hash

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
            bkey_closed  = self.bkey_closed_f
            bkey_opp     = self.bkey_closed_b
            bkey_map_self= self.bkey_to_hash_f
            bkey_map_opp = self.bkey_to_hash_b
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
            bkey_closed  = self.bkey_closed_b
            bkey_opp     = self.bkey_closed_f
            bkey_map_self= self.bkey_to_hash_b
            bkey_map_opp = self.bkey_to_hash_f
            last_tgt     = self.last_target_b
            game         = self.backward_game
            opp_game     = self.forward_game
            opp_g_map    = self.g_f
            cur_anch     = 'anchor_b'
            opp_anch     = 'anchor_f'

        opp_anchor = getattr(self, opp_anch)

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
        # Register meeting keys for this side (forward-frame).
        u_map    = game.decodeMap(u_hash)
        u_fwd    = u_map if is_forward else game.flipGame(u_map)
        fk       = self._full_key(u_fwd)        # full-state key (agent + boxes)
        fkey_closed.add(fk)
        fkey_map_self[fk] = u_hash
        bk       = self._box_key(u_fwd)         # box-only key (legacy criterion)
        bkey_closed.add(bk)
        bkey_map_self[bk] = u_hash

        # Update temporal anchor
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
            last_tgt[v_hash]   = opp_anchor

            # ── Intersection check (O(1)) ─────────────────────────────
            v_fwd = v_map if is_forward else game.flipGame(v_map)
            
            key_v  = self._full_key(v_fwd)
            opp_set = fkey_opp
            opp_map = fkey_map_opp

            if key_v in opp_set:
                opp_hash = opp_map[key_v]
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
                # Satisficing: return immediately on first intersection
                return True, self.reconstruct_path()

            # ── Push successor ────────────────────────────────────────
            
            if self.nn is not None:
                v_f = self._f_score_nn(v_hash, new_g, opp_anchor, game, opp_game)
            else:
                v_f = self._f_score(v_hash, new_g, opp_anchor, game, opp_game)
            self._push(heap, v_f, new_g, v_hash)

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
            fwd_map = self.game.flipGame(bwd_map)
            path_b.append(self.game.encodeMap(fwd_map))
            curr = self.parent_b.get(curr)

        return path_f + path_b

    # ──────────────────────────────────────────────────────────────────
    # Public search interface
    # ──────────────────────────────────────────────────────────────────

    def search(self, max_iterations: int = 10000) -> Optional[List[str]]:
        """
        Execute TTBS bidirectional search.

        Args:
            max_iterations: Expansion budget.

        Returns:
            List of forward-encoded map strings (start → goal), or None.
        """
        self.init_search()
        for _ in range(max_iterations):
            found, path = self.step()
            if found:
                return path
        return None

    def test_search(self, max_iterations: int = 10000) -> Optional[List[str]]:
        """
        Execute TTBS bidirectional search.

        Args:
            max_iterations: Expansion budget.

        Returns:
            List of forward-encoded map strings (start → goal), or None.
        """
        self.init_search()
        for _ in range(max_iterations):
            found, path = self.step()
            if found:
                return self.iteration
        return None