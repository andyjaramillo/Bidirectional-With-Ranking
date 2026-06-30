"""Genuine ranking loss for the bidirectional F2F search.

The forward ranking paper (Chrestien et al., "Optimize Planning Heuristics to
Rank", NeurIPS 2023) trains a heuristic so that, at each step of the search,
the on-(solution-)path state has the LOWEST merit in the open list — the search
then expands it first ("perfect ranking", their Definition 1). The merit is
f = alpha*g + beta*h:  L* uses alpha=beta=1 (rank on g+h, for A*); L_gbfs uses
alpha=0 (rank on h, for GBFS). See rank_forward/losses.py for the forward port.

This module carries that condition over to our bidirectional TTBS, where the
heuristic is the PAIRWISE NN h(s, d) and the search scores f(s)=g(s)+h(s, d)
against a MOVING anchor d. The "genuine" adaptation (the design agreed with the
user):

  * Per-step anchor (faithful). When a path parent ``u_i`` is expanded, the
    search scores ALL of ``u_i``'s children against one ``opp_anchor`` — the one
    live at that expansion (the survivor-scoring pass uses it). We logged that
    anchor (``s.log_expansions``). The on-path child ``u_{i+1}`` must have the
    lowest merit among ``u_i``'s children under THAT anchor — i.e. against the
    exact F2F target the search actually used at that step.
  * Negatives = ``u_i``'s off-path children — the open-list approximation the
    paper uses, and exactly what rank_forward/losses.py uses — read from the
    induced transition graph ``s.edges_*``.
  * Both frontiers. The forward segment (start->meeting) is scored against the
    live backward anchor per step; the backward segment (goal->meeting) against
    the live forward anchor per step.
  * Pure ranking. The only objective is the sum over steps and siblings of
    ``softplus(f_on - f_neg)`` — penalised when the on-path child is not strictly
    below an off-path sibling.

We rank on the RAW h (no clamp), matching the paper and rank_forward/losses.py,
so gradients flow even where h would be clamped at search time; for a sane
distance heuristic h>0, so the search-time ``clamp_min(0)`` is a no-op there.

Nothing here is Sokoban-specific: it consumes hashes, the induced graph, the
per-step anchors, and the model's pairwise ``forward_batch``.
"""
import torch
import torch.nn.functional as F


def _segment_groups(s, chain, is_forward):
    """For each expanded on-path parent in ``chain`` that has off-path siblings,
    yield (target, other_board, child_boards, g_list) where ``child_boards[0]``
    is the on-path child and the rest are the off-path siblings, all in the
    active frontier's game frame and scored against the per-step anchor."""
    if len(chain) < 2:
        return
    if is_forward:
        active_game, opp_game = s.forward_game, s.backward_game
        edges, g_map, anchor_log = s.edges_f, s.g_f, s.expand_anchor_f
    else:
        active_game, opp_game = s.backward_game, s.forward_game
        edges, g_map, anchor_log = s.edges_b, s.g_b, s.expand_anchor_b
    target = active_game.target
    for i in range(len(chain) - 1):
        u = chain[i]
        on_child = chain[i + 1]
        kids = edges.get(u)
        if not kids:
            continue
        negs = sorted(k for k in kids if k != on_child)
        if not negs:
            continue
        # The opponent anchor that was live when u was expanded — i.e. what the
        # search scored u's children against. _anchor_other flips it into the
        # active frame exactly as the search's f-scorers do (None -> goal_map).
        other = s._anchor_other(anchor_log.get(u), active_game, opp_game)
        child_hashes = [on_child] + negs
        child_boards = [active_game.decodeMap(h) for h in child_hashes]
        # All of u's freshly generated children enter the open list at the same
        # depth g = g(u)+1 — exactly the f the search assigns when it pushes
        # them — so g is UNIFORM across the group. (Consequence: in f=g+h the g
        # term cancels in every f_on - f_sibling difference, so L* and L_gbfs
        # coincide for these same-depth, same-anchor negatives. We still wire g
        # in faithfully below.)
        g_parent = g_map.get(u, 0)
        g_list = [g_parent + 1] * len(child_hashes)
        yield target, other, child_boards, g_list


def bidir_ranking_loss(model, s, use_g=True, gbfs=False):
    """Genuine per-step ranking loss for one solved bidirectional search ``s``
    (which must have run with ``log_expansions=True``).

    Returns a scalar loss tensor (with grad), or ``None`` if the search found no
    path or has no rankable step. ``gbfs=True`` (or ``use_g=False``) gives
    L_gbfs (rank on h only); otherwise L* (rank on g+h). NOTE: because the
    negatives are same-depth siblings of one parent (all at g=g(u)+1), the g
    term cancels in every pairwise difference, so L* and L_gbfs produce
    IDENTICAL gradients here — the flag is kept only for API parity with the
    forward losses. The loss therefore shapes only the RELATIVE h of the
    on-path child vs its siblings; the absolute scale of h (which matters for
    the f=g+h search) is left free — the pure-ranking trade-off the toggle
    exists to measure.
    """
    fwd, bwd = s.reconstruct_segments()
    if not fwd:
        return None
    alpha = 0.0 if (gbfs or not use_g) else 1.0
    beta = 1.0

    groups = (list(_segment_groups(s, fwd, True))
              + list(_segment_groups(s, bwd, False)))
    if not groups:
        return None

    # One batched forward pass over every child board across all groups.
    states, targets, others, spans, gs = [], [], [], [], []
    cursor = 0
    for target, other, child_boards, g_list in groups:
        k = len(child_boards)
        states.extend(child_boards)
        targets.extend([target] * k)
        others.extend([other] * k)
        spans.append((cursor, cursor + k))
        gs.append(g_list)
        cursor += k

    h_all = model.forward_batch(states, targets, others)
    if h_all.ndim == 0:
        h_all = h_all.unsqueeze(0)

    terms = []
    for (a, b), g_list in zip(spans, gs):
        h = h_all[a:b]
        if alpha != 0.0:
            g = torch.tensor(g_list, dtype=h.dtype, device=h.device)
            f = alpha * g + beta * h
        else:
            f = beta * h
        # On-path child is index 0; every off-path sibling is a negative.
        # Want f[0] strictly below each f[1:], penalise the violation.
        terms.append(F.softplus(f[0] - f[1:]).sum())

    return torch.stack(terms).sum()
