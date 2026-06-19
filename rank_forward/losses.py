"""The five loss functions, in PyTorch.

All operate per problem instance (one minibatch = one instance, as the paper
requires). ``h`` values come from one batched forward pass over exactly the
states a loss needs.

Notation (unit action costs):  g(s_i)=i on the path, hstar(s_i)=L-i,
off-path sibling of s_p has g = p+1.  Merit r(s_i,s_j)=alpha(g_i-g_j)+beta(h_i-h_j).
We want every on-path s_i to have strictly lower merit than the off-path s_j in
the open list, i.e. r(s_i,s_j) < 0, penalised by softplus(r) = log(1+exp(r)).

  L*    : ranking, alpha=beta=1  (rank on f=g+h)   -- for A*
  Lgbfs : ranking, alpha=0,beta=1 (rank on h)      -- for GBFS
  Lrt   : softplus over consecutive on-path pairs (Eq. 6), on-path only
  L2    : (h - hstar)^2, on-path only (regression to cost-to-goal)
  Lbe   : Bellman loss of Stahlberg et al. (ref [49])

Ranking pairing follows Definition 1 / Eq. (1) literally: an off-path sibling
of s_p stays in the open list from step p+1 onward, so it is contrasted with
EVERY later on-path state s_i for i in [p+1, L] (the "full" triangular pairing).
Set ``local_pairs=True`` to contrast each sibling only with its same-depth
competitor s_{p+1} (a cheaper approximation).
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

RANKING = ("lstar", "lgbfs")
ALL_LOSSES = ("lstar", "lgbfs", "lrt", "l2", "bellman")


def _h(model, inst, states: List[np.ndarray]) -> torch.Tensor:
    """Batched heuristic values h(states) as a (len(states),) grad tensor."""
    n = len(states)
    return model.forward_batch(states, [inst.target] * n, [inst.goal_ctx] * n)


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return x.mean() if x.numel() else x.sum()
    return x.sum()


def _ranking_loss(model, inst, alpha: float, beta: float, reduction: str,
                  local_pairs: bool) -> torch.Tensor:
    L = inst.L
    path = inst.path
    off = inst.off_states
    if not off:
        # No off-path competitors -> no ranking constraints; keep grad graph.
        return _h(model, inst, path).sum() * 0.0

    h_all = _h(model, inst, list(path) + list(off))
    h_path = h_all[:L + 1]                      # (L+1,)
    h_off = h_all[L + 1:]                       # (M,)

    g_path = torch.arange(L + 1, dtype=h_all.dtype)
    parent = torch.tensor(inst.off_parent, dtype=torch.long)
    g_off = (parent + 1).to(h_all.dtype)

    f_path = alpha * g_path + beta * h_path     # (L+1,)
    f_off = alpha * g_off + beta * h_off        # (M,)

    if local_pairs:
        # Contrast each sibling only with its same-depth competitor s_{p+1}.
        comp = f_path[parent + 1]               # (M,)
        r = comp - f_off                        # want comp < f_off
        return _reduce(F.softplus(r), reduction)

    # Full Definition-1 pairing: sibling of s_p vs every s_i, i in [p+1, L].
    i_idx = torch.arange(L + 1).unsqueeze(1)    # (L+1, 1)
    mask = i_idx >= (parent + 1).unsqueeze(0)   # (L+1, M)
    D = f_path.unsqueeze(1) - f_off.unsqueeze(0)  # (L+1, M) = r(s_i, s_j)
    return _reduce(F.softplus(D[mask]), reduction)


def lstar_loss(model, inst, reduction="sum", local_pairs=False):
    return _ranking_loss(model, inst, 1.0, 1.0, reduction, local_pairs)


def lgbfs_loss(model, inst, reduction="sum", local_pairs=False):
    return _ranking_loss(model, inst, 0.0, 1.0, reduction, local_pairs)


def lrt_loss(model, inst, reduction="sum"):
    """Eq. (6): rank consecutive on-path states so h decreases toward the goal."""
    h_path = _h(model, inst, inst.path)
    diff = h_path[1:] - h_path[:-1]             # deeper minus shallower
    return _reduce(F.softplus(diff), reduction)


def l2_loss(model, inst, reduction="sum"):
    """Eq. (5): regress h to cost-to-goal hstar(s_i) = L - i (unit costs)."""
    L = inst.L
    h_path = _h(model, inst, inst.path)
    hstar = torch.arange(L, -1, -1, dtype=h_path.dtype)   # [L, L-1, ..., 0]
    return _reduce((h_path - hstar) ** 2, reduction)


def bellman_loss(model, inst, reduction="sum"):
    """Bellman loss (ref [49]):
        sum_s max{1 + min_{s'} h(s') - h(s), 0}
             + max{0, hstar(s) - h(s)} + max{0, h(s) - 2 hstar(s)}
    over non-goal path states s, with s' ranging over the children of s."""
    L = inst.L
    # Batch = path states + all per-node off-children (so every child has an h).
    states = list(inst.path)
    child_slices = []                           # (start, end) into off block per node i
    off_block = []
    cursor = L + 1
    for i in range(L):
        offs = inst.node_off[i]
        child_slices.append((cursor, cursor + len(offs)))
        off_block.extend(offs)
        cursor += len(offs)
    h_all = _h(model, inst, states + off_block)
    h_path = h_all[:L + 1]

    terms = []
    for i in range(L):                          # non-goal path nodes
        hstar_i = float(L - i)
        h_i = h_path[i]
        # children of s_i: on-path successor s_{i+1} plus off-path siblings.
        a, b = child_slices[i]
        child_h = torch.cat([h_path[i + 1:i + 2], h_all[a:b]])
        res = F.relu(1.0 + child_h.min() - h_i)
        reg = F.relu(hstar_i - h_i) + F.relu(h_i - 2.0 * hstar_i)
        terms.append(res + reg)
    return _reduce(torch.stack(terms), reduction)


def instance_loss(model, inst, loss_name: str, reduction="sum",
                  local_pairs=False) -> torch.Tensor:
    loss_name = loss_name.lower()
    if loss_name == "lstar":
        return lstar_loss(model, inst, reduction, local_pairs)
    if loss_name == "lgbfs":
        return lgbfs_loss(model, inst, reduction, local_pairs)
    if loss_name == "lrt":
        return lrt_loss(model, inst, reduction)
    if loss_name == "l2":
        return l2_loss(model, inst, reduction)
    if loss_name == "bellman":
        return bellman_loss(model, inst, reduction)
    raise ValueError(f"unknown loss {loss_name!r}; choose from {ALL_LOSSES}")
