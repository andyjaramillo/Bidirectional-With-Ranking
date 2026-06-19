"""Dataset: load the solvable Sokoban benchmark, split train/eval, and build
(and cache) the per-instance training trajectories.

The train/eval split is a deterministic prefix/suffix of the SAME ordered list
the bidirectional run uses (game.getData.get_solvable_data), so both methods
see identical instances. The expensive part — solving each training instance
optimally to get its path + off-path siblings — is cached to disk keyed by
(n_total, n_eval, solve_cap, use_deadlock), so repeated training runs over
different losses reuse one solve pass.
"""
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

from game.getData import get_solvable_data
from .trajectory import build_instance, Instance


def load_split(n_total: int, n_eval: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return (train_boards, eval_boards): train = first n_total-n_eval, eval =
    the next n_eval. Disjoint, deterministic."""
    boards = get_solvable_data(limit=n_total)
    if len(boards) < n_total:
        raise RuntimeError(
            f"requested {n_total} puzzles but solvable dataset has {len(boards)}")
    n_train = n_total - n_eval
    return boards[:n_train], boards[n_train:n_total]


def build_train_instances(train_boards: List[np.ndarray], solve_cap: int,
                          use_deadlock: bool, cache_path: Optional[str] = None
                          ) -> List[Instance]:
    """Solve each training board optimally and assemble its Instance, skipping
    any that fail to solve within ``solve_cap``. Cached to ``cache_path``."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    instances: List[Instance] = []
    n = len(train_boards)
    for i, b in enumerate(train_boards):
        inst = build_instance(b, max_iterations=solve_cap, use_deadlock=use_deadlock)
        if inst is not None:
            instances.append(inst)
        if (i + 1) % 100 == 0:
            print(f"  built {i+1}/{n} instances "
                  f"({len(instances)} solved)", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(instances, f)
    return instances


def cache_key(cache_dir: str, n_total: int, n_eval: int, solve_cap: int,
              use_deadlock: bool) -> str:
    name = f"train_n{n_total}_ev{n_eval}_cap{solve_cap}_dl{int(use_deadlock)}.pkl"
    return os.path.join(cache_dir, name)
