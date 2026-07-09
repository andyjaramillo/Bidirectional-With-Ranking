import os
import pickle
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Flat transition buffer for heuristic learning.

    Each entry is (state, target, other_state, distance, puzzle_id) where
    ``puzzle_id`` is an optional tag identifying which puzzle this pair was
    mined from (used for re-mining). ``sample()`` strips the tag, so training
    code sees 4-element tuples.

    Prioritized replay (optional): ``sample_per()`` draws entries proportional
    to (|regression error| + eps)^alpha instead of uniformly (Schaul et al.
    2015, proportional variant). Priorities live in a numpy circular array
    mirroring the deque's FIFO eviction (``_prio_head`` = physical slot of
    logical index 0), so the default ``sample()`` path is untouched — same
    entries, same RNG stream — when PER is off. New entries enter at the
    current max priority so each is seen at least once before being down-
    weighted; sampled entries have their priority refreshed by the caller via
    ``update_priorities()``.
    """

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        # Prioritized-replay bookkeeping (inert unless sample_per is used).
        self._prio = np.zeros(capacity, dtype=np.float64)
        self._prio_head = 0
        self._max_prio = 1.0

    def add(self, state, target, other, distance, puzzle_id=None):
        cap = self.buffer.maxlen
        if len(self.buffer) == cap:
            # Deque evicts logical index 0 (physical slot _prio_head); the new
            # entry recycles that slot as the new logical LAST element.
            self._prio[self._prio_head] = self._max_prio
            self._prio_head = (self._prio_head + 1) % cap
        else:
            self._prio[(self._prio_head + len(self.buffer)) % cap] = self._max_prio
        self.buffer.append((
            np.array(state, copy=True),
            target,
            np.array(other, copy=True),
            float(distance),
            puzzle_id,
        ))

    def add_pairs_from_path(self, decoded_path, target,
                            num_random_pairs=None, symmetric=True,
                            include_endpoints=True, rng=None,
                            puzzle_id=None):
        """Pair-based labeling: store (s_i, s_j, |i-j|) tuples from a path.

        - ``num_random_pairs`` random (i, j) pairs are sampled.
        - Pairs involving the start (i=0) and goal (i=L-1) are always
          included when ``include_endpoints`` is True.
        - If ``symmetric``, each pair is added in both orderings.
        - All entries are tagged with ``puzzle_id``.
        Returns the number of entries added.
        """
        rng = rng or random
        L = len(decoded_path)
        if L < 2:
            return 0
        if num_random_pairs is None:
            num_random_pairs = 2 * L

        pairs = set()
        if include_endpoints:
            for k in range(1, L):
                pairs.add((0, k))
            for k in range(0, L - 1):
                pairs.add((k, L - 1))
        for _ in range(num_random_pairs):
            i, j = rng.randrange(L), rng.randrange(L)
            if i != j:
                pairs.add((min(i, j), max(i, j)))

        added = 0
        for i, j in pairs:
            dist = j - i
            self.add(decoded_path[i], target, decoded_path[j], dist, puzzle_id)
            added += 1
            if symmetric:
                self.add(decoded_path[j], target, decoded_path[i], dist, puzzle_id)
                added += 1
        return added

    def sample(self, batch_size):
        """Return a random minibatch WITHOUT the puzzle_id tag."""
        n = min(batch_size, len(self.buffer))
        return [e[:4] for e in random.sample(self.buffer, n)]

    def _logical_prios(self):
        """Priorities aligned with the deque's logical order."""
        n, cap = len(self.buffer), self.buffer.maxlen
        idx = (self._prio_head + np.arange(n)) % cap
        return self._prio[idx]

    def sample_per(self, batch_size, alpha=0.6, beta=0.4):
        """Prioritized minibatch (proportional PER, with replacement).

        Draws logical indices with probability P(i) ∝ prio_i^alpha and returns
        (samples, logical_indices, is_weights) where the importance-sampling
        weights w_i = (N·P(i))^(−beta), max-normalized, undo the sampling bias
        when applied per-sample to the regression term. Caller must refresh the
        drawn entries' priorities with ``update_priorities`` right after the
        forward pass (before any further ``add``, which shifts logical
        indices). Consumes numpy's global RNG (seeded by the driver).
        """
        n = len(self.buffer)
        k = min(batch_size, n)
        probs = self._logical_prios() ** alpha
        probs /= probs.sum()
        chosen = np.random.choice(n, size=k, replace=True, p=probs)
        w = (n * probs[chosen]) ** (-beta)
        w /= w.max()
        samples = [self.buffer[int(i)][:4] for i in chosen]
        return samples, chosen, w

    def update_priorities(self, logical_indices, errors, eps=0.5):
        """Set priority (|error| + eps) for the given logical indices."""
        n, cap = len(self.buffer), self.buffer.maxlen
        for i, err in zip(logical_indices, errors):
            i = int(i)
            if 0 <= i < n:
                pr = abs(float(err)) + eps
                self._prio[(self._prio_head + i) % cap] = pr
                if pr > self._max_prio:
                    self._max_prio = pr

    def remove_by_tag(self, puzzle_id):
        """Drop every entry tagged with ``puzzle_id``. Returns count dropped."""
        prios = self._logical_prios()
        kept, kept_p = [], []
        for e, pi in zip(self.buffer, prios):
            if e[-1] != puzzle_id:
                kept.append(e)
                kept_p.append(pi)
        dropped = len(self.buffer) - len(kept)
        if dropped:
            self.buffer = deque(kept, maxlen=self.buffer.maxlen)
            self._prio[:] = 0.0
            if kept_p:
                self._prio[:len(kept_p)] = kept_p
            self._prio_head = 0
        return dropped

    def oldest_tag(self):
        """puzzle_id of the front (oldest) entry, or None if empty."""
        return self.buffer[0][-1] if self.buffer else None

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((list(self.buffer), self.buffer.maxlen), f)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data, maxlen = pickle.load(f)
        self.buffer = deque(data, maxlen=maxlen)
        # Priorities are not persisted: restart every loaded entry at max.
        self._prio = np.zeros(maxlen, dtype=np.float64)
        self._prio[:len(self.buffer)] = self._max_prio
        self._prio_head = 0
        return True
