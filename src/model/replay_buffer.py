import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        pass

    def add(self, state, target, goal_map, cost_to_go):
        """
            goal map is the final state on the other search instance

        """

        self.buffer.append((
            np.array(state, copy=True),
            target,
            np.array(goal_map, copy=True),
            cost_to_go
        ))

    def add_pairs_from_path(self, decoded_path, target, num_random_pairs=None, include_endpoint=True):
        """
            include_endpoints is needed since all pairwise of the states tied back to the associated 
            goal state

        """
        L = len(decoded_path)
        pairs = set() ## set so we dont have duplicates

        if include_endpoint:
            for k in range(0,L):
                pairs.add((0, k))
            for k in range(0, L-1):
                pairs.add((k, L-1))

        ## now we have all endpoints, we need the in between pairs

        if num_random_pairs is None:
            num_random_pairs = 2*L
        

        ## we first append integer pairings to make use of the unique attribute in sets

        for _ in range(num_random_pairs):
            i = random.randrange(L)
            j = random.randrange(L)
            if i == j:
                continue
            pairs.add((min(i,j), max(i,j)))

        ## now we can loop over the pairs and append the actual states into the buffer

        for i,j in pairs:
            dist = i - j
            self.add(decoded_path[i], target, decoded_path[j], dist)

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)
