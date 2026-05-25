import random
from src.algos.Algo import Algo
from src.environment.SokobanGame import SokobanGame
import heapq
import random
from src.utils.sample_random import samp_rand_norm
import sys
class SearchFrontier(Algo):
    """
    Manages the search frontier for the Anchor Search algorithm.
    """
    def __init__(self, start_state, isBackward, nn=None):
        """
        Initialize the SearchFrontier.
        
        Args:
            start_state (np.ndarray): The initial state for this frontier.
            isBackward (bool): Whether this is a backward search frontier.
            nn (NN, optional): Neural network for heuristic evaluation.
        """
        self.start_state = start_state
        self.game = SokobanGame(start_state, isBackward)
        self.nn = nn
        # self.side = side
        self.isBackward = isBackward

        # inital_start_states = SokobanGame.initalizeRandomStates(start_state, 3)
        # inital_start_states.append(start_state)
        inital_start_states = [start_state]

        self.open = inital_start_states
        # Maps hash -> {"g": cost, "parent": state, "loc": index_in_open}
        self.open_closed_set = set()
        self.open_closed = {}
        for i in range(len(inital_start_states)): 
            # print(inital_start_states[i])
            self.open_closed_set.add(self.game.getStateHash(inital_start_states[i]))
            self.open_closed[self.game.encodeMap(inital_start_states[i])] =  {
                    "g": 0, 
                    "parent": None, 
                    "loc": 0
            }
            
        self.parentMap = {}
        self.anchor = start_state
        self.anchor_h = -1
        self.anchor_g = 0
        self.sample_count = 10 # Equivalent to 'sampleCount' in C++
        self.rendezvous = None
        self.valid_solution = True

 
    def step(self, other_front, anchor_selection="Temporal"):
        """
        Perform one expansion step of the anchor search.
        
        Args:
            other_front (SearchFrontier): The opposite search frontier.
            anchor_selection (str): Method for selecting the next anchor.
            
        Returns:
            str: Status of the step ("SUCCESS", "FAILED", "FINISHED_EARLY", or "CONTINUE").
        """
        if not self.open:
            self.valid_solution = False
            return "FAILED" # Search failed


        min_dist = float('inf')
        best_candidate_idx = len(self.open) - 1
        
        # Determine how many to sample from the end of the list
        # samples = len(self.open)
        for i in range(len(self.open)):
            current_state = self.open[i]
            curr_hash = self.game.encodeMap(current_state)
            # Use your MWPM Manhattan heuristic h(s, d)
            dist = 0
            if self.nn != None:
                dist = self.nn(current_state, self.game.target, self.game.flipGame(other_front.anchor)) + self.open_closed[curr_hash]["g"]

            else:
                dist = (self.game.evaluateBoard(current_state, other_front.anchor) + self.open_closed[curr_hash]["g"])  

            
            best_hash = self.game.encodeMap(self.open[best_candidate_idx])
            
            # Tie-breaking: smaller h, then larger g
            if (dist < min_dist or 
               (dist == min_dist and self.open_closed[curr_hash]["g"] > self.open_closed[best_hash]["g"])):
                min_dist = dist
                best_candidate_idx = i

        # 2. POP BEST CANDIDATE (Efficient swap-and-pop)
        best_candidate = self.open[best_candidate_idx]
        best_hash = self.game.encodeMap(best_candidate)
        self.game.puzzle = best_candidate
        # Swap with last element to pop in O(1)
        last_state = self.open[-1]
        last_hash = self.game.encodeMap(last_state)
        self.open[best_candidate_idx] = last_state
        self.open_closed[last_hash]["loc"] = best_candidate_idx
        
        self.open.pop()
        self.open_closed[best_hash]["loc"] = -1 # Mark as no longer in Open


        # 3. INTERSECTION CHECK (Move this UP to before expansion)
        # Check if the node we just popped is in the other side's visited set
        if self.open_closed_set.isdisjoint(other_front.open_closed_set) == False:
            z = self.open_closed_set.intersection(other_front.open_closed_set)
            arbitrary_item = self.game.reconstruct_game(self.anchor, z.pop())
            self.parentMap[self.game.encodeMap(arbitrary_item)] = self.game.encodeMap(best_candidate)
            self.rendezvous = arbitrary_item
            other_front.rendezvous = other_front.game.flipGame(arbitrary_item)
            return "SUCCESS"
        

        # 4. EXPAND SUCCESSORS
        player_loc = self.game.getPlayerLocation(best_candidate)
        directions = self.game.availableStates(player_loc)
        successors = []
        for direction_and_movement in directions:
            direction = direction_and_movement[0]
            new_map = self.game.move(player_loc, direction_and_movement)
            
            if new_map is None  :
                continue
            if self.game.isGoal(new_map) == True:
                return "FINISHED_EARLY"

            successors.append(new_map)

        for neighbor in successors:
            nhash = self.game.encodeMap(neighbor)
            # Standard g-cost update
            new_g = self.open_closed[best_hash]["g"] + 1 
            
            if nhash in self.open_closed:
                if new_g < self.open_closed[nhash]["g"]:
                    self.open_closed[nhash]["g"] = new_g
                    self.open_closed[nhash]["parent"] = best_candidate
                    # RE-OPEN if it was closed
                    # if self.open_closed[nhash]["loc"] == -1:
                    #     self.open.append(neighbor)
                    #     self.open_closed[nhash]["loc"] = len(self.open) - 1
                    #     self.parentMap[nhash] = best_hash
            else:
                # print("nade with", nhash)
                # New state discovered
                self.open.append(neighbor)
                self.parentMap[nhash] = best_hash
                self.open_closed[nhash] = {
                    "g": new_g,
                    "parent": best_candidate,
                    "loc": len(self.open) - 1
                }
                self.open_closed_set.add(self.game.getStateHash(neighbor))
        # print(self.game.encodeMap(best_candidate))
        # 4. ANCHOR SELECTION LOGIC
        if anchor_selection == "Temporal":
            self.anchor = best_candidate
        elif anchor_selection == "Closest":
            # Distance to the other side's starting position
            hh = self.game.evaluateBoard(best_candidate, other_front.start_state)
            if hh < self.anchor_h or self.anchor_h < 0:
                self.anchor = best_candidate
                self.anchor_h = hh
                self.anchor_g = self.open_closed[best_hash]["g"]
            elif hh == self.anchor_h:
                if self.open_closed[best_hash]["g"] < self.anchor_g:
                    self.anchor = best_candidate
                    self.anchor_g = self.open_closed[best_hash]["g"]
        elif anchor_selection == "Random":
            if self.open:
                self.anchor = random.choice(self.open)


        return "CONTINUE"
    def reconstructPath(self):
        """
        Reconstruct the path from the start state to the rendezvous point.
        
        Returns:
            list: List of encoded maps on the path.
        """
        inital_state = self.game.encodeMap(self.start_state)
        current_game = self.game.encodeMap(self.rendezvous)
        path = [current_game]
        visit = []
        while current_game != inital_state:
            if current_game in visit:
                print("cycle")
                visit.append(current_game)
                return visit
            if current_game in self.parentMap  :

                parent = self.parentMap[current_game]
                # print(parent)
                path.append(parent)
                visit.append(current_game)
                current_game = parent
            else:
                print("ERROR: ", len(path), self.game.encodeMap(current_game), self.game.encodeMap(self.rendezvous) in self.parentMap)
                sys.exit(1)
        # path.reverse()
        return path
    
    def flipPath(self, path):
        """
        Flip the encoded maps in a path for bidirectional consistency.
        
        Args:
            path (list): List of encoded maps.
            
        Returns:
            list: Flipped path.
        """
        new_path = []
        for encodedPuzzle in path:
            new_path.append(self.game.encodeMap(self.game.flipGame(self.game.decodeMap(encodedPuzzle))))
        return new_path