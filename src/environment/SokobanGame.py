from typing import Tuple
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
class SokobanGame:
    """
    Representation of the Sokoban game logic, supporting both forward and backward movements.
    """
    def __init__(self, puzzle, isBackward = False):
        """
        Initialize the Sokoban game.
        
        Args:
            puzzle (np.ndarray): The initial board layout.
            isBackward (bool): Whether the game is for backward search.
        """
        self.puzzle = puzzle
        self.target = list(zip(*(np.where(puzzle == 2))))
        # if isBackward:
        #     self.actions = [self.Up(self.target), self.Down(self.target), self.Left(self.target),self.Right(self.target), self.PullUp(self.target), self.PullDown(self.target), self.PullLeft(self.target), self.PullRight(self.target)]
        # else:
        #     self.actions = [self.Up(self.target), self.Down(self.target), self.Left(self.target),self.Right(self.target)]
        self.goal_map = self.initializeBackwardPuzzle(puzzle)
        # print(self.encodeMap(self.goal_map))
        self.action_map = {
            (-1, 0): self.Up(self.target, isBackward),      # Up
            (1, 0): self.Down(self.target, isBackward),    # Down
            (0, -1): self.Left(self.target, isBackward),   # Left (Note: Changed from (0, 1) in your original logic)
            (0, 1): self.Right(self.target, isBackward),   # Right (Note: Changed from (0, -1) in your original logic)
        }
        self.isBackward = isBackward
        if self.isBackward == False:
            player_loc = np.where(self.goal_map == 3)
            self.goal_map[player_loc[0][0]][player_loc[1][0]] = 1
            self.goal_map[7][5] = 3
        else:
            player_loc = np.where(self.goal_map == 3)
            self.goal_map[player_loc[0][0]][player_loc[1][0]] = 1
            self.goal_map[3][1] = 3
        # If backward search, add the 'Pull' actions to the map
        if isBackward:
            # Note: Assuming Pull actions use the same directional tuples for their 'move' logic
            self.pull_action_map = {
                (-1, 0): self.PullUp(self.target),    # PullUp (Overrides Up)
                (1, 0): self.PullDown(self.target),  # PullDown (Overrides Down)
                (0, -1): self.PullLeft(self.target), # PullLeft (Overrides Left)
                (0, 1): self.PullRight(self.target), # PullRight (Overrides Right)
            }
        # if isBidrectional == True:
        #     self.back_targets = list(zip(*(np.where(puzzle == 4))))
        #     self.backwardPuzzle = self.initializeBackwardPuzzle(self.puzzle)

    @staticmethod
    def initializeBackwardPuzzle(board):
        """
        Create a backward puzzle by swapping targets and boxes.
        
        Args:
            board (np.ndarray): Original board.
            
        Returns:
            np.ndarray: Modified board for backward search.
        """
        new_copy = copy.deepcopy(board)
        for row in range(len(board)):
            for col in range(len(board[0])):
                if new_copy[row][col] == 4:
                    new_copy[row][col] = 2
                elif new_copy[row][col] == 2:
                    new_copy[row][col] = 4
        return new_copy
    
    # ## This is not on initialization of the backward search.
    def flipGame(self, board):
        """
        Flip the game board for bidirectional search visualization or logic.
        
        Args:
            board (np.ndarray): The board to flip.
            
        Returns:
            np.ndarray: Flipped board.
        """
        new_copy = copy.deepcopy(board)

        for row in range(len(board)):
            for col in range(len(board[0])):
                if new_copy[row][col] == 2:
                    new_copy[row][col] = 1
        for target in self.target:
            if  new_copy[target[0]][target[1]] != 3 and new_copy[target[0]][target[1]] != 4:
                new_copy[target[0]][target[1]] = 2
        return new_copy
    
    def initalizeRandomStates(state, number):
        """
        Initialize a number of random states by moving the player to different empty locations.
        
        Args:
            state (np.ndarray): The base state.
            number (int): Number of random states to generate.
            
        Returns:
            list: List of generated random states.
        """
        avaiable_locations = list(zip(*np.where(state == 1)))
 
        player_loc = list(zip(*np.where(state == 3)))[0]
        visited = set()
        states = []
        for i in range(number):
            rand_indx = random.randint(0, len(avaiable_locations) - 1)
            while rand_indx in visited:
                rand_indx = random.randint(0, len(avaiable_locations) - 1)
            visited.add(rand_indx)
            new_state = copy.deepcopy(state)
            new_state[player_loc[0]][player_loc[1]] = 1
            new_loc = avaiable_locations[rand_indx]
            new_state[new_loc[0]][new_loc[1]] = 3
            states.append(new_state)
        
        return states
    
    def compareGames(self, game1, game2):
        """
        Compare two game boards for equality based on box and player locations.
        
        Args:
            game1 (np.ndarray): First game board.
            game2 (np.ndarray): Second game board.
            
        Returns:
            bool: True if boards are equivalent, False otherwise.
        """
        boxes_1 = list(zip(*np.where(game1 == 4)))
        boxes_2 = list(zip(*np.where(game2 == 4)))
        player_1 = list(zip(*np.where(game1 == 3)))[0]
        player_2 = list(zip(*np.where(game2 == 3)))[0]
        statement = (set(boxes_1) == set(boxes_2)) and player_1 == player_2
        return statement
    
    def getStateHash(self, game):
        """
        Generate a unique hash for the game state based on player and box locations. No target box locations
        
        Args:
            game (np.ndarray): The game board.
            
        Returns:
            str: Hash string representing the state.
        """
        boxes_1 = list(zip(*np.where(game == 4)))
        player_1 = list(zip(*np.where(game == 3)))[0]
        all_coords = [player_1] + boxes_1
        return "".join(str(val) for tup in all_coords for val in tup)
    

    def reconstruct_game(self, current_game, state_hash):
        """
        Reconstruct a game board from its state hash.
        
        Args:
            current_game (np.ndarray): Base board layout (walls, etc.).
            state_hash (str): Hash string representing player and box positions.
            
        Returns:
            np.ndarray: Reconstructed game board.
        """
        # 1. Start with a copy of the current board
        new_board = np.array(current_game).copy()
        
        # 2. "Clean" the board: Convert all Players(3) and Boxes(4) back to Floor(1)
        # Note: This assumes you don't have targets covered in this specific array.
        # If targets CAN be covered, we'd need a way to know where they were.
        new_board[new_board == 3] = 1
        new_board[new_board == 4] = 1
        
        # 3. Parse the hash (Assuming 1-digit coordinates for 10x10)
        coords = [int(d) for d in state_hash]
        
        # 4. Place Player
        py, px = coords[0], coords[1]
        new_board[py][px] = 3
        
        # 5. Place Boxes
        for i in range(2, len(coords), 2):
            by, bx = coords[i], coords[i+1]
            new_board[by][bx] = 4
            
        return new_board
    


    def successorInVisited(self,decodedMap, visited):
        """
        Check if a successor state has already been visited.
        
        Args:
            decodedMap (np.ndarray): The successor state.
            visited (set): Set of visited state hashes.
            
        Returns:
            bool: True if visited, False otherwise.
        """
        for vis in visited:
            decoded_vis = self.decodeMap(vis)
            if self.compareGames(decodedMap, decoded_vis):
                return True
        return False
                        
    
    def hasDeadlock(self, board):
        """
        Check if any boxes are in a deadlock position (e.g., trapped in a corner).
        
        Args:
            board (np.ndarray): The game board.
            
        Returns:
            bool: True if a deadlock is detected, False otherwise.
        """
        ### To Be Implemented
        box_locs = list(zip(*(np.where(board == 4))))
        surround = [(-1, 0), (0, 1), (1,0), (0,-1) ]
        rows, cols = board.shape

        for r, c in box_locs:
            blocked_count = 0
            
            # Check the four surrounding positions
            for dr, dc in surround:
                nr, nc = r + dr, c + dc  # New row, new column
                
                # Check for boundary issues (this box is next to the board edge)
                if not (0 <= nr < rows and 0 <= nc < cols):
                    # Count the boundary as a blocked side
                    blocked_count += 1
                    continue
                
                # Check if the surrounding cell contains an immovable object:
                # Wall (1), or another Box/Box-on-Target (4 or 5)
                if board[nr, nc] == 0:
                    blocked_count += 1
            
            # If 3 or 4 sides are blocked, it's a three-sided trap.
            if blocked_count >= 3:
                # We found a box that is trapped
                return True
                
        # No boxes were found to be trapped on three or more sides
        return False

    @staticmethod
    def decodeMap(map_string):
        """
        Decode a string representation of the map back into a 2D numpy array.
        
        Args:
            map_string (str): Encoded map string.
            
        Returns:
            np.ndarray: Decoded 10x10 map.
        """
        int_list = [int(digit) for digit in map_string]
            
        # 2. Convert the list to a 1D NumPy array
        flat_numpy_array = np.array(int_list)
        
        # 3. Reshape the 1D array back to the original dimensions
        decoded_map = flat_numpy_array.reshape((10,10))
        # You might want to convert it back to a standard Python list of lists 
        # if that was the original format:
        # decoded_map_list = decoded_map.tolist()
        return decoded_map
    @staticmethod
    def encodeMap(map):
        """
        Encode a 2D map array into a compact string representation.
        
        Args:
            map (np.ndarray): The map array.
            
        Returns:
            str: Encoded map string.
        """
        numpyMap = np.array(map)
        flattened = np.ravel(numpyMap)
        board_string = "".join(str(x) for x in flattened)
        return board_string
    def isGoal(self, board):
        """
        Check if the current board configuration is a goal state.
        
        Args:
            board (np.ndarray): The game board.
            
        Returns:
            bool: True if all boxes are on targets, False otherwise.
        """
        boxes = list(zip(*(np.where(board == 4))))
        for box_loc in boxes:
            if box_loc not in self.target:
                return False
        return True
    def evaluateBoard(self, board, open_set_state=None):
        """
        Heuristic evaluation of the board using Manhattan distance matching.
        
        Args:
            board (np.ndarray): Current game board.
            open_set_state (np.ndarray, optional): State to compare against if meeting in the middle.
            
        Returns:
            float: Heuristic cost estimate.
        """
        block_locations = list(zip(*np.where(board == 4)))
        distances = 0
        targets = self.target
        if open_set_state is not None:
            s_boxes = list(zip(*np.where(board == 4)))
            d_boxes = list(zip(*np.where(open_set_state == 4)))
            s_d_matrix = np.zeros((len(s_boxes), len(d_boxes)))
            for sIndex, sbox in enumerate(s_boxes):
                for dIndex, dbox in enumerate(d_boxes):
                    s_d_matrix[sIndex][dIndex] =  abs(dbox[0] - sbox[0]) + abs(dbox[1] - sbox[1])
            row_ind, col_ind = linear_sum_assignment(s_d_matrix)
            h_boxes = s_d_matrix[row_ind, col_ind].sum()
            return h_boxes
        else:
            s_boxes = list(zip(*np.where(board == 4)))
            d_boxes = list(zip(*np.where(self.goal_map == 4)))
            s_d_matrix = np.zeros((len(s_boxes), len(d_boxes)))
            for sIndex, sbox in enumerate(s_boxes):
                for dIndex, dbox in enumerate(d_boxes):
                    s_d_matrix[sIndex][dIndex] =  abs(dbox[0] - sbox[0]) + abs(dbox[1] - sbox[1])
            row_ind, col_ind = linear_sum_assignment(s_d_matrix)
            h_boxes = s_d_matrix[row_ind, col_ind].sum()
            return h_boxes
            # for loc_tuple in block_locations:
            #     min_distance_for_box = float('inf')
            #     for targ_tuple in targets:
            #         distance = abs(targ_tuple[0] - loc_tuple[0]) + abs(targ_tuple[1] - loc_tuple[1])
            #         if distance < min_distance_for_box:
            #             min_distance_for_box = distance
            #     distances += min_distance_for_box
            # return distances
        # min_
        # for block in 


    def availableStates(self, current_location: Tuple[int,int]):
        """
        Find available moves from the current player location.
        
        Args:
            current_location (tuple): Player's (row, col) position.
            
        Returns:
            list: List of available (direction, action_object) pairs.
        """
        # do a grid of surrounding 8 check
        surrounding_blocks = [(-1, 0), (0, 1), (1,0), (0,-1) ]
        availableTuples = []
        for block in surrounding_blocks:
            new_tuple = (current_location[0] + block[0], current_location[1] + block[1])
            new_loc = self.puzzle[new_tuple[0]][new_tuple[1]]
            if new_loc != 0:
                availableTuples.append(block)
        tuple_map = []
        for tuple in availableTuples:
            tuple_map.append((tuple, self.action_map[tuple]))
        if self.isBackward:
            for tuple in surrounding_blocks:
                tuple_map.append((tuple, self.pull_action_map[tuple]))
        return tuple_map
    def getPlayerLocation(self, board):
        """
        Locate the player on the board.
        
        Args:
            board (np.ndarray): The game board.
            
        Returns:
            tuple: Player's (row, col) position.
        """
        loc = np.where(board == 3)
        player_location: Tuple[int,int] = (loc[0][0], loc[1][0])
        return player_location
    def move(self,current_loc, direction_and_movement):
        """
        Execute a move on the game board.
        
        Args:
            current_loc (tuple): Current player position.
            direction_and_movement (tuple): (direction, action_object) pair.
            
        Returns:
            np.ndarray: New board state if valid, None otherwise.
        """
        movement = direction_and_movement[1]
        # print(movement)
        #action_object = self.action_map.get(direction)
        
        # 3. Execute the move
        new_board = movement.moveAndUpdateBoard(current_loc, self.puzzle)
    
        if new_board is None:
            return None
        return new_board

    class Move:
        """
        Base class for player movement actions.
        """
        def __init__(self, target_locations, isBackward):
            """
            Initialize a movement action.
            
            Args:
                target_locations (list): List of goal target locations.
                isBackward (bool): Whether the move is part of a backward search.
            """
            self.target_locations = target_locations
            self.isBackward = isBackward
            pass
        def canMove(self, currentLocation, board): #bool to see if moving in that direction is possible
            raise NotImplementedError
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            raise NotImplementedError
        def resetTargetSpots(self,board):
            for target_loc in self.target_locations:
                if board[target_loc[0]][target_loc[1]] != 4 and board[target_loc[0]][target_loc[1]] != 3:
                    board[target_loc[0]][target_loc[1]]= 2
            return board
    class Up(Move):
        """
        Action for moving the player up.
        """
        def canMove(self, currentLocation, board) -> bool:
            new_loc = (currentLocation[0] -1, currentLocation[1])
            res = True
            if board[new_loc[0]][new_loc[1]] == 0: # if wall is there
                res = False
            ## next check if its a block
            if self.isBackward and board[new_loc[0]][new_loc[1]] == 4:
                return False
            if board[new_loc[0]][new_loc[1]] == 4 and (board[new_loc[0] -1][new_loc[1]] == 0 or board[new_loc[0] - 1][new_loc[1]] == 4):
                res = False # because it means it wants to move the block but the block cant move and cant move two blocks at once
            return res
        def moveAndUpdateBoard(self, currentLocation, board):
            if not self.canMove(currentLocation, board):
                return None
            new_loc = (currentLocation[0] - 1, currentLocation[1])
            new_copy = copy.deepcopy(board)
            res = None
            if new_copy[new_loc[0]][new_loc[1]] == 1 or new_copy[new_loc[0]][new_loc[1]] == 2: # open space
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res =  new_copy
            ## next check if its a block
            if new_copy[new_loc[0]][new_loc[1]] == 4:
                new_copy[new_loc[0] - 1][new_loc[1]] = 4
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            res = self.resetTargetSpots(new_copy)
            return res

    class Down(Move):
        """
        Action for moving the player down.
        """
        def canMove(self, currentLocation, board):
            new_loc = (currentLocation[0] + 1, currentLocation[1])
            res = True
            if board[new_loc[0]][new_loc[1]] == 0: # if wall is there
                res =  False
            ## next check if its a block
            if self.isBackward and board[new_loc[0]][new_loc[1]] == 4:
                return False
            if board[new_loc[0]][new_loc[1]] == 4 and (board[new_loc[0] + 1][new_loc[1]] == 0 or board[new_loc[0] + 1][new_loc[1]] == 4):
                res =  False # because it means it wants to move the block but the block cant move and cant move two blocks at once
            return res
        def moveAndUpdateBoard(self, currentLocation, board):
            if not self.canMove(currentLocation, board):
                return None
            new_loc = (currentLocation[0] + 1, currentLocation[1])
            new_copy = copy.deepcopy(board)
            res = None
            if board[new_loc[0]][new_loc[1]] == 1 or new_copy[new_loc[0]][new_loc[1]] == 2: # open space
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            ## next check if its a block
            if board[new_loc[0]][new_loc[1]] == 4:
                new_copy[new_loc[0] + 1][new_loc[1]] = 4
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            res = self.resetTargetSpots(new_copy)
            return res
    class Left(Move):
        """
        Action for moving the player left.
        """
        def canMove(self, currentLocation, board):
            new_loc = (currentLocation[0], currentLocation[1] -1)
            res = True
            if board[new_loc[0]][new_loc[1]] == 0: # if wall is there
                res = False
            ## next check if its a block
            if self.isBackward and board[new_loc[0]][new_loc[1]] == 4:
                return False
            if board[new_loc[0]][new_loc[1]] == 4 and (board[new_loc[0]][new_loc[1] - 1] == 0 or board[new_loc[0]][new_loc[1] - 1] == 4):
                res = False # because it means it wants to move the block but the block cant move and cant move two blocks at once
            return res
        def moveAndUpdateBoard(self, currentLocation, board):
            if not self.canMove(currentLocation, board):
                return None
            new_loc = (currentLocation[0], currentLocation[1] - 1)
            new_copy = copy.deepcopy(board)
            res = None
            if board[new_loc[0]][new_loc[1]] == 1 or new_copy[new_loc[0]][new_loc[1]] == 2: # open space
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res =  new_copy
            ## next check if its a block
            if board[new_loc[0]][new_loc[1]] == 4:
                new_copy[new_loc[0]][new_loc[1] - 1] = 4
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            res = self.resetTargetSpots(new_copy)
            return res

    class Right(Move):
        """
        Action for moving the player right.
        """
        def canMove(self, currentLocation, board):
            new_loc = (currentLocation[0], currentLocation[1] + 1)
            res = True
            if board[new_loc[0]][new_loc[1]] == 0: # if wall is there
                res = False
            ## next check if its a block
            if self.isBackward and board[new_loc[0]][new_loc[1]] == 4:
                return False
            if board[new_loc[0]][new_loc[1]] == 4 and (board[new_loc[0]][new_loc[1] + 1] == 0 or board[new_loc[0]][new_loc[1] + 1] == 4):
                res = False # because it means it wants to move the block but the block cant move and cant move two blocks at once
            return res
        def moveAndUpdateBoard(self, currentLocation, board):

            if not self.canMove(currentLocation, board):
                return None
            new_loc = (currentLocation[0], currentLocation[1] + 1)
            new_copy = copy.deepcopy(board)
            res = None
            if board[new_loc[0]][new_loc[1]] == 1 or new_copy[new_loc[0]][new_loc[1]] == 2: # open space

                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            ## next check if its a block
            if board[new_loc[0]][new_loc[1]] == 4:
                new_copy[new_loc[0]][new_loc[1] + 1] = 4
                new_copy[new_loc[0]][new_loc[1]] = 3
                new_copy[currentLocation[0]][currentLocation[1]] = 1
                res = new_copy
            res = self.resetTargetSpots(new_copy)
            return res
    class Pull:
        """
        Base class for pulling actions used in backward search.
        """
        def __init__(self, target_locations):
            """
            Initialize a pulling action.
            
            Args:
                target_locations (list): List of goal target locations.
            """
            self.target_locations = target_locations
            pass
        def canPull(self, currentLocation, board): #bool to see if moving in that direction is possible
            raise NotImplementedError
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            raise NotImplementedError
        def resetTargetSpots(self,board):
            for target_loc in self.target_locations:
                if board[target_loc[0]][target_loc[1]] != 4 and board[target_loc[0]][target_loc[1]] != 3:
                    board[target_loc[0]][target_loc[1]]= 2
            return board
        
    class PullUp(Pull):
        """
        Action for pulling a box up.
        """
        def canPull(self, currentLocation, board): #bool to see if moving in that direction is possible
            ## first check if there is a block to pull
            res = True
            block_location = (currentLocation[0] + 1, currentLocation[1])

            if board[block_location[0]][block_location[1]] != 4:
                res =  False
            ## next see if there is a block above us
            move_to_spot = (currentLocation[0] - 1, currentLocation[1])
            if board[move_to_spot[0]][move_to_spot[1]] != 1 and board[move_to_spot[0]][move_to_spot[1]] != 2:
                res = False
            return res
            
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            if not self.canPull(currentLocation, board):
                return None
            new_copy = copy.deepcopy(board)
            new_loc = (currentLocation[0] - 1, currentLocation[1])
            
            new_copy[new_loc[0]][new_loc[1]] = 3
            new_copy[currentLocation[0]][currentLocation[1]] = 4
            new_copy[currentLocation[0] + 1][currentLocation[1]] = 1
            new_copy = self.resetTargetSpots(new_copy)
            return new_copy
        
    class PullDown(Pull):
        """
        Action for pulling a box down.
        """
        def canPull(self, currentLocation, board): #bool to see if moving in that direction is possible
            block_location = (currentLocation[0] - 1, currentLocation[1])
            if board[block_location[0]][block_location[1]] != 4:
                return False
            move_to_spot = (currentLocation[0] + 1, currentLocation[1])
            if board[move_to_spot[0]][move_to_spot[1]] != 1 and board[move_to_spot[0]][move_to_spot[1]] != 2:
                return False
            return True
        
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            if not self.canPull(currentLocation, board):
                return None
            new_copy = copy.deepcopy(board)
            new_loc = (currentLocation[0] + 1, currentLocation[1])

            new_copy[new_loc[0]][new_loc[1]] = 3
            new_copy[currentLocation[0]][currentLocation[1]] = 4
            new_copy[currentLocation[0] - 1][currentLocation[1]] = 1
            new_copy = self.resetTargetSpots(new_copy)
            return new_copy
        
    class PullLeft(Pull):
        """
        Action for pulling a box left.
        """
        def canPull(self, currentLocation, board): #bool to see if moving in that direction is possible
            block_location = (currentLocation[0], currentLocation[1] +1)
            if board[block_location[0]][block_location[1]] != 4:
                return False
            move_to_spot = (currentLocation[0], currentLocation[1] - 1)
            if board[move_to_spot[0]][move_to_spot[1]] != 1 and board[move_to_spot[0]][move_to_spot[1]] != 2:
                return False
            return True
        
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            if not self.canPull(currentLocation, board):
                return None
            new_copy = copy.deepcopy(board)
            new_loc = (currentLocation[0], currentLocation[1] - 1)

            new_copy[new_loc[0]][new_loc[1]] = 3
            new_copy[currentLocation[0]][currentLocation[1]] = 4
            new_copy[currentLocation[0]][currentLocation[1] + 1] = 1
            new_copy = self.resetTargetSpots(new_copy)
            return new_copy
        
    class PullRight(Pull):
        """
        Action for pulling a box right.
        """
        def canPull(self, currentLocation, board): #bool to see if moving in that direction is possible
            block_location = (currentLocation[0], currentLocation[1] -1)
            if board[block_location[0]][block_location[1]] != 4:
                return False
            move_to_spot = (currentLocation[0], currentLocation[1] + 1)
            if board[move_to_spot[0]][move_to_spot[1]] != 1 and board[move_to_spot[0]][move_to_spot[1]] != 2:
                return False
            return True
        
        def moveAndUpdateBoard(self, currentLocation, board) -> Tuple[list[list[int]], Tuple[int,int]]: # tuple is for if a target was hit, otherwise just None
            if not self.canPull(currentLocation, board):
                return None
            new_copy = copy.deepcopy(board)
            new_loc = (currentLocation[0], currentLocation[1] + 1)
            new_copy[new_loc[0]][new_loc[1]] = 3
            new_copy[currentLocation[0]][currentLocation[1]] = 4
            new_copy[currentLocation[0]][currentLocation[1] - 1] = 1
            new_copy = self.resetTargetSpots(new_copy)
            return new_copy