from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
class Algo(ABC):

    # @abstractmethod
    # def __init__(self, state, isBackward, nn=None):
    #     pass

    @abstractmethod
    def step(self, other_front: Algo, **kwargs):
        return NotImplementedError("Must be implemented by subclass")
    
    @abstractmethod
    def reconstructPath(self) -> List[str]:
        return NotImplementedError("Must be implemented by subclass")
    
    def fullPath(self, other_front: Algo) -> List[str]:
        front_path = self.reconstructPath()
        back_path = other_front.reconstructPath()
        back_path_f_oriented = front_path + other_front.flipPath(back_path)
        return back_path_f_oriented

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