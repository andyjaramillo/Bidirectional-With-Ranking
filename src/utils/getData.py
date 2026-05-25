import numpy as np
from io import TextIOWrapper

class StateBox103Box:
    duplicate: list
    file_pointer: TextIOWrapper
    index: int
    def __init__(self):
        self.duplicate = []
        self.file_pointer = open("data/states10_3box.txt", "r")

    def __iter__(self):
        state = next(self.file_pointer)
        while state in self.duplicate:

            self.index += 1
        
        arr=np.asarray(state)
        temp=np.reshape(arr, (10,10))
        return temp
    
    def __exit__(self, exc_type, exc, tb):
        self.file_pointer.close()
        

def get_data(useDummyData=False):
    """
    Load game states from a text file.
    
    Args:
        useDummyData (bool): Whether to use the small test dataset.
        
    Returns:
        list: List of 10x10 numpy arrays representing game states.
    """
    all_states=[]
    f = None
    if useDummyData:
        f=open("test_box.txt", "r")
    else:
        f=open("data/states10_3box.txt", "r")

    array_s=[]
    array_a=[]
    duplicate = []
    index = []

    i=0
    k=0
    for line in f:
        array_s.append([int(x) for x in line.split()])
        if array_s[i] not in duplicate:
            arr=np.asarray(array_s[i])
            temp=np.reshape(arr, (10,10))
            all_states.append(temp)
            duplicate.append(array_s[i])
            index.append(i)

    f.close()
    return all_states
    


def get_paths():
    """
    Load paths from a text file.clear
    
    Returns:
        list: List of paths, where each path is a list of game states.
    """
    all_paths=[]
    f=open("paths10_3box.txt", "r")
    array_s=[]
    for line in f:
        array_s.append([int(x) for x in line.split()])
        arr=np.asarray(array_s[:-1])
        all_paths.append(arr)
        array_s=[]
    return all_paths

def decode_path(path, decode_fn):
    return [decode_fn(state) for state in path]