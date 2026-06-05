import os
import numpy as np

# Repo-root /data directory, resolved relative to this file so data loads
# regardless of the current working directory.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_DIR = _DATA_DIR  # public alias


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
        f=open(os.path.join(_DATA_DIR, "test_box.txt"), "r")
    else:
        f=open(os.path.join(_DATA_DIR, "states10_3box.txt"), "r")

    array_s=[]
    array_a=[]
    duplicate = []
    index = []

    i=0
    k=0
    for line in f:
        if k < 10:
            k+=1
            continue
        array_s.append([int(x) for x in line.split()])
        #print(i)
        if array_s[i] not in duplicate:
            arr=np.asarray(array_s[i])
            temp=np.reshape(arr, (10,10))
            all_states.append(temp)
            duplicate.append(array_s[i])
            index.append(i)
        i+=1
        # if i > 10:
        #     break
        # break

    return all_states
    f.close()


def get_solvable_data(limit=None):
    """Load the solvable-only benchmark produced by
    ``analysis/build_solvable_benchmark.py`` (instances the bidirectional
    search can actually meet on; the player-goal-pinned unsolvable artifacts
    are excluded). Same return type as ``get_data`` — a list of 10x10 arrays.

    Args:
        limit (int|None): keep only the first ``limit`` boards.

    Returns:
        list[np.ndarray]: solvable 10x10 boards.
    """
    path = os.path.join(_DATA_DIR, "solvable10_3box.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"    PYTHONPATH=. python analysis/build_solvable_benchmark.py")
    boards = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boards.append(np.reshape(np.asarray([int(x) for x in line.split()]), (10, 10)))
            if limit is not None and len(boards) >= limit:
                break
    return boards


def get_paths():
    """
    Load paths from a text file.
    
    Returns:
        list: List of paths, where each path is a list of game states.
    """
    all_paths=[]
    f=open(os.path.join(_DATA_DIR, "paths10_3box.txt"), "r")
    array_s=[]
    for line in f:
        array_s.append([int(x) for x in line.split()])
        arr=np.asarray(array_s[:-1])
        all_paths.append(arr)
        array_s=[]
    return all_paths