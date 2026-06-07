import sys
import os

# Get the absolute path of the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./src"))
# sys.path.insert(0, parent_dir)

from src.algos.TTBS import TTBS
from src.algos.AnchorSearch import SearchFrontier
from src.utils.getData import get_data
data = get_data()


paths_txt = open("solveable_paths10_3box.txt", "a")
res = ""
for state in data:
    search = TTBS(puzzle=state)
    search.search()
    path = search.reconstruct_path()
    path_str = ",".join(path)
    paths_txt.write(path_str + "\n")

paths_txt.close()