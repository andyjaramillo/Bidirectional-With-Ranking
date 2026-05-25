
from src.algos.TTBS import TTBS
from src.algos.AnchorSearch import SearchFrontier as AnchorSearch
from src.algos.astar import Astar


ALGO_REGISTRY = {
    "anchor": AnchorSearch,
    "astar": Astar,
    "ttbs": TTBS
}