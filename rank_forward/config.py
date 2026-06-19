"""Env-overridable configuration, in the spirit of learning/online_run.py."""
import os


def _int(name, default):
    return int(os.environ.get(name, str(default)))


def _float(name, default):
    return float(os.environ.get(name, str(default)))


def _str(name, default):
    return os.environ.get(name, default)


def _bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


# Data / split
N_TOTAL = _int("N_TOTAL", 2000)         # puzzles taken from solvable10_3box.txt
N_EVAL = _int("N_EVAL", 200)            # held-out suffix used for evaluation
SEED = _int("SEED", 0)

# Search
ALG = _str("ALG", "astar").lower()      # astar | gbfs   (eval search)
MAX_ITERS = _int("MAX_ITERS", 10000)    # expansion budget, same default as online_run
USE_DEADLOCK = _bool("USE_DEADLOCK", False)  # match bidirectional forward side if True
SOLVE_CAP = _int("SOLVE_CAP", 200000)   # budget for optimal label generation

# Heuristic net
MODEL = _str("MODEL", "smallcnn").lower()   # smallcnn | smallcnn_attn | coat
MODEL_CHANNELS = _int("MODEL_CHANNELS", 32)

# Loss / training
LOSS = _str("LOSS", "lstar").lower()    # lstar | lgbfs | lrt | l2 | bellman
REDUCTION = _str("REDUCTION", "sum").lower()   # sum (paper) | mean
LOCAL_PAIRS = _bool("LOCAL_PAIRS", False)      # ranking: full Eq.1 pairing vs local
LR = _float("LR", 1e-3)
WEIGHT_DECAY = _float("WEIGHT_DECAY", 0.0)
STEPS = _int("STEPS", 20000)            # total SGD steps (1 step = 1 instance)
EVAL_EVERY = _int("EVAL_EVERY", 5000)   # steps between held-out evals
EVAL_SUBSET = _int("EVAL_SUBSET", 50)   # #eval puzzles during periodic eval
CKPT_EVERY = _int("CKPT_EVERY", 5000)

CKPT_DIR = _str("CKPT_DIR", os.path.join(os.path.dirname(__file__), "ckpt"))
CACHE_DIR = _str("CACHE_DIR", os.path.join(os.path.dirname(__file__), "cache"))
