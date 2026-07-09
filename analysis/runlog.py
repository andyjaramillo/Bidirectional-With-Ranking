"""Run-logging layer: persist every trained model + its per-instance eval
results under a stable directory key, so metrics never require retraining.

Motivation: table/intersection/ratio metrics were repeatedly recomputed by
RE-TRAINING because neither models nor per-instance eval data were saved.
Here each run is one (domain, method, seed, config) and lives at
    runs/<domain>/<method>/seed<seed>_<confighash>/
containing:
    config.json      the full config dict (+ the hash it keyed on)
    model.pt         state_dict (+ optimizer/step if given)
    train_curve.csv  optional training curve
    eval/<set>.json  per-instance {"iters": [...], "solved": [bool,...]}
A top-level runs/index.jsonl appends one summary line per (run, eval set).

Metrics tools (analysis/run_metrics.py) read ONLY the eval JSONs — tables,
both-solved intersection, ratios — so nothing is ever retrained to remeasure.
Domain-agnostic: consumes only dicts/lists, no Sokoban/tile specifics.
"""
import csv
import hashlib
import json
import os
import time

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")


def config_hash(config: dict, length: int = 8) -> str:
    """Stable short hash of a config dict (order-independent)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:length]


def run_dir(domain: str, method: str, seed: int, config: dict) -> str:
    d = os.path.join(RUNS_DIR, domain, method,
                     f"seed{seed}_{config_hash(config)}")
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    return d


def save_config(run: str, config: dict) -> None:
    payload = dict(config)
    payload["_config_hash"] = config_hash(config)
    payload["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_model(run: str, model, optimizer=None, step=None) -> None:
    import torch
    payload = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = step
    torch.save(payload, os.path.join(run, "model.pt"))


def load_model(run: str, model) -> object:
    """Load saved weights into an already-constructed model; returns it."""
    import torch
    payload = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
    sd = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(sd)
    model.eval()
    return model


def save_train_curve(run: str, rows, header) -> None:
    with open(os.path.join(run, "train_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def record_eval(run: str, testset: str, iters, solved,
                domain=None, method=None, seed=None, extra=None) -> dict:
    """Persist per-instance results and append a summary line to the index.
    iters: list[int]; solved: list[bool] (same length)."""
    import statistics
    iters = [int(x) for x in iters]
    solved = [bool(x) for x in solved]
    sv = [it for it, ok in zip(iters, solved) if ok]
    summary = {
        "run": run, "domain": domain, "method": method, "seed": seed,
        "testset": testset, "n": len(iters), "solved": sum(solved),
        "median": statistics.median(sv) if sv else None,
        "mean": statistics.mean(sv) if sv else None,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        summary.update(extra)
    with open(os.path.join(run, "eval", f"{testset}.json"), "w") as f:
        json.dump({"iters": iters, "solved": solved, "summary": summary}, f)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, "index.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


def load_eval(run: str, testset: str) -> dict:
    with open(os.path.join(run, "eval", f"{testset}.json")) as f:
        return json.load(f)


def read_index() -> list:
    path = os.path.join(RUNS_DIR, "index.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
