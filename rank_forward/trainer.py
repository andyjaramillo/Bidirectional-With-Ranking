"""Training loop: one SGD step per problem instance, one persistent Adam
optimizer (the paper's protocol). Trains the SAME network that is later used
in the search — no separate frozen copy.
"""
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch

from rank_forward.losses import instance_loss
from rank_forward.trajectory import Instance


def train(instances: List[Instance], model, loss_name: str, *,
          steps: int, lr: float, weight_decay: float = 0.0,
          reduction: str = "sum", local_pairs: bool = False,
          seed: int = 0, eval_every: int = 0, ckpt_every: int = 0,
          ckpt_path: Optional[str] = None, eval_boards=None, eval_alg="astar",
          eval_max_iters: int = 10000, eval_subset: int = 50):
    """Train ``model`` in place. Returns the model. ``instances`` is the list
    of training Instances (already solved/assembled)."""
    if not instances:
        raise RuntimeError("no training instances")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    order = list(range(len(instances)))
    model.train()

    t0 = time.time()
    running = 0.0
    pos = 0
    rng.shuffle(order)
    for step in range(1, steps + 1):
        if pos >= len(order):
            rng.shuffle(order)
            pos = 0
        inst = instances[order[pos]]
        pos += 1

        opt.zero_grad()
        loss = instance_loss(model, inst, loss_name, reduction=reduction,
                             local_pairs=local_pairs)
        loss.backward()
        opt.step()
        running += float(loss.detach())

        if step % 200 == 0:
            print(f"  step {step:>6d}/{steps}  loss={running/200:8.4f}  "
                  f"dt={time.time()-t0:6.1f}s", flush=True)
            running = 0.0

        if eval_every and step % eval_every == 0 and eval_boards is not None:
            _periodic_eval(model, eval_boards, eval_alg, eval_max_iters,
                           eval_subset, step)
            model.train()

        if ckpt_every and ckpt_path and step % ckpt_every == 0:
            _save(model, ckpt_path)

    if ckpt_path:
        _save(model, ckpt_path)
    return model


def _periodic_eval(model, eval_boards, alg, max_iters, subset, step):
    from rank_forward.forward_run import evaluate   # lazy import (avoids cycle)
    model.eval()
    sub = eval_boards[:subset]
    res = evaluate(model, sub, alg=alg, max_iters=max_iters)
    print(f"  [eval @ {step}] alg={alg} solved={res['solved']}/{len(sub)} "
          f"mean_iters={res['mean_iters']:.1f} median_iters={res['median_iters']:.1f}",
          flush=True)


def _save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
