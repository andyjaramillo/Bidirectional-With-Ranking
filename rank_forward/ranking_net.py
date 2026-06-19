"""Heuristic network factory for the forward ranking baseline.

Default architecture is our existing ``SmallCNN`` (learning/nn.py, ~23.6k
params) — the SAME encoder the bidirectional method uses. Holding the encoder
identical is what makes the comparison clean: any difference in node
expansions is attributable to the *loss* and the *search direction*, not to
network capacity.

The paper's larger CoAT net (7 conv + 4 self-attention blocks, the ``NN`` class
in learning/nn.py) is available for a paper-faithful ablation via name="coat",
but is far heavier on CPU and confounds the capacity variable, so it is not the
default.
"""
from learning.nn import build_model, NN


def build_forward_model(name: str = "smallcnn", channels: int = 32, **kw):
    """Return a heuristic net taking (state, target, goal_state) -> scalar.

    name: "smallcnn" (default, recommended), "smallcnn_attn", or "coat"
    (the paper's attention net; heavier).
    """
    name = name.lower()
    if name in ("coat", "nn", "paper"):
        # The paper's grid net: 7 conv + 4 CoAT blocks, scalar head.
        return NN(dim=180)
    return build_model(name, channels=channels, **kw)
