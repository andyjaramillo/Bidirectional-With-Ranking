"""rank_forward — a PyTorch reimplementation of the unidirectional
forward-search ranking-heuristic method from

    Chrestien, Edelkamp, Komenda, Pevný,
    "Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal",
    NeurIPS 2023.  https://arxiv.org/abs/2310.19463

It exists as a *baseline* for our bidirectional front-to-front method: a
forward A*/GBFS planner whose heuristic h(s,theta) is trained by a RANKING
loss (L*, L_gbfs) instead of regression to cost-to-goal, evaluated on the same
Sokoban instances and the same node-expansion metric so the two are directly
comparable.

See README.md for the design and the (deliberate) differences from both the
paper and the original (TensorFlow) reference implementation.
"""
