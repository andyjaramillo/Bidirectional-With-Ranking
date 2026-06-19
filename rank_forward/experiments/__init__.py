"""Experiment drivers for the rank_forward baseline.

These are one-off run scripts (not part of the importable library): they wire
rank_forward's pieces (and, for the head-to-head, the bidirectional
learning/online_run) into full train+eval comparisons. Run from the repo root,
e.g.

    PYTHONPATH=. python rank_forward/experiments/head_to_head.py

See README.md in this folder for what each one does and its env knobs.
"""
