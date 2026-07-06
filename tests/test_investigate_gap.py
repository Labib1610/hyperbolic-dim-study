"""Fast tests for the gap investigation (no full probe sweeps)."""

import networkx as nx
import numpy as np

from src.data import all_pairs_tree_distances
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.diagnostics.investigate_gap import (
    degree_cv,
    node_levels,
    train_poincare_rich,
    tree_delta,
)


def test_perfect_tree_delta_zero():
    tree = make_synthetic_tree(2, 3)  # 15 nodes, pure tree
    assert abs(tree_delta(tree)) < 1e-9


def test_irregularity_proxy_star_vs_path():
    # star: one hub + leaves (high degree variance); path: uniform degrees.
    star = nx.star_graph(5)  # node 0 center + 5 leaves
    path = nx.path_graph(6)
    cv_star = degree_cv(star)
    cv_path = degree_cv(path)
    assert cv_star > cv_path
    assert cv_path >= 0.0


def test_node_levels_mapping():
    # A-B, A-C, B-D  rooted at A: levels A=0, B=1, C=1, D=2
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D")])
    levels = node_levels(g, "A")
    assert levels["A"] == 0
    assert levels["B"] == 1 and levels["C"] == 1
    assert levels["D"] == 2


def test_per_level_alpha_param_count():
    tree = make_synthetic_tree(2, 4)  # depth 4 -> 5 levels
    root = tree.graph["root"]
    n_levels = max(node_levels(tree, root).values()) + 1
    _coords, info = train_poincare_rich(tree, 2, seed=0, alpha_mode="per_level", epochs=10)
    assert info["n_alpha"] == n_levels
    assert info["n_alpha"] >= 5


def test_end_to_end_global_and_per_level_finite():
    tree = make_synthetic_tree(2, 4)  # 31 nodes
    for mode in ("global", "per_level"):
        coords, info = train_poincare_rich(tree, 2, seed=0, alpha_mode=mode, epochs=20)
        assert np.isfinite(coords).all()
        assert np.isfinite(info["train_dist"]) and np.isfinite(info["val_dist"])
    # per_level has >= as many alpha params as global (which is 1)
    _c, gi = train_poincare_rich(tree, 2, seed=0, alpha_mode="global", epochs=5)
    _c, li = train_poincare_rich(tree, 2, seed=0, alpha_mode="per_level", epochs=5)
    assert li["n_alpha"] >= gi["n_alpha"]
