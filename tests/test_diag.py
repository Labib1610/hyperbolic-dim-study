"""Fast tests for the plateau diagnostic (no full probe sweeps)."""

import numpy as np

from src import data
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.metrics import evaluate_embedding
from src.trainer import train_embedding


def test_synthetic_tree_structure():
    tree = make_synthetic_tree(2, 3)
    assert tree.number_of_nodes() == 15
    assert tree.number_of_edges() == 14  # nodes - 1 => tree
    import networkx as nx

    assert nx.is_connected(tree)
    root = tree.graph["root"]
    # depth: max root-to-leaf distance == 3
    depths = nx.single_source_shortest_path_length(tree, root)
    assert max(depths.values()) == 3
    # every internal node has exactly 2 children
    child_count = {n: 0 for n in tree}
    for parent, _child in nx.bfs_edges(tree, root):
        child_count[parent] += 1
    internal = [c for c in child_count.values() if c > 0]
    assert all(c == 2 for c in internal)
    assert len(internal) == 7  # 1 root + 2 + 4 internal nodes


def test_synthetic_tree_integrates_with_data():
    tree = make_synthetic_tree(2, 3)
    root = tree.graph["root"]
    # a leaf is at tree-distance == depth from the root
    import networkx as nx

    depths = nx.single_source_shortest_path_length(tree, root)
    leaf = max(depths, key=depths.get)
    assert data.tree_distance(tree, root, leaf) == 3
    stats = data.subtree_stats(tree, root)
    assert stats["num_nodes"] == 15
    assert stats["depth"] == 3
    assert stats["max_branching"] == 2


def test_end_to_end_tiny(tmp_path):
    tree = make_synthetic_tree(2, 3)
    root = tree.graph["root"]
    data.save_subtree(tree, root, str(tmp_path / "subtrees" / "synth"))
    cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "subtree": "synth",
        "seed": 0,
        "train": {
            "geometry": "poincare", "dim": 2, "loss": "distortion", "hybrid_lambda": 1.0,
            "n_negatives": 3, "pairs_per_step": 512, "full_pairs_threshold": 150,
            "epochs": 30, "lr": 0.05, "optimizer": "radam",
            "early_stop_patience": 10000, "seed": 0,
        },
    }
    run = train_embedding("synth", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))
    coords = np.load(run.dir / "artifacts" / "coords.npy")
    m = evaluate_embedding(coords, tree, "poincare")
    for key in ["distortion_avg_scaled", "distortion_worst", "map", "delta_relative"]:
        assert np.isfinite(m[key])
    assert 0.0 <= m["map"] <= 1.0
