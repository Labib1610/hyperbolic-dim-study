"""Metrics tests with hand-computed ground truth (δ, distortion, MAP) + integration."""

import math

import networkx as nx
import numpy as np
import pytest

from src import data
from src.metrics import (
    delta_hyperbolicity_exact,
    delta_hyperbolicity_sampled,
    distortion,
    evaluate_embedding,
    mean_average_precision,
)
from src.trainer import train_embedding


def _abcde_tree():
    """A/B/C/D/E: edges A-B, A-C, B-D, B-E. Sorted node order -> idx A,B,C,D,E."""
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    return g


def _abcde_Dt():
    apsp = data.all_pairs_tree_distances(_abcde_tree())
    assert apsp["nodes"] == ["A", "B", "C", "D", "E"]
    return apsp["matrix"].astype(float)


# ============================ δ-hyperbolicity ================================
def test_delta_tree_is_zero():
    # 4-leaf tree metric: sums 4, 6, 6 -> (6-6)/2 = 0.
    idx = {"w": 0, "x": 1, "y": 2, "z": 3}
    D = np.zeros((4, 4))

    def setd(a, b, val):
        D[idx[a], idx[b]] = D[idx[b], idx[a]] = val

    setd("w", "x", 2)
    setd("y", "z", 2)
    setd("w", "y", 3)
    setd("w", "z", 3)
    setd("x", "y", 3)
    setd("x", "z", 3)
    assert abs(delta_hyperbolicity_exact(D) - 0.0) < 1e-9


def test_delta_square_is_0414():
    # Unit-square corners w=(0,0), x=(1,0), y=(1,1), z=(0,1):
    # sides=1, diagonals=sqrt(2). Sums 2, 2.828, 2 -> (2.828-2)/2 = 0.41421.
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    assert abs(delta_hyperbolicity_exact(D) - 0.41421356) < 1e-4


def test_delta_exact_equals_sampled_small():
    rng = np.random.RandomState(3)
    M = rng.rand(12, 12)
    D = (M + M.T) / 2
    np.fill_diagonal(D, 0.0)
    # n_samples covers all C(12,4)=495 quadruples -> falls back to exact.
    exact = delta_hyperbolicity_exact(D)
    sampled = delta_hyperbolicity_sampled(D, n_samples=100000, seed=0)
    assert abs(exact - sampled) < 1e-9


# ============================ distortion =====================================
def test_distortion_perfect_embedding():
    # Dg exactly proportional to Dt -> scaled avg ~0, worst = 1.
    Dt = _abcde_Dt()
    Dg = 2.0 * Dt  # perfect up to scale
    res = distortion(Dg, Dt)
    assert abs(res["distortion_avg_scaled"]) < 1e-6
    assert abs(res["distortion_worst"] - 1.0) < 1e-6


def test_distortion_known_imperfect():
    # 1D embedding on a line; positions A=0,B=1,C=2,D=1.5,E=3. Euclidean distances.
    # ratios (Dg/Dt): A-C=2/1=2 (max), C-D=0.5/3=0.16667 (min) -> worst = 2/0.16667 = 12.0
    Dt = _abcde_Dt()
    pos = np.array([[0.0], [1.0], [2.0], [1.5], [3.0]])
    Dg = np.abs(pos - pos.T)
    res = distortion(Dg, Dt)
    assert abs(res["distortion_worst"] - 12.0) < 1e-3


# ============================ MAP ============================================
def test_map_perfect_is_one():
    # Using the tree metric itself as the "geometric" distances: every node's
    # neighbors (Dt=1) are strictly nearer than all Dt>=2 nodes -> MAP = 1.0.
    Dt = _abcde_Dt()
    assert abs(mean_average_precision(Dt, Dt) - 1.0) < 1e-9


def test_map_one_inversion():
    # Perturb the C-E entry so C and E each get one non-neighbor placed nearest.
    #   AP(A)=AP(B)=AP(D)=1 (neighbors still strictly nearest).
    #   Node C: dists A=1,B=2,D=3,E=0.5 -> order E,A,... relevant {A} at rank2 -> AP=1/2=0.5
    #   Node E: dists A=2,B=1,C=0.5,D=2 -> order C,B,... relevant {B} at rank2 -> AP=1/2=0.5
    #   MAP = (1+1+0.5+1+0.5)/5 = 0.8
    Dt = _abcde_Dt()
    Dg = Dt.copy()
    Dg[2, 4] = Dg[4, 2] = 0.5  # C index 2, E index 4
    assert abs(mean_average_precision(Dg, Dt) - 0.8) < 1e-9


# ============================ integration ====================================
def test_integration_evaluate_trained_embedding(tmp_path):
    g = _abcde_tree()
    data.save_subtree(g, "A", str(tmp_path / "subtrees" / "tiny"))
    cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "subtree": "tiny",
        "seed": 0,
        "train": {
            "geometry": "poincare", "dim": 2, "loss": "distortion", "hybrid_lambda": 1.0,
            "n_negatives": 3, "pairs_per_step": 512, "full_pairs_threshold": 150,
            "epochs": 100, "lr": 0.05, "optimizer": "radam",
            "early_stop_patience": 10000, "seed": 0,
        },
    }
    run = train_embedding("tiny", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))
    coords = np.load(run.dir / "artifacts" / "coords.npy")

    m = evaluate_embedding(coords, g, "poincare")
    for key in [
        "distortion_avg", "distortion_avg_scaled", "distortion_worst",
        "map", "delta", "delta_relative", "scale", "num_nodes",
    ]:
        assert key in m, f"missing metric {key}"
        assert np.isfinite(m[key])
    assert 0.0 <= m["map"] <= 1.0
    assert m["distortion_worst"] >= 1.0
    assert m["num_nodes"] == 5


def test_evaluate_run_writes_back(tmp_path):
    import json

    g = _abcde_tree()
    data.save_subtree(g, "A", str(tmp_path / "subtrees" / "tiny"))
    cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "subtree": "tiny",
        "seed": 0,
        "train": {
            "geometry": "poincare", "dim": 2, "loss": "distortion", "hybrid_lambda": 1.0,
            "n_negatives": 3, "pairs_per_step": 512, "full_pairs_threshold": 150,
            "epochs": 50, "lr": 0.05, "optimizer": "radam",
            "early_stop_patience": 10000, "seed": 0,
        },
    }
    run = train_embedding("tiny", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))

    from src.metrics import evaluate_run

    evaluate_run(run.dir)
    written = json.loads((run.dir / "metrics.json").read_text())
    assert "map" in written and "delta_relative" in written

    import pandas as pd

    df = pd.read_csv(tmp_path / "results.csv")
    row = df[df["run_id"] == run.dir.name].iloc[0]
    assert np.isfinite(row["distortion_worst"])
    assert np.isfinite(row["map"])
