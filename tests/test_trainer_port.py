"""Tests for the Stage-6 trainer port: learnable-α, R2 stopping, backward-compat."""

import networkx as nx
import numpy as np
import pytest

from src import data
from src.trainer import (
    _fixed_val_pairs, _pair_ids, _sample_train_pairs, convergence_stop_epoch,
    train_embedding,
)


def _tiny_tree(tmp_path, name="tiny"):
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    data.save_subtree(g, "A", str(tmp_path / "subtrees" / name))


def _cfg(tmp_path, **train_over):
    train = {
        "geometry": "poincare", "dim": 2, "loss": "distortion", "hybrid_lambda": 1.0,
        "n_negatives": 3, "pairs_per_step": 512, "full_pairs_threshold": 150,
        "lr": 0.05, "optimizer": "radam", "seed": 0,
        "scale": "learnable", "stopping": "fixed_epochs", "fixed_epochs": 200,
        "epoch_cap": 8000, "patience_epochs": 500, "min_delta_rel": 0.005,
    }
    train.update(train_over)
    return {"paths": {"data_dir": str(tmp_path)}, "subtree": "tiny", "seed": 0, "train": train}


def _run(tmp_path, **train_over):
    _tiny_tree(tmp_path)
    cfg = _cfg(tmp_path, **train_over)
    return train_embedding("tiny", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))


# --- convergence helper (hand-checked) ----------------------------------------
def test_convergence_stop_epoch_handchecked():
    # patience 3, min_delta 1%: improves through epoch 3, then flat -> stop at index 6.
    vals = [1.0, 0.5, 0.25, 0.2, 0.199, 0.1988, 0.1987, 0.1986]
    assert convergence_stop_epoch(vals, patience_epochs=3, min_delta_rel=0.01) == 6
    # never plateaus -> returns len
    assert convergence_stop_epoch([1.0, 0.5, 0.2, 0.1], patience_epochs=3, min_delta_rel=0.01) == 4


# --- alpha receives gradients + moves -----------------------------------------
def test_alpha_moves_from_init(tmp_path):
    run = _run(tmp_path, scale="learnable", stopping="fixed_epochs", fixed_epochs=100)
    alpha = run.metrics["alpha"]
    assert np.isfinite(alpha) and alpha > 0
    assert abs(alpha - 1.0) > 1e-3  # moved from the alpha≈1 init


def test_fixed_scale_has_no_alpha_effect(tmp_path):
    # scale=fixed records alpha=1.0 (unused) and reproduces old scale=1 behaviour.
    run = _run(tmp_path, scale="fixed", stopping="fixed_epochs", fixed_epochs=100)
    assert run.metrics["alpha"] == 1.0
    curve = run.curves["train_loss"]
    assert curve[-1] < curve[0]  # still learns


# --- R2 convergence stopping actually stops before the cap --------------------
def test_convergence_stops_before_cap(tmp_path):
    run = _run(tmp_path, scale="learnable", stopping="convergence",
               patience_epochs=50, min_delta_rel=0.01, epoch_cap=8000)
    assert run.metrics["epochs_ran"] < 8000  # converged early on the tiny tree
    assert "epochs_to_converge" in run.metrics


# --- fixed_epochs runs exactly that many --------------------------------------
def test_fixed_epochs_runs_exact(tmp_path):
    run = _run(tmp_path, stopping="fixed_epochs", fixed_epochs=123)
    assert run.metrics["epochs_ran"] == 123


# --- backward-compat: fixed scale + fixed epochs overfits the tiny tree -------
def test_backward_compat_overfit(tmp_path):
    run = _run(tmp_path, scale="fixed", stopping="fixed_epochs", fixed_epochs=500)
    curve = run.curves["train_loss"]
    assert curve[-1] < 0.25 * curve[0]  # reproduces old overfit-sanity behaviour


# --- determinism: same seed -> same converged metrics -------------------------
def test_determinism(tmp_path):
    r1 = _run(tmp_path, stopping="convergence", patience_epochs=50, epoch_cap=2000)
    r2 = _run(tmp_path, stopping="convergence", patience_epochs=50, epoch_cap=2000)
    assert abs(r1.metrics["final_train_loss"] - r2.metrics["final_train_loss"]) < 1e-6
    assert r1.metrics["epochs_ran"] == r2.metrics["epochs_ran"]
    assert abs(r1.metrics["alpha"] - r2.metrics["alpha"]) < 1e-6


# --- learnable beats fixed on a deeper synthetic tree (the fix works) ---------
def test_learnable_beats_fixed_recovery(tmp_path):
    from src.diagnostics.diag_plateau import make_synthetic_tree
    from src.metrics import evaluate_embedding

    tree = make_synthetic_tree(2, 5)  # 63-node perfect tree
    data.save_subtree(tree, tree.graph["root"], str(tmp_path / "subtrees" / "synth"))

    def run_mode(scale):
        cfg = {
            "paths": {"data_dir": str(tmp_path)}, "subtree": "synth", "seed": 0,
            "train": {"geometry": "poincare", "dim": 2, "loss": "distortion",
                      "hybrid_lambda": 1.0, "n_negatives": 3, "pairs_per_step": 512,
                      "full_pairs_threshold": 150, "lr": 0.05, "optimizer": "radam",
                      "seed": 0, "scale": scale, "stopping": "fixed_epochs", "fixed_epochs": 800},
        }
        run = train_embedding("synth", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))
        coords = np.load(run.dir / "artifacts" / "coords.npy")
        return evaluate_embedding(coords, tree, "poincare", delta_samples=500)["map"]

    assert run_mode("learnable") > run_mode("fixed")


# --- large_tree scalability path -----------------------------------------------
def test_fixed_val_pairs_deterministic_and_capped():
    N = 100
    p1, ids1 = _fixed_val_pairs(N, 10000, seed=0)
    p2, ids2 = _fixed_val_pairs(N, 10000, seed=0)
    assert np.array_equal(p1, p2)  # fixed/seeded
    assert np.array_equal(ids1, ids2)
    # capped at 20% of all pairs (N*(N-1)/2 = 4950 -> <= 990)
    assert len(p1) <= (N * (N - 1) // 2) // 5
    # all distinct, all valid unordered pairs
    assert len(set(map(tuple, p1))) == len(p1)
    assert (p1[:, 0] < p1[:, 1]).all()


def test_sample_train_pairs_excludes_val():
    import torch

    N = 200
    _, val_ids = _fixed_val_pairs(N, 500, seed=1)
    rng = np.random.RandomState(3)
    batch = _sample_train_pairs(rng, N, 256, val_ids, torch.device("cpu")).numpy()
    assert batch.shape == (256, 2)
    assert (batch[:, 0] != batch[:, 1]).all()
    lo = np.minimum(batch[:, 0], batch[:, 1])
    hi = np.maximum(batch[:, 0], batch[:, 1])
    assert not np.isin(_pair_ids(lo, hi, N), val_ids).any()  # no val pairs leaked into train


def _large_tree_cfg(tmp_path, **over):
    from src.diagnostics.diag_plateau import make_synthetic_tree

    tree = make_synthetic_tree(2, 6)  # 127 nodes
    data.save_subtree(tree, tree.graph["root"], str(tmp_path / "subtrees" / "big"))
    train = {
        "geometry": "poincare", "dim": 8, "loss": "distortion", "hybrid_lambda": 1.0,
        "n_negatives": 3, "pairs_per_step": 256, "full_pairs_threshold": 150,
        "lr": 0.05, "optimizer": "radam", "seed": 0,
        "scale": "learnable", "stopping": "fixed_epochs", "fixed_epochs": 60,
        "large_tree": True, "val_subsample": 2000,
    }
    train.update(over)
    return {"paths": {"data_dir": str(tmp_path)}, "subtree": "big", "seed": 0, "train": train}


def test_large_tree_runs_finite(tmp_path):
    cfg = _large_tree_cfg(tmp_path)
    run = train_embedding("big", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))
    assert np.isfinite(run.curves["train_loss"][-1])
    assert np.isfinite(run.metrics["best_val_loss"])
    assert np.isfinite(run.metrics["alpha"]) and run.metrics["alpha"] > 0
    assert run.metrics["epochs_ran"] == 60


def test_large_tree_deterministic(tmp_path):
    r1 = train_embedding("big", _large_tree_cfg(tmp_path), base_dir=str(tmp_path / "r1"), results_dir=str(tmp_path / "a"))
    r2 = train_embedding("big", _large_tree_cfg(tmp_path), base_dir=str(tmp_path / "r2"), results_dir=str(tmp_path / "b"))
    assert abs(r1.metrics["final_train_loss"] - r2.metrics["final_train_loss"]) < 1e-6
    assert abs(r1.metrics["alpha"] - r2.metrics["alpha"]) < 1e-6
