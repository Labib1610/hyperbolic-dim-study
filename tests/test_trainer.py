"""Trainer tests on a tiny hand-built 5-node tree (fast, dim=2)."""

import networkx as nx
import numpy as np
import pytest

from src import data
from src.trainer import train_embedding


def _tiny_tree_dir(tmp_path):
    """Build A-B, A-C, B-D, B-E and save it as a loadable subtree."""
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    path = tmp_path / "subtrees" / "tiny"
    data.save_subtree(g, "A", str(path))
    return path


def _cfg(tmp_path, geometry="poincare", loss="distortion", epochs=500, patience=10_000):
    # These tests exercise the fixed-budget path (stopping=fixed_epochs) to keep
    # their original intent and speed; the new convergence default is covered by
    # tests/test_trainer_port.py.
    return {
        "paths": {"data_dir": str(tmp_path)},
        "subtree": "tiny",
        "seed": 0,
        "train": {
            "geometry": geometry,
            "dim": 2,
            "loss": loss,
            "hybrid_lambda": 1.0,
            "n_negatives": 3,
            "pairs_per_step": 512,
            "full_pairs_threshold": 150,
            "stopping": "fixed_epochs",
            "fixed_epochs": epochs,
            "epochs": epochs,
            "lr": 0.05,
            "optimizer": "radam" if geometry == "poincare" else "adam",
            "early_stop_patience": patience,
            "seed": 0,
        },
    }


def _run(tmp_path, **kw):
    _tiny_tree_dir(tmp_path)
    cfg = _cfg(tmp_path, **kw)
    return train_embedding(
        "tiny", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path)
    )


def test_overfit_sanity_poincare_distortion(tmp_path):
    run = _run(tmp_path, geometry="poincare", loss="distortion", epochs=500)
    curve = run.curves["train_loss"]
    initial, final = curve[0], curve[-1]
    assert final < 0.25 * initial, f"loss did not drop enough: {initial} -> {final}"


def test_euclidean_parity_learns(tmp_path):
    run = _run(tmp_path, geometry="euclidean", loss="distortion", epochs=500)
    curve = run.curves["train_loss"]
    assert curve[-1] < curve[0], "euclidean loss did not decrease"
    assert np.isfinite(curve[-1])


@pytest.mark.parametrize("loss", ["distortion", "ranking", "hybrid"])
def test_all_loss_modes_finite(tmp_path, loss):
    run = _run(tmp_path, geometry="poincare", loss=loss, epochs=30)
    assert np.isfinite(run.curves["train_loss"][-1])
    assert np.isfinite(run.metrics["best_val_loss"])


def test_no_nan_parameters(tmp_path):
    run = _run(tmp_path, geometry="poincare", loss="hybrid", epochs=50)
    coords = np.load(run.dir / "artifacts" / "coords.npy")
    assert np.isfinite(coords).all(), "coords contain NaN/inf after training"


def test_tracking_integration(tmp_path):
    run = _run(tmp_path, geometry="poincare", loss="distortion", epochs=20)
    assert (run.dir / "artifacts" / "coords.npy").exists()
    assert (run.dir / "curves" / "train_loss.json").exists()
    results = tmp_path / "results.csv"
    assert results.exists()
    df = __import__("pandas").read_csv(results)
    assert len(df) == 1
    assert df.iloc[0]["train.geometry"] == "poincare"


def test_determinism(tmp_path):
    r1 = _run(tmp_path, geometry="poincare", loss="distortion", epochs=100)
    r2 = _run(tmp_path, geometry="poincare", loss="distortion", epochs=100)
    assert abs(r1.metrics["final_train_loss"] - r2.metrics["final_train_loss"]) < 1e-6


def test_invalid_optimizer_combo(tmp_path):
    _tiny_tree_dir(tmp_path)
    cfg = _cfg(tmp_path, geometry="poincare")
    cfg["train"]["optimizer"] = "adam"  # incompatible with poincare
    with pytest.raises(ValueError):
        train_embedding("tiny", cfg, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path))
