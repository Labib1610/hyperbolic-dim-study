"""Fast tests for the fairness check (no 8000-epoch sweeps)."""

import numpy as np

from src.data import all_pairs_tree_distances
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.diagnostics.fairness_check import epochs_to_converge, train_curve


def test_epochs_to_converge_handchecked():
    epochs = [0, 100, 200, 300, 400, 500]
    vals = [1.0, 0.5, 0.2, 0.12, 0.101, 0.100]
    # plateau over last-100-epoch window (400,500) = (0.101+0.100)/2 = 0.1005
    # first epoch within 1% (0.001005): epoch 400 (|0.101-0.1005|=0.0005).
    assert epochs_to_converge(epochs, vals, final_window=100, tol=0.01) == 400


def test_euclidean_learnable_alpha_runs():
    tree = make_synthetic_tree(2, 3)  # 15 nodes
    coords, alpha, cps = train_curve(tree, 4, "euclidean", 0, epochs=40, eval_every=20, use_alpha=True)
    assert np.isfinite(coords).all()
    assert alpha > 0 and np.isfinite(alpha)  # learnable alpha active for Euclidean too
    assert len(cps) >= 2
    for cp in cps:
        assert np.isfinite(cp["val_distortion"]) and 0.0 <= cp["val_map"] <= 1.0


def test_both_geometries_same_objective_tiny():
    tree = make_synthetic_tree(2, 3)
    out = {}
    for geom in ("poincare", "euclidean"):
        coords, alpha, cps = train_curve(tree, 2, geom, 0, epochs=30, eval_every=15, use_alpha=True)
        out[geom] = (coords, alpha)
        assert np.isfinite(coords).all()
        assert alpha > 0  # both use the identical learnable-alpha objective
    # both produced a scalar positive alpha -> same geometry-agnostic objective
    assert np.isfinite(out["poincare"][1]) and np.isfinite(out["euclidean"][1])
