"""Fast tests for the fix-prototype sandbox (no full oracle sweep)."""

import geoopt
import numpy as np
import torch

from src.diagnostics.diag_plateau import make_synthetic_tree
from src.diagnostics.fix_prototype import (
    default_cfg,
    init_coords,
    norm_stats,
    train_poincare_fixed,
)


def test_init_modes_norm_ranges():
    manifold = geoopt.PoincareBall()
    origin = init_coords(50, 4, "origin", manifold)
    ball = init_coords(50, 4, "uniform_ball", manifold)
    randn = init_coords(50, 4, "scaled_randn", manifold)

    on = origin.norm(dim=-1)
    bn = ball.norm(dim=-1)
    rn = randn.norm(dim=-1)
    # all strictly inside the ball
    for n in (on, bn, rn):
        assert (n < 1.0).all()
    # origin: tiny
    assert on.max() < 0.01
    # uniform_ball: spread roughly in [0.1, 0.8]
    assert bn.min() > 0.05 and bn.max() < 0.85
    assert bn.max() - bn.min() > 0.2  # genuinely spread
    # scaled_randn: bigger than origin
    assert rn.max() > on.max()


def test_learnable_alpha_receives_gradient():
    tree = make_synthetic_tree(2, 3)  # 15 nodes
    cfg = default_cfg(scale="learnable", epochs=20, lr=0.05)
    coords, info = train_poincare_fixed(tree, 2, cfg, seed=0)
    # alpha is a finite positive scalar after training and differs from softplus(0)=0.693
    assert "alpha" in info
    assert info["alpha"] > 0
    assert np.isfinite(info["alpha"])
    assert abs(info["alpha"] - 0.6931) > 1e-3  # it moved


def test_winning_fix_beats_baseline_tiny():
    # On a small oracle, the learnable-scale fix should beat the origin baseline MAP.
    tree = make_synthetic_tree(2, 3)  # 15 nodes
    base_cfg = default_cfg(epochs=300, lr=0.05)
    fix_cfg = default_cfg(init="uniform_ball", scale="learnable", epochs=300, lr=0.05)

    from src.metrics import evaluate_embedding

    base_coords, _ = train_poincare_fixed(tree, 2, base_cfg, seed=0)
    fix_coords, info = train_poincare_fixed(tree, 2, fix_cfg, seed=0)

    base_m = evaluate_embedding(base_coords, tree, "poincare", delta_samples=500)
    fix_m = evaluate_embedding(fix_coords, tree, "poincare", delta_samples=500)

    for m in (base_m, fix_m):
        for k in ["distortion_avg_scaled", "distortion_worst", "map"]:
            assert np.isfinite(m[k])
    assert fix_m["map"] >= base_m["map"], (base_m["map"], fix_m["map"])


def test_norm_stats_shape():
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [0.95, 0.0]])
    s = norm_stats(coords)
    assert abs(s["max"] - 0.95) < 1e-9
    assert 0.0 <= s["frac_gt_0.9"] <= 1.0
