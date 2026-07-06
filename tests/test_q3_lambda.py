"""Fast tests for the Q3 λ sweep (scoring only; no retraining)."""

import numpy as np
import pandas as pd

from src.experiments import exp_q3_hyperlex as q3
from src.experiments import exp_q3_lambda as lam


def test_score_b_lambda0_equals_score_a():
    d = np.array([1.0, 2.0, 0.5])
    gd = np.array([0.3, -0.2, 0.1])
    # Score B at λ=0 == Score A == -d_G
    assert np.allclose(lam.score_b(d, gd, 0.0), -d)


def test_peak_lambda_handchecked():
    lams = [0.0, 1.0, 2.0, 5.0, 10.0]
    rhos = [0.0, 0.2, 0.5, 0.4, 0.3]  # peaks at λ=2 (interior)
    plam, prho, interior = lam.peak_lambda(lams, rhos)
    assert plam == 2.0 and abs(prho - 0.5) < 1e-12 and interior is True
    # monotone rising -> peak at edge, not interior
    plam2, _, interior2 = lam.peak_lambda(lams, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert plam2 == 10.0 and interior2 is False


def test_rescoring_deterministic():
    node_index = {"a.n.01": 0, "b.n.01": 1, "c.n.01": 2}
    coords = np.array([[0.1, 0.0], [0.4, 0.1], [0.6, 0.2]])

    class _S:
        def __init__(self, n): self._n = n
        def name(self): return self._n

    orig = q3.word_noun_synsets
    q3.word_noun_synsets = lambda w: {"x": [_S("a.n.01")], "y": [_S("c.n.01")]}.get(w, [])
    q3._word_idx_cache.clear()
    try:
        cov = pd.DataFrame([{"word1": "x", "word2": "y", "rating": 5.0}])
        s1 = q3.score_pairs(coords, "poincare", node_index, cov)
        s2 = q3.score_pairs(coords, "poincare", node_index, cov)
        assert np.allclose(s1["d_min"], s2["d_min"])
        assert np.allclose(s1["gen_diff"], s2["gen_diff"])
    finally:
        q3.word_noun_synsets = orig
        q3._word_idx_cache.clear()


def test_stability_verdict_detects_flip():
    best = pd.DataFrame([
        {"geometry": "poincare", "dim": 8, "rho_best": 0.4},
        {"geometry": "euclidean", "dim": 8, "rho_best": 0.3},  # P beats E here -> flip
        {"geometry": "poincare", "dim": 16, "rho_best": 0.3},
        {"geometry": "euclidean", "dim": 16, "rho_best": 0.4},
    ])
    flips = lam.stability_verdict(best)
    assert len(flips) == 1 and flips[0][0] == 8
