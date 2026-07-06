"""Fast checks for the paper-figure regeneration (plots from saved CSVs)."""

import os

import pytest

from src.figures import make_paper_figures as mpf

HAS_DATA = os.path.exists("outputs/exp_final_aggregated.csv")

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="exp_final_aggregated.csv not present")


def test_style_helper_runs():
    plt = mpf.set_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    plt.close(fig)


def test_all_figures_generated_nonempty(tmp_path):
    mpf.make_all()
    expected = [
        "figQ1a_distortion_vs_dim", "figQ1b_map_vs_dim",
        "figQ2_adv_vs_branching", "figQ2a_adv_vs_depth", "figQ2b_adv_vs_degcv",
        "figQ3_rho_vs_dim", "figQ3_rho_vs_lambda", "figQ3_generality_radius",
    ]
    for name in expected:
        pdf = mpf.OUT / f"{name}.pdf"
        png = mpf.OUT / f"{name}.png"
        assert pdf.exists() and pdf.stat().st_size > 0, f"missing/empty {pdf}"
        assert png.exists() and png.stat().st_size > 0, f"missing/empty {png}"


def test_q2_branching_r_matches_paper():
    agg = mpf.load_exp_final()
    adv = mpf.q2_advantage(agg, dim=8)
    import numpy as np
    r = float(np.corrcoef(adv["avg_branching"], adv["adv"])[0, 1])
    assert abs(r - 0.64) < 0.02, f"branching r={r} not within 0.02 of paper's 0.64"
