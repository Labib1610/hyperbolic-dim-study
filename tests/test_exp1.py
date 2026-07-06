"""Exp1 tests: grid integrity, resume, aggregation, figure smoke. Fast (tiny grid)."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import data
from src.experiments import exp1_dimension_sweep as exp1


def test_grid_integrity():
    grid = exp1.build_grid()
    assert len(grid) == 240
    assert {c["dim"] for c in grid} == {2, 8, 32, 128}
    assert {c["seed"] for c in grid} == {0, 1, 2, 3, 4}
    assert {c["geometry"] for c in grid} == {"poincare", "euclidean"}
    assert {c["loss"] for c in grid} == {"distortion"}
    assert {c["subtree"] for c in grid} == set(exp1.SUBTREES)
    assert len(exp1.SUBTREES) == 6
    # no duplicate combos
    keys = {(c["subtree"], c["dim"], c["geometry"], c["seed"]) for c in grid}
    assert len(keys) == 240


def _tiny_subtree(tmp_path, name="tiny"):
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    data.save_subtree(g, "A", str(tmp_path / "subtrees" / name))


def _count_run_dirs(runs_dir):
    return len([d for d in runs_dir.iterdir() if d.is_dir()]) if runs_dir.exists() else 0


def test_resume_skips_completed(tmp_path):
    _tiny_subtree(tmp_path)
    kw = dict(
        dims=[2, 8], seeds=[0], subtrees=["tiny"], geometries=["poincare"],
        epochs=3, base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path),
        data_dir=str(tmp_path), progress=False,
    )
    s1 = exp1.run_sweep(**kw)
    assert s1["launched"] == 2 and s1["failed"] == 0
    n_after_first = _count_run_dirs(tmp_path / "runs")
    assert n_after_first == 2

    s2 = exp1.run_sweep(**kw)
    assert s2["launched"] == 0
    assert s2["skipped"] == 2
    # No new run dirs created on the resumed call.
    assert _count_run_dirs(tmp_path / "runs") == n_after_first


def _synthetic_results(tmp_path):
    rows = []
    # one group: subtree=T, poincare, dim=2, seeds 0,1 with distortion_worst 2 and 4
    for seed, dw, m in [(0, 2.0, 0.8), (1, 4.0, 0.6)]:
        rows.append({
            "experiment": exp1.EXPERIMENT_TAG, "subtree": "T", "train.geometry": "poincare",
            "train.dim": 2, "train.seed": seed, "train.loss": "distortion",
            "distortion_worst": dw, "distortion_avg_scaled": 0.1, "map": m,
            "delta_relative": 0.2, "subtree_depth": 3, "subtree_avg_branching": 2.0,
        })
    csv = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_aggregation_means(tmp_path):
    csv = _synthetic_results(tmp_path)
    out = tmp_path / "agg.csv"
    agg = exp1.aggregate_exp1(results_path=str(csv), out_path=str(out))
    assert out.exists()
    row = agg[(agg["subtree"] == "T") & (agg["train.geometry"] == "poincare") & (agg["train.dim"] == 2)].iloc[0]
    # mean(2,4)=3 ; std of {2,4} (pandas sample std, ddof=1) = sqrt(2) ~ 1.4142
    assert abs(row["distortion_worst_mean"] - 3.0) < 1e-9
    assert abs(row["distortion_worst_std"] - np.std([2.0, 4.0], ddof=1)) < 1e-9
    assert abs(row["map_mean"] - 0.7) < 1e-9
    assert row["n_seeds"] == 2
    assert row["subtree_depth"] == 3


def _synthetic_agg():
    rows = []
    for subtree, depth in [("shallow", 2), ("deep", 8)]:
        for geom in ["poincare", "euclidean"]:
            for dim in [2, 8, 32, 128]:
                base = 0.1 if geom == "poincare" else 0.3
                rows.append({
                    "subtree": subtree, "train.geometry": geom, "train.dim": dim,
                    "distortion_avg_scaled_mean": base / (dim ** 0.2),
                    "distortion_avg_scaled_std": 0.01,
                    "map_mean": 0.4 + 0.1 * np.log2(dim) / 8, "map_std": 0.02,
                    "distortion_worst_mean": 5.0, "distortion_worst_std": 0.5,
                    "delta_relative_mean": 0.2, "delta_relative_std": 0.01,
                    "subtree_depth": depth, "subtree_avg_branching": 2.0, "n_seeds": 5,
                })
    return pd.DataFrame(rows)


def test_figures_smoke(tmp_path):
    agg = _synthetic_agg()
    outdir = tmp_path / "figs"
    exp1.make_figures(agg, outdir=str(outdir), subtrees=["shallow", "deep"])
    for name in ["fig1a_distortion_vs_dim", "fig1b_map_vs_dim",
                 "fig1c_dimension_savings", "fig1d_advantage_vs_depth"]:
        assert (outdir / f"{name}.png").exists()
        assert (outdir / f"{name}.pdf").exists()


def test_report_generation(tmp_path):
    agg = _synthetic_agg()
    path = tmp_path / "report.md"
    text = exp1.generate_report(agg, path=str(path), subtrees=["shallow", "deep"])
    assert path.exists()
    assert "Experiment 1" in text
    assert "Q1" in text and "Q2" in text
