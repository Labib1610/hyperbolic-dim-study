"""Fast tests for exp_final (grid, resume, aggregation, Q2 correlation). No real grid."""

import networkx as nx
import numpy as np
import pandas as pd

from src import data
from src.experiments import exp_final as ef


def test_grid_count():
    grid = ef.build_grid()
    assert len(grid) == len(ef.SUBTREES) * len(ef.DIMS) * len(ef.SEEDS) * len(ef.GEOMETRIES)
    assert len(ef.SUBTREES) == 18
    assert set(ef.DIMS) == {2, 4, 8, 16, 32, 64, 128}
    assert len(ef.SEEDS) == 5
    # no duplicate combos
    keys = {(c["subtree"], c["dim"], c["geometry"], c["seed"]) for c in grid}
    assert len(keys) == len(grid)


def test_degree_cv_star_vs_path():
    assert ef.degree_cv(nx.star_graph(6)) > ef.degree_cv(nx.path_graph(6))


def _tiny_suite(tmp_path):
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    data.save_subtree(g, "A", str(tmp_path / "subtrees" / "tiny"))
    pd.DataFrame([{"name": "tiny", "root": "r.n.01", "mode": "bfs", "target_size": 5,
                   "degree_cv": 0.5, "num_nodes": 5, "depth": 2, "avg_branching": 2.0,
                   "max_branching": 2, "num_leaves": 3, "diameter": 3}]).to_csv(
        tmp_path / "subtrees" / "summary.csv", index=False)


def test_resume_skips_completed(tmp_path):
    _tiny_suite(tmp_path)
    kw = dict(subtrees=["tiny"], dims=[2], seeds=[0], geometries=["poincare"],
              base_dir=str(tmp_path / "runs"), results_dir=str(tmp_path), data_dir=str(tmp_path))
    # override cfg to a tiny fixed-epoch budget by monkeypatching load_config default via cfg build.
    import src.experiments.exp_final as m

    orig = m._build_config

    def fast_cfg(combo, base_cfg, struct_row, data_dir):
        cfg = orig(combo, base_cfg, struct_row, data_dir)
        cfg["train"].update({"stopping": "fixed_epochs", "fixed_epochs": 5})
        return cfg

    m._build_config = fast_cfg
    try:
        s1 = m.run_grid(**kw)
        assert s1["launched"] == 1 and s1["failed"] == 0
        n1 = len(list((tmp_path / "runs").iterdir()))
        s2 = m.run_grid(**kw)
        assert s2["launched"] == 0 and s2["skipped"] == 1
        assert len(list((tmp_path / "runs").iterdir())) == n1  # no new run dirs
    finally:
        m._build_config = orig


def test_aggregation_means(tmp_path):
    rows = []
    for seed, dw, mp in [(0, 2.0, 0.8), (1, 4.0, 0.6)]:
        rows.append({"experiment": ef.EXPERIMENT_TAG, "subtree": "T", "train.geometry": "poincare",
                     "train.dim": 8, "train.seed": seed, "distortion_worst": dw,
                     "distortion_avg_scaled": 0.1, "map": mp, "delta_relative": 0.2, "alpha": 0.5,
                     "epochs_to_converge": 100, "subtree_depth": 5, "subtree_avg_branching": 2.0,
                     "subtree_degree_cv": 1.0, "root": "r.n.01"})
    csv = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    agg = ef.aggregate_final(str(csv), str(tmp_path / "agg.csv"))
    row = agg.iloc[0]
    assert abs(row["distortion_worst_mean"] - 3.0) < 1e-9
    assert abs(row["distortion_worst_std"] - np.std([2.0, 4.0], ddof=1)) < 1e-9
    assert abs(row["map_mean"] - 0.7) < 1e-9
    assert row["n_seeds"] == 2


def test_pearson_handchecked():
    # perfectly correlated -> 1.0 ; anti -> -1.0
    assert abs(ef.pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    assert abs(ef.pearson([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9


def test_figures_and_report_smoke(tmp_path):
    # synthetic aggregated table: 2 subtrees, 2 geoms, 2 dims
    rows = []
    for name, depth, dcv in [("shallow", 2, 0.5), ("deep", 8, 2.0)]:
        for geom in ["poincare", "euclidean"]:
            for dim in [2, 8]:
                base = 0.9 if geom == "poincare" else 0.6
                rows.append({"subtree": name, "train.geometry": geom, "train.dim": dim,
                             "map_mean": base, "map_std": 0.01,
                             "distortion_avg_scaled_mean": 0.1, "distortion_avg_scaled_std": 0.01,
                             "distortion_worst_mean": 5.0, "distortion_worst_std": 0.1,
                             "delta_relative_mean": 0.2, "delta_relative_std": 0.01,
                             "alpha_mean": 0.5, "epochs_to_converge_mean": 100,
                             "subtree_depth": depth, "subtree_avg_branching": 2.0,
                             "subtree_degree_cv": dcv, "root": "animal.n.01", "n_seeds": 5})
    agg = pd.DataFrame(rows)
    ef.make_figures(agg, outdir=str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "figQ1a_distortion_vs_dim.png").exists()
    assert (tmp_path / "figs" / "figQ2a_adv_vs_depth.png").exists()
    text = ef.generate_report(agg, path=str(tmp_path / "rep.md"))
    assert "Q1" in text and "Q2" in text
