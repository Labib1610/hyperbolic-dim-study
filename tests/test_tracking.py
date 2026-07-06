"""Tests for the experiment-tracking layer. Uses tmp_path so real outputs/ is untouched."""

import json

import numpy as np
import pandas as pd

from src import tracking
from src.tracking import Run, demo_run, filter_results, flatten_dict, load_results


def _dirs(tmp_path):
    return {"base_dir": tmp_path / "runs", "results_dir": tmp_path}


def test_flatten_dict():
    assert flatten_dict({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert flatten_dict({"x": {"y": {"z": 3}}}) == {"x.y.z": 3}


def test_run_creates_dir_and_files(tmp_path):
    run = Run({"geometry": {"type": "poincare"}, "seed": 1}, name="unit", **_dirs(tmp_path))
    assert run.dir.exists()
    assert (run.dir / "config.yaml").exists()
    meta_path = run.dir / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "git_commit" in meta  # value may be a hash or "unknown"
    assert meta["run_id"] == run.dir.name
    assert "python_version" in meta


def test_log_metrics_merges(tmp_path):
    run = Run({"seed": 1}, name="merge", **_dirs(tmp_path))
    run.log_metrics({"a": 1})
    run.log_metrics({"b": 2})
    assert run.metrics == {"a": 1, "b": 2}
    on_disk = json.loads((run.dir / "metrics.json").read_text())
    assert on_disk == {"a": 1, "b": 2}


def test_log_curve_and_artifacts(tmp_path):
    run = Run({"seed": 1}, name="curve", **_dirs(tmp_path))
    run.log_curve("loss", [1.0, 0.5])
    assert (run.dir / "curves" / "loss.json").exists()
    run.save_array("coords", np.zeros((3, 2)))
    assert (run.dir / "artifacts" / "coords.npy").exists()

    src = tmp_path / "some.edges"
    src.write_text("a b\n")
    stored = run.log_artifact(src)
    assert stored.exists()
    assert stored.parent.name == "artifacts"


def test_save_figure_png_and_pdf(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Run({"seed": 1}, name="fig", **_dirs(tmp_path))
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    run.save_figure("curve", fig)
    plt.close(fig)
    assert (run.dir / "figures" / "curve.png").exists()
    assert (run.dir / "figures" / "curve.pdf").exists()


def test_finish_appends_results(tmp_path):
    run = Run({"geometry": {"type": "poincare"}, "seed": 1}, name="fin", **_dirs(tmp_path))
    run.log_metrics({"map": 0.9})
    run.finish()

    csv = tmp_path / "results.csv"
    jsonl = tmp_path / "results.jsonl"
    assert csv.exists() and jsonl.exists()
    df = load_results(csv)
    assert len(df) == 1
    assert df.iloc[0]["geometry.type"] == "poincare"
    assert df.iloc[0]["map"] == 0.9
    assert df.iloc[0]["status"] == "completed"
    # jsonl has exactly one line.
    assert len([ln for ln in jsonl.read_text().splitlines() if ln.strip()]) == 1


def test_demo_twice_schema_union(tmp_path):
    dirs = _dirs(tmp_path)
    demo_run(**dirs)
    # Second run introduces a NEW config key to exercise schema-union safety.
    demo_run(**dirs, extra_config={"new_flag": 123})

    df = load_results(tmp_path / "results.csv")
    assert len(df) == 2
    # Union of columns preserved; old row has NaN for the new column, not corruption.
    assert "new_flag" in df.columns
    assert pd.isna(df.iloc[0]["new_flag"])
    assert df.iloc[1]["new_flag"] == 123
    # Original columns still intact for both rows.
    assert (df["geometry.type"] == "poincare").all()
    assert (df["map"] == 0.87).all()


def test_load_results_absent_returns_empty(tmp_path):
    df = load_results(tmp_path / "nope.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_filter_results(tmp_path):
    dirs = _dirs(tmp_path)
    demo_run(**dirs)
    demo_run(**dirs, extra_config={"geometry": {"type": "euclidean", "dim": 5}})
    df = load_results(tmp_path / "results.csv")

    poincare = filter_results(df, **{"geometry.type": "poincare"})
    assert len(poincare) == 1
    assert poincare.iloc[0]["geometry.type"] == "poincare"
    # Non-existent column -> empty selection, no error.
    assert filter_results(df, nosuchcol="x").empty
