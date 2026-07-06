"""Experiment 1 — Dimension sweep (Q1 efficiency, previews Q2).

Orchestrates train + evaluate across a grid of (dimension, seed, subtree, geometry)
and aggregates the master results table into figures + a report. This is an
experiment RUNNER only: it calls the tested Stage 3 trainer and Stage 4 metrics
without modifying them.

Scientific question (Q1): does hyperbolic geometry reach at LOW dimension the
quality Euclidean needs HIGH dimension to match? Preview of Q2: does the Poincaré
advantage widen on deeper subtrees?

Grid: dims {2,8,32,128} × seeds {0..4} × 6 subtrees × {poincare,euclidean},
loss=distortion  →  240 runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.metrics import evaluate_run
from src.tracking import load_results
from src.trainer import train_embedding

# --- the exact grid --------------------------------------------------------
DIMS = [2, 8, 32, 128]
SEEDS = [0, 1, 2, 3, 4]
SUBTREES = ["bfs_s50", "bfs_s200", "deep_s100", "balanced_s100", "deep_s200", "balanced_s200"]
GEOMETRIES = ["poincare", "euclidean"]
LOSS = "distortion"
EXPERIMENT_TAG = "exp1_dimension_sweep"
DEFAULT_EPOCHS = 500  # >=500 so low-dim isn't under-trained

METRICS = ["distortion_worst", "distortion_avg_scaled", "map", "delta_relative"]


def build_grid(dims=DIMS, seeds=SEEDS, subtrees=SUBTREES, geometries=GEOMETRIES, loss=LOSS):
    """Enumerate all grid combos as a list of dicts (deterministic order)."""
    grid = []
    for subtree in subtrees:
        for dim in dims:
            for geometry in geometries:
                for seed in seeds:
                    grid.append(
                        {"subtree": subtree, "dim": dim, "geometry": geometry,
                         "seed": seed, "loss": loss}
                    )
    return grid


# ---------------------------------------------------------------------------
# Task 1/2 — runner with resume + tagging
# ---------------------------------------------------------------------------


def _subtree_stats(data_dir: str, subtree: str) -> dict:
    meta = json.loads((Path(data_dir) / "subtrees" / subtree / "meta.json").read_text())
    return meta.get("stats", {})


def _build_config(combo: dict, epochs: int, data_dir: str, base_cfg: dict) -> dict:
    """Build a full training config for one combo, tagged for this experiment."""
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("paths", {})
    cfg["paths"]["data_dir"] = data_dir
    cfg["subtree"] = combo["subtree"]
    cfg["seed"] = combo["seed"]
    cfg["train"].update(
        {
            "geometry": combo["geometry"],
            "dim": combo["dim"],
            "loss": combo["loss"],
            "seed": combo["seed"],
            "epochs": epochs,
            "optimizer": "radam" if combo["geometry"] == "poincare" else "adam",
        }
    )
    # Structure info tagged INTO the row so Q2 analysis needs no rejoin.
    stats = _subtree_stats(data_dir, combo["subtree"])
    cfg["experiment"] = EXPERIMENT_TAG
    cfg["subtree_depth"] = stats.get("depth")
    cfg["subtree_avg_branching"] = stats.get("avg_branching")
    return cfg


def _completed_combos(df: pd.DataFrame) -> set:
    """Set of (geometry, dim, subtree, seed, loss) already completed with metrics."""
    done = set()
    req = ["train.geometry", "train.dim", "subtree", "train.seed", "train.loss", "distortion_worst"]
    if df.empty or any(c not in df.columns for c in req):
        return done
    has_exp = "experiment" in df.columns
    for _, row in df.iterrows():
        if has_exp and row.get("experiment") != EXPERIMENT_TAG:
            continue
        if pd.isna(row["distortion_worst"]):
            continue
        done.add(
            (row["train.geometry"], int(row["train.dim"]), row["subtree"],
             int(row["train.seed"]), row["train.loss"])
        )
    return done


def run_sweep(
    dims=DIMS, seeds=SEEDS, subtrees=SUBTREES, geometries=GEOMETRIES, loss=LOSS,
    epochs=DEFAULT_EPOCHS, base_dir="outputs/runs", results_dir="outputs",
    data_dir="data", config_path="configs/default.yaml", progress=True,
) -> dict:
    """Run the grid with resume support and per-run failure isolation."""
    grid = build_grid(dims, seeds, subtrees, geometries, loss)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "results.csv"
    fail_log = results_dir / "exp1_failures.log"

    base_cfg = load_config(config_path)
    done = _completed_combos(load_results(results_csv))

    total = len(grid)
    launched = skipped = 0
    failures: list[str] = []
    start = time.time()

    try:
        from tqdm import tqdm

        bar = tqdm(total=total, disable=not progress, desc="exp1")
    except Exception:
        bar = None

    for processed, combo in enumerate(grid, start=1):
        key = (combo["geometry"], combo["dim"], combo["subtree"], combo["seed"], combo["loss"])
        if key in done:
            skipped += 1
        else:
            try:
                cfg = _build_config(combo, epochs, data_dir, base_cfg)
                run = train_embedding(
                    combo["subtree"], cfg, base_dir=base_dir, results_dir=str(results_dir)
                )
                evaluate_run(run.dir)
                done.add(key)
                launched += 1
            except Exception as exc:  # noqa: BLE001 — isolate one bad run
                msg = f"{key}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                with open(fail_log, "a") as f:
                    f.write(msg + "\n")

        if bar is not None:
            bar.update(1)
        if progress and processed % 10 == 0:
            elapsed = time.time() - start
            rate = (launched + skipped) / max(elapsed, 1e-9)
            remaining = total - processed
            eta = remaining / max(rate, 1e-9)
            print(
                f"  [{processed}/{total}] launched={launched} skipped={skipped} "
                f"failed={len(failures)} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    if bar is not None:
        bar.close()

    summary = {
        "total": total, "launched": launched, "skipped": skipped,
        "failed": len(failures), "failures": failures, "elapsed_s": time.time() - start,
    }
    print(
        f"Sweep done: {launched} launched, {skipped} skipped, {len(failures)} failed, "
        f"{summary['elapsed_s']:.0f}s."
    )
    return summary


# ---------------------------------------------------------------------------
# Task 3 — aggregation
# ---------------------------------------------------------------------------


def aggregate_exp1(results_path="outputs/results.csv", out_path="outputs/exp1_aggregated.csv") -> pd.DataFrame:
    """Aggregate mean/std across seeds per (subtree, geometry, dim); save tidy CSV."""
    df = load_results(results_path)
    if "experiment" in df.columns:
        df = df[df["experiment"] == EXPERIMENT_TAG]
    if df.empty:
        raise ValueError("no exp1 rows found to aggregate")

    agg_spec = {}
    for m in METRICS:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")
    agg_spec["subtree_depth"] = ("subtree_depth", "first")
    agg_spec["subtree_avg_branching"] = ("subtree_avg_branching", "first")
    agg_spec["n_seeds"] = ("map", "count")

    agg = (
        df.groupby(["subtree", "train.geometry", "train.dim"])
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["subtree_depth", "subtree", "train.geometry", "train.dim"])
        .reset_index(drop=True)
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    return agg


# ---------------------------------------------------------------------------
# Task 4 — figures
# ---------------------------------------------------------------------------


def _save_fig(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")


def _val(agg: pd.DataFrame, subtree: str, geom: str, dim: int, col: str) -> float:
    r = agg[(agg["subtree"] == subtree) & (agg["train.geometry"] == geom) & (agg["train.dim"] == dim)]
    return float(r.iloc[0][col]) if not r.empty else np.nan


def _dimension_savings(agg: pd.DataFrame, subtree: str, dims) -> int | None:
    """Smallest Euclidean dim whose mean scaled distortion <= Poincaré-at-dim-2.

    Returns ``None`` if Euclidean never matches within the grid (report as '>max').
    """
    p2 = _val(agg, subtree, "poincare", 2, "distortion_avg_scaled_mean")
    for d in sorted(dims):
        e = _val(agg, subtree, "euclidean", d, "distortion_avg_scaled_mean")
        if not np.isnan(e) and not np.isnan(p2) and e <= p2:
            return d
    return None


def make_figures(agg: pd.DataFrame, outdir="outputs/figures/exp1", dims=DIMS, subtrees=SUBTREES) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    # order subtrees by depth for readability
    order = sorted(
        [s for s in subtrees if (agg["subtree"] == s).any()],
        key=lambda s: _val_first(agg, s, "subtree_depth"),
    )
    colors = {"poincare": "tab:blue", "euclidean": "tab:red"}

    def _panel_grid(metric_mean, metric_std, title, fname, logy=False):
        n = len(order)
        ncols, nrows = 3, int(np.ceil(n / 3))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
        for i, subtree in enumerate(order):
            ax = axes[i // ncols][i % ncols]
            for geom in GEOMETRIES:
                xs, ys, es = [], [], []
                for d in sorted(dims):
                    m = _val(agg, subtree, geom, d, metric_mean)
                    s = _val(agg, subtree, geom, d, metric_std)
                    if not np.isnan(m):
                        xs.append(d); ys.append(m); es.append(0.0 if np.isnan(s) else s)
                if xs:
                    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
                    ax.plot(xs, ys, "-o", color=colors[geom], label=geom)
                    ax.fill_between(xs, ys - es, ys + es, color=colors[geom], alpha=0.2)
            depth = _val_first(agg, subtree, "subtree_depth")
            ax.set_title(f"{subtree} (depth={depth:g})", fontsize=9)
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("dimension")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig(fig, outdir, fname)
        plt.close(fig)

    _panel_grid("distortion_avg_scaled_mean", "distortion_avg_scaled_std",
                "Fig 1a — scaled avg distortion vs dimension", "fig1a_distortion_vs_dim")
    _panel_grid("map_mean", "map_std", "Fig 1b — MAP vs dimension", "fig1b_map_vs_dim")

    # Fig 1c — dimension savings vs depth
    fig, ax = plt.subplots(figsize=(6, 4))
    for subtree in order:
        depth = _val_first(agg, subtree, "subtree_depth")
        sav = _dimension_savings(agg, subtree, dims)
        y = sav if sav is not None else max(dims) * 2  # ">max"
        ax.scatter(depth, y, s=60)
        label = f"{subtree}\n{'>'+str(max(dims)) if sav is None else sav}"
        ax.annotate(label, (depth, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(max(dims), ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("subtree depth")
    ax.set_ylabel("Euclidean dim to match Poincaré@dim2")
    ax.set_yscale("log", base=2)
    ax.set_title("Fig 1c — dimension savings vs depth (Q2 preview)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, outdir, "fig1c_dimension_savings")
    plt.close(fig)

    # Fig 1d — advantage vs structure at fixed low dim (8)
    fig, ax = plt.subplots(figsize=(6, 4))
    for subtree in order:
        depth = _val_first(agg, subtree, "subtree_depth")
        e = _val(agg, subtree, "euclidean", 8, "distortion_avg_scaled_mean")
        p = _val(agg, subtree, "poincare", 8, "distortion_avg_scaled_mean")
        if not (np.isnan(e) or np.isnan(p)):
            ax.scatter(depth, e - p, s=60)
            ax.annotate(subtree, (depth, e - p), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("subtree depth")
    ax.set_ylabel("(Euclidean - Poincaré) scaled distortion @dim8")
    ax.set_title("Fig 1d — advantage vs structure (dim=8)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, outdir, "fig1d_advantage_vs_depth")
    plt.close(fig)


def _val_first(agg: pd.DataFrame, subtree: str, col: str) -> float:
    r = agg[agg["subtree"] == subtree]
    return float(r.iloc[0][col]) if not r.empty else np.nan


# ---------------------------------------------------------------------------
# Task 5 — auto-generated analysis report
# ---------------------------------------------------------------------------


def generate_report(agg: pd.DataFrame, path="outputs/exp1_report.md", dims=DIMS, subtrees=SUBTREES) -> str:
    order = sorted(
        [s for s in subtrees if (agg["subtree"] == s).any()],
        key=lambda s: _val_first(agg, s, "subtree_depth"),
    )
    lo, hi = min(dims), max(dims)
    lines = ["# Experiment 1 — Dimension Sweep: auto-generated report", ""]

    lines.append(f"Grid: dims={dims}, seeds averaged, loss=distortion. Low dim={lo}, high dim={hi}.")
    lines.append("")

    # Headline table
    lines.append(f"## Poincaré vs Euclidean — scaled distortion & MAP at dim {lo} and dim {hi}")
    lines.append("")
    lines.append(f"| subtree | depth | P dist@{lo} | E dist@{lo} | P dist@{hi} | E dist@{hi} | P MAP@{hi} | E MAP@{hi} |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in order:
        depth = _val_first(agg, s, "subtree_depth")
        row = [
            f"{_val(agg, s, 'poincare', lo, 'distortion_avg_scaled_mean'):.3f}",
            f"{_val(agg, s, 'euclidean', lo, 'distortion_avg_scaled_mean'):.3f}",
            f"{_val(agg, s, 'poincare', hi, 'distortion_avg_scaled_mean'):.3f}",
            f"{_val(agg, s, 'euclidean', hi, 'distortion_avg_scaled_mean'):.3f}",
            f"{_val(agg, s, 'poincare', hi, 'map_mean'):.3f}",
            f"{_val(agg, s, 'euclidean', hi, 'map_mean'):.3f}",
        ]
        lines.append(f"| {s} | {depth:g} | " + " | ".join(row) + " |")
    lines.append("")

    # Q1: does Poincaré win at low dim; does Euclidean catch up?
    lines.append("## Q1 — efficiency (low-dim advantage & catch-up)")
    lines.append("")
    for s in order:
        p_lo = _val(agg, s, "poincare", lo, "distortion_avg_scaled_mean")
        e_lo = _val(agg, s, "euclidean", lo, "distortion_avg_scaled_mean")
        p_hi = _val(agg, s, "poincare", hi, "distortion_avg_scaled_mean")
        e_hi = _val(agg, s, "euclidean", hi, "distortion_avg_scaled_mean")
        sav = _dimension_savings(agg, s, dims)
        sav_str = f">{hi}" if sav is None else str(sav)
        lines.append(
            f"- **{s}** (depth {_val_first(agg, s, 'subtree_depth'):g}): "
            f"dist@{lo} P={p_lo:.3f} vs E={e_lo:.3f}; dist@{hi} P={p_hi:.3f} vs E={e_hi:.3f}. "
            f"Euclidean dim to match Poincaré@dim{lo}: **{sav_str}**."
        )
    lines.append("")

    # Q2 preview: does advantage grow with depth?
    lines.append("## Q2 preview — does the advantage grow with subtree depth?")
    lines.append("")
    lines.append("| subtree | depth | (E - P) scaled distortion @dim8 | dim-savings |")
    lines.append("|---|---|---|---|")
    for s in order:
        depth = _val_first(agg, s, "subtree_depth")
        e8 = _val(agg, s, "euclidean", 8, "distortion_avg_scaled_mean")
        p8 = _val(agg, s, "poincare", 8, "distortion_avg_scaled_mean")
        sav = _dimension_savings(agg, s, dims)
        sav_str = f">{hi}" if sav is None else str(sav)
        lines.append(f"| {s} | {depth:g} | {e8 - p8:+.3f} | {sav_str} |")
    lines.append("")

    # Trainer health: MAP at high dim
    best_map = max((_val(agg, s, "poincare", hi, "map_mean") for s in order), default=float("nan"))
    lines.append("## Trainer health")
    lines.append("")
    lines.append(
        f"- Best Poincaré MAP at dim {hi}: **{best_map:.3f}** "
        f"({'>0.6 OK' if best_map > 0.6 else 'below 0.6 — inspect'})."
    )
    lines.append("")

    # Honesty: any Euclidean wins
    wins = []
    for s in order:
        for d in dims:
            e = _val(agg, s, "euclidean", d, "distortion_avg_scaled_mean")
            p = _val(agg, s, "poincare", d, "distortion_avg_scaled_mean")
            if not (np.isnan(e) or np.isnan(p)) and e < p:
                wins.append(f"{s}@dim{d} (E={e:.3f} < P={p:.3f})")
    lines.append("## Where Euclidean beats Poincaré (scaled distortion, honest)")
    lines.append("")
    lines.append("- " + ("; ".join(wins) if wins else "None — Poincaré <= Euclidean at every (subtree, dim)."))
    lines.append("")

    text = "\n".join(lines)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exp1 dimension sweep runner + analysis.")
    p.add_argument("--run", action="store_true", help="Run the sweep grid.")
    p.add_argument("--analyze-only", action="store_true", help="Skip sweep; only aggregate+figures+report.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    if args.run and not args.analyze_only:
        run_sweep(epochs=args.epochs, progress=not args.no_progress)

    agg = aggregate_exp1()
    make_figures(agg)
    generate_report(agg)
    print("Aggregated -> outputs/exp1_aggregated.csv")
    print("Figures    -> outputs/figures/exp1/")
    print("Report     -> outputs/exp1_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
