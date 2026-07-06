"""Experiment FINAL — the fair results run (Q1 efficiency + Q2 structure law).

Uses the verified production trainer (learnable-α + R2 convergence, Stage 6) to
produce the paper's numbers across ~18 subtrees spanning 3 WordNet roots × 7 dims
× 5 seeds × 2 geometries (~1260 runs). Runner + analysis only — no edits to
trainer/geometry/metrics.

Q1: hyperbolic reaches at LOW dim the quality Euclidean needs HIGH dim for.
Q2 (novel): the hyperbolic advantage is a predictable function of hierarchy
    structure (depth / branching / degree_cv).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import (
    all_pairs_tree_distances, build_wordnet_noun_graph, dag_to_tree, load_subtree,
    sample_subtree, save_subtree, subtree_stats,
)
from src.metrics import evaluate_run
from src.tracking import load_results
from src.trainer import train_embedding

# --- experiment grid -------------------------------------------------------
ROOTS = ["animal.n.01", "plant.n.02", "artifact.n.01"]
MODES = ["bfs", "deep", "balanced"]
SIZES = [100, 200]
DIMS = [2, 4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
GEOMETRIES = ["poincare", "euclidean"]
EXPERIMENT_TAG = "exp_final"
METRICS = ["distortion_worst", "distortion_avg_scaled", "map", "delta_relative"]


def _short(root: str) -> str:
    return root.split(".")[0]


def subtree_name(root: str, mode: str, size: int) -> str:
    return f"{_short(root)}_{mode}_s{size}"


def all_subtree_names() -> list[str]:
    return [subtree_name(r, m, s) for r in ROOTS for m in MODES for s in SIZES]


SUBTREES = all_subtree_names()  # 18


# ---------------------------------------------------------------------------
# Structural scalar
# ---------------------------------------------------------------------------


def degree_cv(tree: nx.Graph) -> float:
    """Coefficient of variation of node degree (irregularity proxy, Stage 5.7)."""
    deg = np.array([d for _, d in tree.degree()], dtype=float)
    return float(deg.std() / deg.mean()) if deg.mean() > 0 else 0.0


# ---------------------------------------------------------------------------
# Task 0 — build the expanded subtree suite
# ---------------------------------------------------------------------------


def build_subtree_suite(data_dir="data", base_seed=42) -> pd.DataFrame:
    """Build ~18 subtrees across 3 roots and regenerate summary.csv (+ degree_cv)."""
    out_dir = Path(data_dir) / "subtrees"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    mode_off = {"bfs": 0, "deep": 1, "balanced": 2}
    for root in ROOTS:
        print(f"  building graph for {root} ...", flush=True)
        raw = build_wordnet_noun_graph(root)
        tree_graph = dag_to_tree(raw, root)
        for mode in MODES:
            for size in SIZES:
                seed = base_seed * 1000 + mode_off[mode] * 100 + size
                tree = sample_subtree(tree_graph, target_size=size, mode=mode,
                                      seed=seed, root_synset_name=root)
                name = subtree_name(root, mode, size)
                save_subtree(tree, root, str(out_dir / name))
                stats = subtree_stats(tree, root)
                rows.append({"name": name, "root": root, "mode": mode, "target_size": size,
                             "degree_cv": degree_cv(tree), **stats})
                print(f"    {name:24s} N={stats['num_nodes']:4d} depth={stats['depth']:2d} "
                      f"avg_br={stats['avg_branching']:.2f} deg_cv={degree_cv(tree):.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def load_structure(data_dir="data") -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "subtrees" / "summary.csv")


# ---------------------------------------------------------------------------
# Grid + config
# ---------------------------------------------------------------------------


def build_grid(subtrees=None, dims=DIMS, seeds=SEEDS, geometries=GEOMETRIES):
    subtrees = subtrees or SUBTREES
    grid = []
    for name in subtrees:
        for dim in dims:
            for geom in geometries:
                for seed in seeds:
                    grid.append({"subtree": name, "dim": dim, "geometry": geom, "seed": seed})
    return grid


def _build_config(combo, base_cfg, struct_row, data_dir) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("paths", {})["data_dir"] = data_dir
    cfg["subtree"] = combo["subtree"]
    cfg["seed"] = combo["seed"]
    cfg["train"].update({
        "geometry": combo["geometry"], "dim": combo["dim"], "loss": "distortion",
        "seed": combo["seed"], "optimizer": "radam" if combo["geometry"] == "poincare" else "adam",
        # production fair defaults (also in default.yaml, set explicitly for clarity):
        "scale": "learnable", "stopping": "convergence",
        "patience_epochs": 500, "min_delta_rel": 0.005, "epoch_cap": 8000,
    })
    cfg["experiment"] = EXPERIMENT_TAG
    cfg["root"] = struct_row["root"]
    cfg["subtree_depth"] = int(struct_row["depth"])
    cfg["subtree_avg_branching"] = float(struct_row["avg_branching"])
    cfg["subtree_degree_cv"] = float(struct_row["degree_cv"])
    return cfg


def _completed_combos(df) -> set:
    done = set()
    req = ["train.geometry", "train.dim", "subtree", "train.seed", "distortion_worst"]
    if df.empty or any(c not in df.columns for c in req):
        return done
    has_exp = "experiment" in df.columns
    for _, r in df.iterrows():
        if has_exp and r.get("experiment") != EXPERIMENT_TAG:
            continue
        if pd.isna(r["distortion_worst"]):
            continue
        done.add((r["train.geometry"], int(r["train.dim"]), r["subtree"], int(r["train.seed"])))
    return done


# ---------------------------------------------------------------------------
# Task 1 — sanity run + ETA
# ---------------------------------------------------------------------------


def run_sanity(struct, base_dir="outputs/runs", results_dir="outputs", data_dir="data",
               dims=(2, 8, 128), seeds=(0, 1)):
    """Run a small sanity grid (shallow + deep subtree), report P-advantage + project ETA."""
    base_cfg = load_config("configs/default.yaml")
    # pick shallowest and deepest subtree by depth
    shallow = struct.loc[struct["depth"].idxmin()]
    deep = struct.loc[struct["depth"].idxmax()]
    picks = {shallow["name"]: shallow, deep["name"]: deep}
    print(f"Sanity subtrees: shallow={shallow['name']} (depth {shallow['depth']}), "
          f"deep={deep['name']} (depth {deep['depth']})")

    per_dim_times = {d: [] for d in dims}
    rows = []
    for name, srow in picks.items():
        for dim in dims:
            for geom in GEOMETRIES:
                for seed in seeds:
                    cfg = _build_config({"subtree": name, "dim": dim, "geometry": geom, "seed": seed},
                                        base_cfg, srow, data_dir)
                    t0 = time.time()
                    run = train_embedding(name, cfg, base_dir=base_dir, results_dir=results_dir)
                    m = evaluate_run(run.dir)
                    dt = time.time() - t0
                    per_dim_times[dim].append(dt)
                    rows.append({"subtree": name, "depth": int(srow["depth"]), "dim": dim,
                                 "geometry": geom, "seed": seed, "map": m["map"],
                                 "distortion_avg_scaled": m["distortion_avg_scaled"],
                                 "epochs_to_converge": run.metrics["epochs_to_converge"],
                                 "time_s": dt})
    df = pd.DataFrame(rows)

    # low-dim advantage on the shallow tree at dim2
    def mean_map(name, dim, geom):
        s = df[(df.subtree == name) & (df.dim == dim) & (df.geometry == geom)]
        return float(s["map"].mean()) if not s.empty else float("nan")

    print("\n--- Sanity: low-dim Poincaré advantage ---")
    for name in picks:
        for dim in dims:
            p, e = mean_map(name, dim, "poincare"), mean_map(name, dim, "euclidean")
            print(f"  {name:24s} dim{dim:3d}: P MAP={p:.3f}  E MAP={e:.3f}  (P-E={p-e:+.3f})")

    # ETA projection: bucket grid dims to measured dims.
    t2 = np.mean(per_dim_times[2]) if per_dim_times[2] else 0.0
    t8 = np.mean(per_dim_times[8]) if per_dim_times[8] else 0.0
    t128 = np.mean(per_dim_times[128]) if per_dim_times[128] else 0.0
    bucket = {2: t2, 4: t8, 8: t8, 16: t8, 32: t128, 64: t128, 128: t128}
    runs_per_dim = len(SUBTREES) * len(SEEDS) * len(GEOMETRIES)  # 180
    total_s = sum(bucket[d] * runs_per_dim for d in DIMS)
    print("\n--- ETA projection for the full grid ---")
    print(f"  measured mean per-run: dim2={t2:.1f}s  dim8={t8:.1f}s  dim128={t128:.1f}s")
    print(f"  full grid = {len(SUBTREES)} subtrees × {len(DIMS)} dims × {len(SEEDS)} seeds × "
          f"{len(GEOMETRIES)} geoms = {len(build_grid())} runs")
    print(f"  projected wall-clock ≈ {total_s/3600:.1f} h  ({total_s/60:.0f} min)")
    if total_s / 3600 > 30:
        print("  WARNING: projected > 30h — consider trimming seeds/dims before the full run.")
    df.to_csv(Path(results_dir) / "exp_final_sanity.csv", index=False)
    return df, total_s / 3600


# ---------------------------------------------------------------------------
# Task 2 — full grid (implemented; executed in the next stage, not here)
# ---------------------------------------------------------------------------


def run_grid(base_dir="outputs/runs", results_dir="outputs", data_dir="data",
             subtrees=None, dims=DIMS, seeds=SEEDS, geometries=GEOMETRIES, progress_every=20):
    grid = build_grid(subtrees, dims, seeds, geometries)
    struct = load_structure(data_dir).set_index("name")
    base_cfg = load_config("configs/default.yaml")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fail_log = results_dir / "exp_final_failures.log"
    done = _completed_combos(load_results(results_dir / "results.csv"))

    total = len(grid)
    launched = skipped = 0
    failures = []
    start = time.time()
    for i, combo in enumerate(grid, 1):
        key = (combo["geometry"], combo["dim"], combo["subtree"], combo["seed"])
        if key in done:
            skipped += 1
        else:
            try:
                cfg = _build_config(combo, base_cfg, struct.loc[combo["subtree"]], data_dir)
                run = train_embedding(combo["subtree"], cfg, base_dir=base_dir,
                                      results_dir=str(results_dir))
                evaluate_run(run.dir)
                done.add(key)
                launched += 1
            except Exception as exc:  # noqa: BLE001 — isolate one bad run
                msg = f"{key}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                with open(fail_log, "a") as f:
                    f.write(msg + "\n")
        if i % progress_every == 0:
            el = time.time() - start
            rate = (launched + skipped) / max(el, 1e-9)
            eta = (total - i) / max(rate, 1e-9)
            print(f"  [{i}/{total}] launched={launched} skipped={skipped} failed={len(failures)} "
                  f"elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
    summary = {"total": total, "launched": launched, "skipped": skipped, "failed": len(failures),
               "failures": failures, "elapsed_s": time.time() - start}
    print(f"Grid done: {launched} launched, {skipped} skipped, {len(failures)} failed, "
          f"{summary['elapsed_s']/60:.0f} min.")
    return summary


# ---------------------------------------------------------------------------
# Task 3 — aggregate
# ---------------------------------------------------------------------------


def aggregate_final(results_path="outputs/results.csv", out_path="outputs/exp_final_aggregated.csv"):
    df = load_results(results_path)
    if "experiment" in df.columns:
        df = df[df["experiment"] == EXPERIMENT_TAG]
    if df.empty:
        raise ValueError("no exp_final rows to aggregate")
    spec = {}
    for m in METRICS:
        spec[f"{m}_mean"] = (m, "mean")
        spec[f"{m}_std"] = (m, "std")
    spec["alpha_mean"] = ("alpha", "mean")
    spec["epochs_to_converge_mean"] = ("epochs_to_converge", "mean")
    for col in ["subtree_depth", "subtree_avg_branching", "subtree_degree_cv", "root"]:
        spec[col] = (col, "first")
    spec["n_seeds"] = ("map", "count")
    agg = (df.groupby(["subtree", "train.geometry", "train.dim"]).agg(**spec)
           .reset_index().sort_values(["subtree_depth", "subtree", "train.geometry", "train.dim"])
           .reset_index(drop=True))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    return agg


# ---------------------------------------------------------------------------
# Q2 correlation helpers
# ---------------------------------------------------------------------------


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(x) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _adv_table(agg, dim=8):
    """Hyperbolic advantage per subtree at a fixed dim + structural columns."""
    rows = []
    for name in agg["subtree"].unique():
        p = agg[(agg.subtree == name) & (agg["train.geometry"] == "poincare") & (agg["train.dim"] == dim)]
        e = agg[(agg.subtree == name) & (agg["train.geometry"] == "euclidean") & (agg["train.dim"] == dim)]
        if p.empty or e.empty:
            continue
        rows.append({
            "subtree": name, "root": p.iloc[0]["root"], "depth": p.iloc[0]["subtree_depth"],
            "avg_branching": p.iloc[0]["subtree_avg_branching"], "degree_cv": p.iloc[0]["subtree_degree_cv"],
            "adv_map": float(p.iloc[0]["map_mean"] - e.iloc[0]["map_mean"]),
            "adv_dist": float(e.iloc[0]["distortion_avg_scaled_mean"] - p.iloc[0]["distortion_avg_scaled_mean"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures + report (Q1/Q2) — used by the analysis step after the grid
# ---------------------------------------------------------------------------


def make_figures(agg, outdir="outputs/figures/exp_final"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    colors = {"poincare": "tab:blue", "euclidean": "tab:red"}
    dims = sorted(agg["train.dim"].unique())

    # Q1a/Q1b: averaged panels (distortion & MAP vs dim)
    for metric, fname in [("distortion_avg_scaled_mean", "figQ1a_distortion_vs_dim"),
                          ("map_mean", "figQ1b_map_vs_dim")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for geom in GEOMETRIES:
            ys = [agg[(agg["train.geometry"] == geom) & (agg["train.dim"] == d)][metric].mean() for d in dims]
            ax.plot(dims, ys, "-o", color=colors[geom], label=geom)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("dimension"); ax.set_ylabel(metric)
        ax.set_title(metric + " vs dim (mean over subtrees)"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / f"{fname}.png", dpi=150, bbox_inches="tight")
        fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight"); plt.close(fig)

    # Q2a/Q2b: advantage vs structure
    adv = _adv_table(agg, dim=8)
    root_colors = {r: c for r, c in zip(ROOTS, ["tab:green", "tab:orange", "tab:purple"])}
    for xcol, fname in [("depth", "figQ2a_adv_vs_depth"), ("degree_cv", "figQ2b_adv_vs_degcv")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for root in ROOTS:
            s = adv[adv.root == root]
            ax.scatter(s[xcol], s["adv_map"], label=_short(root), color=root_colors[root], s=50)
        if len(adv) >= 2:
            z = np.polyfit(adv[xcol], adv["adv_map"], 1)
            xs = np.linspace(adv[xcol].min(), adv[xcol].max(), 50)
            ax.plot(xs, np.polyval(z, xs), "--", color="gray")
        ax.set_xlabel(xcol); ax.set_ylabel("hyperbolic advantage (P-E MAP) @dim8")
        r = pearson(adv[xcol], adv["adv_map"])
        ax.set_title(f"Q2: advantage vs {xcol} (Pearson r={r:.2f})"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / f"{fname}.png", dpi=150, bbox_inches="tight")
        fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight"); plt.close(fig)


def generate_report(agg, path="outputs/exp_final_report.md"):
    dims = sorted(agg["train.dim"].unique())
    lo, hi = min(dims), max(dims)
    L = ["# Experiment FINAL — Q1 efficiency + Q2 structure law", ""]

    def val(name, geom, dim, col):
        s = agg[(agg.subtree == name) & (agg["train.geometry"] == geom) & (agg["train.dim"] == dim)]
        return float(s.iloc[0][col]) if not s.empty else float("nan")

    # Q1 — dimension savings
    L += ["## Q1 — efficiency", "", "| subtree | depth | P MAP@2 | E dim to match P@2 | P-E MAP@8 |",
          "|---|---|---|---|---|"]
    for name in agg["subtree"].unique():
        depth = agg[agg.subtree == name].iloc[0]["subtree_depth"]
        p2 = val(name, "poincare", lo, "map_mean")
        sav = ">%d" % hi
        for d in dims:
            if val(name, "euclidean", d, "map_mean") >= p2:
                sav = str(d); break
        adv8 = val(name, "poincare", 8, "map_mean") - val(name, "euclidean", 8, "map_mean")
        L.append(f"| {name} | {depth:g} | {p2:.3f} | {sav} | {adv8:+.3f} |")
    # convergence fairness
    pe = agg[agg["train.geometry"] == "poincare"]["epochs_to_converge_mean"].mean()
    ee = agg[agg["train.geometry"] == "euclidean"]["epochs_to_converge_mean"].mean()
    L += ["", f"Convergence fairness: mean epochs_to_converge P={pe:.0f} vs E={ee:.0f}.", ""]

    # Q2 — structure law
    L += ["## Q2 — structure law (advantage @dim8)", ""]
    adv = _adv_table(agg, dim=8)
    L += ["| subtree | root | depth | avg_branching | degree_cv | adv (P-E MAP) |",
          "|---|---|---|---|---|---|"]
    for _, r in adv.sort_values("depth").iterrows():
        L.append(f"| {r['subtree']} | {_short(r['root'])} | {r['depth']:g} | {r['avg_branching']:.2f} "
                 f"| {r['degree_cv']:.2f} | {r['adv_map']:+.3f} |")
    L += [""]
    for xcol in ["depth", "avg_branching", "degree_cv"]:
        pr, sr = pearson(adv[xcol], adv["adv_map"]), spearman(adv[xcol], adv["adv_map"])
        L.append(f"- advantage vs **{xcol}**: Pearson r={pr:.3f}, Spearman ρ={sr:.3f}")
    best = max(["depth", "avg_branching", "degree_cv"], key=lambda c: abs(pearson(adv[c], adv["adv_map"])))
    L += ["", f"Best structural predictor of the advantage: **{best}** "
          f"(|Pearson|={abs(pearson(adv[best], adv['adv_map'])):.3f}).", ""]

    Path(path).write_text("\n".join(L))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Experiment FINAL runner + analysis.")
    p.add_argument("--build-subtrees", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--run", action="store_true", help="Run the full grid (long).")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="outputs")
    args = p.parse_args(argv)

    if args.build_subtrees:
        print("Building subtree suite...")
        struct = build_subtree_suite(data_dir=args.data_dir)
        print("\nStructural spread:")
        for col in ["depth", "avg_branching", "degree_cv", "num_nodes"]:
            print(f"  {col}: [{struct[col].min():.2f}, {struct[col].max():.2f}]")
    if args.sanity:
        struct = load_structure(args.data_dir)
        run_sanity(struct, results_dir=args.results_dir, data_dir=args.data_dir)
    if args.run:
        run_grid(results_dir=args.results_dir, data_dir=args.data_dir)
    if args.analyze:
        agg = aggregate_final(f"{args.results_dir}/results.csv", f"{args.results_dir}/exp_final_aggregated.csv")
        make_figures(agg)
        print(generate_report(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
