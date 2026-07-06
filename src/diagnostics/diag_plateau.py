"""Diagnostic: is the Stage-5 Poincaré high-dim plateau optimization or real?

This is an INVESTIGATION, not a pipeline stage. It must NOT modify
``src/geometry.py``, ``src/trainer.py``, or ``src/metrics.py`` — it only calls
them with different configs and adds a synthetic-tree generator. If a probe
reveals a real trainer bug, we STOP and report with evidence rather than fixing
it here.

Probes:
    1. Synthetic perfect tree (DECISIVE): on a tree we build to be perfectly
       hyperbolic-embeddable, Poincaré MUST win at low dim. If it can't, the
       trainer — not the geometry — is the problem.
    2. Full-pairs vs sampled-pairs on deep_s200 (the plateau coincides with the
       sampled-pairs regime).
    3. Boundary-saturation check (norm distribution of Poincaré points).
    4. Optimization knob sweep (epochs / lr / pairs_per_step) on the worst case.

All runs are tracked and tagged ``experiment="diag_plateau"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import save_subtree
from src.metrics import evaluate_run
from src.tracking import load_results
from src.trainer import train_embedding

DIMS = [2, 8, 32, 128]
SEEDS = [0, 1, 2]
DIAG_TAG = "diag_plateau"
DIAG_DATA_DIR = "outputs/diag_trees"
FIG_DIR = "outputs/figures/diag"


# ---------------------------------------------------------------------------
# Synthetic perfect tree
# ---------------------------------------------------------------------------


def make_synthetic_tree(branching: int, depth: int) -> nx.Graph:
    """Build a perfect ``branching``-ary tree of the given ``depth`` (height).

    Returns an undirected tree with string node ids (matching the WordNet
    convention) and ``graph['root']`` set, so :mod:`src.data` distance/stats
    functions consume it unchanged. Node count = (b^(d+1) - 1) / (b - 1).
    """
    g = nx.balanced_tree(branching, depth)  # root is int 0
    g = nx.relabel_nodes(g, {n: f"n{n}" for n in g.nodes()})
    g.graph["root"] = "n0"
    return g


def _save_tree(tree: nx.Graph, name: str, data_dir: str = DIAG_DATA_DIR) -> None:
    save_subtree(tree, tree.graph["root"], str(Path(data_dir) / "subtrees" / name))


# ---------------------------------------------------------------------------
# Config + train/eval helpers (config-only; no trainer edits)
# ---------------------------------------------------------------------------


def _make_cfg(
    subtree, geometry, dim, seed, data_dir, probe, *, epochs=500, lr=0.01,
    pairs_per_step=512, full_pairs_threshold=150, patience=50, knob="baseline",
):
    cfg = load_config("configs/default.yaml")
    cfg["subtree"] = subtree
    cfg["seed"] = seed
    cfg.setdefault("paths", {})["data_dir"] = data_dir
    cfg["train"].update(
        {
            "geometry": geometry, "dim": dim, "loss": "distortion", "seed": seed,
            "epochs": epochs, "lr": lr, "pairs_per_step": pairs_per_step,
            "full_pairs_threshold": full_pairs_threshold, "early_stop_patience": patience,
            "optimizer": "radam" if geometry == "poincare" else "adam",
        }
    )
    cfg["experiment"] = DIAG_TAG
    cfg["probe"] = probe
    cfg["knob"] = knob
    return cfg


def _train_eval(cfg, subtree, base_dir, results_dir):
    run = train_embedding(subtree, cfg, base_dir=base_dir, results_dir=results_dir)
    metrics = evaluate_run(run.dir)
    return run, metrics


def _pick(metrics: dict) -> dict:
    return {
        "distortion_avg_scaled": metrics["distortion_avg_scaled"],
        "distortion_worst": metrics["distortion_worst"],
        "map": metrics["map"],
    }


# ---------------------------------------------------------------------------
# Probe 1 — synthetic perfect tree (DECISIVE)
# ---------------------------------------------------------------------------


def probe1(base_dir="outputs/runs", results_dir="outputs", data_dir=DIAG_DATA_DIR, seeds=SEEDS):
    trees = [("synth_b2d6", 2, 6), ("synth_b3d4", 3, 4)]
    rows = []
    for name, b, d in trees:
        tree = make_synthetic_tree(b, d)
        _save_tree(tree, name, data_dir)
        n_nodes = tree.number_of_nodes()
        for geom in ("poincare", "euclidean"):
            for dim in DIMS:
                for seed in seeds:
                    cfg = _make_cfg(name, geom, dim, seed, data_dir, probe=1)
                    _run, m = _train_eval(cfg, name, base_dir, results_dir)
                    rows.append({"tree": name, "branching": b, "depth": d, "n_nodes": n_nodes,
                                 "geometry": geom, "dim": dim, "seed": seed, **_pick(m)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "diag_probe1.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Probe 2 — full-pairs vs sampled-pairs on deep_s200
# ---------------------------------------------------------------------------


def probe2(base_dir="outputs/runs", results_dir="outputs", data_dir="data", seeds=SEEDS):
    rows = []
    for mode, thr in [("sampled", 150), ("full", 100000)]:
        for dim in (8, 32, 128):
            for seed in seeds:
                cfg = _make_cfg("deep_s200", "poincare", dim, seed, data_dir, probe=2,
                                full_pairs_threshold=thr, knob=mode)
                _run, m = _train_eval(cfg, "deep_s200", base_dir, results_dir)
                rows.append({"mode": mode, "dim": dim, "seed": seed, **_pick(m)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "diag_probe2.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Probe 3 — boundary saturation
# ---------------------------------------------------------------------------


def _find_stage5_coords(results_dir, runs_root, subtree, geometry, dim, seed):
    df = load_results(Path(results_dir) / "results.csv")
    req = ["run_id", "subtree", "train.geometry", "train.dim", "train.seed"]
    if df.empty or any(c not in df.columns for c in req):
        return None
    sel = df[
        (df["subtree"] == subtree) & (df["train.geometry"] == geometry)
        & (df["train.dim"] == dim) & (df["train.seed"] == seed)
    ]
    for _, row in sel.iterrows():
        p = Path(runs_root) / row["run_id"] / "artifacts" / "coords.npy"
        if p.exists():
            return np.load(p)
    return None


def probe3(base_dir="outputs/runs", results_dir="outputs", data_dir="data", seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = 1e-5  # matches src.geometry.EPS / projection boundary at 1 - eps
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dim in zip(axes, (8, 128)):
        coords = _find_stage5_coords(results_dir, base_dir, "deep_s200", "poincare", dim, seed)
        source = "stage5"
        if coords is None:
            cfg = _make_cfg("deep_s200", "poincare", dim, seed, data_dir, probe=3)
            run, _ = _train_eval(cfg, "deep_s200", base_dir, results_dir)
            coords = np.load(run.dir / "artifacts" / "coords.npy")
            source = "retrained"
        norms = np.linalg.norm(coords, axis=1)
        rows.append({
            "dim": dim, "source": source, "n": int(len(norms)),
            "norm_min": float(norms.min()), "norm_median": float(np.median(norms)),
            "norm_max": float(norms.max()),
            "frac_gt_0.99": float((norms > 0.99).mean()),
            "frac_gt_0.999": float((norms > 0.999).mean()),
            "n_at_boundary": int((norms > 1 - eps).sum()),
        })
        ax.hist(norms, bins=30, range=(0, 1), color="tab:blue", alpha=0.8)
        ax.axvline(0.99, ls="--", color="red", alpha=0.6)
        ax.set_title(f"deep_s200 poincare dim{dim} ‖x‖ ({source})")
        ax.set_xlabel("point norm")
        ax.set_ylabel("count")
    fig.tight_layout()
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(FIG_DIR) / "probe3_norm_hist.png", dpi=150, bbox_inches="tight")
    fig.savefig(Path(FIG_DIR) / "probe3_norm_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "diag_probe3.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Probe 4 — optimization knob sweep
# ---------------------------------------------------------------------------


def probe4(base_dir="outputs/runs", results_dir="outputs", data_dir="data", seeds=SEEDS):
    # NOTE: the "init radius" knob from the spec is NOT config-exposed — the
    # trainer hardcodes uniform(-1e-3, 1e-3) in build_embedding. Changing it
    # would require editing protected code, so it is reported as a limitation
    # rather than run here.
    knob_configs = {
        "baseline": {},
        "epochs3x": {"epochs": 1500, "patience": 150},
        "lr0.005": {"lr": 0.005},
        "lr0.02": {"lr": 0.02},
        "lr0.05": {"lr": 0.05},
        "pairs2048": {"pairs_per_step": 2048},
    }
    rows = []
    for knob, kw in knob_configs.items():
        for seed in seeds:
            cfg = _make_cfg("deep_s200", "poincare", 128, seed, data_dir, probe=4, knob=knob, **kw)
            _run, m = _train_eval(cfg, "deep_s200", base_dir, results_dir)
            rows.append({"knob": knob, "dim": 128, "seed": seed, **_pick(m)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "diag_probe4.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Stage-5 reference lookups (for honest comparison)
# ---------------------------------------------------------------------------


def _stage5_mean(results_dir, subtree, geometry, dim, col):
    df = load_results(Path(results_dir) / "results.csv")
    if "experiment" in df.columns:
        df = df[df["experiment"] == "exp1_dimension_sweep"]
    sel = df[(df["subtree"] == subtree) & (df["train.geometry"] == geometry) & (df["train.dim"] == dim)]
    return float(sel[col].mean()) if not sel.empty else float("nan")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(results_dir="outputs", path="outputs/diag_plateau_report.md") -> str:
    rd = Path(results_dir)
    L = ["# Diagnostic — Poincaré high-dim plateau: optimization or real?", ""]

    def _load(n):
        p = rd / f"diag_probe{n}.csv"
        return pd.read_csv(p) if p.exists() else None

    p1, p2, p3, p4 = _load(1), _load(2), _load(3), _load(4)

    # ---- Probe 1 ----
    L += ["## Probe 1 — synthetic perfect tree (DECISIVE)", ""]
    if p1 is not None:
        L.append("| tree | dim | P dist_scaled | E dist_scaled | P worst | E worst | P MAP | E MAP |")
        L.append("|---|---|---|---|---|---|---|---|")
        p1v = "healthy"
        for tree in sorted(p1["tree"].unique()):
            for dim in DIMS:
                def gv(geom, col):
                    s = p1[(p1.tree == tree) & (p1.geometry == geom) & (p1.dim == dim)]
                    return float(s[col].mean()) if not s.empty else float("nan")
                L.append(
                    f"| {tree} | {dim} | {gv('poincare','distortion_avg_scaled'):.3f} | "
                    f"{gv('euclidean','distortion_avg_scaled'):.3f} | {gv('poincare','distortion_worst'):.2f} | "
                    f"{gv('euclidean','distortion_worst'):.2f} | {gv('poincare','map'):.3f} | {gv('euclidean','map'):.3f} |"
                )
        # verdict: compare at dim2 (low) and dim128 (high), averaged over trees
        def m(geom, dim):
            return float(p1[(p1.geometry == geom) & (p1.dim == dim)]["distortion_avg_scaled"].mean())
        p2d, e2d, p128, e128 = m("poincare", 2), m("euclidean", 2), m("poincare", 128), m("euclidean", 128)
        wins_low = p2d < e2d
        no_plateau = p128 <= e128 + 1e-9
        verdict = (
            "Poincaré wins at low dim AND does not plateau above Euclidean at high dim "
            "→ trainer HEALTHY; the WordNet plateau would be structural (about WordNet)."
            if wins_low and no_plateau else
            "Poincaré also plateaus/loses on the perfect tree → OPTIMIZATION artifact; "
            "Stage-5 geometry comparison is unfair and must be redone after a fix."
        )
        L += ["", f"Mean over trees: dist_scaled P@2={p2d:.3f} E@2={e2d:.3f}; P@128={p128:.3f} E@128={e128:.3f}.",
              "", f"**Verdict:** {verdict}", ""]
    else:
        L += ["_not run_", ""]

    # ---- Probe 2 ----
    L += ["## Probe 2 — full-pairs vs sampled-pairs on deep_s200 (Poincaré)", ""]
    if p2 is not None:
        L.append("| dim | sampled dist_scaled | full dist_scaled | sampled MAP | full MAP |")
        L.append("|---|---|---|---|---|")
        improved = False
        for dim in (8, 32, 128):
            def gv(mode, col):
                s = p2[(p2["mode"] == mode) & (p2.dim == dim)]
                return float(s[col].mean()) if not s.empty else float("nan")
            sd, fd = gv("sampled", "distortion_avg_scaled"), gv("full", "distortion_avg_scaled")
            L.append(f"| {dim} | {sd:.3f} | {fd:.3f} | {gv('sampled','map'):.3f} | {gv('full','map'):.3f} |")
            if fd < sd - 0.02:
                improved = True
        L += ["", f"**Verdict:** {'Full-pairs materially improves Poincaré → sampling is a major cause.' if improved else 'Full-pairs does NOT materially help → sampling is not the main cause.'}", ""]
    else:
        L += ["_not run_", ""]

    # ---- Probe 3 ----
    L += ["## Probe 3 — boundary saturation (Poincaré deep_s200)", ""]
    if p3 is not None:
        L.append("| dim | source | norm_min | norm_median | norm_max | frac>0.99 | frac>0.999 | n_at_boundary |")
        L.append("|---|---|---|---|---|---|---|---|")
        saturated = False
        for _, r in p3.iterrows():
            L.append(
                f"| {int(r['dim'])} | {r['source']} | {r['norm_min']:.4f} | {r['norm_median']:.4f} | "
                f"{r['norm_max']:.4f} | {r['frac_gt_0.99']:.3f} | {r['frac_gt_0.999']:.3f} | {int(r['n_at_boundary'])} |"
            )
            if r["frac_gt_0.99"] > 0.2:
                saturated = True
        L += ["", f"**Verdict:** {'Substantial fraction crushed at the boundary → boundary saturation present (optimization artifact).' if saturated else 'Points are NOT crushed at the boundary → boundary saturation is not the cause.'}", ""]
    else:
        L += ["_not run_", ""]

    # ---- Probe 4 ----
    L += ["## Probe 4 — optimization knob sweep (deep_s200, Poincaré, dim 128)", ""]
    if p4 is not None:
        e_ref = _stage5_mean(results_dir, "deep_s200", "euclidean", 128, "distortion_avg_scaled")
        e_ref_map = _stage5_mean(results_dir, "deep_s200", "euclidean", 128, "map")
        L.append("| knob | mean dist_scaled | best dist_scaled | mean MAP | best MAP |")
        L.append("|---|---|---|---|---|")
        best_dist, best_map = float("inf"), 0.0
        for knob in p4["knob"].unique():
            s = p4[p4["knob"] == knob]
            md, bd = float(s["distortion_avg_scaled"].mean()), float(s["distortion_avg_scaled"].min())
            mm, bm = float(s["map"].mean()), float(s["map"].max())
            best_dist = min(best_dist, bd)
            best_map = max(best_map, bm)
            L.append(f"| {knob} | {md:.3f} | {bd:.3f} | {mm:.3f} | {bm:.3f} |")
        L += ["",
              f"Best Poincaré dim128 across knobs: dist_scaled={best_dist:.3f}, MAP={best_map:.3f}.",
              f"Euclidean@dim128 (Stage-5): dist_scaled={e_ref:.3f}, MAP={e_ref_map:.3f}.",
              f"Limitation: the 'init radius' knob is not config-exposed (trainer hardcodes 1e-3); not run.",
              "", f"**Verdict:** {'Better training closes most of the gap → Stage-5 was an under-training artifact.' if best_dist <= e_ref + 0.03 else 'Even the best knob does NOT close the gap to Euclidean → not simply under-training.'}", ""]
    else:
        L += ["_not run_", ""]

    text = "\n".join(L)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Poincaré plateau diagnostic.")
    p.add_argument("--probe", choices=["1", "2", "3", "4", "all"], default="all")
    p.add_argument("--base-dir", default="outputs/runs")
    p.add_argument("--results-dir", default="outputs")
    args = p.parse_args(argv)

    kw = dict(base_dir=args.base_dir, results_dir=args.results_dir)
    if args.probe in ("1", "all"):
        print("Probe 1 (synthetic perfect tree)...")
        probe1(**kw)
    if args.probe in ("2", "all"):
        print("Probe 2 (full vs sampled pairs)...")
        probe2(**kw)
    if args.probe in ("3", "all"):
        print("Probe 3 (boundary saturation)...")
        probe3(**kw)
    if args.probe in ("4", "all"):
        print("Probe 4 (optimization knobs)...")
        probe4(**kw)

    text = generate_report(results_dir=args.results_dir)
    print("\nReport -> outputs/diag_plateau_report.md\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
