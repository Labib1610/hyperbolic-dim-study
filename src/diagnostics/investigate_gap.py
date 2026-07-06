"""Sandbox investigation: is the real-data Poincaré gap under-training or real structure?

Stage 5.6 recovered Poincaré on a synthetic perfect tree via a learnable scale α,
but on real WordNet ``deep_s200`` a gap to Euclidean remained at high dim. This
investigation decides whether that residual gap is:
    (A) remaining under-training (closes with more epochs / richer α / lr tuning), or
    (B) genuine structure (real WordNet is less tree-like than a perfect tree).

SANDBOX: everything here; reuses the Stage 5.6 sandbox trainer helpers and
``src.metrics`` / ``src.data``. Does NOT edit ``trainer.py`` / ``geometry.py`` /
``metrics.py``. A richer α-parameterization trainer is added here (sandbox freedom).

Structural proxy: pure trees are all exactly 0-hyperbolic (δ=0), so δ on the tree
metric cannot discriminate subtrees. We instead use the **coefficient of variation
of node degree** (degree_cv) as a principled "distance from a clean/uniform tree":
high degree-CV = presence of high-degree gateway hubs = deviation from the uniform
exponential branching that hyperbolic space natively models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import geoopt

from src.data import all_pairs_tree_distances, load_subtree
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.diagnostics.fix_prototype import (
    _device, _evaluate, _seed_all, init_coords, train_euclidean_ref,
)
from src.geometry import PoincareBall
from src.metrics import delta_hyperbolicity, pairwise_distances
from src.tracking import Run

TAG = "investigate_gap"
FIG_DIR = "outputs/figures/gap"
DELTA_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def degree_cv(tree: nx.Graph) -> float:
    """Coefficient of variation of node degree (irregularity / gateway-skew proxy)."""
    deg = np.array([d for _, d in tree.degree()], dtype=float)
    return float(deg.std() / deg.mean()) if deg.mean() > 0 else 0.0


def tree_delta(tree: nx.Graph) -> float:
    """δ-hyperbolicity of the tree-distance metric (== 0 for any pure tree)."""
    D = all_pairs_tree_distances(tree)["matrix"].astype(float)
    return float(delta_hyperbolicity(D, exact_threshold=60, n_samples=20000, seed=0))


def node_levels(tree: nx.Graph, root: str) -> dict:
    """Map each node to its depth (level) from ``root``."""
    return nx.single_source_shortest_path_length(tree, root)


def _level_index(tree: nx.Graph, nodes: list, root: str) -> np.ndarray:
    levels = node_levels(tree, root)
    return np.array([levels[n] for n in nodes], dtype=np.int64)


# ---------------------------------------------------------------------------
# Richer sandbox Poincaré trainer (global / per-level / per-node α)
# ---------------------------------------------------------------------------


def train_poincare_rich(
    tree, dim: int, seed: int, *, init="uniform_ball", alpha_mode="global",
    epochs=500, lr=0.01, lr_schedule="const", pairs_per_step=512,
    full_pairs_threshold=150, val_frac=0.1,
):
    """Poincaré sandbox trainer with a learnable scale (global/per-level/per-node).

    The per-pair scale is the geometric mean ``sqrt(s_u * s_v)`` of node scales
    ``s_i = softplus(raw_i)``, where ``raw`` is one scalar (global), one per depth
    level (per_level), or one per node (per_node). Returns (coords, info) with the
    loss curve, α summary, and train vs held-out scaled distortion (for the
    generalization check).
    """
    _seed_all(seed)
    device = _device()
    geom = PoincareBall()
    manifold = geoopt.PoincareBall()

    apsp = all_pairs_tree_distances(tree)
    nodes = apsp["nodes"]
    N = len(nodes)
    Dt = torch.tensor(apsp["matrix"], dtype=torch.float32, device=device)
    root = tree.graph.get("root", nodes[0])
    level_idx = torch.tensor(_level_index(tree, nodes, root), device=device)
    num_levels = int(level_idx.max().item()) + 1

    X = geoopt.ManifoldParameter(
        init_coords(N, dim, init, manifold).to(device), manifold=manifold
    )
    if alpha_mode == "global":
        raw = nn.Parameter(torch.zeros(1, device=device))
    elif alpha_mode == "per_level":
        raw = nn.Parameter(torch.zeros(num_levels, device=device))
    elif alpha_mode == "per_node":
        raw = nn.Parameter(torch.zeros(N, device=device))
    else:
        raise ValueError(f"unknown alpha_mode {alpha_mode!r}")
    opt = geoopt.optim.RiemannianAdam([X, raw], lr=lr)

    def node_scale():
        s = torch.nn.functional.softplus(raw)
        if alpha_mode == "global":
            return s.expand(N)
        if alpha_mode == "per_level":
            return s[level_idx]
        return s  # per_node

    # train/val split of unordered pairs
    iu = torch.triu_indices(N, N, offset=1)
    all_pairs = iu.t().contiguous()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(all_pairs.shape[0])
    n_val = min(int(round(all_pairs.shape[0] * val_frac)), all_pairs.shape[0] - 1)
    val_pairs = all_pairs[perm[:n_val]].to(device)
    train_pairs = all_pairs[perm[n_val:]].to(device)
    use_full = N <= full_pairs_threshold

    curve = []
    best_loss = float("inf")
    best_coords = X.detach().cpu().numpy().copy()

    for epoch in range(epochs):
        if lr_schedule == "cosine":
            cur = lr * 0.5 * (1 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
            for g in opt.param_groups:
                g["lr"] = cur

        opt.zero_grad()
        if use_full:
            batch = train_pairs
        else:
            k = min(pairs_per_step, train_pairs.shape[0])
            sel = rng.choice(train_pairs.shape[0], size=k, replace=False)
            batch = train_pairs[torch.as_tensor(sel, device=device)]
        u, v = batch[:, 0], batch[:, 1]
        s = node_scale()
        scale_uv = torch.sqrt(s[u] * s[v])
        denom = (scale_uv * Dt[u, v]).clamp_min(1e-6)
        loss = torch.mean((geom.dist(X[u], X[v]) / denom - 1.0) ** 2)
        loss.backward()
        opt.step()

        lv = float(loss.detach())
        curve.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_coords = X.detach().cpu().numpy().copy()

    # α summary + generalization: scaled distortion on train vs held-out pairs.
    s_final = node_scale().detach().cpu().numpy()
    Dg = pairwise_distances(best_coords, geom)
    Dt_np = apsp["matrix"].astype(float)

    def _scaled_dist(pairs_t):
        p = pairs_t.cpu().numpy()
        dg = Dg[p[:, 0], p[:, 1]]
        dt = Dt_np[p[:, 0], p[:, 1]]
        ratio = dg / dt
        scale = ratio.mean()
        return float(np.mean(np.abs(dg - scale * dt) / (scale * dt)))

    info = {
        "curve": curve, "best_loss": best_loss,
        "alpha_mean": float(s_final.mean()), "alpha_std": float(s_final.std()),
        "n_alpha": int(raw.numel()),
        "train_dist": _scaled_dist(train_pairs), "val_dist": _scaled_dist(val_pairs),
    }
    return best_coords, info


# ---------------------------------------------------------------------------
# Run helper (train + evaluate + track)
# ---------------------------------------------------------------------------


def _run_poincare(tree, subtree_name, dim, seed, probe, base_dir, results_dir, **cfg):
    coords, info = train_poincare_rich(tree, dim, seed, **cfg)
    m = _evaluate(coords, tree, "poincare")
    rec = {
        "subtree": subtree_name, "dim": dim, "seed": seed, "geometry": "poincare",
        "alpha_mode": cfg.get("alpha_mode", "global"), "epochs": cfg.get("epochs", 500),
        "lr": cfg.get("lr", 0.01), "lr_schedule": cfg.get("lr_schedule", "const"),
        "distortion_avg_scaled": m["distortion_avg_scaled"],
        "distortion_worst": m["distortion_worst"], "map": m["map"],
        "alpha_mean": info["alpha_mean"], "alpha_std": info["alpha_std"],
        "n_alpha": info["n_alpha"], "train_dist": info["train_dist"], "val_dist": info["val_dist"],
    }
    run = Run({"experiment": TAG, "probe": probe, "subtree": subtree_name,
               "geometry": "poincare", "dim": dim, "seed": seed, **cfg},
              name=f"{probe}_{subtree_name}_{cfg.get('alpha_mode','global')}_d{dim}_s{seed}",
              base_dir=base_dir, results_dir=results_dir)
    run.log_curve("train_loss", info["curve"])
    run.save_array("coords", coords)
    run.log_metrics(m)
    run.finish()
    return rec


def _run_euclidean(tree, subtree_name, dim, seed):
    coords, _info = train_euclidean_ref(tree, dim, seed)
    m = _evaluate(coords, tree, "euclidean")
    return {"subtree": subtree_name, "dim": dim, "seed": seed, "geometry": "euclidean",
            "alpha_mode": "-", "distortion_avg_scaled": m["distortion_avg_scaled"],
            "distortion_worst": m["distortion_worst"], "map": m["map"]}


# ---------------------------------------------------------------------------
# Probe A — under-training probes on deep_s200
# ---------------------------------------------------------------------------


def probe_a(base_dir="outputs/runs", results_dir="outputs", data_dir="data", seeds=(0, 1, 2)):
    tree, _ = load_subtree(f"{data_dir}/subtrees/deep_s200")
    rows = []
    kw = dict(probe="A", base_dir=base_dir, results_dir=results_dir)

    # A1 — more epochs (global alpha), dims 32 & 128
    for dim in (32, 128):
        for epochs in (500, 2500, 5000):
            for seed in seeds:
                rows.append(_run_poincare(tree, "deep_s200", dim, seed,
                                          alpha_mode="global", epochs=epochs, **kw) | {"knob": f"epochs{epochs}"})
    # A2 — richer alpha (per_level / per_node) at base epochs
    for dim in (32, 128):
        for mode in ("per_level", "per_node"):
            for seed in seeds:
                rows.append(_run_poincare(tree, "deep_s200", dim, seed,
                                          alpha_mode=mode, epochs=500, **kw) | {"knob": f"alpha_{mode}"})
    # A3 — lr sweep + cosine at dim128 (global)
    base_lr = 0.01
    for mult in (0.5, 2.0, 5.0):
        for seed in seeds:
            rows.append(_run_poincare(tree, "deep_s200", 128, seed, alpha_mode="global",
                                      epochs=500, lr=base_lr * mult, **kw) | {"knob": f"lr{mult}x"})
    for seed in seeds:
        rows.append(_run_poincare(tree, "deep_s200", 128, seed, alpha_mode="global",
                                  epochs=500, lr_schedule="cosine", **kw) | {"knob": "cosine"})

    # Euclidean reference at dim 32 & 128
    for dim in (32, 128):
        for seed in seeds:
            rows.append(_run_euclidean(tree, "deep_s200", dim, seed) | {"knob": "euclidean", "probe": "A"})

    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "gap_probeA.csv", index=False)
    return df


def _best_a_config(dfA):
    """Best-effort Poincaré config at dim128 from Probe A (max mean MAP)."""
    p = dfA[(dfA.geometry == "poincare") & (dfA.dim == 128)]
    g = p.groupby("knob").agg(map_mean=("map", "mean"),
                              dist_mean=("distortion_avg_scaled", "mean")).reset_index()
    g = g.sort_values(["map_mean", "dist_mean"], ascending=[False, True])
    return g.iloc[0]["knob"], float(g.iloc[0]["map_mean"]), float(g.iloc[0]["dist_mean"])


# ---------------------------------------------------------------------------
# Probe B — real-structure probes
# ---------------------------------------------------------------------------

B_SUBTREES = ["bfs_s50", "deep_s100", "deep_s200", "balanced_s200"]


def probe_b(best_knob, base_dir="outputs/runs", results_dir="outputs", data_dir="data", seeds=(0, 1, 2)):
    # map best_knob -> config kwargs
    def knob_cfg(knob):
        if knob.startswith("epochs"):
            return dict(alpha_mode="global", epochs=int(knob[6:]))
        if knob.startswith("alpha_"):
            return dict(alpha_mode=knob[6:], epochs=500)
        if knob.startswith("lr") and knob.endswith("x"):
            return dict(alpha_mode="global", epochs=500, lr=0.01 * float(knob[2:-1]))
        if knob == "cosine":
            return dict(alpha_mode="global", epochs=500, lr_schedule="cosine")
        return dict(alpha_mode="global", epochs=500)

    cfg = knob_cfg(best_knob)
    rows = []
    struct = []
    for name in B_SUBTREES:
        tree, _ = load_subtree(f"{data_dir}/subtrees/{name}")
        struct.append({"subtree": name, "degree_cv": degree_cv(tree), "tree_delta": tree_delta(tree),
                       "depth": all_pairs_tree_distances(tree)["matrix"].max()})
        for seed in seeds:
            rows.append(_run_poincare(tree, name, 128, seed, probe="B",
                                      base_dir=base_dir, results_dir=results_dir, **cfg))
            rows.append(_run_euclidean(tree, name, 128, seed) | {"probe": "B"})
    dfB = pd.DataFrame(rows)
    dfS = pd.DataFrame(struct)
    dfB.to_csv(Path(results_dir) / "gap_probeB.csv", index=False)
    dfS.to_csv(Path(results_dir) / "gap_structure.csv", index=False)
    return dfB, dfS


def _gap_vs_structure(dfB, dfS):
    out = []
    for _, s in dfS.iterrows():
        name = s["subtree"]
        p = dfB[(dfB.subtree == name) & (dfB.geometry == "poincare")]
        e = dfB[(dfB.subtree == name) & (dfB.geometry == "euclidean")]
        out.append({
            "subtree": name, "degree_cv": s["degree_cv"], "depth": s["depth"],
            "P_map": float(p["map"].mean()), "E_map": float(e["map"].mean()),
            "map_gap": float(e["map"].mean() - p["map"].mean()),
            "P_dist": float(p["distortion_avg_scaled"].mean()),
            "E_dist": float(e["distortion_avg_scaled"].mean()),
            "dist_gap": float(p["distortion_avg_scaled"].mean() - e["distortion_avg_scaled"].mean()),
        })
    return pd.DataFrame(out).sort_values("degree_cv").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(dfA, gap_struct, dfA_alpha, outdir=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # (i) Poincaré dim128 MAP under Probe-A knobs vs Euclidean
    p = dfA[(dfA.geometry == "poincare") & (dfA.dim == 128)]
    g = p.groupby("knob")["map"].mean().sort_values()
    e_map = dfA[(dfA.geometry == "euclidean") & (dfA.dim == 128)]["map"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(g.index, g.values, color="tab:blue")
    ax.axvline(e_map, ls="--", color="tab:red", label=f"Euclidean={e_map:.2f}")
    ax.set_xlabel("MAP @dim128 (deep_s200)")
    ax.set_title("Probe A — Poincaré MAP under knobs vs Euclidean")
    ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "probeA_map.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "probeA_map.pdf", bbox_inches="tight"); plt.close(fig)

    # (ii) gap vs structure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(gap_struct["degree_cv"], gap_struct["map_gap"], s=60)
    for _, r in gap_struct.iterrows():
        ax.annotate(r["subtree"], (r["degree_cv"], r["map_gap"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("degree_cv (irregularity)"); ax.set_ylabel("MAP gap (E - P) @dim128")
    ax.set_title("Probe B — gap vs structure"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "probeB_gap_vs_structure.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "probeB_gap_vs_structure.pdf", bbox_inches="tight"); plt.close(fig)

    # (iii) alpha across trees/seeds (global-alpha runs)
    fig, ax = plt.subplots(figsize=(7, 4))
    a = dfA_alpha
    if not a.empty:
        for i, name in enumerate(sorted(a["subtree"].unique())):
            s = a[a.subtree == name]
            ax.scatter([i] * len(s), s["alpha_mean"], s=40, alpha=0.7)
        ax.set_xticks(range(len(a["subtree"].unique())))
        ax.set_xticklabels(sorted(a["subtree"].unique()), rotation=30, fontsize=8)
    ax.set_ylabel("learned alpha (global, mean)")
    ax.set_title("Probe C — alpha across trees/seeds")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "probeC_alpha.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "probeC_alpha.pdf", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(dfA, dfB, dfS, gap_struct, best_knob, path="outputs/investigate_gap_report.md"):
    L = ["# Investigation — real-data Poincaré gap: under-training or real structure?", ""]

    # Probe A
    e128 = dfA[(dfA.geometry == "euclidean") & (dfA.dim == 128)]
    e_map, e_dist = float(e128["map"].mean()), float(e128["distortion_avg_scaled"].mean())
    p128 = dfA[(dfA.geometry == "poincare") & (dfA.dim == 128)]
    base = p128[p128.knob == "epochs500"]
    base_map, base_dist = float(base["map"].mean()), float(base["distortion_avg_scaled"].mean())
    knob, best_map, best_dist = _best_a_config(dfA)

    L += ["## Probe A — under-training (deep_s200, dim128)", "",
          "| knob | dist_scaled | MAP |", "|---|---|---|"]
    for k in sorted(p128["knob"].unique()):
        s = p128[p128.knob == k]
        L.append(f"| {k} | {s['distortion_avg_scaled'].mean():.3f} | {s['map'].mean():.3f} |")
    L.append(f"| euclidean | {e_dist:.3f} | {e_map:.3f} |")
    if best_map >= e_map - 0.05:
        verdict_a = "CLOSES"
    elif best_map >= base_map + 0.05:
        verdict_a = "NARROWS"
    else:
        verdict_a = "DOES NOT MOVE"
    L += ["",
          f"Baseline (epochs500 global α): dist={base_dist:.3f} MAP={base_map:.3f}.",
          f"Best Probe-A knob = **{knob}**: dist={best_dist:.3f} MAP={best_map:.3f}.",
          f"Euclidean@dim128: dist={e_dist:.3f} MAP={e_map:.3f}.",
          f"**Verdict A:** extra effort **{verdict_a}** the gap.", ""]

    # A2 richer alpha
    L += ["### Probe A2 — richer α (global vs per-level vs per-node), dim128", "",
          "| α mode | dist_scaled | MAP | n_α | train_dist | val_dist |", "|---|---|---|---|---|---|"]
    for mode in ("global", "per_level", "per_node"):
        s = p128[p128.alpha_mode == mode]
        if not s.empty:
            L.append(f"| {mode} | {s['distortion_avg_scaled'].mean():.3f} | {s['map'].mean():.3f} | "
                     f"{int(s['n_alpha'].mean())} | {s['train_dist'].mean():.3f} | {s['val_dist'].mean():.3f} |")
    L.append("")

    # Probe B
    L += ["## Probe B — gap vs structure (dim128, best-effort α)", "",
          f"Structural scalar: **degree_cv** (coefficient of variation of node degree). "
          f"All tree metrics have δ=0 (pure trees are 0-hyperbolic), so δ cannot discriminate — "
          f"degree_cv captures gateway-driven irregularity instead.", "",
          "| subtree | degree_cv | depth | P MAP | E MAP | MAP gap (E-P) | dist gap (P-E) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in gap_struct.iterrows():
        L.append(f"| {r['subtree']} | {r['degree_cv']:.2f} | {int(r['depth'])} | {r['P_map']:.3f} | "
                 f"{r['E_map']:.3f} | {r['map_gap']:+.3f} | {r['dist_gap']:+.3f} |")
    if len(gap_struct) >= 3:
        corr = float(np.corrcoef(gap_struct["degree_cv"], gap_struct["map_gap"])[0, 1])
    else:
        corr = float("nan")
    L += ["", f"Pearson corr(degree_cv, MAP gap) = **{corr:.3f}**.",
          f"**Verdict B:** {'gap tracks irregularity → evidence for real structure.' if corr > 0.5 else ('gap does not clearly track irregularity.' if not np.isnan(corr) else 'insufficient points.')}", ""]

    # Probe C
    L += ["## Probe C — learnable-scale stability (global α)", ""]
    glob = pd.concat([dfA[(dfA.alpha_mode == "global")], dfB[dfB.get("alpha_mode") == "global"]]) if "alpha_mode" in dfB.columns else dfA[dfA.alpha_mode == "global"]
    L += ["| subtree | dim | α mean | α std (across seeds) |", "|---|---|---|---|"]
    for (name, dim), s in glob[glob.geometry == "poincare"].groupby(["subtree", "dim"]):
        L.append(f"| {name} | {int(dim)} | {s['alpha_mean'].mean():.3f} | {s['alpha_mean'].std():.4f} |")
    # generalization from A2
    gen = []
    for mode in ("global", "per_level", "per_node"):
        s = p128[p128.alpha_mode == mode]
        if not s.empty:
            gen.append(f"{mode}: train={s['train_dist'].mean():.3f} val={s['val_dist'].mean():.3f} "
                       f"(gap {s['val_dist'].mean()-s['train_dist'].mean():+.3f})")
    L += ["", "Generalization (train vs held-out scaled distortion, dim128): " + "; ".join(gen), ""]

    # Overall
    L += ["## Overall verdict (A vs B)", "",
          f"- Probe A: extra training **{verdict_a}** the deep_s200 gap "
          f"(best MAP {best_map:.3f} vs Euclidean {e_map:.3f}).",
          f"- Probe B: corr(irregularity, gap) = {corr:.3f}.",
          "- Interpretation is for us to decide together; numbers above are the evidence.", ""]

    text = "\n".join(L)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Investigate the real-data Poincaré gap.")
    p.add_argument("--probe", choices=["A", "B", "C", "all"], default="all")
    p.add_argument("--base-dir", default="outputs/runs")
    p.add_argument("--results-dir", default="outputs")
    args = p.parse_args(argv)
    kw = dict(base_dir=args.base_dir, results_dir=args.results_dir)

    print("Probe A (under-training)...")
    dfA = probe_a(**kw)
    best_knob, _, _ = _best_a_config(dfA)
    print(f"Best Probe-A knob: {best_knob}")

    print("Probe B (structure)...")
    dfB, dfS = probe_b(best_knob, **kw)
    gap_struct = _gap_vs_structure(dfB, dfS)

    # global-alpha rows for the Probe-C alpha figure (from A + B)
    alpha_rows = pd.concat([
        dfA[(dfA.geometry == "poincare") & (dfA.alpha_mode == "global")][["subtree", "dim", "seed", "alpha_mean"]],
        dfB[(dfB.geometry == "poincare") & (dfB.get("alpha_mode") == "global")][["subtree", "dim", "seed", "alpha_mean"]]
        if "alpha_mode" in dfB.columns else pd.DataFrame(),
    ], ignore_index=True)

    make_figures(dfA, gap_struct, alpha_rows)
    text = generate_report(dfA, dfB, dfS, gap_struct, best_knob)
    print("\nReport -> outputs/investigate_gap_report.md\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
