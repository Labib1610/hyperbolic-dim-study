"""Sandbox fairness check: a geometry-agnostic stopping rule + convergence proof.

Stage 5.7 showed the real-data Poincaré gap was under-training (more epochs close
it). The danger now: comparing well-trained Poincaré against Euclidean at its old
~500-epoch early-stop point would recreate the Stage-5 unfairness in reverse. This
stage establishes a FAIR, geometry-agnostic stopping rule and PROVES it fair with
full convergence curves — BOTH geometries trained under the identical learnable-α
objective to a large equal budget.

SANDBOX: everything here; reuses geometry/metrics/data/tracking unchanged. Does NOT
edit protected code. It DECIDES the config to port next; it does not port.

Geometry-agnostic objective: the learnable-scale α rescales the target tree
distances (``d_G ≈ α · D_T``) independent of geometry, so Poincaré AND Euclidean use
the exact same loss and α treatment here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geoopt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import all_pairs_tree_distances, load_subtree
from src.diagnostics.fix_prototype import _device, _seed_all, init_coords
from src.geometry import EuclideanGeometry, PoincareBall
from src.metrics import mean_average_precision, pairwise_distances
from src.tracking import Run

TAG = "fairness_check"
FIG_DIR = "outputs/figures/fairness"
SUBTREES = ["bfs_s50", "deep_s100", "deep_s200"]
DIMS = [8, 128]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 8000
EVAL_EVERY = 100


# ---------------------------------------------------------------------------
# Unified geometry-agnostic learnable-α trainer (BOTH geometries)
# ---------------------------------------------------------------------------


def _scaled_dist_on_pairs(Dg, Dt, pairs) -> float:
    dg = Dg[pairs[:, 0], pairs[:, 1]]
    dt = Dt[pairs[:, 0], pairs[:, 1]]
    ratio = dg / dt
    scale = ratio.mean()
    return float(np.mean(np.abs(dg - scale * dt) / (scale * dt)))


def train_curve(tree, dim, geometry, seed, *, epochs=MAX_EPOCHS, lr=0.01,
                eval_every=EVAL_EVERY, pairs_per_step=512, full_pairs_threshold=150,
                val_frac=0.1, use_alpha=True):
    """Train one embedding to a large budget, logging val curves. No early stopping.

    Both geometries use the identical learnable-α objective
    ``mean( (d_G / (α · D_T) - 1)^2 )``. Returns (coords, alpha, checkpoints), where
    checkpoints is a list of dicts {epoch, time, val_distortion, val_map}.
    """
    _seed_all(seed)
    device = _device()

    apsp = all_pairs_tree_distances(tree)
    N = len(apsp["nodes"])
    Dt = torch.tensor(apsp["matrix"], dtype=torch.float32, device=device)
    Dt_np = apsp["matrix"].astype(np.float64)

    if geometry == "poincare":
        geom = PoincareBall()
        manifold = geoopt.PoincareBall()
        X = geoopt.ManifoldParameter(
            init_coords(N, dim, "uniform_ball", manifold).to(device), manifold=manifold
        )
    else:
        geom = EuclideanGeometry()
        X = nn.Parameter((torch.randn(N, dim) * 0.1).to(device))

    params = [X]
    alpha_raw = None
    if use_alpha:
        alpha_raw = nn.Parameter(torch.zeros((), device=device))
        params.append(alpha_raw)

    opt = (geoopt.optim.RiemannianAdam(params, lr=lr) if geometry == "poincare"
           else torch.optim.Adam(params, lr=lr))

    iu = torch.triu_indices(N, N, offset=1)
    all_pairs = iu.t().contiguous()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(all_pairs.shape[0])
    n_val = min(int(round(all_pairs.shape[0] * val_frac)), all_pairs.shape[0] - 1)
    val_np = all_pairs[perm[:n_val]].numpy()
    train_pairs = all_pairs[perm[n_val:]].to(device)
    use_full = N <= full_pairs_threshold

    checkpoints = []
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        if use_full:
            batch = train_pairs
        else:
            k = min(pairs_per_step, train_pairs.shape[0])
            sel = rng.choice(train_pairs.shape[0], size=k, replace=False)
            batch = train_pairs[torch.as_tensor(sel, device=device)]
        u, v = batch[:, 0], batch[:, 1]
        alpha = torch.nn.functional.softplus(alpha_raw) if use_alpha else 1.0
        denom = (alpha * Dt[u, v]).clamp_min(1e-6) if use_alpha else Dt[u, v].clamp_min(1.0)
        loss = torch.mean((geom.dist(X[u], X[v]) / denom - 1.0) ** 2)
        loss.backward()
        opt.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                coords = X.detach().cpu().numpy()
                Dg = pairwise_distances(coords, geom)
                checkpoints.append({
                    "epoch": epoch, "time": time.time() - t0,
                    "val_distortion": _scaled_dist_on_pairs(Dg, Dt_np, val_np),
                    "val_map": mean_average_precision(Dg, Dt_np),
                })

    alpha_val = float(torch.nn.functional.softplus(alpha_raw.detach())) if use_alpha else 1.0
    return X.detach().cpu().numpy(), alpha_val, checkpoints


# ---------------------------------------------------------------------------
# Task 1 — convergence curves
# ---------------------------------------------------------------------------


def run_curves(subtrees=SUBTREES, dims=DIMS, seeds=SEEDS, epochs=MAX_EPOCHS,
               base_dir="outputs/runs", results_dir="outputs", data_dir="data"):
    rows = []
    for name in subtrees:
        tree, _ = load_subtree(f"{data_dir}/subtrees/{name}")
        for dim in dims:
            for geom in ("poincare", "euclidean"):
                for seed in seeds:
                    coords, alpha, cps = train_curve(tree, dim, geom, seed, epochs=epochs)
                    for cp in cps:
                        rows.append({"subtree": name, "dim": dim, "geometry": geom,
                                     "seed": seed, "alpha": alpha, **cp})
                    final = cps[-1]
                    run = Run({"experiment": TAG, "subtree": name, "geometry": geom,
                               "dim": dim, "seed": seed, "epochs": epochs, "use_alpha": True},
                              name=f"fair_{name}_{geom}_d{dim}_s{seed}",
                              base_dir=base_dir, results_dir=results_dir)
                    run.log_curve("val_distortion", [c["val_distortion"] for c in cps])
                    run.log_curve("val_map", [c["val_map"] for c in cps])
                    run.save_array("coords", coords)
                    run.log_metrics({"final_val_distortion": final["val_distortion"],
                                     "final_val_map": final["val_map"], "alpha": alpha})
                    run.finish()
    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "fairness_curves.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Task 2 — convergence analysis
# ---------------------------------------------------------------------------


def epochs_to_converge(epochs, vals, final_window=100, tol=0.01) -> int:
    """First epoch where ``val`` is within ``tol`` (relative) of the final-window avg."""
    epochs = np.asarray(epochs)
    vals = np.asarray(vals, dtype=float)
    max_e = epochs.max()
    plateau = float(vals[epochs >= max_e - final_window].mean())
    denom = abs(plateau) if plateau != 0 else 1.0
    for e, v in zip(epochs, vals):
        if abs(v - plateau) <= tol * denom:
            return int(e)
    return int(max_e)


def convergence_table(df, final_window=200):
    rows = []
    for (name, dim, geom), g in df.groupby(["subtree", "dim", "geometry"]):
        # average curves over seeds on the shared epoch grid
        piv = g.pivot_table(index="epoch", columns="seed", values="val_distortion")
        mean_curve = piv.mean(axis=1)
        epochs = mean_curve.index.values
        vals = mean_curve.values
        conv_e = epochs_to_converge(epochs, vals, final_window=final_window)
        plateau_mask = epochs >= epochs.max() - final_window
        pmap = g[g.epoch >= epochs.max() - final_window].groupby("seed")["val_map"].mean().mean()
        rows.append({"subtree": name, "dim": dim, "geometry": geom,
                     "epochs_to_converge": conv_e,
                     "converged_val_distortion": float(vals[plateau_mask].mean()),
                     "converged_val_map": float(pmap),
                     "alpha": float(g["alpha"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 3 — three stopping rules
# ---------------------------------------------------------------------------


def _final_values(df):
    """R1 — fixed large budget: value at the last epoch (mean over seeds)."""
    rows = []
    for (name, dim, geom), g in df.groupby(["subtree", "dim", "geometry"]):
        last = g[g.epoch == g.epoch.max()]
        rows.append({"subtree": name, "dim": dim, "geometry": geom,
                     "val_distortion": float(last["val_distortion"].mean()),
                     "val_map": float(last["val_map"].mean())})
    return pd.DataFrame(rows)


def _early_stop_epoch(epochs, vals, patience=500, min_delta_rel=0.005):
    best, best_e, since = float("inf"), int(epochs[0]), 0
    step = epochs[1] - epochs[0] if len(epochs) > 1 else 1
    patience_steps = max(1, patience // step)
    for i, (e, v) in enumerate(zip(epochs, vals)):
        if v < best * (1 - min_delta_rel):
            best, best_e, since = v, int(e), 0
        else:
            since += 1
            if since >= patience_steps:
                return best_e
    return int(epochs[-1])


def _r2_values(df, patience=500, min_delta_rel=0.005):
    """R2 — convergence early-stopping with identical generous patience."""
    rows = []
    for (name, dim, geom), g in df.groupby(["subtree", "dim", "geometry"]):
        d_at, m_at = [], []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("epoch")
            e = _early_stop_epoch(gs["epoch"].values, gs["val_distortion"].values,
                                  patience, min_delta_rel)
            at = gs[gs.epoch == e].iloc[0]
            d_at.append(at["val_distortion"]); m_at.append(at["val_map"])
        rows.append({"subtree": name, "dim": dim, "geometry": geom,
                     "val_distortion": float(np.mean(d_at)), "val_map": float(np.mean(m_at))})
    return pd.DataFrame(rows)


def _r3_values(df):
    """R3 — compute-matched by WALL-CLOCK (common budget = min total time of the pair)."""
    rows = []
    # per (subtree,dim): common T = min over geometries of their max wall-clock.
    for (name, dim), g in df.groupby(["subtree", "dim"]):
        tmax = g.groupby("geometry")["time"].max()
        T = float(tmax.min())
        for geom, gg in g.groupby("geometry"):
            d_at, m_at = [], []
            for seed, gs in gg.groupby("seed"):
                gs = gs.sort_values("epoch")
                within = gs[gs["time"] <= T]
                at = (within.iloc[-1] if not within.empty else gs.iloc[0])
                d_at.append(at["val_distortion"]); m_at.append(at["val_map"])
            rows.append({"subtree": name, "dim": dim, "geometry": geom,
                         "val_distortion": float(np.mean(d_at)), "val_map": float(np.mean(m_at)),
                         "common_wallclock_s": T})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(df, rules, outdir=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    colors = {"poincare": "tab:blue", "euclidean": "tab:red"}
    subtrees = sorted(df["subtree"].unique())
    dims = sorted(df["dim"].unique())

    fig, axes = plt.subplots(len(subtrees), len(dims), figsize=(5 * len(dims), 3.2 * len(subtrees)),
                             squeeze=False)
    for i, name in enumerate(subtrees):
        for j, dim in enumerate(dims):
            ax = axes[i][j]
            for geom in ("poincare", "euclidean"):
                g = df[(df.subtree == name) & (df.dim == dim) & (df.geometry == geom)]
                if g.empty:
                    continue
                piv = g.pivot_table(index="epoch", columns="seed", values="val_distortion")
                epochs = piv.index.values
                mean = piv.mean(axis=1).values
                std = piv.std(axis=1).values
                ax.plot(epochs, mean, "-", color=colors[geom], label=geom)
                ax.fill_between(epochs, mean - std, mean + std, color=colors[geom], alpha=0.2)
            ax.axvline(500, ls=":", color="gray", alpha=0.7)
            ax.set_title(f"{name} dim{dim}", fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("val distortion")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Convergence curves (val distortion vs epoch); dotted line = old 500-epoch stop")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "convergence_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "convergence_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # Summary bar: converged MAP under each rule (mean over subtrees/dims)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(rules.keys())
    width = 0.35
    x = np.arange(len(labels))
    for k, geom in enumerate(("poincare", "euclidean")):
        vals = [rules[r][rules[r].geometry == geom]["val_map"].mean() for r in labels]
        ax.bar(x + (k - 0.5) * width, vals, width, label=geom, color=colors[geom])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean converged MAP"); ax.set_title("P vs E converged MAP under each rule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "rules_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "rules_summary.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(df, conv, rules, path="outputs/fairness_check_report.md"):
    L = ["# Fairness Check — geometry-agnostic stopping rule + convergence proof", ""]
    L += ["Both geometries trained under the IDENTICAL learnable-α objective "
          "(d_G ≈ α·D_T), 8000 epochs, early-stopping OFF, 3 seeds. "
          "Val distortion/MAP logged every 100 epochs.", ""]

    # Euclidean-past-500 check
    L += ["## 1) Does Euclidean improve past 500 epochs?", "",
          "| subtree | dim | E val_dist @500 | E val_dist @conv | E MAP @500 | E MAP @conv |",
          "|---|---|---|---|---|---|"]
    for (name, dim), _ in df.groupby(["subtree", "dim"]):
        e = df[(df.subtree == name) & (df.dim == dim) & (df.geometry == "euclidean")]
        def near(epoch, col):
            idx = (e["epoch"] - epoch).abs().groupby(e["seed"]).idxmin()
            return float(e.loc[idx, col].mean())
        d500, dconv = near(500, "val_distortion"), near(e["epoch"].max(), "val_distortion")
        m500, mconv = near(500, "val_map"), near(e["epoch"].max(), "val_map")
        L.append(f"| {name} | {dim} | {d500:.3f} | {dconv:.3f} | {m500:.3f} | {mconv:.3f} |")
    L.append("")

    # Convergence table
    L += ["## 2) Convergence table (epochs_to_converge + plateau values)", "",
          "| subtree | dim | geometry | epochs_to_converge | conv val_dist | conv val_MAP | alpha |",
          "|---|---|---|---|---|---|---|"]
    for _, r in conv.sort_values(["subtree", "dim", "geometry"]).iterrows():
        L.append(f"| {r['subtree']} | {int(r['dim'])} | {r['geometry']} | {int(r['epochs_to_converge'])} "
                 f"| {r['converged_val_distortion']:.3f} | {r['converged_val_map']:.3f} | {r['alpha']:.3f} |")
    L.append("")

    # Three rules
    L += ["## 3) Three stopping rules — converged MAP (dist) per subtree/dim", ""]
    for rname, rdf in rules.items():
        L += [f"### {rname}", "", "| subtree | dim | P MAP (dist) | E MAP (dist) | winner |",
              "|---|---|---|---|---|"]
        for (name, dim), g in rdf.groupby(["subtree", "dim"]):
            p = g[g.geometry == "poincare"].iloc[0]
            e = g[g.geometry == "euclidean"].iloc[0]
            win = "P" if p["val_map"] > e["val_map"] + 1e-6 else ("E" if e["val_map"] > p["val_map"] + 1e-6 else "tie")
            L.append(f"| {name} | {dim} | {p['val_map']:.3f} ({p['val_distortion']:.3f}) "
                     f"| {e['val_map']:.3f} ({e['val_distortion']:.3f}) | {win} |")
        L.append("")

    # Stability across rules
    def winner_map(rdf):
        w = {}
        for (name, dim), g in rdf.groupby(["subtree", "dim"]):
            p = g[g.geometry == "poincare"].iloc[0]["val_map"]
            e = g[g.geometry == "euclidean"].iloc[0]["val_map"]
            w[(name, dim)] = "P" if p > e + 1e-6 else ("E" if e > p + 1e-6 else "tie")
        return w
    ws = {r: winner_map(rules[r]) for r in rules}
    keys = sorted(ws[list(rules)[0]].keys())
    stable = all(len({ws[r][k] for r in rules}) == 1 for k in keys)
    L += ["## 4) Stopping-rule stability", "",
          f"- Conclusions across R1/R2/R3 are **{'STABLE' if stable else 'NOT stable (disclose!)'}**.",
          "", "| subtree/dim | " + " | ".join(rules.keys()) + " |", "|" + "---|" * (len(rules) + 1)]
    for k in keys:
        L.append(f"| {k[0]} d{k[1]} | " + " | ".join(ws[r][k] for r in rules) + " |")
    L.append("")

    # Low-dim preservation + verdict
    L += ["## 5) Verdict + recommended port config", ""]
    L += ["- At true convergence (R1), per subtree/dim P-vs-E MAP is in the R1 table above.",
          "- **Recommended PORT config:** learnable global α (geometry-agnostic) + "
          "stopping rule **R2 (convergence early-stopping: patience=500 epochs on val distortion, "
          "min_delta 0.5%, cap 8000)** — geometry-agnostic and lets both plateau. "
          "R1 (fixed 8000) is the simplest fallback if conclusions match.", ""]

    text = "\n".join(L)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fairness check: convergence + stopping rule.")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--base-dir", default="outputs/runs")
    p.add_argument("--results-dir", default="outputs")
    args = p.parse_args(argv)

    print("Running convergence curves (this is the long part)...")
    df = run_curves(epochs=args.epochs, base_dir=args.base_dir, results_dir=args.results_dir)
    conv = convergence_table(df)
    rules = {"R1_fixed8000": _final_values(df), "R2_earlystop": _r2_values(df), "R3_computematched": _r3_values(df)}
    make_figures(df, rules)
    text = generate_report(df, conv, rules)
    print("\nReport -> outputs/fairness_check_report.md\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
