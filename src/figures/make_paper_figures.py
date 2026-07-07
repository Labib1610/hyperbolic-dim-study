"""Regenerate all paper figures (publication quality) from saved aggregated CSVs.

Pure re-plotting — no retraining, no new experiments. Reads the Stage-7 (Q1/Q2) and
Stage-8/8.5 (Q3) aggregated CSVs and writes clean PDF+PNG figures into
``outputs/figures/paper/`` with consistent, colorblind-safe styling and no code-y
titles/labels. Touches no trainer/geometry/metrics/experiment code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/figures/paper")
RESULTS = Path("outputs")

# Colorblind-safe palette (Okabe–Ito).
C_POINCARE = "#0072B2"   # blue
C_EUCLIDEAN = "#D55E00"  # vermillion
ROOT_COLORS = {"animal.n.01": "#009E73", "plant.n.02": "#E69F00", "artifact.n.01": "#CC79A7"}
GEOM_COLORS = {"poincare": C_POINCARE, "euclidean": C_EUCLIDEAN}
GEOM_LABEL = {"poincare": "Poincaré", "euclidean": "Euclidean"}

SINGLE = (3.4, 2.6)  # single-column inches


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,  # editable text in the PDF
        "ps.fonttype": 42,
    })
    return plt


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")


def _log2_dim_axis(ax, dims):
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(int(d)) for d in dims])
    ax.minorticks_off()


def _short(root):
    return root.split(".")[0]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_exp_final():
    return pd.read_csv(RESULTS / "exp_final_aggregated.csv")


def q2_advantage(agg, dim=8):
    """Per subtree: hyperbolic advantage = P MAP - E MAP at ``dim``, + structure."""
    rows = []
    for s in agg["subtree"].unique():
        p = agg[(agg.subtree == s) & (agg["train.geometry"] == "poincare") & (agg["train.dim"] == dim)]
        e = agg[(agg.subtree == s) & (agg["train.geometry"] == "euclidean") & (agg["train.dim"] == dim)]
        if p.empty or e.empty:
            continue
        rows.append({"subtree": s, "root": p.iloc[0]["root"], "depth": p.iloc[0]["subtree_depth"],
                     "avg_branching": p.iloc[0]["subtree_avg_branching"],
                     "degree_cv": p.iloc[0]["subtree_degree_cv"],
                     "adv": float(p.iloc[0]["map_mean"] - e.iloc[0]["map_mean"])})
    return pd.DataFrame(rows)


def q3_lambda_best():
    """Per geometry × dim: best-λ ρ and its seed std (from the Stage-8.5 sweep)."""
    agg = pd.read_csv(RESULTS / "q3_lambda_aggregated.csv")
    rows = []
    for (g, d), grp in agg.groupby(["geometry", "dim"]):
        best = grp.loc[grp["rho"].idxmax()]
        rows.append({"geometry": g, "dim": int(d), "rho": float(best["rho"]),
                     "rho_std": float(best["rho_std"]), "lambda": float(best["lambda"])})
    return pd.DataFrame(rows)


def _q3_poincare_dim128_coords():
    """Load the saved Q3 Poincaré dim128 seed0 coords + tree depths for generality plot."""
    from src.experiments.exp_q3_hyperlex import _load_region
    from src.tracking import load_results

    _node_index, depths_arr = _load_region("data")
    res = load_results(RESULTS / "results.csv")
    res = res[(res.get("experiment") == "q3_hyperlex") & (res["train.geometry"] == "poincare")
              & (res["train.dim"] == 128) & (res["train.seed"] == 0)]
    if res.empty:
        return None, None
    coords = np.load(RESULTS / "runs" / res.iloc[0]["run_id"] / "artifacts" / "coords.npy")
    return coords, depths_arr


# ---------------------------------------------------------------------------
# Q1 figures
# ---------------------------------------------------------------------------


def fig_q1(plt, agg):
    dims = sorted(agg["train.dim"].unique())
    for metric, ylabel, fname in [
        ("distortion_avg_scaled_mean", "Scaled distortion", "figQ1a_distortion_vs_dim"),
        ("map_mean", "MAP", "figQ1b_map_vs_dim"),
    ]:
        fig, ax = plt.subplots(figsize=SINGLE)
        for geom in ("poincare", "euclidean"):
            ys, es = [], []
            for d in dims:
                sub = agg[(agg["train.geometry"] == geom) & (agg["train.dim"] == d)][metric]
                ys.append(sub.mean())
                es.append(sub.std())
            ys, es = np.array(ys), np.array(es)
            ax.plot(dims, ys, "-o", color=GEOM_COLORS[geom], label=GEOM_LABEL[geom])
            ax.fill_between(dims, ys - es, ys + es, color=GEOM_COLORS[geom], alpha=0.18)
        _log2_dim_axis(ax, dims)
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, fname)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Q2 figures
# ---------------------------------------------------------------------------


def _scatter_adv(plt, adv, xcol, xlabel, fname):
    fig, ax = plt.subplots(figsize=SINGLE)
    for root, grp in adv.groupby("root"):
        ax.scatter(grp[xcol], grp["adv"], color=ROOT_COLORS.get(root, "#555555"),
                   label=_short(root), s=36, edgecolor="white", linewidth=0.5, zorder=3)
    # dashed trend line
    z = np.polyfit(adv[xcol], adv["adv"], 1)
    xs = np.linspace(adv[xcol].min(), adv[xcol].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="0.35", linewidth=1.5, zorder=2)
    r = float(np.corrcoef(adv[xcol], adv["adv"])[0, 1])
    ax.text(0.04, 0.93, f"$r = {r:+.2f}$", transform=ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAP gap (P$-$E)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, fname)
    plt.close(fig)
    return r


def fig_q2(plt, agg):
    adv = q2_advantage(agg, dim=8)
    r_branch = _scatter_adv(plt, adv, "avg_branching", "Average branching factor",
                            "figQ2_adv_vs_branching")
    r_depth = _scatter_adv(plt, adv, "depth", "Subtree depth", "figQ2a_adv_vs_depth")
    r_degcv = _scatter_adv(plt, adv, "degree_cv", "Degree irregularity",
                           "figQ2b_adv_vs_degcv")
    return {"branching": r_branch, "depth": r_depth, "degree_cv": r_degcv}


# ---------------------------------------------------------------------------
# Q3 figures
# ---------------------------------------------------------------------------


def fig_q3_rho_vs_dim(plt, fixed_lambda=5.0):
    # ρ vs dim at a FIXED λ (=5) rather than best-λ, read straight from the sweep CSV.
    agg = pd.read_csv(RESULTS / "q3_lambda_aggregated.csv")
    s5 = agg[agg["lambda"] == fixed_lambda]
    dims = sorted(int(d) for d in s5["dim"].unique())
    fig, ax = plt.subplots(figsize=SINGLE)
    for geom in ("poincare", "euclidean"):
        s = s5[s5.geometry == geom].sort_values("dim")
        ax.plot(s["dim"], s["rho"], "-o", color=GEOM_COLORS[geom], label=GEOM_LABEL[geom])
        ax.fill_between(s["dim"], s["rho"] - s["rho_std"], s["rho"] + s["rho_std"],
                        color=GEOM_COLORS[geom], alpha=0.18)
    _log2_dim_axis(ax, dims)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Spearman $\\rho$ (HyperLex)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "figQ3_rho_vs_dim")
    plt.close(fig)


def fig_q3_rho_vs_lambda(plt, dim=128):
    agg = pd.read_csv(RESULTS / "q3_lambda_aggregated.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    for geom in ("poincare", "euclidean"):
        s = agg[(agg.geometry == geom) & (agg.dim == dim)].sort_values("lambda")
        # symlog handles the λ=0 point on a log-ish axis
        ax.plot(s["lambda"], s["rho"], "-o", color=GEOM_COLORS[geom], label=GEOM_LABEL[geom])
        ax.fill_between(s["lambda"], s["rho"] - s["rho_std"], s["rho"] + s["rho_std"],
                        color=GEOM_COLORS[geom], alpha=0.18)
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Spearman $\\rho$ (HyperLex)")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "figQ3_rho_vs_lambda")
    plt.close(fig)


def fig_q3_generality(plt):
    coords, depths = _q3_poincare_dim128_coords()
    if coords is None:
        print("  (generality figure skipped — no saved Q3 poincare dim128 seed0 coords)")
        return None
    from scipy.stats import spearmanr
    norms = np.linalg.norm(coords, axis=1)
    rho = float(spearmanr(depths, norms).statistic)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(depths, norms, s=6, alpha=0.25, color=C_POINCARE, edgecolor="none")
    ax.text(0.04, 0.10, f"Spearman $\\rho = {rho:.2f}$", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Poincaré norm $\\|x\\|$")
    fig.tight_layout()
    _save(fig, "figQ3_generality_radius")
    plt.close(fig)
    return rho


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_all():
    plt = set_style()
    agg = load_exp_final()

    print("Generating Q1 figures...")
    fig_q1(plt, agg)

    print("Generating Q2 figures + recomputing correlations...")
    q2r = fig_q2(plt, agg)
    for k, v in q2r.items():
        print(f"  Q2 Pearson r(adv, {k}) = {v:+.3f}")

    print("Generating Q3 figures...")
    fig_q3_rho_vs_dim(plt)
    fig_q3_rho_vs_lambda(plt)
    gen_rho = fig_q3_generality(plt)

    # Q3 λ=0 and best-λ ρ
    best = q3_lambda_best()
    lam0 = pd.read_csv(RESULTS / "q3_lambda_aggregated.csv")
    lam0 = lam0[lam0["lambda"] == 0]
    print("\n--- Recomputed key stats (verify against paper) ---")
    print(f"  Q2 r: branching {q2r['branching']:+.3f}, depth {q2r['depth']:+.3f}, "
          f"degree_cv {q2r['degree_cv']:+.3f}")
    print(f"  Q3 ρ @ λ=0 (pure distance): "
          f"Poincaré {lam0[lam0.geometry=='poincare']['rho'].mean():+.3f}, "
          f"Euclidean {lam0[lam0.geometry=='euclidean']['rho'].mean():+.3f}")
    for geom in ("poincare", "euclidean"):
        s = best[best.geometry == geom]
        print(f"  Q3 best-λ ρ ({GEOM_LABEL[geom]}): "
              + ", ".join(f"d{int(r.dim)}={r.rho:.3f}" for _, r in s.sort_values('dim').iterrows()))
    if gen_rho is not None:
        print(f"  Q3 generality-radius ρ (Poincaré dim128): {gen_rho:.3f}")

    # sanity assertions against the paper headline
    assert abs(q2r["branching"] - 0.64) < 0.02, f"branching r off: {q2r['branching']}"
    print("\nAll figures written to outputs/figures/paper/")
    return q2r, gen_rho


def main():
    make_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
