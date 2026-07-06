"""Sandbox fix prototype: recover Poincaré on the oracle (perfect tree).

Stage 5.5 diagnosed the Poincaré plateau as a TRAINING defect: with the distortion
loss targeting ``d_G = D_T`` (tree distances up to ~12) and tiny origin init, points
stall at norm ~0.88 and cannot represent the required large distances. This sandbox
tests three fixes — WITHOUT touching protected code (``trainer.py``/``geometry.py``/
``metrics.py``). It defines its own small Poincaré training loop and reuses
``src.geometry`` for distances and ``src.metrics`` for evaluation.

Fixes (toggleable + combinable):
    1. Spread-out init: origin | uniform_ball | scaled_randn.
    2. Learning-rate burn-in (Nickel–Kiela): reduced lr for the first epochs.
    3. Scale handling: learnable alpha (target d_G ≈ alpha·D_T) or normalized target
       (rescale D_T into a comfortable range) — so the geometry need not jam points
       against the rim to express large distances.

The oracle is the synthetic perfect binary tree (b=2, depth=6, 127 nodes): Sarkar's
construction guarantees near-zero-distortion hyperbolic embeddability, so a correct
trainer MUST let Poincaré win at low dim. A fix "recovers" iff Poincaré reaches low
distortion + high MAP at low dim (esp. dim 2) on this oracle.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import geoopt
import numpy as np
import torch
import torch.nn as nn

from src.data import all_pairs_tree_distances, load_subtree
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.geometry import EuclideanGeometry, PoincareBall
from src.metrics import evaluate_embedding
from src.tracking import Run

FIG_DIR = "outputs/figures/fix"
TAG = "fix_prototype"
# delta is irrelevant to this diagnostic; keep its sampling cheap.
DELTA_SAMPLES = 2000


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def default_cfg(**over) -> dict:
    cfg = {
        "init": "origin",            # origin | uniform_ball | scaled_randn
        "burnin_epochs": 0,
        "burnin_factor": 0.1,
        "scale": "fixed",            # fixed | learnable
        "target_norm": "none",       # none | normalize
        "target_const": 3.0,
        "epochs": 500,
        "lr": 0.01,
        "pairs_per_step": 512,
        "full_pairs_threshold": 150,
    }
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------------------
# Initialization modes (Fix 1)
# ---------------------------------------------------------------------------


def init_coords(N: int, dim: int, mode: str, manifold) -> torch.Tensor:
    """Return an (N, dim) init tensor inside the ball for the chosen mode."""
    if mode == "origin":
        data = torch.empty(N, dim).uniform_(-1e-3, 1e-3)
    elif mode == "uniform_ball":
        directions = torch.randn(N, dim)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radii = torch.empty(N, 1).uniform_(0.1, 0.8)
        data = directions * radii
    elif mode == "scaled_randn":
        data = torch.randn(N, dim) * 0.1
    else:
        raise ValueError(f"unknown init mode {mode!r}")
    return manifold.projx(data)


# ---------------------------------------------------------------------------
# Task 1 — sandbox Poincaré trainer with all fixes
# ---------------------------------------------------------------------------


def train_poincare_fixed(tree, dim: int, cfg: dict, seed: int):
    """Train a Poincaré embedding with the chosen fixes. Returns (coords, info)."""
    _seed_all(seed)
    device = _device()
    geom = PoincareBall()
    manifold = geoopt.PoincareBall()

    apsp = all_pairs_tree_distances(tree)
    N = len(apsp["nodes"])
    Dt = torch.tensor(apsp["matrix"], dtype=torch.float32, device=device)

    # Target distances (Fix 3 — normalized target).
    if cfg["target_norm"] == "normalize":
        Dt_target = Dt / Dt.max().clamp_min(1.0) * cfg["target_const"]
    else:
        Dt_target = Dt

    X = geoopt.ManifoldParameter(
        init_coords(N, dim, cfg["init"], manifold).to(device), manifold=manifold
    )
    params = [X]

    alpha_raw = None
    if cfg["scale"] == "learnable":  # Fix 3 — learnable scale
        alpha_raw = nn.Parameter(torch.zeros((), device=device))
        params.append(alpha_raw)

    opt = geoopt.optim.RiemannianAdam(params, lr=cfg["lr"])

    iu = torch.triu_indices(N, N, offset=1)
    all_pairs = iu.t().contiguous().to(device)
    use_full = N <= cfg["full_pairs_threshold"]
    rng = np.random.RandomState(seed)

    curve = []
    best_loss = float("inf")
    best_coords = X.detach().cpu().numpy().copy()

    for epoch in range(cfg["epochs"]):
        # Fix 2 — burn-in lr schedule.
        lr = cfg["lr"] * (cfg["burnin_factor"] if epoch < cfg["burnin_epochs"] else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        if use_full:
            batch = all_pairs
        else:
            k = min(cfg["pairs_per_step"], all_pairs.shape[0])
            sel = rng.choice(all_pairs.shape[0], size=k, replace=False)
            batch = all_pairs[torch.as_tensor(sel, device=device)]

        u, v = batch[:, 0], batch[:, 1]
        dg = geom.dist(X[u], X[v])
        denom = Dt_target[u, v]
        if alpha_raw is not None:
            alpha = torch.nn.functional.softplus(alpha_raw)
            denom = alpha * denom
        denom = denom.clamp_min(1e-6)
        loss = torch.mean((dg / denom - 1.0) ** 2)
        loss.backward()
        opt.step()

        lv = float(loss.detach())
        curve.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_coords = X.detach().cpu().numpy().copy()

    info = {"curve": curve, "best_loss": best_loss}
    if alpha_raw is not None:
        info["alpha"] = float(torch.nn.functional.softplus(alpha_raw.detach()))
    return best_coords, info


def train_euclidean_ref(tree, dim: int, seed: int, epochs: int = 500, lr: float = 0.01,
                        pairs_per_step: int = 512, full_pairs_threshold: int = 150):
    """Thin Euclidean reference trainer (production-equivalent: randn init, Adam, scale=1)."""
    _seed_all(seed)
    device = _device()
    geom = EuclideanGeometry()

    apsp = all_pairs_tree_distances(tree)
    N = len(apsp["nodes"])
    Dt = torch.tensor(apsp["matrix"], dtype=torch.float32, device=device)

    X = nn.Parameter((torch.randn(N, dim) * 0.01).to(device))
    opt = torch.optim.Adam([X], lr=lr)

    iu = torch.triu_indices(N, N, offset=1)
    all_pairs = iu.t().contiguous().to(device)
    use_full = N <= full_pairs_threshold
    rng = np.random.RandomState(seed)

    curve = []
    best_loss = float("inf")
    best_coords = X.detach().cpu().numpy().copy()
    for _epoch in range(epochs):
        opt.zero_grad()
        if use_full:
            batch = all_pairs
        else:
            k = min(pairs_per_step, all_pairs.shape[0])
            sel = rng.choice(all_pairs.shape[0], size=k, replace=False)
            batch = all_pairs[torch.as_tensor(sel, device=device)]
        u, v = batch[:, 0], batch[:, 1]
        dg = geom.dist(X[u], X[v])
        loss = torch.mean((dg / Dt[u, v].clamp_min(1.0) - 1.0) ** 2)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        curve.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_coords = X.detach().cpu().numpy().copy()
    return best_coords, {"curve": curve, "best_loss": best_loss}


# ---------------------------------------------------------------------------
# Evaluation + tracking helper
# ---------------------------------------------------------------------------


def _evaluate(coords, tree, geometry):
    return evaluate_embedding(coords, tree, geometry, delta_samples=DELTA_SAMPLES, seed=0)


def _track(name, config, coords, metrics, curve, base_dir, results_dir):
    run = Run(config, name=name, base_dir=base_dir, results_dir=results_dir)
    run.log_curve("train_loss", curve)
    run.save_array("coords", coords)
    run.log_metrics(metrics)
    run.finish()
    return run


def norm_stats(coords) -> dict:
    norms = np.linalg.norm(coords, axis=1)
    return {
        "min": float(norms.min()), "median": float(np.median(norms)), "max": float(norms.max()),
        "frac_gt_0.9": float((norms > 0.9).mean()), "frac_gt_0.99": float((norms > 0.99).mean()),
    }


# ---------------------------------------------------------------------------
# Named fix configurations
# ---------------------------------------------------------------------------

ORACLE_CONFIGS = {
    "baseline": default_cfg(),
    "fix1_uniform_ball": default_cfg(init="uniform_ball"),
    "fix1_scaled_randn": default_cfg(init="scaled_randn"),
    "fix2_burnin": default_cfg(burnin_epochs=50),
    "fix3_learnable": default_cfg(scale="learnable"),
    "fix3_normalize": default_cfg(target_norm="normalize"),
    "combo_init_burnin": default_cfg(init="uniform_ball", burnin_epochs=50),
    "combo_init_learnable": default_cfg(init="uniform_ball", scale="learnable"),
    "combo_all": default_cfg(init="uniform_ball", burnin_epochs=50, scale="learnable"),
}


# ---------------------------------------------------------------------------
# Task 2 — oracle experiment
# ---------------------------------------------------------------------------


def run_oracle(dims=(2, 8, 32), seeds=(0, 1, 2), base_dir="outputs/runs", results_dir="outputs"):
    tree = make_synthetic_tree(2, 6)  # 127-node perfect binary tree
    rows = []
    coords_cache = {}

    for cfg_name, cfg in ORACLE_CONFIGS.items():
        for dim in dims:
            for seed in seeds:
                coords, info = train_poincare_fixed(tree, dim, cfg, seed)
                m = _evaluate(coords, tree, "poincare")
                coords_cache[(cfg_name, dim, seed)] = coords
                record = {"config": cfg_name, "geometry": "poincare", "dim": dim, "seed": seed,
                          "distortion_avg_scaled": m["distortion_avg_scaled"],
                          "distortion_worst": m["distortion_worst"], "map": m["map"],
                          "alpha": info.get("alpha", np.nan)}
                rows.append(record)
                _track(f"oracle_{cfg_name}_d{dim}_s{seed}",
                       {"experiment": TAG, "part": "oracle", "config": cfg_name,
                        "geometry": "poincare", "dim": dim, "seed": seed, **cfg},
                       coords, m, info["curve"], base_dir, results_dir)

    # Euclidean reference line.
    for dim in dims:
        for seed in seeds:
            coords, info = train_euclidean_ref(tree, dim, seed)
            m = _evaluate(coords, tree, "euclidean")
            coords_cache[("euclidean", dim, seed)] = coords
            rows.append({"config": "euclidean_ref", "geometry": "euclidean", "dim": dim, "seed": seed,
                         "distortion_avg_scaled": m["distortion_avg_scaled"],
                         "distortion_worst": m["distortion_worst"], "map": m["map"], "alpha": np.nan})
            _track(f"oracle_euclidean_d{dim}_s{seed}",
                   {"experiment": TAG, "part": "oracle", "config": "euclidean_ref",
                    "geometry": "euclidean", "dim": dim, "seed": seed},
                   coords, m, info["curve"], base_dir, results_dir)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "fix_oracle.csv", index=False)
    return df, coords_cache


def pick_winner(df, low_dim=2):
    """Winner = Poincaré config with the highest mean MAP at low dim (tie: lower dist)."""
    import pandas as pd

    pois = df[(df.geometry == "poincare") & (df.dim == low_dim)]
    grp = pois.groupby("config").agg(map_mean=("map", "mean"),
                                     dist_mean=("distortion_avg_scaled", "mean")).reset_index()
    grp = grp.sort_values(["map_mean", "dist_mean"], ascending=[False, True])
    return grp.iloc[0]["config"]


# ---------------------------------------------------------------------------
# Task 4 — real-data confirmation on deep_s200
# ---------------------------------------------------------------------------


def run_deep_s200(winner_config, dims=(8, 32, 128), seeds=(0, 1, 2),
                  base_dir="outputs/runs", results_dir="outputs", data_dir="data"):
    tree, _meta = load_subtree(f"{data_dir}/subtrees/deep_s200")
    configs = {"baseline": ORACLE_CONFIGS["baseline"], winner_config: ORACLE_CONFIGS[winner_config]}
    rows = []
    for cfg_name, cfg in configs.items():
        for dim in dims:
            for seed in seeds:
                coords, info = train_poincare_fixed(tree, dim, cfg, seed)
                m = _evaluate(coords, tree, "poincare")
                rows.append({"config": cfg_name, "geometry": "poincare", "dim": dim, "seed": seed,
                             "distortion_avg_scaled": m["distortion_avg_scaled"],
                             "distortion_worst": m["distortion_worst"], "map": m["map"]})
                _track(f"deep_{cfg_name}_d{dim}_s{seed}",
                       {"experiment": TAG, "part": "deep_s200", "config": cfg_name,
                        "geometry": "poincare", "subtree": "deep_s200", "dim": dim, "seed": seed, **cfg},
                       coords, m, info["curve"], base_dir, results_dir)
    # Euclidean reference on deep_s200.
    for dim in dims:
        for seed in seeds:
            coords, info = train_euclidean_ref(tree, dim, seed)
            m = _evaluate(coords, tree, "euclidean")
            rows.append({"config": "euclidean_ref", "geometry": "euclidean", "dim": dim, "seed": seed,
                         "distortion_avg_scaled": m["distortion_avg_scaled"],
                         "distortion_worst": m["distortion_worst"], "map": m["map"]})
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "fix_deep_s200.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Task 5 — figures + report
# ---------------------------------------------------------------------------


def make_figures(oracle_df, outdir=FIG_DIR):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dims = sorted(oracle_df["dim"].unique())
    configs = list(ORACLE_CONFIGS.keys()) + ["euclidean_ref"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, (metric, title) in enumerate(
        [("distortion_avg_scaled", "scaled avg distortion"), ("map", "MAP")]
    ):
        ax = axes[col]
        for cfg_name in configs:
            xs, ys = [], []
            for d in dims:
                s = oracle_df[(oracle_df.config == cfg_name) & (oracle_df.dim == d)]
                if not s.empty:
                    xs.append(d); ys.append(float(s[metric].mean()))
            if xs:
                style = "--" if cfg_name == "euclidean_ref" else "-o"
                lw = 2.5 if cfg_name == "euclidean_ref" else 1.5
                ax.plot(xs, ys, style, label=cfg_name, linewidth=lw)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("dimension")
        ax.set_title(f"Oracle (perfect b2d6 tree): {title} vs dim")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "oracle_fixes.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "oracle_fixes.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_report(oracle_df, winner, winner_norms, baseline_norms, deep_df,
                    path="outputs/fix_prototype_report.md"):
    import pandas as pd

    dims = sorted(oracle_df["dim"].unique())
    L = ["# Fix Prototype — recovering Poincaré on the oracle", ""]

    def gv(df, cfg, dim, col):
        s = df[(df.config == cfg) & (df.dim == dim)]
        return float(s[col].mean()) if not s.empty else float("nan")

    # Oracle table
    L += ["## Oracle (perfect binary tree b=2, depth=6, 127 nodes)", "",
          "Mean over 3 seeds. dist = distortion_avg_scaled; MAP higher is better.", ""]
    header = "| config | " + " | ".join(f"dist@{d} / MAP@{d}" for d in dims) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(dims) + 1))
    for cfg in list(ORACLE_CONFIGS.keys()) + ["euclidean_ref"]:
        cells = [f"{gv(oracle_df, cfg, d, 'distortion_avg_scaled'):.3f} / {gv(oracle_df, cfg, d, 'map'):.3f}" for d in dims]
        L.append(f"| {cfg} | " + " | ".join(cells) + " |")
    L.append("")

    # Winner + money question
    lo = dims[0]
    p_win_map = gv(oracle_df, winner, lo, "map")
    p_win_dist = gv(oracle_df, winner, lo, "distortion_avg_scaled")
    e_map = gv(oracle_df, "euclidean_ref", lo, "map")
    e_dist = gv(oracle_df, "euclidean_ref", lo, "distortion_avg_scaled")
    base_map = gv(oracle_df, "baseline", lo, "map")
    beats = p_win_dist < e_dist and p_win_map >= e_map - 1e-9
    L += ["## Winner", "",
          f"- **Winning config: `{winner}`**.",
          f"- Oracle @dim{lo}: winner P dist={p_win_dist:.3f} MAP={p_win_map:.3f}  vs  "
          f"baseline P MAP={base_map:.3f}  vs  Euclidean dist={e_dist:.3f} MAP={e_map:.3f}.",
          f"- **Money question — does Poincaré@dim{lo} beat Euclidean@dim{lo}?** "
          f"{'YES' if beats else 'NO'} (P dist {p_win_dist:.3f} vs E {e_dist:.3f}; P MAP {p_win_map:.3f} vs E {e_map:.3f}).",
          ""]

    # Norm migration
    L += ["## Point-norm migration (oracle @dim2)", "",
          "| config | min | median | max | frac>0.9 | frac>0.99 |",
          "|---|---|---|---|---|---|",
          f"| baseline | {baseline_norms['min']:.3f} | {baseline_norms['median']:.3f} | "
          f"{baseline_norms['max']:.3f} | {baseline_norms['frac_gt_0.9']:.3f} | {baseline_norms['frac_gt_0.99']:.3f} |",
          f"| {winner} | {winner_norms['min']:.3f} | {winner_norms['median']:.3f} | "
          f"{winner_norms['max']:.3f} | {winner_norms['frac_gt_0.9']:.3f} | {winner_norms['frac_gt_0.99']:.3f} |",
          ""]

    # deep_s200 confirmation
    L += ["## Real-data confirmation on deep_s200", "",
          "| dim | baseline P dist/MAP | winner P dist/MAP | Euclidean dist/MAP |",
          "|---|---|---|---|"]
    for d in sorted(deep_df["dim"].unique()):
        L.append(
            f"| {d} | {gv(deep_df,'baseline',d,'distortion_avg_scaled'):.3f} / {gv(deep_df,'baseline',d,'map'):.3f} "
            f"| {gv(deep_df,winner,d,'distortion_avg_scaled'):.3f} / {gv(deep_df,winner,d,'map'):.3f} "
            f"| {gv(deep_df,'euclidean_ref',d,'distortion_avg_scaled'):.3f} / {gv(deep_df,'euclidean_ref',d,'map'):.3f} |"
        )
    L.append("")

    # Which fix mattered / is Fix 3 in the winner
    fix3_in_winner = ORACLE_CONFIGS[winner]["scale"] == "learnable" or ORACLE_CONFIGS[winner]["target_norm"] == "normalize"
    L += ["## Notes", "",
          f"- Winner config toggles: {ORACLE_CONFIGS[winner]}.",
          f"- Is Fix 3 (scale handling) part of the winner? **{'YES' if fix3_in_winner else 'NO'}** "
          f"(if yes it may become part of the published method, not just a bug fix).", ""]

    text = "\n".join(L)
    Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Poincaré fix prototype (sandbox).")
    p.add_argument("--part", choices=["oracle", "deep", "all"], default="all")
    p.add_argument("--base-dir", default="outputs/runs")
    p.add_argument("--results-dir", default="outputs")
    args = p.parse_args(argv)
    kw = dict(base_dir=args.base_dir, results_dir=args.results_dir)

    print("Running oracle sweep...")
    oracle_df, cache = run_oracle(**kw)
    winner = pick_winner(oracle_df)
    print(f"Winner: {winner}")

    baseline_norms = norm_stats(cache[("baseline", 2, 0)])
    winner_norms = norm_stats(cache[(winner, 2, 0)])

    deep_df = None
    if args.part in ("deep", "all"):
        print("Running deep_s200 confirmation...")
        deep_df = run_deep_s200(winner, **kw)

    make_figures(oracle_df)
    if deep_df is None:
        import pandas as pd

        deep_df = pd.DataFrame(columns=["config", "dim", "distortion_avg_scaled", "map"])
    text = generate_report(oracle_df, winner, winner_norms, baseline_norms, deep_df)
    print("\nReport -> outputs/fix_prototype_report.md\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
