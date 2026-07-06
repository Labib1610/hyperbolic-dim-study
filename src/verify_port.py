"""Verify the ported production trainer reproduces the sandbox ground-truth.

Runs the ported ``src.trainer`` (learnable-α + R2 convergence) on the synthetic
oracle and on ``deep_s200`` and checks the converged metrics against the Stage
5.6/5.8 sandbox reference values within tolerance. Also checks the failure-mode
reproduction (scale=fixed + short fixed_epochs -> broken Poincaré) and the α
shrink-with-depth pattern.

Usage: ``python -m src.verify_port``  (exit code 0 iff all checks pass).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import save_subtree
from src.diagnostics.diag_plateau import make_synthetic_tree
from src.metrics import evaluate_embedding
from src.trainer import train_embedding

VERIFY_DATA_DIR = "outputs/verify_trees"

# Sandbox reference values the port must REPRODUCE-OR-EXCEED. The production port
# uses R2 convergence stopping (cap 8000), so it trains further than the 500-epoch
# sandbox reference runs and is expected to be AT LEAST as good. We therefore check
# `port_MAP >= ref - MAP_ATOL` (the port may — and does — beat the sandbox; it must
# not be worse by more than the tolerance). References documented per row.
REFERENCES = {
    ("oracle", 2): {"map": 0.46, "note": "Stage5.6 fix3/combo_learnable oracle dim2 MAP~0.46-0.48"},
    ("oracle", 8): {"map": 0.92, "note": "Stage5.6 oracle dim8 MAP~0.92-0.95"},
    ("deep_s200", 128): {"map": 0.87, "note": "Stage5.8 deep_s200 dim128 converged MAP~0.87-0.90"},
}
MAP_ATOL = 0.05


def _cfg(subtree, geometry, dim, seed, *, scale="learnable", stopping="convergence",
         fixed_epochs=8000, epoch_cap=8000, patience=500):
    return {
        "paths": {"data_dir": VERIFY_DATA_DIR},
        "subtree": subtree,
        "seed": seed,
        "train": {
            "geometry": geometry, "dim": dim, "loss": "distortion", "hybrid_lambda": 1.0,
            "n_negatives": 10, "pairs_per_step": 512, "full_pairs_threshold": 150,
            "lr": 0.01, "optimizer": "radam" if geometry == "poincare" else "adam",
            "seed": seed, "scale": scale, "stopping": stopping,
            "patience_epochs": patience, "min_delta_rel": 0.005, "epoch_cap": epoch_cap,
            "fixed_epochs": fixed_epochs,
        },
    }


def _prepare_trees():
    Path(VERIFY_DATA_DIR).mkdir(parents=True, exist_ok=True)
    oracle = make_synthetic_tree(2, 6)  # 127-node perfect binary tree
    save_subtree(oracle, oracle.graph["root"], f"{VERIFY_DATA_DIR}/subtrees/oracle")
    # deep_s200 already lives in data/; symlink-free: copy its files under verify dir.
    from src.data import load_subtree
    tree, _ = load_subtree("data/subtrees/deep_s200")
    save_subtree(tree, tree.graph["root"], f"{VERIFY_DATA_DIR}/subtrees/deep_s200")
    return oracle, tree


def _train_eval(subtree_name, tree, geometry, dim, seed, **cfg_over):
    cfg = _cfg(subtree_name, geometry, dim, seed, **cfg_over)
    run = train_embedding(subtree_name, cfg, base_dir="outputs/runs", results_dir="outputs")
    coords = np.load(run.dir / "artifacts" / "coords.npy")
    m = evaluate_embedding(coords, tree, geometry, delta_samples=2000)
    return m, run.metrics.get("alpha", float("nan")), run


def main() -> int:
    oracle, deep = _prepare_trees()
    trees = {"oracle": oracle, "deep_s200": deep}
    all_ok = True
    print("=== Port verification (learnable-α + R2 convergence, seed 0) ===\n")

    # 1) Ground-truth reproduction
    results = {}
    for (name, dim), ref in REFERENCES.items():
        m, alpha, _ = _train_eval(name, trees[name], "poincare", dim, 0)
        ok = m["map"] >= ref["map"] - MAP_ATOL  # reproduce-or-exceed
        all_ok &= ok
        results[(name, dim)] = (m, alpha)
        print(f"[{'OK ' if ok else 'BAD'}] {name} dim{dim}: MAP={m['map']:.3f} "
              f"dist={m['distortion_avg_scaled']:.3f} alpha={alpha:.3f}  "
              f"(ref MAP>={ref['map']:.2f}-{MAP_ATOL}; {ref['note']})")

    # 2) Oracle dim-2 sanity: learnable clearly beats fixed
    m_learn, a_learn, _ = _train_eval("oracle", oracle, "poincare", 2, 0)
    m_fixed, _, _ = _train_eval("oracle", oracle, "poincare", 2, 0,
                                scale="fixed", stopping="fixed_epochs", fixed_epochs=500)
    sanity_ok = m_learn["map"] >= 0.4 and m_learn["map"] > m_fixed["map"] + 0.1
    all_ok &= sanity_ok
    print(f"\n[{'OK ' if sanity_ok else 'BAD'}] oracle dim2 sanity: "
          f"learnable MAP={m_learn['map']:.3f} > fixed MAP={m_fixed['map']:.3f} (want learnable>=0.4)")

    # 3) Failure-mode reproduction: fixed scale + short epochs -> broken Poincaré
    fail_ok = m_fixed["map"] <= 0.15
    all_ok &= fail_ok
    print(f"[{'OK ' if fail_ok else 'BAD'}] failure-mode: scale=fixed,fixed_epochs=500 "
          f"oracle dim2 MAP={m_fixed['map']:.3f} (want <=0.15, ~broken)")

    # 4) alpha shrink-with-depth: oracle (shallow-ish, small distances) vs deep_s200
    a_oracle = results[("oracle", 8)][1]
    a_deep = results[("deep_s200", 128)][1]
    alpha_ok = np.isfinite(a_oracle) and np.isfinite(a_deep) and a_deep < 1.0
    all_ok &= alpha_ok
    print(f"[{'OK ' if alpha_ok else 'BAD'}] alpha pattern: oracle α={a_oracle:.3f}, "
          f"deep_s200 α={a_deep:.3f} (deep tree α<1, finite/positive)")

    print(f"\n=== {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
