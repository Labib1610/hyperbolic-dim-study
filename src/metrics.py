"""Evaluation metrics: distortion, MAP, and δ-hyperbolicity.

This stage consumes learned coordinates (``coords.npy``) + a subtree and produces
the numbers the paper reports. No training, data sampling, or geometry changes
here — geometry distances come from :mod:`src.geometry`, tree distances from
:mod:`src.data`.

Metric roles:
    * **distortion** (primary headline): how faithfully geometric distances match
      tree distances (average + worst-case multiplicative).
    * **MAP** (primary, comparability): are each node's true tree-neighbors its
      nearest points? (Nickel–Kiela graph-reconstruction convention.)
    * **δ-hyperbolicity** (DIAGNOSTIC, not the x-axis): the Gromov four-point δ of
      the *embedding's* distance matrix. A tree metric has δ=0; Euclidean
      embeddings distort toward δ>0 while hyperbolic ones stay more tree-like.
      Reported as evidence, not as an independent variable.
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np
import yaml

from src.data import all_pairs_tree_distances, load_subtree
from src.geometry import EuclideanGeometry, PoincareBall
from src.tracking import atomic_write_json, atomic_write_text, load_results


def _resolve_geometry(geometry):
    """Accept a geometry object or a string name and return a geometry object."""
    if isinstance(geometry, str):
        if geometry == "poincare":
            return PoincareBall()
        if geometry == "euclidean":
            return EuclideanGeometry()
        raise ValueError(f"unknown geometry {geometry!r}")
    return geometry


# ---------------------------------------------------------------------------
# Task 1 — pairwise geometric distance matrix
# ---------------------------------------------------------------------------


def pairwise_distances(coords, geometry) -> np.ndarray:
    """Full symmetric ``(N, N)`` matrix of geometric distances between ``coords``.

    Relies on Stage 2 clamping so Poincaré distances are finite even near the
    boundary. The diagonal is forced to exactly 0.
    """
    import torch

    geom = _resolve_geometry(geometry)
    X = torch.as_tensor(np.asarray(coords), dtype=torch.float64)
    N = X.shape[0]
    A = X.unsqueeze(1).expand(N, N, -1)
    B = X.unsqueeze(0).expand(N, N, -1)
    D = geom.dist(A, B).detach().cpu().numpy().astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Task 2 — distortion
# ---------------------------------------------------------------------------


def distortion(Dg: np.ndarray, Dt: np.ndarray) -> dict:
    """Average (raw + scale-fitted) and worst-case multiplicative distortion.

    Over all pairs ``u < v`` with ``Dt > 0``:
        * ``distortion_avg`` — raw (scale=1): ``mean |Dg - Dt| / Dt``.
        * ``distortion_avg_scaled`` — with the global ``scale = mean(Dg/Dt)`` that
          sets the mean ratio to 1 (embeddings are only defined up to scale):
          ``mean |Dg - scale*Dt| / (scale*Dt)``.
        * ``distortion_worst`` — ``max(ratio) / min(ratio)`` where
          ``ratio = Dg/Dt`` (scale-invariant); ``1.0`` is perfect.
    """
    N = Dg.shape[0]
    iu = np.triu_indices(N, k=1)
    dg = Dg[iu].astype(np.float64)
    dt = Dt[iu].astype(np.float64)
    mask = dt > 0
    dg, dt = dg[mask], dt[mask]

    ratio = dg / dt
    scale = float(np.mean(ratio))
    return {
        "distortion_avg": float(np.mean(np.abs(dg - dt) / dt)),
        "distortion_avg_scaled": float(np.mean(np.abs(dg - scale * dt) / (scale * dt))),
        "distortion_worst": float(ratio.max() / ratio.min()),
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Task 3 — MAP (mean average precision)
# ---------------------------------------------------------------------------


def mean_average_precision(Dg: np.ndarray, Dt: np.ndarray) -> float:
    """Mean average precision for graph reconstruction (Nickel–Kiela convention).

    For each node ``u``, rank all other nodes by geometric distance ascending; the
    relevant set is ``u``'s tree-neighbors (``Dt == 1``). ``AP(u)`` averages the
    precision at the ranks where each relevant neighbor appears. ``MAP`` averages
    ``AP`` over nodes with ≥1 neighbor. ``1.0`` means every node's true neighbors
    are strictly its nearest points.
    """
    N = Dg.shape[0]
    aps = []
    for u in range(N):
        relevant = {j for j in np.where(Dt[u] == 1)[0] if j != u}
        if not relevant:
            continue
        others = [j for j in range(N) if j != u]
        others.sort(key=lambda j: (Dg[u, j], j))  # ascending; deterministic tie-break
        found = 0
        precisions = []
        for rank, j in enumerate(others, start=1):
            if j in relevant:
                found += 1
                precisions.append(found / rank)
        aps.append(float(np.mean(precisions)))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Task 4 — δ-hyperbolicity (Gromov four-point)
# ---------------------------------------------------------------------------


def _delta_quad(D: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    s1 = D[i, j] + D[k, l]
    s2 = D[i, k] + D[j, l]
    s3 = D[i, l] + D[j, k]
    largest, middle, _ = sorted((s1, s2, s3), reverse=True)
    return (largest - middle) / 2.0


def delta_hyperbolicity_exact(D: np.ndarray) -> float:
    """Exact Gromov δ: max over ALL quadruples. O(N^4) — small N only (≤ ~60)."""
    N = D.shape[0]
    if N < 4:
        return 0.0
    best = 0.0
    for i, j, k, l in itertools.combinations(range(N), 4):
        d = _delta_quad(D, i, j, k, l)
        if d > best:
            best = d
    return float(best)


def delta_hyperbolicity_sampled(D: np.ndarray, n_samples: int = 50000, seed: int = 0) -> float:
    """Sampled Gromov δ: max over ``n_samples`` random quadruples (under-estimate).

    If ``n_samples`` covers all quadruples, falls back to the exact computation so
    the result is deterministic and equals :func:`delta_hyperbolicity_exact`.
    """
    N = D.shape[0]
    if N < 4:
        return 0.0
    if n_samples >= comb(N, 4):
        return delta_hyperbolicity_exact(D)
    rng = np.random.RandomState(seed)
    best = 0.0
    for _ in range(n_samples):
        i, j, k, l = rng.choice(N, size=4, replace=False)
        d = _delta_quad(D, i, j, k, l)
        if d > best:
            best = d
    return float(best)


def delta_hyperbolicity(
    D: np.ndarray, exact_threshold: int = 60, n_samples: int = 50000, seed: int = 0
) -> float:
    """Dispatch: exact δ for ``N <= exact_threshold``, else sampled."""
    if D.shape[0] <= exact_threshold:
        return delta_hyperbolicity_exact(D)
    return delta_hyperbolicity_sampled(D, n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Task 5 — aggregate evaluator
# ---------------------------------------------------------------------------


def evaluate_embedding(coords, subtree, geometry, delta_samples: int = 50000, seed: int = 0) -> dict:
    """Compute all metrics for ``coords`` embedded under ``geometry`` for ``subtree``.

    ``coords`` rows MUST align with ``all_pairs_tree_distances(subtree)['nodes']``
    (the sorted node order the trainer used). Returns distortion (raw + scaled +
    worst), MAP, δ, relative δ, the fitted scale, and node count.
    """
    apsp = all_pairs_tree_distances(subtree)
    Dt = apsp["matrix"].astype(np.float64)
    Dg = pairwise_distances(coords, geometry)

    result = distortion(Dg, Dt)
    result["map"] = mean_average_precision(Dg, Dt)
    delta = delta_hyperbolicity(Dg, exact_threshold=60, n_samples=delta_samples, seed=seed)
    diam = float(Dg.max())
    result["delta"] = delta
    result["delta_relative"] = delta / diam if diam > 0 else 0.0
    result["num_nodes"] = int(Dg.shape[0])
    return result


# ---------------------------------------------------------------------------
# Task 6 — evaluate a saved run + write metrics back
# ---------------------------------------------------------------------------


def evaluate_run(run_dir, delta_samples: int = 50000, seed: int = 0) -> dict:
    """Evaluate a saved run and write metrics back into its records.

    Loads ``artifacts/coords.npy`` and ``config.yaml`` to recover geometry +
    subtree, recomputes metrics, then updates the run's ``metrics.json`` and its
    row in the master ``outputs/results.csv`` (adding the metric columns) using
    atomic writers.
    """
    run_dir = Path(run_dir)
    coords = np.load(run_dir / "artifacts" / "coords.npy")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    geometry = cfg["train"]["geometry"]
    subtree_name = cfg.get("subtree", "bfs_s50")
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    tree, _meta = load_subtree(f"{data_dir}/subtrees/{subtree_name}")

    metrics = evaluate_embedding(coords, tree, geometry, delta_samples=delta_samples, seed=seed)

    # Merge into metrics.json.
    metrics_path = run_dir / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing.update(metrics)
    atomic_write_json(metrics_path, existing)

    # Update the master results.csv row (outputs/results.csv is two levels up).
    csv_path = run_dir.parent.parent / "results.csv"
    df = load_results(csv_path)
    rid = run_dir.name
    if "run_id" in df.columns and (df["run_id"] == rid).any():
        for key, value in metrics.items():
            df.loc[df["run_id"] == rid, key] = value
        atomic_write_text(csv_path, df.to_csv(index=False))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _iter_run_dirs(runs_root: Path):
    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and (d / "artifacts" / "coords.npy").exists():
            yield d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate embedding runs.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Evaluate a single run directory.")
    g.add_argument("--all", action="store_true", help="Evaluate all runs under outputs/runs/.")
    p.add_argument("--runs-root", default="outputs/runs", help="Root of run directories.")
    p.add_argument("--delta-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    def _summ(run_dir, m):
        print(
            f"{Path(run_dir).name}\n"
            f"    dist_worst={m['distortion_worst']:.4f}  dist_avg_scaled={m['distortion_avg_scaled']:.4f}"
            f"  MAP={m['map']:.4f}  delta_rel={m['delta_relative']:.4f}  N={m['num_nodes']}"
        )

    if args.run:
        m = evaluate_run(args.run, delta_samples=args.delta_samples, seed=args.seed)
        _summ(args.run, m)
    else:
        runs_root = Path(args.runs_root)
        if not runs_root.exists():
            print(f"No runs root at {runs_root}")
            return 0
        count = 0
        for d in _iter_run_dirs(runs_root):
            m = evaluate_run(d, delta_samples=args.delta_samples, seed=args.seed)
            _summ(d, m)
            count += 1
        print(f"\nEvaluated {count} run(s); master table updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
