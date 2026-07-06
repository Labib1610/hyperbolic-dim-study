"""Embedding trainer: learn subtree coordinates via Riemannian/Euclidean optimization.

Given a subtree (from :mod:`src.data`), this learns per-node coordinates in either
the Poincaré ball or Euclidean space so that geometric distances reflect tree
distances, under one of three loss modes (``distortion``, ``ranking``, ``hybrid``).
Every run records itself through :mod:`src.tracking`.

Design seam (important): the trainer produces **learned coordinates + loss curves +
a checkpoint** only. Evaluation metrics (distortion, mAP, δ-hyperbolicity) are
Stage 4 and consume ``coords.npy`` — they are deliberately NOT computed here. The
loss is training signal, not an evaluation metric.

Poincaré optimization uses geoopt's ``ManifoldParameter`` + ``RiemannianAdam``
(validated in Stage 2 to match our distance to 1e-4). Distances in the loss are
computed with our own :mod:`src.geometry` so the reference geometry is exercised.
"""

from __future__ import annotations

import argparse
import copy
import math

import geoopt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

# raw_alpha init so that softplus(raw_alpha) == 1.0 (alpha ≈ 1 at the start).
_INIT_RAW_ALPHA = math.log(math.expm1(1.0))

from src.config import get_device, load_config, set_seed
from src.data import all_pairs_tree_distances, load_subtree
from src.geometry import EuclideanGeometry, PoincareBall
from src.tracking import Run


# ---------------------------------------------------------------------------
# Config convenience: attribute access over nested dicts
# ---------------------------------------------------------------------------


class DotDict(dict):
    """Dict subclass allowing recursive attribute access (``cfg.train.dim``)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return DotDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


def _to_plain(obj):
    """Recursively convert DotDicts to plain dicts (for serialization/tracking)."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Task 2/3 — model + optimizer
# ---------------------------------------------------------------------------


def build_embedding(N: int, dim: int, geometry: str, device=None):
    """Create the coordinate parameter for ``N`` nodes in ``dim`` dimensions.

    Poincaré: a ``geoopt.ManifoldParameter`` on the ball, initialised in a tiny
    ball (radius ~1e-3) near the origin for stable early training. Euclidean: a
    plain ``nn.Parameter`` with small random init.
    """
    device = device or get_device()
    # Build the init data on CPU (portable, seeded RNG) then move to device BEFORE
    # wrapping — wrapping the moved tensor yields a leaf Parameter on the device,
    # whereas `Parameter(...).to(device)` would produce a non-leaf the optimizer
    # rejects.
    if geometry == "poincare":
        manifold = geoopt.PoincareBall()
        data = torch.empty(N, dim).uniform_(-1e-3, 1e-3)
        data = manifold.projx(data)
        param = geoopt.ManifoldParameter(data.to(device), manifold=manifold)
    elif geometry == "euclidean":
        data = torch.randn(N, dim) * 0.01
        param = nn.Parameter(data.to(device))
    else:
        raise ValueError(f"unknown geometry {geometry!r} (expected poincare|euclidean)")
    return param


def build_optimizer(params, cfg):
    """Build the optimizer matching the geometry/optimizer combo in ``cfg.train``.

    ``params`` may be a single parameter or a list (e.g. coordinates + the
    learnable ``raw_alpha``). RiemannianAdam holds the mixed group: the Poincaré
    ``ManifoldParameter`` gets a Riemannian update, while the plain Euclidean
    ``raw_alpha`` gets a standard Adam update — geoopt dispatches per parameter.
    """
    if not isinstance(params, (list, tuple)):
        params = [params]
    geometry = cfg.train.geometry
    optimizer = cfg.train.optimizer
    lr = cfg.train.lr
    if geometry == "poincare":
        if optimizer != "radam":
            raise ValueError("poincare geometry requires optimizer='radam' (RiemannianAdam)")
        return geoopt.optim.RiemannianAdam(list(params), lr=lr)
    if geometry == "euclidean":
        if optimizer != "adam":
            raise ValueError("euclidean geometry requires optimizer='adam'")
        return torch.optim.Adam(list(params), lr=lr)
    raise ValueError(f"unknown geometry {geometry!r}")


def _make_geometry(geometry: str):
    return PoincareBall() if geometry == "poincare" else EuclideanGeometry()


# ---------------------------------------------------------------------------
# Task 4 — loss functions
# ---------------------------------------------------------------------------


def distortion_loss(X, geom, D_T, pairs, alpha=None):
    """Mode A (primary): relative-distortion loss over the given node ``pairs``.

    With ``alpha is None`` (``scale='fixed'``, legacy): fits ``d_G ≈ D_T`` via
    ``mean( (d_G / D_T - 1)^2 )`` (denominator clamped to ``>= 1``).

    With a learnable ``alpha`` (``scale='learnable'``, the fair default): fits
    ``d_G ≈ alpha * D_T``, i.e. ``mean( (d_G / (alpha * D_T) - 1)^2 )``. The learned
    scale rescales the target tree-distances into a range the geometry can
    represent (Stages 5.5-5.8), fixing the near-boundary infeasibility of scale=1.
    """
    u, v = pairs[:, 0], pairs[:, 1]
    dg = geom.dist(X[u], X[v])
    if alpha is None:
        dt = D_T[u, v].clamp_min(1.0)
        return torch.mean((dg / dt - 1.0) ** 2)
    denom = (alpha * D_T[u, v]).clamp_min(1e-6)
    return torch.mean((dg / denom - 1.0) ** 2)


def ranking_loss(X, geom, edges, neg_pool, n_negatives, rng):
    """Mode B (Nickel–Kiela): softmax ranking loss with sampled negatives.

    For each edge ``(u, v)``, ``n_negatives`` non-neighbor negatives ``v'`` are
    sampled; the loss pushes ``d_G(x_u, x_v)`` below the negatives via a stable
    softmax over ``{v} ∪ Neg(u)``. Returned as the mean over edges (see module /
    report notes: mean rather than the spec's raw sum, so it stays O(1) and keeps
    the hybrid combination balanced — optimization is equivalent up to a constant).
    """
    if len(edges) == 0:
        return X.sum() * 0.0
    total = X.new_zeros(())
    for u, v in edges:
        pool = neg_pool[u]
        if len(pool) == 0:
            continue
        replace = len(pool) < n_negatives
        negs = rng.choice(pool, size=n_negatives, replace=replace)
        cand = torch.tensor([v, *negs.tolist()], device=X.device, dtype=torch.long)
        xu = X[u].unsqueeze(0).expand(cand.shape[0], -1)
        logits = -geom.dist(xu, X[cand])  # (1 + n_negatives,); index 0 is positive
        log_z = torch.logsumexp(logits, dim=0)  # stable log-sum-exp
        total = total + (log_z - logits[0])
    return total / len(edges)


# ---------------------------------------------------------------------------
# Task 5 — training loop
# ---------------------------------------------------------------------------


def _all_unordered_pairs(N: int) -> torch.Tensor:
    iu = torch.triu_indices(N, N, offset=1)
    return iu.t().contiguous()  # (P, 2)


def _build_neg_pool(N: int, neighbors: list[set[int]]) -> list[np.ndarray]:
    """For each node, the array of candidate negatives (non-neighbors, excl. self)."""
    all_idx = set(range(N))
    pools = []
    for u in range(N):
        cand = sorted(all_idx - neighbors[u] - {u})
        pools.append(np.array(cand, dtype=np.int64))
    return pools


def _split_rows(n: int, rng, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val index split; always keeps >=1 train row."""
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    n_val = min(n_val, n - 1)  # never let train be empty
    return perm[n_val:], perm[:n_val]


# --- large-tree scalability helpers (used only when train.large_tree is True) ---
# The default (small-tree) path enumerates all O(N^2) pairs and evaluates val on the
# full held-out set every epoch — fine for the <=few-thousand-node Q1/Q2 subtrees but
# O(N^2)/epoch, so it doesn't scale past ~1k nodes. The large_tree path instead:
#   (1) builds only the dense int distance matrix (skips the O(N^2)-memory dists dict),
#   (2) samples training pairs directly each step (no full enumeration),
#   (3) evaluates val distortion on a FIXED, seeded subsample (same pairs every epoch).


def _tree_distance_matrix(tree):
    """Dense int32 tree-distance matrix (N BFS) without the O(N^2)-memory dists dict."""
    nodes = sorted(tree.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.int32)
    for src in nodes:
        i = index[src]
        for dst, d in nx.single_source_shortest_path_length(tree, src).items():
            matrix[i, index[dst]] = d
    return nodes, index, matrix


def _pair_ids(lo, hi, N):
    return lo.astype(np.int64) * N + hi.astype(np.int64)


def _fixed_val_pairs(N: int, count: int, seed: int):
    """A fixed (seeded) set of distinct unordered pairs, constant across epochs.

    Capped at 20% of all possible pairs so the training sampler always has pairs to
    draw. Returns (pairs (V,2) int64, sorted encoded ids for exclusion).
    """
    max_pairs = N * (N - 1) // 2
    count = min(count, max_pairs // 5) if max_pairs >= 5 else max_pairs
    vrng = np.random.RandomState(seed + 999)
    seen: set[tuple[int, int]] = set()
    lo_list, hi_list = [], []
    while len(lo_list) < count:
        us = vrng.randint(0, N, size=2 * count + 16)
        vs = vrng.randint(0, N, size=2 * count + 16)
        for u, v in zip(us, vs):
            if u == v:
                continue
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            if key in seen:
                continue
            seen.add(key)
            lo_list.append(key[0])
            hi_list.append(key[1])
            if len(lo_list) >= count:
                break
    lo = np.array(lo_list, dtype=np.int64)
    hi = np.array(hi_list, dtype=np.int64)
    return np.stack([lo, hi], axis=1), np.sort(_pair_ids(lo, hi, N))


def _sample_train_pairs(rng, N: int, k: int, val_ids, device):
    """Directly sample k random distinct unordered pairs, excluding the val subsample."""
    collected = []
    need = k
    while need > 0:
        us = rng.randint(0, N, size=2 * need + 16)
        vs = rng.randint(0, N, size=2 * need + 16)
        m = us != vs
        us, vs = us[m], vs[m]
        lo = np.minimum(us, vs)
        hi = np.maximum(us, vs)
        keep = ~np.isin(_pair_ids(lo, hi, N), val_ids)
        us, vs = us[keep], vs[keep]
        t = min(need, len(us))
        if t > 0:
            collected.append(np.stack([us[:t], vs[:t]], axis=1))
            need -= t
    return torch.as_tensor(np.concatenate(collected, axis=0), device=device)


def convergence_stop_epoch(vals, patience_epochs: int, min_delta_rel: float) -> int:
    """Epoch index at which R2 convergence stopping would trigger for ``vals``.

    Tracks a reference best; each epoch without a relative improvement of at least
    ``min_delta_rel`` increments a no-improve counter, and stopping triggers once
    it reaches ``patience_epochs``. Returns ``len(vals)`` if it never triggers.
    Pure/inspectable so the training loop and tests share identical logic.
    """
    ref = float("inf")
    no_improve = 0
    for i, v in enumerate(vals):
        if v < ref * (1.0 - min_delta_rel):
            ref = v
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience_epochs:
                return i
    return len(vals)


def train_embedding(subtree_name: str, cfg, base_dir="outputs/runs", results_dir=None) -> Run:
    """Train one embedding and return its completed :class:`~src.tracking.Run`."""
    cfg = DotDict(_to_plain(cfg))
    tcfg = cfg.train
    seed = tcfg.get("seed", cfg.get("seed", 42))
    set_seed(seed)
    device = get_device()
    rng = np.random.RandomState(seed)

    # --- data ---
    data_dir = cfg.get("paths", {}).get("data_dir", "data")
    tree, _meta = load_subtree(f"{data_dir}/subtrees/{subtree_name}")
    loss_mode = tcfg.loss
    large_tree = tcfg.get("large_tree", False)

    # Tree-distance matrix. large_tree skips the O(N^2)-memory dists dict that
    # all_pairs_tree_distances also builds (the dense int matrix is kept — modest).
    if large_tree:
        nodes, index, dmatrix = _tree_distance_matrix(tree)
    else:
        apsp = all_pairs_tree_distances(tree)
        nodes, index, dmatrix = apsp["nodes"], apsp["index"], apsp["matrix"]
    N = len(nodes)
    D_T = torch.tensor(dmatrix, dtype=torch.float32, device=device)

    edges = [(index[a], index[b]) for a, b in tree.edges()]
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    # neg_pool is O(N^2); only ranking/hybrid need it.
    neg_pool = _build_neg_pool(N, neighbors) if loss_mode in ("ranking", "hybrid") else None

    # --- pair sets ---
    if large_tree:
        # Fixed, seeded val subsample (constant across epochs); training pairs sampled
        # directly each step (no O(N^2) enumeration, no full-val-per-epoch cost).
        val_subsample = tcfg.get("val_subsample", 10000)
        val_pairs_np, val_ids = _fixed_val_pairs(N, val_subsample, seed)
        val_pairs = torch.as_tensor(val_pairs_np, device=device)
        train_pairs = None
        use_full_pairs = False
    else:
        all_pairs = _all_unordered_pairs(N).to(device)
        tr_p, va_p = _split_rows(all_pairs.shape[0], rng)
        train_pairs, val_pairs = all_pairs[tr_p], all_pairs[va_p]
        val_ids = None
        use_full_pairs = N <= tcfg.full_pairs_threshold

    e_tr, e_va = _split_rows(len(edges), rng) if edges else (np.array([]), np.array([]))
    train_edges = [edges[i] for i in e_tr]
    val_edges = [edges[i] for i in e_va]

    # --- model + learnable scale (Stage 6 port) ---
    X = build_embedding(N, tcfg.dim, tcfg.geometry, device=device)
    geom = _make_geometry(tcfg.geometry)

    use_alpha = tcfg.get("scale", "learnable") == "learnable"
    if use_alpha:
        raw_alpha = nn.Parameter(torch.tensor(_INIT_RAW_ALPHA, device=device))
        opt = build_optimizer([X, raw_alpha], cfg)
    else:
        raw_alpha = None
        opt = build_optimizer(X, cfg)

    def current_alpha():
        return torch.nn.functional.softplus(raw_alpha) if use_alpha else None

    def compute_loss(training: bool):
        parts = []
        if loss_mode in ("distortion", "hybrid"):
            if large_tree:
                batch = (_sample_train_pairs(rng, N, tcfg.pairs_per_step, val_ids, device)
                         if training else val_pairs)
            else:
                pairs_pool = train_pairs if training else val_pairs
                if training and not use_full_pairs:
                    k = min(tcfg.pairs_per_step, pairs_pool.shape[0])
                    sel = rng.choice(pairs_pool.shape[0], size=k, replace=False)
                    batch = pairs_pool[torch.as_tensor(sel, device=device)]
                else:
                    batch = pairs_pool
            parts.append(distortion_loss(X, geom, D_T, batch, alpha=current_alpha()))
        if loss_mode in ("ranking", "hybrid"):
            # alpha applies to the distortion target; ranking uses distances directly.
            edge_pool = train_edges if training else val_edges
            r = ranking_loss(X, geom, edge_pool, neg_pool, tcfg.n_negatives, rng)
            parts.append(tcfg.hybrid_lambda * r if loss_mode == "hybrid" else r)
        return sum(parts)

    # --- stopping configuration (R2 convergence vs fixed epochs) ---
    stopping = tcfg.get("stopping", "convergence")
    if stopping == "convergence":
        max_epochs = tcfg.get("epoch_cap", 8000)
        patience = tcfg.get("patience_epochs", 500)
        min_delta_rel = tcfg.get("min_delta_rel", 0.005)
    elif stopping == "fixed_epochs":
        # legacy `epochs` key is honoured as the fixed budget fallback.
        max_epochs = tcfg.get("fixed_epochs", tcfg.get("epochs", 8000))
        patience = None
        min_delta_rel = None
    else:
        raise ValueError(f"unknown stopping {stopping!r} (expected convergence|fixed_epochs)")

    # --- run tracking ---
    name = f"{tcfg.geometry}_d{tcfg.dim}_{loss_mode}_{subtree_name}"
    run = Run(_to_plain(cfg), name=name, base_dir=base_dir, results_dir=results_dir)

    train_curve, val_curve = [], []
    best_val = float("inf")
    best_coords = X.detach().cpu().numpy().copy()
    best_epoch = 0
    ref_val = float("inf")   # reference for the min_delta convergence check
    no_improve = 0

    for epoch in range(max_epochs):
        X.requires_grad_(True)
        opt.zero_grad()
        train_loss = compute_loss(training=True)
        train_loss.backward()
        opt.step()

        with torch.no_grad():
            val_loss = compute_loss(training=False)
        tl, vl = float(train_loss.detach()), float(val_loss.detach())
        train_curve.append(tl)
        val_curve.append(vl)

        # Keep the absolute best-val checkpoint (independent of the stopping rule).
        if vl < best_val:
            best_val = vl
            best_coords = X.detach().cpu().numpy().copy()
            best_epoch = epoch

        if stopping == "convergence":
            if vl < ref_val * (1.0 - min_delta_rel):
                ref_val = vl
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    epochs_ran = len(train_curve)
    final_alpha = float(current_alpha().detach()) if use_alpha else 1.0

    run.log_curve("train_loss", train_curve)
    run.log_curve("val_loss", val_curve)
    run.save_array("coords", best_coords)
    run.log_metrics(
        {
            "final_train_loss": train_curve[-1],
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_to_converge": best_epoch,
            "epochs_ran": epochs_ran,
            "alpha": final_alpha,
            "num_nodes": N,
        }
    )
    run.finish()
    return run


# ---------------------------------------------------------------------------
# Task 6 — CLI
# ---------------------------------------------------------------------------


def _apply_overrides(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})
    if args.subtree is not None:
        cfg["subtree"] = args.subtree
    mapping = {
        "geometry": "geometry", "dim": "dim", "loss": "loss", "epochs": "epochs",
        "lr": "lr", "hybrid_lambda": "hybrid_lambda", "n_negatives": "n_negatives",
        "pairs_per_step": "pairs_per_step", "seed": "seed", "optimizer": "optimizer",
        "scale": "scale", "stopping": "stopping", "patience_epochs": "patience_epochs",
        "min_delta_rel": "min_delta_rel", "epoch_cap": "epoch_cap", "fixed_epochs": "fixed_epochs",
    }
    for arg_name, cfg_key in mapping.items():
        val = getattr(args, arg_name)
        if val is not None:
            cfg["train"][cfg_key] = val
    # Auto-pick the optimizer for the geometry unless explicitly overridden.
    if args.optimizer is None:
        cfg["train"]["optimizer"] = "radam" if cfg["train"]["geometry"] == "poincare" else "adam"
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a subtree embedding.")
    p.add_argument("--subtree", default=None, help="Subtree dir under data/subtrees/.")
    p.add_argument("--geometry", default=None, choices=["poincare", "euclidean"])
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--loss", default=None, choices=["distortion", "ranking", "hybrid"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hybrid_lambda", type=float, default=None)
    p.add_argument("--n_negatives", type=int, default=None)
    p.add_argument("--pairs_per_step", type=int, default=None)
    p.add_argument("--optimizer", default=None, choices=["radam", "adam"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scale", default=None, choices=["learnable", "fixed"])
    p.add_argument("--stopping", default=None, choices=["convergence", "fixed_epochs"])
    p.add_argument("--patience_epochs", type=int, default=None)
    p.add_argument("--min_delta_rel", type=float, default=None)
    p.add_argument("--epoch_cap", type=int, default=None)
    p.add_argument("--fixed_epochs", type=int, default=None)
    p.add_argument("--config", default="configs/default.yaml")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    subtree = cfg.get("subtree", "bfs_s50")

    run = train_embedding(subtree, cfg)
    print(f"Run directory: {run.dir}")
    print(f"  final_train_loss   = {run.metrics['final_train_loss']:.6f}")
    print(f"  best_val_loss      = {run.metrics['best_val_loss']:.6f}")
    print(f"  epochs_ran         = {run.metrics['epochs_ran']}")
    print(f"  epochs_to_converge = {run.metrics['epochs_to_converge']}")
    print(f"  alpha              = {run.metrics['alpha']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
