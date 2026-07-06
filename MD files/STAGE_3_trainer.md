# Stage 3 — Embedding Trainer (Riemannian optimization, 3 loss modes)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet), as a function of hierarchy structure (δ-hyperbolicity).**

Stages 0 (env), 1 (data), 2 (geometry core), 2.5 (experiment tracking) are complete and tested. This stage builds the **trainer**: it learns coordinates for the nodes of a subtree so that geometric distances reflect tree-distances, in EITHER the Poincaré ball OR Euclidean space (chosen by config), optimized with the appropriate (Riemannian or standard) optimizer, under one of THREE loss modes. Every run records itself via `src/tracking.py`.

**Do NOT implement evaluation metrics (distortion, MAP, δ-hyperbolicity) in this stage** — those are Stage 4. The trainer only needs the *loss* (which is training signal), not the evaluation metrics. Keep a clean seam: trainer produces learned coordinates + loss curves + a saved checkpoint; Stage 4 will consume the coordinates and compute metrics.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active, torch (cu128, sm_120 verified), `geoopt` 0.5.1 available.
- Use existing modules: `src/geometry.py` (`PoincareBall`, `EuclideanGeometry`), `src/data.py` (load subtrees, tree distances), `src/tracking.py` (`Run`), `src/config.py` (`set_seed`, `get_device`).
- **Use `geoopt` for the optimizer and manifold parameter** (its `RiemannianAdam` + `ManifoldParameter` on `geoopt.PoincareBall`). We validated in Stage 2 that geoopt's Poincaré distance matches ours to 1e-4, so it is safe to rely on. Our own `src/geometry.py` remains the reference used in tests and metrics.

---

## Loss functions (implement all three EXACTLY as specified)

Let the subtree have `N` nodes. Coordinates are a parameter tensor `X` of shape `(N, dim)`. For Poincaré, `X` is a `geoopt.ManifoldParameter` on the ball (initialized near origin, e.g. uniform in a small radius like 1e-3, then projected). For Euclidean, `X` is a plain `nn.Parameter` (small random init). `geom` is the geometry object providing `dist`.

We precompute, from `src/data.py`, the all-pairs tree-distance matrix `D_T` (shape `(N,N)`, integer hops), and the edge set `E` (parent/child pairs). Node ↔ index mapping is fixed.

### Mode A: `distortion` (PRIMARY)
Direct relative-distortion loss over sampled pairs:
```
L_dist = mean over sampled pairs (u,v), u<v of  ( d_G(x_u, x_v) / D_T[u,v]  - 1 )^2
```
- Sample a batch of node pairs each step (e.g. `pairs_per_step` random unordered pairs with `D_T > 0`), OR use all pairs if N is small (N ≤ ~150 → all pairs is cheap; make this automatic with a `full_pairs_threshold`).
- This directly targets our headline distortion metric.

### Mode B: `ranking` (Nickel–Kiela, for comparability)
For each true edge `(u, v) ∈ E`, sample `n_negatives` negative nodes `v'` that are NOT tree-neighbors of `u`. Loss:
```
L_rank = - sum over edges (u,v) of  log(  exp(-d_G(x_u,x_v)) / sum_{v' in {v} ∪ Neg(u)} exp(-d_G(x_u,x_v'))  )
```
- Implement the softmax in a numerically stable way (subtract max of the negative-distances logits, i.e. standard log-sum-exp trick).
- `n_negatives` configurable (default 10).

### Mode C: `hybrid`
```
L = L_dist + hybrid_lambda * L_rank
```
- `hybrid_lambda` configurable (default 1.0). Both sub-losses computed on their respective sampling schemes each step.

All three must be differentiable w.r.t. `X` and produce finite gradients (Stage 2 stability work ensures `dist` is NaN-safe; still, clamp `D_T` denominator to ≥1 since tree-distances are ≥1 by construction anyway).

---

## Task 1 — Config schema
Extend config (YAML, loadable via `src/config.py`) with a `train` section, e.g.:
```yaml
train:
  geometry: poincare        # poincare | euclidean
  dim: 5
  loss: distortion          # distortion | ranking | hybrid
  hybrid_lambda: 1.0
  n_negatives: 10
  pairs_per_step: 512
  full_pairs_threshold: 150 # if N <= this, use all pairs for distortion loss
  epochs: 500
  lr: 0.01
  optimizer: radam          # radam (RiemannianAdam) for poincare; adam for euclidean
  early_stop_patience: 50   # stop if val loss no-improve for this many epochs
  seed: 42
subtree: bfs_s50            # which subtree dir under data/subtrees/ to load
```

## Task 2 — Model / parameter setup
Function `build_embedding(N, dim, geometry) -> parameter`:
- Poincaré: `geoopt.ManifoldParameter(small_init, manifold=geoopt.PoincareBall())`, init points in a tiny ball (radius ~1e-3) so early training is stable near the origin.
- Euclidean: `nn.Parameter(torch.randn(N, dim) * 0.01)`.
Move to device from `get_device()`.

## Task 3 — Optimizer setup
Function `build_optimizer(param, cfg)`:
- Poincaré + `radam` → `geoopt.optim.RiemannianAdam([param], lr=cfg.lr)`.
- Euclidean + `adam` → `torch.optim.Adam([param], lr=cfg.lr)`.
- Assert compatible combos; error clearly otherwise.

## Task 4 — Loss module
Implement the three losses above as functions taking `(X, geom, D_T, edges, cfg, rng)` and returning a scalar loss tensor. Factor shared pieces (pair sampling, negative sampling) into helpers. Ensure log-sum-exp stability in ranking.

## Task 5 — Training loop
Function `train_embedding(subtree_name, cfg) -> Run`:
1. `set_seed(cfg.seed)`, pick device.
2. Load subtree via `src/data.py`; build `D_T`, `edges`, node-index map.
3. Build embedding + optimizer per geometry.
4. Hold out a small fraction of pairs as a **validation set** (e.g. 10% of pairs) used only to compute a val loss for early stopping — never trained on. (For ranking mode, hold out a fraction of edges.)
5. Start a `Run(config=cfg_as_dict, name=f"{cfg.train.geometry}_d{cfg.train.dim}_{cfg.train.loss}_{subtree_name}")`.
6. Loop epochs: compute train loss, `backward`, `optimizer.step()`. Every epoch (or every K steps) compute val loss. Track both curves.
7. Early stop on val-loss patience; keep best-val checkpoint (best coordinates).
8. On finish: `log_curve("train_loss", ...)`, `log_curve("val_loss", ...)`, `save_array("coords", best_coords.detach().cpu().numpy())`, `log_metrics({"final_train_loss":..., "best_val_loss":..., "epochs_ran":...})`, `run.finish()`.
   - NOTE: distortion/MAP metrics are NOT computed here (Stage 4). Only loss-based numbers.
9. Return the `Run`.

## Task 6 — CLI
`python -m src.trainer --subtree bfs_s50 --geometry poincare --dim 5 --loss distortion` runs one training and prints the run dir + final losses. All config keys overridable by flags; defaults from `configs/default.yaml`.

## Task 7 — Tests (`tests/test_trainer.py`)
Fast, tiny tests (use a hand-built 5-node tree like Stage 1's A/B/C/D/E, dim=2, few epochs):
- **Overfit sanity (Poincaré, distortion):** on the tiny tree with dim=2 and ~500 epochs, the train loss should drop substantially (assert final < 25% of initial). A tiny tree must be embeddable → loss should go low. This is the key "the trainer actually learns" test.
- **Euclidean parity:** same tiny tree, geometry=euclidean, runs without error and loss also decreases.
- **All three loss modes run** without NaN for a few epochs each (assert final loss is finite for distortion, ranking, hybrid).
- **No-NaN guarantee:** assert no parameter is NaN after training (guards the boundary-stability work).
- **Tracking integration:** assert a run dir was created with `coords.npy` artifact and a row appended to a temp `results.csv`.
- **Determinism:** same seed → same final loss (atol small).
Keep total test runtime modest (tiny tree, low epochs where possible; the overfit test can use more epochs but dim=2, N=5 is instant).

---

## Definition of Done — checklist
- [ ] `src/trainer.py` implements build_embedding, build_optimizer, the 3 losses, train loop, CLI.
- [ ] Poincaré uses geoopt `ManifoldParameter` + `RiemannianAdam`; Euclidean uses `nn.Parameter` + `Adam`.
- [ ] All three loss modes (`distortion`, `ranking`, `hybrid`) implemented per spec with stable log-sum-exp for ranking.
- [ ] Overfit-sanity test passes (loss drops to <25% of initial on the tiny tree, Poincaré distortion).
- [ ] Euclidean parity runs and learns.
- [ ] No NaN parameters after any run; determinism test passes.
- [ ] Each run writes coords.npy + curves + a results.csv row via `src/tracking.py`.
- [ ] `pytest tests/test_trainer.py` passes.

## Report back to the human
Report: (1) for the tiny 5-node tree, the initial vs final train loss for Poincaré-distortion (to confirm it learns), (2) confirmation all three loss modes ran finite, (3) a sample run directory listing showing `coords.npy` + curves, (4) the determinism check result, and (5) any geoopt API details you had to adapt (exact class/optimizer names/signatures used), since Stage 4 and later will build on this.
```
