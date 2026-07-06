# Stage 4 — Evaluation Metrics (distortion, MAP, δ-hyperbolicity)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet), as a function of hierarchy structure.**

Stages 0–3 + 2.5 are complete: we can load subtrees, embed them (Poincaré or Euclidean, 3 loss modes), and every run saves `coords.npy` + records into `outputs/results.csv`. This stage implements the **evaluation metrics** that turn learned coordinates into the numbers our paper reports. **No training, no data sampling changes, no geometry changes here** — this stage consumes `coords.npy` + a subtree and outputs metric values.

### The three metrics and their roles (important framing)
- **Distortion** (primary, our headline axis): how faithfully geometric distances match tree-distances.
- **MAP** (mean average precision, primary for graph reconstruction / comparability to prior work): are each node's true tree-neighbors its nearest points?
- **δ-hyperbolicity** (DIAGNOSTIC, not the x-axis): computed on the *embedding's distance matrix*. A tree metric has δ=0; a Euclidean embedding distorts distances so its δ>0, while a hyperbolic embedding stays more tree-like (lower δ). We report δ as evidence that hyperbolic embeddings preserve tree-likeness — NOT as the independent variable. (The independent variables for the Q2 study are branching factor and depth, already computed in Stage 1.)

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active.
- Use `src/geometry.py` (`PoincareBall`, `EuclideanGeometry`) for distances, `src/data.py` for subtree + tree distances, `numpy`/`scipy`.

---

## Objective / Definition of Done
Implement `src/metrics.py` with functions to compute distortion (avg + worst-case), MAP, and δ-hyperbolicity (exact for small N, sampled for large N), all validated against **hand-computed ground-truth values listed below**. Provide an `evaluate_embedding(coords, subtree, geometry) -> dict` that returns all metrics, plus a CLI to evaluate a saved run. All checklist items pass; `pytest tests/test_metrics.py` passes.

---

## Task 1 — Pairwise geometric distance matrix
Function `pairwise_distances(coords, geometry) -> np.ndarray (N,N)`:
- Given coordinates (numpy or torch) and a geometry object, return the full symmetric matrix of geometric distances (Poincaré or Euclidean per the geometry).
- Must be finite everywhere (rely on Stage 2 clamping for Poincaré). Diagonal = 0.

## Task 2 — Distortion (primary)
Given the geometric distance matrix `Dg` and the tree-distance matrix `Dt` (from `src/data.py`), over all pairs u<v (Dt>0):

**Average distortion:**
```
D_avg = mean over pairs of  | Dg[u,v] - scale * Dt[u,v] | / (scale * Dt[u,v])
```
where `scale` is the single global constant that best aligns the two (fit `scale` by minimizing this, or use the closed-form that sets mean ratio to 1: `scale = mean(Dg/Dt)`; document the choice). Report both the raw version (scale=1) AND the scale-fitted version, since geometry embeddings are only defined up to scale.

**Worst-case (multiplicative) distortion:**
```
ratio[u,v] = Dg[u,v] / Dt[u,v]
D_worst = max(ratio) / min(ratio)     # over pairs with Dt>0
```
- `D_worst = 1.0` is perfect; larger is worse. (This matches the definition we derived by hand.)

## Task 3 — MAP (mean average precision for graph reconstruction)
For each node u, rank all other nodes by geometric distance ascending. u's "relevant" set = its true tree-neighbors (nodes at tree-distance 1, i.e. parent + children). Compute average precision for u:
```
AP(u) = mean over each relevant neighbor r of  (precision at the rank where r appears)
precision at rank k = (# relevant nodes within the k nearest) / k
```
MAP = mean of AP(u) over all nodes u that have ≥1 neighbor. `MAP = 1.0` means every node's true neighbors are its nearest points. (This is the standard Nickel–Kiela reconstruction metric; implement it to match that convention.)

## Task 4 — δ-hyperbolicity (diagnostic)
Operates on ANY distance matrix `D` (N,N) — we'll call it on the *embedding's* `Dg`.
Definition (four-point condition): for four points, form the three pairwise-sum matchings S1,S2,S3; sort desc; `δ_quad = (S_largest - S_middle)/2`. Global `δ = max over all quadruples`.

Implement two paths:
- `delta_hyperbolicity_exact(D)`: exact max over all quadruples. Only for small N (≤ ~60); O(N^4).
- `delta_hyperbolicity_sampled(D, n_samples, seed)`: sample `n_samples` random quadruples, return the max δ_quad found (an under-estimate that approaches true δ as samples grow). Use for large N.
- `delta_hyperbolicity(D, exact_threshold=60, ...)`: dispatch automatically.
- Optionally also report the **relative δ**: `δ / diam(D)` (diameter-normalized), which is scale-invariant and comparable across subtrees — report this too.

## Task 5 — Aggregate evaluator
`evaluate_embedding(coords, subtree, geometry, delta_samples=50000, seed=0) -> dict` returning:
`{ "distortion_avg", "distortion_avg_scaled", "distortion_worst", "map", "delta", "delta_relative", "scale", "num_nodes" }`.

## Task 6 — Evaluate a saved run (CLI + tracking integration)
- `evaluate_run(run_dir) -> dict`: load `coords.npy` from the run's artifacts, read its `config.yaml` to know geometry + subtree, recompute all metrics, and **write them back**: update that run's `metrics.json` AND update its row in `outputs/results.csv` (add the metric columns). Use the Stage 2.5 tracking helpers / atomic writers so this is safe.
- CLI: `python -m src.metrics --run <run_dir>` evaluates one run; `python -m src.metrics --all` evaluates every run under `outputs/runs/` that has a `coords.npy` and updates the master table. Print a summary.

---

## Task 7 — Hand-computed ground-truth tests (`tests/test_metrics.py`)
Use small, fully hand-checkable inputs. **These exact values were derived by hand and MUST match.**

**δ tests (the key correctness tests):**
- **T-δ-1 (tree → 0):** Build the 4-leaf tree metric from the project notes:
  points w,x,y,z with distances d(w,x)=2, d(y,z)=2, d(w,y)=3, d(w,z)=3, d(x,y)=3, d(x,z)=3.
  Assert `delta_hyperbolicity_exact(D) == 0` (atol 1e-9). (Sums are 4,6,6 → (6−6)/2 = 0.)
- **T-δ-2 (square → 0.414):** Four unit-square corners w,x,y,z with sides=1 and diagonals=√2.
  Assert `delta_hyperbolicity_exact(D) ≈ 0.41421` (atol 1e-4). (Sums 2, 2.828, 2 → (2.828−2)/2.)
- **T-δ-3 (exact == sampled on small):** on a random 12-point distance matrix, `delta_hyperbolicity_sampled` with all quadruples sampled equals `delta_hyperbolicity_exact` (atol 1e-9).

**Distortion tests:**
- **T-dist-1 (perfect embedding → distortion 0 / worst 1):** construct coords whose geometric distances are exactly proportional to tree distances (e.g. embed a path graph on a line with correct spacing); assert `D_avg (scaled) ≈ 0` and `D_worst ≈ 1.0` (atol 1e-6).
- **T-dist-2 (known imperfect):** the tiny A/B/C/D/E tree with a hand-chosen embedding where you can compute a couple of ratios by hand; assert `D_worst` equals the hand value (atol 1e-3). Document the arithmetic in a comment.

**MAP tests:**
- **T-map-1 (perfect → 1.0):** an embedding where every node's true tree-neighbors are strictly its nearest points → assert `MAP == 1.0`.
- **T-map-2 (one inversion):** perturb one node so a non-neighbor becomes nearer than a true neighbor; assert MAP drops below 1.0 to the specific hand-computed value (document the AP arithmetic for the affected node in a comment).

**Integration:**
- **T-int-1:** train a tiny embedding (reuse Stage 3 on the 5-node tree, few epochs, Poincaré, distortion), run `evaluate_embedding`, assert all metric keys present and finite, `MAP` in [0,1], `D_worst ≥ 1`.

---

## Definition of Done — checklist
- [ ] `src/metrics.py` implements pairwise_distances, distortion (avg raw+scaled, worst), MAP, δ (exact+sampled+dispatch+relative), evaluate_embedding, evaluate_run, CLI.
- [ ] T-δ-1 (tree→0) and T-δ-2 (square→0.41421) pass — confirms δ matches our hand math.
- [ ] exact==sampled δ test passes on small input.
- [ ] Distortion perfect-embedding test (worst→1.0) and known-imperfect test pass.
- [ ] MAP perfect→1.0 and one-inversion tests pass.
- [ ] `evaluate_run` writes metrics back into `metrics.json` and the master `results.csv` row.
- [ ] `python -m src.metrics --all` evaluates existing runs and updates the master table.
- [ ] `pytest tests/test_metrics.py` passes.

## Report back to the human
Report: (1) the δ values for the tree (should be 0) and the square (should be 0.41421) to confirm against hand math; (2) after running `--all` on the runs created in Stage 3, paste the resulting metric rows — specifically compare **Poincaré vs Euclidean distortion_worst, MAP, and delta_relative on `bfs_s50`** (this is our first real apples-to-apples geometry comparison and the first hint of whether the hypothesis holds); (3) any metric where Euclidean unexpectedly beats Poincaré (worth noting honestly); (4) confirmation all tests pass.
```
