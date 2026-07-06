# Stage 6 — Port: Learnable-α Objective + R2 Stopping into the Production Trainer

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies.**

After a long, careful diagnostic arc (Stages 5, 5.5–5.8) we established the correct, FAIR training method:
- **Objective:** learnable **global** scale α — the loss fits `d_G ≈ α · D_T` with α a learned positive scalar (softplus-parameterized), applied identically to BOTH geometries (it rescales target tree-distances into a range the geometry can represent; it is geometry-agnostic). This fixed the Poincaré failure where fixed scale=1 demanded infeasible near-boundary distances.
- **Stopping rule R2:** convergence early-stopping — patience = 500 epochs on validation distortion, min_delta = 0.5% relative, cap = 8000 epochs. Geometry-agnostic; lets both geometries fully converge; avoids the overfitting that a fixed 8000-epoch budget induces in Euclidean.

Proven results at fair convergence (from Stage 5.8, our ground-truth to reproduce): Poincaré wins 5/6 (R2) with a large low-dim margin (e.g. `bfs_s50` dim8: P MAP 0.790 vs E 0.381; dim128: P 1.000 vs E 0.116). Riemannian optimization converges SLOWER (Poincaré ~4000–7700 epochs vs Euclidean ~300–1400) but reaches a better optimum.

**This stage ports that method into `src/trainer.py`** — the first deliberate modification of previously-protected code. Because everything downstream depends on it, the port MUST reproduce the sandbox (`src/diagnostics/fix_prototype.py` / `fairness_check.py`) numbers on known cases before we trust it. Do NOT change `src/geometry.py` or `src/metrics.py`.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. The sandbox implementations in `src/diagnostics/` are the reference; port their logic into the production trainer cleanly (don't just import sandbox code — make it first-class in `trainer.py`).

---

## Task 1 — Extend the trainer config
Add to the `train` config (keep old keys working — backward compatible):
```yaml
train:
  # ... existing keys ...
  scale: learnable        # learnable | fixed   (default: learnable; 'fixed' = old scale=1 behavior)
  stopping: convergence   # convergence | fixed_epochs   (default: convergence = R2)
  # R2 params:
  patience_epochs: 500
  min_delta_rel: 0.005
  epoch_cap: 8000
  # fixed-epochs fallback:
  fixed_epochs: 8000
```
- `scale: fixed` + `stopping: fixed_epochs` with a small epoch count must reproduce the OLD (pre-fix) behavior exactly — this preserves the ability to demonstrate the failure mode for the paper.
- Defaults (`scale: learnable`, `stopping: convergence`) give the correct method.

## Task 2 — Learnable-α objective (both geometries)
- Add a learnable scalar `alpha` (parameterize as `alpha = softplus(raw_alpha)`, `raw_alpha` a trainable `nn.Parameter`, init so alpha≈1) to the optimization, for BOTH poincare and euclidean.
- The distortion loss becomes `mean( (d_G / (alpha * D_T) - 1)^2 )` over sampled/all pairs. (Ranking/hybrid losses: apply alpha consistently if used; for this project's main path loss=distortion.)
- `alpha` is optimized jointly with coordinates. For Poincaré, coordinates use RiemannianAdam; `raw_alpha` is a Euclidean parameter — ensure the optimizer setup handles a mixed parameter group (RiemannianAdam can hold Euclidean params too, or use a second Adam for alpha; pick one, document it, and make sure alpha actually receives gradients).
- Record final learned `alpha` in the run's metrics.

## Task 3 — R2 convergence stopping
- Implement convergence early-stopping on the VALIDATION distortion (the held-out-pairs val loss the trainer already computes):
  - Track best val distortion; if no relative improvement ≥ `min_delta_rel` for `patience_epochs` consecutive epochs, stop.
  - Hard cap at `epoch_cap`.
  - Keep the best-val checkpoint (as the trainer already does).
- `stopping: fixed_epochs` mode: train exactly `fixed_epochs`, no early stop (for controlled experiments / failure-mode reproduction).
- Record `epochs_ran` and `epochs_to_converge` (the best-val epoch) in metrics.

## Task 4 — Preserve determinism + tracking
- Same seeding discipline as before (torch CPU+CUDA + isolated RNG for sampling). Same tracking integration (coords.npy, curves incl. val, metrics incl. alpha + epochs_to_converge, results.csv row).
- The new config keys flow into the run config (Stage 2.5 schema-union safety already handles new columns).

## Task 5 — VERIFY against sandbox ground-truth (the critical part)
Add `tests/test_trainer_port.py` and a verification script `src/verify_port.py`.

**Ground-truth reproduction (must match sandbox within tolerance):**
On the synthetic oracle `make_synthetic_tree(2, 6)` (127-node perfect binary tree) and on `deep_s200`, with `scale=learnable, stopping=convergence`, Poincaré, seed 0:
- Assert the ported trainer's converged MAP / distortion match the Stage-5.6/5.8 sandbox numbers within a reasonable tolerance (atol ~0.05 on MAP, given seed/impl differences — document the exact sandbox reference values used).
- **Oracle low-dim sanity (the decisive one):** ported Poincaré at dim 2 on the perfect tree must achieve MAP ≳ 0.4 and clearly beat a `scale=fixed` run on the same tree (reproduces the fix working). 
- **Failure-mode reproduction:** `scale=fixed, stopping=fixed_epochs=500` on the oracle reproduces the BROKEN behavior (Poincaré MAP ≲ 0.1) — confirms we can still demonstrate the failure for the paper.
- **alpha sanity:** learned alpha is finite, positive, and (on deep trees) < 1, matching the interpretable pattern (alpha shrinks for deeper/larger-distance trees).

**Fast unit tests:**
- alpha receives gradients and moves from init.
- convergence-stopping triggers on a synthetic plateauing val curve at the right epoch (hand-checked).
- backward-compat: `scale=fixed, stopping=fixed_epochs` reproduces old trainer loss trajectory on the tiny 5-node tree (within tolerance).
- determinism: same seed → same converged metrics.

## Task 6 — Regression: existing tests still pass
- Run the FULL existing suite. All previously-passing tests (geometry, data, tracking, metrics, and the original trainer overfit-sanity) must STILL pass. If the port changes default behavior in a way that breaks an old test, update the test intentionally and NOTE it in the report (don't silently change tested behavior).

---

## Definition of Done — checklist
- [ ] `src/trainer.py` supports `scale` (learnable|fixed) and `stopping` (convergence|fixed_epochs), defaults = learnable + convergence.
- [ ] Learnable-α implemented for BOTH geometries; alpha receives gradients; recorded in metrics.
- [ ] R2 convergence stopping implemented with patience/min_delta/cap; epochs_to_converge recorded.
- [ ] Old behavior reproducible via `scale=fixed, stopping=fixed_epochs` (failure-mode preserved).
- [ ] `src/verify_port.py` reproduces sandbox ground-truth (oracle recovery + deep_s200) within tolerance.
- [ ] Oracle dim-2 sanity passes (learnable beats fixed, MAP recovered).
- [ ] Failure-mode reproduction passes (fixed-scale short-epoch → broken Poincaré).
- [ ] `pytest` FULL suite passes (regression-clean); any intentional test updates noted.

## Report back to the human
1. **Port verification:** ported-trainer converged MAP/distortion for Poincaré on the oracle (dim 2, 8) and deep_s200 (dim 128) vs the sandbox reference numbers — do they match within tolerance?
2. **Failure-mode still reproducible?** `scale=fixed, stopping=fixed_epochs=500` on the oracle → Poincaré MAP (should be low, ~broken).
3. **alpha values** learned on the oracle vs deep_s200 (confirm the interpretable shrink-with-depth pattern).
4. **Regression:** full suite pass count; any tests intentionally updated (and why).
5. Confirm the default path is learnable-α + R2, and the config to use for the fair Stage-5 rerun.
Do NOT start the full rerun in this stage — that's the next MD. This stage only ports + verifies.
```
