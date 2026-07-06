# Stage 5.6 — Fix Prototype: Recover Poincaré on the Oracle (sandbox)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies.**

Stage 5.5 diagnosed the Poincaré plateau as a **training defect**, not a geometric result. Root cause (well-evidenced): with the distortion loss targeting `d_G = D_T` (tree distances up to ~12) and tiny origin init (`uniform(-1e-3, 1e-3)`), Poincaré points stall at norm ≤ ~0.88 and never migrate outward toward the boundary where deep-tree leaves must live. Two points at norm 0.88 are only ~2.8 apart in hyperbolic distance, so the model **physically cannot represent** the required distances. Loss floors ~0.40, MAP collapses to ~random. This is the classic naive-Poincaré-embedding failure. On a **synthetic perfect tree** (Sarkar's construction guarantees near-zero-distortion hyperbolic embeddability), Poincaré should trivially win — but currently fails. That perfect tree is our **oracle**: a fix is validated iff Poincaré recovers on it.

**This is a SANDBOX prototype. Do NOT edit `src/trainer.py`, `src/geometry.py`, or `src/metrics.py`.** Implement all fixes inside a NEW file `src/diagnostics/fix_prototype.py` that defines its own small training routine (it may copy/adapt the trainer's loop internally — that's fine in the sandbox). Once we identify the winning fix from this stage's results, a LATER stage will port it into `src/trainer.py` cleanly. Reuse `src/geometry.py` (PoincareBall/Euclidean) and `src/metrics.py` (evaluate) unchanged.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse geometry, metrics, tracking, and the `make_synthetic_tree` from Stage 5.5 diagnostics.

---

## The three fixes to test (implement each as a toggle)

### Fix 1 — Spread-out initialization
Instead of `uniform(-1e-3, 1e-3)`, initialize points spread across radii. Provide options:
- `init="origin"` (baseline, ±1e-3),
- `init="uniform_ball"` (sample directions uniformly, radii uniform in e.g. [0.1, 0.8], then project),
- `init="scaled_randn"` (randn × 0.1, projected).
Mechanism: gives points room to separate so outward gradients have structure.

### Fix 2 — Learning-rate burn-in
- `burnin_epochs` (e.g. 0 baseline vs 50), during which lr = `lr * burnin_factor` (e.g. 0.1×), then restore full lr.
Mechanism: lets points migrate outward stably before fine-tuning (Nickel–Kiela recipe).

### Fix 3 — Learnable scale / normalized target (dark-horse; may be the better method)
The loss currently fixes scale=1: `(d_G / D_T - 1)^2`. Two variants:
- `scale="learnable"`: introduce a single positive scalar parameter `alpha` (softplus-parameterized), target becomes matching `d_G ≈ alpha * D_T`; optimize `alpha` jointly. Loss: `(d_G / (alpha * D_T) - 1)^2`.
- `target_norm="normalize"`: rescale tree distances into a comfortable range (e.g. divide D_T by its max, times a constant like 3) BEFORE fitting, so required geometric distances don't demand norms ~0.9997.
Mechanism: geometry needn't jam points against the rim to express large distances → better-posed objective.

All three must be independently toggleable AND combinable.

---

## Task 1 — Sandbox trainer
In `src/diagnostics/fix_prototype.py`, implement `train_poincare_fixed(tree, dim, cfg, seed)` that:
- builds a Poincaré `geoopt.ManifoldParameter` with the chosen `init`,
- optionally adds a learnable `alpha` parameter (Fix 3) to the optimizer,
- runs the distortion loss (with scale handling per Fix 3) using the SAME geometry.dist as production,
- applies burn-in lr schedule per Fix 2,
- returns learned coords (+ alpha if used) and loss curve.
Also provide `train_euclidean_ref(...)` (reuse production trainer or a thin copy) for the reference line.

## Task 2 — Oracle experiment (the validation)
On the synthetic perfect tree **b=2, depth=6 (127 nodes)** — the exact oracle Stage 5.5 used — run:
- **Baseline** (init=origin, no burn-in, scale=1): reproduce the failure (expect P MAP ~0.04, dist_scaled ~0.43).
- **Each fix alone:** Fix1(uniform_ball), Fix1(scaled_randn), Fix2(burnin), Fix3(learnable), Fix3(normalize).
- **Combinations:** best-init + burnin; best-init + Fix3; all-three.
- dims {2, 8, 32}, 3 seeds each. (dim 2 matters most: Sarkar says a perfect binary tree embeds in 2D hyperbolic with near-zero distortion — Poincaré SHOULD shine at dim 2, which is the cleanest proof of recovery.)
- Evaluate distortion_avg_scaled, distortion_worst, MAP for each.
- Include the Euclidean reference at the same dims.

**Success criterion (state explicitly in report):** a fix "recovers" Poincaré if, on this oracle, Poincaré reaches **low distortion and high MAP at LOW dim (esp. dim 2)** — ideally MAP ≳ 0.8 and dist_scaled well below the 0.43 baseline — and no longer plateaus below Euclidean. The strongest possible evidence is Poincaré at dim 2 BEATING Euclidean at dim 2 (that's the whole point of hyperbolic geometry for trees).

## Task 3 — Point-norm sanity
For the winning configuration, report the learned point-norm distribution on the oracle (min/median/max, frac>0.9, frac>0.99). We EXPECT a healthy fix to push some points out toward the boundary (norms approaching but not pinned at 1) — this directly confirms the mechanism (points can now migrate outward). Contrast with the baseline's stuck-at-0.88.

## Task 4 — Confirm on ONE real WordNet subtree
Take `deep_s200` (a worst plateau case from Stage 5). Run baseline vs the winning fix, Poincaré, dims {8, 32, 128}, 3 seeds. Report whether the fix improves Poincaré distortion/MAP toward (or past) Euclidean. (This checks the fix generalizes from synthetic to real data before we port it.)

## Task 5 — Report + figures
- Auto-write `outputs/fix_prototype_report.md` with: the full oracle table (all fixes × dims), the winner, the norm distributions (baseline vs winner), and the deep_s200 confirmation.
- Figure: distortion_scaled & MAP vs dim on the oracle, one line per fix + Euclidean ref, saved to `outputs/figures/fix/`.
- Tag all runs `experiment="fix_prototype"` via tracking.

## Tests (`tests/test_fix_prototype.py`, fast)
- The three init modes produce points with the expected norm ranges (origin: tiny; uniform_ball: spread in [0.1,0.8]; all inside the ball).
- Fix 3 learnable `alpha` is a positive scalar that receives gradients (assert it changes after a few steps).
- A tiny end-to-end: oracle b2d3 (15 nodes), Poincaré dim2, winning-fix config, few epochs → runs, finite metrics, and beats the origin-baseline MAP on that tiny tree (sanity that the fix helps even at toy scale).
- Do NOT run the full oracle sweep in tests.

---

## Definition of Done — checklist
- [ ] `src/diagnostics/fix_prototype.py` implements all 3 fixes (toggleable + combinable) in a sandbox trainer; no edits to protected code.
- [ ] Oracle experiment run: baseline reproduces failure; each fix + combinations evaluated at dims {2,8,32}, 3 seeds.
- [ ] Success criterion explicitly evaluated: does any config make Poincaré win at LOW dim on the oracle (esp. dim 2, MAP ≳ 0.8)?
- [ ] Point-norm distribution reported for the winner (shows outward migration).
- [ ] deep_s200 confirmation run: does the fix help on real data?
- [ ] `outputs/fix_prototype_report.md` + figures generated.
- [ ] `pytest tests/test_fix_prototype.py` passes.

## Report back to the human (MOST IMPORTANT)
1. **Baseline reproduces failure?** (P MAP/dist on oracle at dim 2/8 ≈ the Stage 5.5 crippled numbers.)
2. **Winner:** which fix (or combination) recovers Poincaré on the oracle? Give the oracle table for baseline vs each fix at dim 2, 8, 32 (dist_scaled + MAP).
3. **The money question:** does the winning fix make **Poincaré at dim 2 beat Euclidean at dim 2** on the perfect tree? (If yes → the geometry advantage is real and was purely masked by the training bug.)
4. **Norm migration:** do the winner's points now spread outward (max norm, frac>0.9) vs the stuck-at-0.88 baseline?
5. **Real-data check:** on deep_s200, does the fix move Poincaré toward/past Euclidean?
6. Which fix mattered most, and is Fix 3 (learnable scale) part of the winner — since if so it may become part of our published method, not just a bug fix.
Do NOT over-claim; report numbers, we decide together whether to port it and re-run Stage 5.
```
