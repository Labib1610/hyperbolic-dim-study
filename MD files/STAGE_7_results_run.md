# Stage 7 — The Fair Results Run (Q1 efficiency + Q2 structure law)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet).**

The trainer is now FAIR and verified (Stage 6): learnable global α objective (geometry-agnostic) + R2 convergence stopping (patience 500, min_delta 0.5%, cap 8000), applied identically to both geometries. Sandbox and port agree; failure mode preserved. This is the **real results run** that produces the paper's numbers. Everything before was earning the right to run this fairly.

**Two claims this run must support:**
- **Q1 (efficiency):** hyperbolic reaches at LOW dimension the quality Euclidean needs HIGH dimension for; the advantage is largest at low dim and shrinks as dim grows.
- **Q2 (structure law — our NOVEL core):** the size of the hyperbolic advantage is a PREDICTABLE FUNCTION of the hierarchy's structure (depth / branching / irregularity). More tree-like, deeper hierarchies → larger hyperbolic advantage.

This is an **experiment RUNNER + analysis**. Do NOT modify `trainer.py`/`geometry.py`/`metrics.py` (now all verified). Only orchestrate + aggregate + plot. If a bug surfaces, STOP and report.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Use the production trainer defaults (learnable-α + R2). Reuse Stage 1 subtree tooling, Stage 4 metrics, Stage 2.5 tracking, Stage 5's runner patterns (resume, failure isolation, aggregation).

---

## Task 0 — Build the expanded subtree suite (structure spread across roots)
Extend/verify the Stage-1 sampler to produce **~18 subtrees across 3 WordNet roots** for structural diversity:
- Roots: `animal.n.01`, `plant.n.02`, `artifact.n.01` (fall back to another populous noun root if any is too small; document choices).
- Per root, sample ~6 subtrees spanning structure: mix of modes {bfs (shallow/broad), deep (deep/narrow), balanced} × sizes {100, 200} (adjust to hit ~6/root).
- Save under `data/subtrees/` with the Stage-1 stats sidecars. Regenerate `summary.csv` including every subtree's depth, avg_branching, max_branching, num_leaves, diameter, and (NEW) a structural irregularity scalar `degree_cv` (coefficient of variation of node degree — from Stage 5.7).
- **Report the structural spread:** min/max of depth, avg_branching, degree_cv across the 18 — we NEED wide spread for Q2. If spread is narrow, adjust sampling before the big run.

## Task 1 — Sanity run FIRST (catch bugs before the overnight job)
Before the full grid: run 2 subtrees (one shallow, one deep) × dims {2, 8, 128} × 2 geometries × 2 seeds with the production trainer. Confirm:
- Poincaré recovers (low-dim advantage present; e.g. shallow-tree dim2 P MAP ≫ E MAP), matching the Stage-5.8/6 ballpark.
- Runtimes → project the FULL-grid ETA. **Print the projected total wall-clock.** If projected > ~30h, STOP and report so we can trim (fewer seeds/dims) before committing.

## Task 2 — The full grid (with resume + failure isolation)
Grid:
- **Subtrees:** all ~18.
- **Dims:** `{2, 4, 8, 16, 32, 64, 128}` (7).
- **Seeds:** `{0,1,2,3,4}` (5).
- **Geometries:** `poincare`, `euclidean`.
- Total ≈ 18 × 7 × 5 × 2 = **~1260 runs**.
- Trainer: production defaults (learnable-α + R2, cap 8000).
- Tag every run `experiment="exp_final"` and carry subtree `depth`, `avg_branching`, `degree_cv`, `root` into each row.
- **Resume:** skip combos already completed with metrics in `results.csv`. **Failure isolation:** log + continue on any single-run error (`outputs/exp_final_failures.log`). Progress + ETA every N runs.

## Task 3 — Aggregate
- `aggregate_final()`: filter `experiment=exp_final`; group by (subtree, geometry, dim); mean+std over seeds for `distortion_avg_scaled`, `distortion_worst`, `map`, `delta_relative`, and record `alpha` mean, `epochs_to_converge` mean.
- Save `outputs/exp_final_aggregated.csv`.

## Task 4 — Q1 figures & numbers
- **Fig Q1a:** distortion_avg_scaled vs dim (log x), P vs E, std bands, small-multiples over subtrees (or a representative subset + an averaged panel).
- **Fig Q1b:** MAP vs dim, same layout.
- **Q1 headline metric — dimension savings:** for each subtree, the smallest Euclidean dim whose mean MAP ≥ Poincaré's MAP at dim 2 (and separately at dim 4). Tabulate. Also the inverse: at fixed low dim (2, 4, 8), the P−E MAP advantage.
- **Convergence-fairness panel:** report mean epochs_to_converge for P vs E (shows P needs more epochs — the fairness story) so the paper can preempt the "did you train both fairly?" question.

## Task 5 — Q2 figures & numbers (THE NOVEL CORE)
- Define the **hyperbolic advantage** per subtree at a fixed low dim (dim 8 primary; also dim 4): `adv = (P MAP − E MAP)` and a distortion version `(E dist − P dist)`.
- **Fig Q2a:** scatter of `adv` vs subtree **depth**, colored by root, with a fitted trend + correlation (Pearson & Spearman).
- **Fig Q2b:** `adv` vs **degree_cv** (irregularity) and vs **avg_branching** — whichever structural axis predicts best.
- **Report which structural variable predicts the advantage best** (depth vs branching vs degree_cv), with correlation coefficients and a simple linear fit. THIS is the Q2 law. Honest reporting: if no variable predicts well, say so (still a finding — "advantage is roughly structure-independent within this range").
- Multi-root check: does the trend hold WITHIN each root and ACROSS roots (not just an artifact of one taxonomy)?

## Task 6 — Auto-report
`outputs/exp_final_report.md` with: subtree suite summary + structural spread; Q1 verdict (savings numbers, low-dim advantage, does E catch up by dim 128, convergence-epochs fairness note); Q2 verdict (which structural variable predicts advantage, correlations, within/across-root robustness); and an honest "where Euclidean wins / ties" section. Numbers pulled from the aggregated CSV — nothing hand-typed.

## Tests (`tests/test_exp_final.py`, fast — do NOT run the real grid)
- Grid enumerates the expected combo count.
- Resume: tiny fake grid run twice → second run skips completed.
- Aggregation: synthetic results table → correct group means/stds (hand-check one).
- Q2 helper: given a synthetic (structure, advantage) table with a known correlation, the reported Pearson matches (hand-check).
- Figures generate without error.

---

## Definition of Done — checklist
- [ ] ~18 subtrees across 3 roots built; structural spread reported (wide depth/branching/degree_cv range).
- [ ] Sanity run done; full-grid ETA projected and reported BEFORE the big run.
- [ ] Full grid (~1260 runs) completed (or documented partial + resumable), resume + failure isolation working.
- [ ] `outputs/exp_final_aggregated.csv` produced.
- [ ] Q1 figures (a,b) + dimension-savings table + convergence-fairness note.
- [ ] Q2 figures (a,b) + best-predictor structural variable with correlations + within/across-root check.
- [ ] `outputs/exp_final_report.md` auto-generated from real numbers.
- [ ] `pytest tests/test_exp_final.py` passes.

## Report back to the human
1. **Subtree suite:** the structural spread (depth/branching/degree_cv min–max) across the 18 — is it wide enough for Q2?
2. **Sanity + ETA:** projected full-grid wall-clock; did the sanity run reproduce the low-dim Poincaré advantage?
3. **Completion:** how many of ~1260 runs finished; failures.
4. **Q1 verdict:** dimension-savings per subtree (Euclidean dim to match Poincaré@dim2/4); low-dim P−E advantage; does Euclidean catch up by dim128? Mean epochs_to_converge P vs E (fairness).
5. **Q2 verdict (most important):** which structural variable predicts the hyperbolic advantage best, with Pearson/Spearman; does the trend hold within AND across roots? Paste the Q2 table.
6. Paste `outputs/exp_final_report.md`.
Do NOT over-interpret — report numbers; we interpret together and decide how the paper frames Q1/Q2.
```
