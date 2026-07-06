# Stage 8.5 — λ Sweep: Lock Q3 + Distance-vs-Direction Analysis

## Context (read first)
Project: **hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet).**

Q3 (Stage 8) returned an honest NEGATIVE result: on HyperLex graded lexical entailment, Euclidean matches/beats Poincaré at every dim; the geometric advantage does NOT transfer downstream. The working signal is a *directional generality term* (Score B: distance + λ·(‖w1‖−‖w2‖)), not symmetric distance (Score A ≈ ρ0). BUT the λ tuning **saturated at the grid ceiling (λ=5)** — the direction term dominated and distance contributed ~nothing. Before we finalize Q3 for the paper, we must confirm the conclusion is stable across a FULL λ range, and turn this into a clean **distance-vs-direction analysis** that strengthens (not just patches) the finding.

**This is a SCORING-ONLY analysis — NO retraining.** The embeddings from Stage 8 are saved; λ only affects how we *score* HyperLex pairs from existing coordinates. Reuse the saved Stage-8 runs. Do NOT modify trainer/geometry/metrics. New code in `src/experiments/exp_q3_lambda.py`.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse saved Q3 embeddings (`experiment="q3_hyperlex"` runs / their `coords.npy`), the HyperLex loader + scoring from Stage 8, `scipy.stats`.

---

## Task 1 — Full λ sweep on existing embeddings
For each saved Q3 embedding (both geometries × dims {8,16,32,64,128} × 3 seeds):
- Recompute Score B ρ across a WIDE λ grid: `λ ∈ {0, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100}` (0 = pure distance = Score A; large = pure direction).
- Record Spearman ρ (and Pearson) vs HyperLex for every (geometry, dim, seed, λ).
- This is pure re-scoring: load coords, compute pair scores, correlate. Fast.

## Task 2 — The distance-vs-direction curve (the new finding)
- Aggregate: for each geometry × dim, plot **ρ vs λ** (mean±std over seeds). This shows how entailment prediction shifts from distance-dominated (λ=0) to direction-dominated (λ large).
- Report the key numbers:
  - ρ at λ=0 (pure distance) for both geometries — expected ≈ 0 (confirms symmetric distance carries almost no entailment signal).
  - ρ at the BEST λ for each geometry/dim — does it plateau, or keep rising past λ=5 (the old ceiling)?
  - The λ where ρ peaks — is it interior (a real optimum) or still at the grid edge (extend further if so)?
- **Stability check for the Q3 conclusion:** across the full λ range, does Euclidean still match/beat Poincaré at every dim? If yes → Q3 negative conclusion is robust. If the ranking flips at some λ → report exactly where and by how much (honest, important).

## Task 3 — Interpretation numbers for the paper
- Quantify "direction matters, curvature doesn't": at each geometry's best λ, the P-vs-E ρ gap. And the ratio of best-λ ρ to λ=0 ρ (how much of the signal is directional).
- Tie to the generality-radius result (Stage 8: depth↔norm ρ = P 0.735, E 0.75–0.88): does the geometry with the STRONGER generality-radius (Euclidean) also win entailment? (Supports "the generality signal, not curvature, drives downstream entailment.")

## Task 4 — Report + figures
- `outputs/q3_lambda_report.md`: the ρ-vs-λ table/curve, λ=0 vs best-λ numbers, the stability verdict (does Euclidean still win across all λ?), peak-λ location, and the direction-vs-curvature interpretation.
- Figures under `outputs/figures/q3_lambda/`: (i) ρ vs λ, one line per geometry (at a representative dim, e.g. 128, and a small-multiples over dims); (ii) best-λ ρ vs dim, P vs E (the locked headline).
- Tag `experiment="q3_lambda"`. Numbers auto-pulled.

## Tests (`tests/test_q3_lambda.py`, fast)
- Score B at λ=0 exactly equals Score A (hand-check on a tiny embedding).
- ρ-vs-λ helper returns the correct peak λ on a synthetic curve (hand-checked).
- Re-scoring is deterministic given fixed coords.
- Do NOT retrain in tests.

---

## Definition of Done — checklist
- [ ] `src/experiments/exp_q3_lambda.py` sweeps λ ∈ {0,...,100} on all saved Q3 embeddings (no retraining).
- [ ] ρ-vs-λ curves produced for both geometries × dims.
- [ ] λ=0 (pure distance) ρ reported ≈ 0 for both (confirms distance carries little entailment).
- [ ] Peak-λ identified; if still at grid edge, grid extended until interior peak or clearly plateaued.
- [ ] Stability verdict: does Euclidean match/beat Poincaré across the FULL λ range? (locks Q3)
- [ ] `outputs/q3_lambda_report.md` + figures generated.
- [ ] `pytest tests/test_q3_lambda.py` passes.

## Report back to the human
1. **λ=0 (pure distance) ρ** for P and E — is it ≈0 (distance carries little entailment signal)?
2. **Best-λ ρ** per geometry/dim, and WHERE the peak λ is (interior optimum, or did extending past 5 change things?).
3. **Stability verdict:** across the full λ range, does Euclidean still match/beat Poincaré at every dim? Any λ where Poincaré wins?
4. **Direction-vs-curvature:** how much of the entailment signal is directional (best-λ ρ vs λ=0 ρ), and does the stronger-generality-radius geometry (E) win — supporting "generality signal, not curvature, drives entailment"?
5. Paste `outputs/q3_lambda_report.md`.
Do NOT over-claim; report numbers. This locks Q3 so we can write it honestly.
```
