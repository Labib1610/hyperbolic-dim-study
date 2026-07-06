# Stage 5 — Experiment 1: Dimension Sweep (Q1 efficiency, previews Q2)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet), as a function of hierarchy structure.**

Stages 0–4 + 2.5 are complete and tested: we can embed subtrees (Poincaré/Euclidean, 3 losses) and evaluate them (distortion, MAP, δ). Stage 4 gave a first result: on `bfs_s50` (shallow), Poincaré beat Euclidean on every metric (worst-distortion 9.3 vs 58.8, MAP 0.28 vs 0.15). Now we run the first **systematic experiment**.

This stage is an **experiment RUNNER + analysis**, not new mechanism. It orchestrates many train+evaluate runs across a grid, aggregates the master results table, and produces the first paper figure. **Do not modify geometry, trainer, or metrics logic** — only call them. If you find a bug in them, STOP and report rather than silently editing (we protect tested code).

### The scientific question (Q1)
Does hyperbolic geometry reach at LOW dimension the embedding quality that Euclidean needs HIGH dimension to match? Concretely: plot distortion (and MAP) vs dimension for both geometries; we expect Poincaré's curve to be far better at low dim and the gap to shrink as dimension grows. We ALSO preview Q2: does the Poincaré advantage widen on deeper subtrees?

---

## Experiment grid (exact)
- **Dimensions:** `{2, 8, 32, 128}`
- **Seeds:** `{0, 1, 2, 3, 4}` (5 seeds)
- **Subtrees (6, spanning structure):** `bfs_s50`, `bfs_s200`, `deep_s100`, `balanced_s100`, `deep_s200`, `balanced_s200`
- **Geometries:** `poincare`, `euclidean`
- **Loss:** `distortion` (our primary; ranking/hybrid explored in a later ablation, not here)
- Total = 4 × 5 × 6 × 2 = **240 runs**.
- Training config: reuse Stage 3 defaults (epochs, lr, early stopping). Keep epochs adequate (e.g. ≥500) so low-dim isn't under-trained; document the value used.

## Objective / Definition of Done
`src/experiments/exp1_dimension_sweep.py` runs the full grid (with res..resume support), every run recorded via tracking, then aggregates and produces figures + a tidy results file. All checklist items pass.

---

## Task 1 — Runner with resume
- Function `run_sweep(cfg)` iterating the grid. For each combo: build config, call the Stage 3 `train_embedding`, then Stage 4 `evaluate_run` on the resulting run dir (so metrics land in `results.csv`).
- **Resume support (important for a 240-run job):** before launching a combo, check whether a completed run with that exact (geometry, dim, subtree, seed, loss) already exists in `outputs/results.csv` with all metric columns populated; if so, skip. This lets the job be re-run after interruption without redoing work.
- Progress bar + periodic print (e.g. every 10 runs: how many done/remaining, elapsed, ETA).
- Wrap each run in try/except: on failure, log the combo + error to `outputs/exp1_failures.log` and CONTINUE (one bad run shouldn't kill 239 others). Report failures at the end.

## Task 2 — Tag runs for this experiment
- Add `experiment: "exp1_dimension_sweep"` into each run's config so we can filter later. Also tag `subtree_depth` and `subtree_avg_branching` (read from the subtree's stats JSON) INTO each run's config/row, so the results table carries structure info for Q2 analysis without a rejoin.

## Task 3 — Aggregation
- After the sweep, `aggregate_exp1()`:
  - Load `outputs/results.csv`, filter `experiment == exp1_dimension_sweep`.
  - Group by (subtree, geometry, dim); compute mean and std across the 5 seeds for: `distortion_worst`, `distortion_avg_scaled`, `map`, `delta_relative`.
  - Save tidy aggregated table to `outputs/exp1_aggregated.csv`.

## Task 4 — Figures (save PNG+PDF via tracking figure saver conventions, into `outputs/figures/exp1/`)
1. **Fig 1a — distortion vs dimension:** x=dim (log scale), y=`distortion_avg_scaled`, two lines (poincare/euclidean) with std error bands, one panel PER subtree (small-multiples grid of 6). 
2. **Fig 1b — MAP vs dimension:** same layout, y=MAP. (Confirms MAP climbs with dim — sanity that trainer is healthy and low-dim MAP was a dimension choice, not a bug.)
3. **Fig 1c — the "efficiency gap" summary:** for each subtree, compute the **dimension-savings**: the smallest Euclidean dim whose mean distortion_avg_scaled is ≤ the Poincaré distortion at dim=2 (i.e., "Euclidean needs dimension D to match Poincaré's dim-2 quality"). If Euclidean never matches within the grid, report ">128". Plot/tabulate this savings against subtree depth — THIS is the Q2 preview.
4. **Fig 1d — advantage vs structure:** x=subtree depth, y=(Euclidean − Poincaré) distortion_avg_scaled at dim=8 (fixed low dim), to see if the advantage widens with depth.

## Task 5 — Analysis report (`outputs/exp1_report.md`, auto-generated)
Auto-write a short markdown summarizing, with numbers pulled from the aggregated table:
- Does Poincaré beat Euclidean at every dim, or does Euclidean catch up — at what dim?
- Does the gap shrink as dim grows (expected)?
- Does the Poincaré advantage (at fixed low dim) widen with subtree depth (Q2 preview)?
- Does MAP climb to reasonable levels (>0.6?) at higher dim, confirming trainer health?
- Any subtree where Euclidean wins anything (report honestly).

## Task 6 — Tests (`tests/test_exp1.py`)
- **Grid-integrity test:** the grid enumerates exactly 240 combos with the specified values.
- **Resume test:** run a TINY fake sweep (2 dims × 1 seed × 1 subtree, few epochs) twice; second invocation skips already-completed combos (assert no new run dirs created on second call).
- **Aggregation test:** on a small synthetic results table, `aggregate_exp1` computes correct means/stds (hand-check one group).
- **Figure smoke test:** figures generate without error and files exist (don't assert on pixels).
Keep tests fast (tiny fake grid / synthetic tables — do NOT run the real 240 in tests).

---

## Definition of Done — checklist
- [ ] `src/experiments/exp1_dimension_sweep.py` runs the full 240-run grid with resume + failure isolation.
- [ ] Each run tagged `experiment=exp1_dimension_sweep` and carries subtree depth/branching in its row.
- [ ] `outputs/exp1_aggregated.csv` produced (mean/std over seeds).
- [ ] Figures 1a–1d saved as PNG+PDF under `outputs/figures/exp1/`.
- [ ] `outputs/exp1_report.md` auto-generated with the analysis bullets filled from real numbers.
- [ ] `pytest tests/test_exp1.py` passes (grid, resume, aggregation, figure smoke).
- [ ] Full run completed (or, if very slow, at least a documented subset completed and the rest resumable) — report actual completion status.

## Report back to the human
Report:
1. **How many of 240 runs completed**, total wall-clock time, any failures (from the log).
2. **The aggregated headline:** for each subtree, Poincaré vs Euclidean `distortion_avg_scaled` and `MAP` at dim 2 and dim 128 (a compact table).
3. **Q1 verdict:** does hyperbolic win big at low dim and does Euclidean catch up by dim 128? At roughly what dimension does Euclidean match Poincaré-at-dim-2 (the savings number), per subtree.
4. **Q2 preview:** does the savings / advantage grow with subtree depth? (This is the most important thing to look at.)
5. **Trainer-health:** does MAP reach >0.6 at high dim on at least some subtrees?
6. Paste `outputs/exp1_report.md`.
Do NOT hand-interpret beyond the numbers — just report them; we will interpret together.
```
