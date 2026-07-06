# Stage 5.5 — Diagnostic: Is the Poincaré Plateau Optimization or Real?

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies.**

Stage 5 (240-run dimension sweep) produced a surprising, hypothesis-contradicting result: on larger WordNet subtrees (N=100/200, which use **sampled pairs** in the loss), **Poincaré embeddings plateau** — their distortion/MAP stop improving past dim ~8–32, while **Euclidean keeps improving** and overtakes Poincaré at high dim. Yet the `delta_relative` diagnostic shows Poincaré embeddings ARE more tree-like at every dim. On the small tree (bfs_s50, N=50, **full pairs**) Poincaré behaves correctly throughout.

This diagnostic determines whether that plateau is:
- **(A) an optimization artifact** (Poincaré under-trained / boundary-saturated / hurt by pair-sampling), meaning our comparison is currently unfair, OR
- **(B) a real geometric outcome** at these scales.

**This is an INVESTIGATION, not a pipeline stage.** It must NOT modify `src/geometry.py`, `src/trainer.py`, or `src/metrics.py` (protected, tested code). It only ADDS a new script `src/diagnostics/diag_plateau.py` that calls existing code with different configs, plus a synthetic-tree generator in that same diagnostics file. If a probe reveals a real bug in the trainer, STOP and report it with evidence — do not fix it here; we will decide the fix together.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse Stage 1 (data/trees), Stage 3 (`train_embedding`), Stage 4 (`evaluate_embedding`/metrics), Stage 2.5 (tracking).

---

## The decisive question
On a tree we CONSTRUCT to be perfectly hyperbolic-embeddable, Poincaré MUST achieve near-zero distortion at low dimension (this is mathematical fact, e.g. Sarkar's construction). If our trainer cannot make Poincaré win there, the trainer is the problem — not the geometry. Probe 1 is decisive; Probes 2–4 localize the cause.

---

## Probe 1 — Synthetic perfect tree (DECISIVE)
Add `make_synthetic_tree(branching, depth)` to the diagnostics module: build a clean, complete `networkx` tree with the given branching factor and depth (a perfect b-ary tree). Return it as an undirected tree compatible with `src/data.py`'s distance/stats functions (same node-indexing convention so Stage 3/4 consume it unchanged).

Run, for a moderate synthetic tree (e.g. branching=2, depth=6 → 127 nodes; and branching=3, depth=4 → 121 nodes, comparable to our N~100 WordNet subtrees):
- Train BOTH geometries across dims {2, 8, 32, 128}, loss=distortion, 3 seeds, using the SAME default training config as Stage 5.
- Evaluate distortion_avg_scaled, distortion_worst, MAP.

**Interpretation to report (do not spin):**
- If Poincaré achieves low distortion (e.g. distortion_avg_scaled well below Euclidean at low dim, and does NOT plateau above Euclidean at high dim) → **trainer is healthy; the WordNet plateau is structural, a real finding about WordNet.**
- If Poincaré ALSO plateaus / loses on the *synthetic perfect tree* → **trainer/optimization is the culprit; Stage 5's geometry comparison is unfair and must be redone after a fix.**

## Probe 2 — Full-pairs vs sampled-pairs on a larger tree
The plateau coincides with the sampled-pairs regime (N>full_pairs_threshold). On ONE problematic WordNet subtree (`deep_s200`):
- Run Poincaré at dims {8, 32, 128}, 3 seeds, TWICE: once with pair-sampling as-is, once forcing **full pairs** (temporarily raise `full_pairs_threshold` above N via config only — NOT a code edit; if the trainer can't be forced to full-pairs by config alone, report that as a limitation rather than editing the trainer).
- Compare: does full-pairs remove the plateau (Poincaré keeps improving with dim)?

**Report:** whether full-pairs materially improves Poincaré's high-dim distortion/MAP. If yes → sampling is a major cause.

## Probe 3 — Boundary saturation check
For the Poincaré runs on `deep_s200` at dim {8, 128} (reuse saved `coords.npy` from Stage 5 if available, else retrain 1 seed):
- Compute the distribution of point norms ‖x‖: report min, median, max, and the fraction with ‖x‖ > 0.99 and > 0.999.
- Also report how many points sit within `EPS` of the projection boundary.

**Interpretation:** a large fraction of points crushed at ‖x‖≈1 indicates boundary saturation — a classic Poincaré-SGD failure where gradients vanish and points can't rearrange. If present → optimization artifact, fixable via lr / init scale / learning-rate warmup / curvature scaling.

## Probe 4 — Optimization knob sweep (does Poincaré improve with better training?)
On `deep_s200`, Poincaré, dim=128 (the worst plateau case), 3 seeds each, vary ONE knob at a time from the Stage 5 default (all via config/CLI, no code edits):
- epochs: default vs 3× more (with early-stopping patience raised proportionally).
- lr: default vs {0.5×, 2×, 5×}.
- pairs_per_step: default vs 4× more.
- init radius: default (1e-3) vs smaller (1e-5) — sometimes tiny init helps Poincaré.
Report the best distortion_avg_scaled / MAP achieved for Poincaré under any knob setting, vs the Stage 5 baseline and vs Euclidean@dim128.

**Interpretation:** if any knob lets Poincaré close most of the gap to Euclidean at dim 128 → the Stage 5 result was an under-training artifact.

---

## Task — Orchestration, tracking, report
- Put everything in `src/diagnostics/diag_plateau.py` with a CLI: `--probe {1,2,3,4,all}`.
- Every training run recorded via tracking, tagged `experiment="diag_plateau"` + which probe.
- Save a consolidated `outputs/diag_plateau_report.md` auto-filled with the numbers from each probe and a one-line verdict per probe.
- Save any figures (e.g. Probe 1 distortion-vs-dim for synthetic tree; Probe 3 norm histograms) under `outputs/figures/diag/`.

## Tests (`tests/test_diag.py`, keep fast)
- `make_synthetic_tree(2, 3)` returns a valid tree: 15 nodes, is connected, edges == nodes−1, depth == 3, every internal node has exactly 2 children.
- Synthetic tree integrates with `src/data.py` distance functions (tree_distance root-to-leaf == depth).
- A tiny end-to-end: synthetic tree (branching2 depth3), Poincaré dim2, few epochs, `evaluate_embedding` returns finite metrics.
- Do NOT run the full probe sweeps in tests.

---

## Definition of Done — checklist
- [ ] `src/diagnostics/diag_plateau.py` added; no edits to geometry/trainer/metrics.
- [ ] Probe 1 (synthetic perfect tree) run for both geometries across dims; verdict reported.
- [ ] Probe 2 (full vs sampled pairs on deep_s200) run; verdict reported.
- [ ] Probe 3 (boundary norm distribution) computed and reported.
- [ ] Probe 4 (optimization knob sweep) run; best Poincaré vs baseline vs Euclidean reported.
- [ ] `outputs/diag_plateau_report.md` auto-generated with per-probe numbers + verdicts.
- [ ] `pytest tests/test_diag.py` passes.

## Report back to the human (MOST IMPORTANT PART)
Report, per probe, with numbers pulled from the saved report:
1. **Probe 1 verdict (decisive):** On the synthetic perfect tree, does Poincaré win at low dim and avoid the high-dim plateau? Give distortion_avg_scaled for P vs E at dim 2, 8, 128. State plainly: does this point to (A) trainer-healthy/structural-finding or (B) optimization-artifact?
2. **Probe 2:** Does full-pairs remove the plateau on deep_s200? Numbers before/after.
3. **Probe 3:** What fraction of Poincaré points are at ‖x‖>0.99 / >0.999? Is boundary saturation present?
4. **Probe 4:** Best Poincaré dim-128 distortion/MAP under any knob vs Stage 5 baseline vs Euclidean. Did better training close the gap?
5. **Overall:** which hypothesis (A optimization-artifact vs B real-result) does the combined evidence support? Do NOT over-claim — if mixed, say so. We interpret and decide the fix together.
```
