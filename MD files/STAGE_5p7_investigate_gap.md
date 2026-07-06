# Stage 5.7 — Investigation: Real-Data Gap — Under-training or Real Structure?

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies.**

Stage 5.6 result: the **learnable-scale α** fix decisively recovers Poincaré on a *synthetic perfect tree* — Poincaré@dim2 beats Euclidean@dim2 (MAP 0.479 vs 0.100), proving the hyperbolic advantage is real and Stage 5's "Euclidean wins" was a training artifact. Root cause confirmed: the fixed scale=1 distortion target demands infeasible large geometric distances; a learned α (~0.2–0.33) rescales targets into a reachable range while preserving ratios. **Fix 3 (learnable scale) did essentially all the work**; init/burn-in were minor.

BUT on **real WordNet `deep_s200`**, the fix helps a lot (dim128 MAP 0.23→0.55) yet does NOT close the gap: Euclidean still leads at dim≥32 (dim128 E MAP 0.975 vs P 0.545). This investigation answers ONE question decisively:

> **Is the residual real-data Poincaré gap due to (A) remaining under-training, or (B) genuine structure — real WordNet being less tree-like than a perfect tree?**

If (A): finish the fix, then port + fair Stage-5 rerun. If (B): that IS our finding — "hyperbolic's advantage erodes predictably with deviation from tree-likeness" (our Q2, reborn with a mechanism). This investigation also validates whether the **learnable-scale objective is stable/principled enough to be part of our published method.**

**Still a SANDBOX investigation.** Work in `src/diagnostics/investigate_gap.py`. Reuse the Stage 5.6 sandbox trainer (`fix_prototype.py`) and `metrics.py`, `data.py`. Do NOT edit `src/trainer.py`, `src/geometry.py`, `src/metrics.py`. If a probe implies a trainer change, report it — we port later, once.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse `train_poincare_fixed`, `train_euclidean_ref`, `make_synthetic_tree`, `evaluate_embedding`, δ from `metrics.py`, subtrees from `data.py`.

---

## Probe A — Under-training probes (push Poincaré as hard as is FAIR)
On `deep_s200`, Poincaré with the Stage-5.6 winner (uniform_ball init + learnable α), dim {32, 128}, 3 seeds, vary ONE thing at a time from the winner baseline:

A1. **Many more epochs:** winner epochs vs 5× and 10× (raise early-stop patience proportionally). Does MAP/dist keep improving toward Euclidean, or has it truly converged?
A2. **Richer scale parameterization** (the key suspect — WordNet has uneven density; one global α may bottleneck):
   - `alpha_global` (current), vs
   - `alpha_per_level` (a learned scale per tree depth-level; map each node to its depth, use the depth's α), vs
   - `alpha_per_node` (a learned scale offset per node — most flexible; watch for overfitting).
A3. **High-dim lr schedule:** at dim128, try lr {0.5×, 2×, 5×} and a cosine decay. Sometimes high-dim Poincaré needs a different lr than low-dim.

**Report:** the BEST Poincaré dim128 dist/MAP achievable under any Probe-A setting, vs the Stage-5.6 winner baseline, vs Euclidean@dim128. State plainly: does honest extra effort CLOSE, NARROW, or NOT MOVE the gap?

## Probe B — Real-structure probes (is the gap explained by non-tree-likeness?)
B1. **Measure δ of the trees themselves.** Compute δ-hyperbolicity (and δ_relative = δ/diameter) on the *tree-distance metric* of: the synthetic oracle (b2d6), and the real subtrees `bfs_s50`, `deep_s100`, `deep_s200`, `balanced_s200`. (A perfect tree's tree-metric has δ=0 exactly; real WordNet subtrees, once we include multiple root-paths / the dag_to_tree simplification, may still be exactly 0 as pure trees — so ALSO compute δ on an alternative metric that captures real irregularity: e.g. the *shortest-path metric on the original DAG before dag_to_tree* if available, OR the metric induced by node degrees/branching irregularity. Document exactly which metric you use for δ and why. The goal: a per-subtree scalar that captures "how far from a clean tree this hierarchy is.")
   - If pure trees all give δ=0, use a structural irregularity proxy instead and say so: e.g. branching-factor variance, or Gini of subtree sizes, or the fraction of "gateway" high-degree nodes. Pick one principled proxy and justify it.
B2. **Correlate gap with structure.** For each real subtree (`bfs_s50` depth2, `deep_s100` depth4, `deep_s200` depth7, `balanced_s200` depth9), compute the **best-effort** Poincaré-vs-Euclidean gap at dim128 (using the Probe-A best settings) and plot/tabulate that gap against the structural scalar from B1. Does a LARGER gap track LESS tree-like structure?

**Report:** does the residual gap correlate with a structural measure? If yes → evidence for (B) real structure. If the gap vanishes under Probe A regardless of structure → evidence for (A) under-training.

## Probe C — Learnable-scale stability (is it method-worthy?)
C1. **α convergence:** across seeds and across the trees above, does the learned α converge to consistent values, or is it erratic? Report α mean/std per tree.
C2. **α vs dimension:** does α systematically change with dim? (Interpretable behavior supports method framing.)
C3. **Does per-level/per-node α (from A2) generalize** or overfit — compare train vs held-out-pairs distortion for global vs per-level vs per-node α.

**Report:** is learnable-scale stable, interpretable, and generalizing (→ method-worthy), or a finicky knob (→ keep as fix only)?

## Task — Orchestration + report
- `src/diagnostics/investigate_gap.py`, CLI `--probe {A,B,C,all}`. Tag runs `experiment="investigate_gap"`.
- Auto-write `outputs/investigate_gap_report.md`: Probe-A gap-closure verdict, Probe-B correlation (table + the structural scalar used), Probe-C stability numbers, and an overall (A vs B) verdict with the numbers behind it.
- Figures under `outputs/figures/gap/`: (i) Poincaré dim128 MAP under Probe-A settings vs Euclidean; (ii) gap-vs-structure scatter from B2; (iii) α values across trees/seeds from C1.

## Tests (`tests/test_investigate_gap.py`, fast)
- δ on a perfect tree metric == 0 (reuse Stage 4 correctness), and the chosen irregularity proxy is computable and differs across two hand-built trees (a star vs a path).
- `alpha_per_level` builds the correct node→level→α mapping on a tiny tree (hand-check one mapping).
- A tiny end-to-end: deep-ish toy tree, global vs per_level α, few epochs, both finite; per_level has ≥ as many α params as levels.
- Do NOT run full sweeps in tests.

---

## Definition of Done — checklist
- [ ] `src/diagnostics/investigate_gap.py` added; protected code untouched.
- [ ] Probe A run: best-effort Poincaré dim128 vs Euclidean reported; gap closure verdict (close/narrow/none).
- [ ] Probe A2 tested global vs per-level vs per-node α.
- [ ] Probe B: structural scalar defined + justified; gap-vs-structure correlation reported.
- [ ] Probe C: α stability/interpretability/generalization reported.
- [ ] `outputs/investigate_gap_report.md` + 3 figures generated.
- [ ] `pytest tests/test_investigate_gap.py` passes.

## Report back to the human (MOST IMPORTANT)
1. **Probe A verdict:** does honest extra training (epochs / richer α / lr) CLOSE, NARROW, or NOT MOVE the deep_s200 dim128 gap? Give best Poincaré dist/MAP vs Euclidean.
2. **Did richer α (per-level / per-node) help** specifically? This is the prime suspect — report its effect.
3. **Probe B:** which structural scalar did you use, and does the residual gap correlate with it across subtrees? (Table.)
4. **Probe C:** is α stable/interpretable across seeds & trees (method-worthy) or finicky (fix-only)?
5. **Overall verdict:** (A) under-training or (B) real structure — or a mix — WITH the numbers. If mixed, quantify how much each explains.
Do NOT over-claim. We interpret together and then decide: port the finished fix + fair Stage-5 rerun, or reframe the thesis around structure-dependence.
```
