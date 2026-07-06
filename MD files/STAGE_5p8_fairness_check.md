# Stage 5.8 — Fairness Check: A Geometry-Agnostic Stopping Rule

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies.**

Stage 5.7 verdict: the real-data Poincaré gap was **under-training** — with learnable global α and ~5× more epochs, Poincaré reaches parity/superiority with Euclidean (deep_s200 dim128: P MAP 0.962 vs E 0.975; at 5000 epochs P distortion beats E). Richer α didn't help; the gap didn't track structure. learnable-α is stable/interpretable/method-worthy.

**The danger this stage addresses:** Stage 5.7's fix was "give Poincaré more epochs." If we now compare well-trained Poincaré (2500–5000 epochs) against Euclidean at its old early-stopping point (~500), we recreate the Stage-5 unfairness in REVERSE — well-trained hyperbolic vs under-trained Euclidean. A reviewer will (rightly) reject that. Before porting anything or re-running Stage 5, we must establish a **fair, geometry-agnostic stopping rule** and PROVE it fair with convergence curves.

**Key question:** at a large, EQUAL training budget, does Euclidean also improve, plateau, or overfit? Only if we compare BOTH at genuine convergence is the result defensible.

**Still SANDBOX.** Work in `src/diagnostics/fairness_check.py`, reusing the Stage-5.6 learnable-α sandbox trainer (`fix_prototype.py`). Do NOT edit `src/trainer.py`, `src/geometry.py`, `src/metrics.py`. This stage DECIDES the config we will port next; it does not port.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse `train_poincare_fixed` (learnable-α), `train_euclidean_ref`, `evaluate_embedding`, subtrees from `data.py`, tracking.
- IMPORTANT: Euclidean must ALSO get the learnable-α scale treatment for a fair comparison (the scale fix is geometry-agnostic — it rescales target tree-distances, independent of geometry). If the Euclidean reference doesn't currently support learnable α, add it in the sandbox so BOTH geometries use the identical objective. Confirm this in the report.

---

## Task 1 — Full convergence curves (the core evidence)
On 3 representative subtrees — `bfs_s50` (shallow, N=50), `deep_s100` (medium), `deep_s200` (deep, N=200) — at dims {8, 128}, BOTH geometries, learnable-α objective, **3 seeds**:
- Train to a LARGE fixed budget: **8000 epochs**, early-stopping DISABLED (or patience set so high it never triggers), logging val-distortion and val-MAP every K epochs (e.g. every 50).
- Save the full per-epoch val curves.

Produce, per (subtree, dim), a convergence figure: val-distortion vs epoch for Poincaré and Euclidean (mean ± std band over seeds). These curves are the paper's fairness evidence.

## Task 2 — Where does each geometry converge?
From the curves, compute for each (subtree, dim, geometry):
- `epochs_to_converge`: first epoch where val-distortion is within 1% (relative) of its final-100-epoch average (i.e., effectively plateaued).
- `converged_val_distortion` and `converged_val_MAP`: the plateau values.
- Report a table. The question to answer explicitly: **does Euclidean keep improving past 500 epochs, or is it already plateaued there?** And **how many epochs does each geometry need?** (We expect Poincaré to need more; we must SHOW Euclidean is given enough to converge too.)

## Task 3 — Candidate stopping rules (explore, then pick the fair one)
Evaluate THREE geometry-agnostic rules; for each, both geometries get the SAME rule:
- **R1 — fixed large budget:** both train exactly 8000 epochs, report final.
- **R2 — convergence early-stopping:** both use identical generous patience (e.g. patience=500 epochs on val-distortion, min_delta relative 0.5%), high cap 8000.
- **R3 — compute-matched:** both train to the SAME wall-clock or same #gradient-steps budget (note: Poincaré steps are slightly costlier; report both step-count and wall-clock).
For each rule, report Poincaré-vs-Euclidean converged distortion/MAP on the 3 subtrees × 2 dims. 

**Fairness criteria to check for each rule:**
- Does the rule let BOTH geometries reach their plateau (from Task 2)? A fair rule must not cut either short.
- Is the rule specified WITHOUT reference to geometry (no per-geometry epoch numbers)? 
- Are the P-vs-E conclusions STABLE across R1/R2/R3? If the winner flips depending on the rule, that's a red flag we must disclose.

Recommend the rule that (a) provably lets both converge, (b) is geometry-agnostic, (c) gives stable conclusions — and note if all three agree (ideal).

## Task 4 — The honest verdict
State plainly, with numbers:
- At genuine convergence (both geometries fully trained), does Poincaré match / beat / lose to Euclidean, per subtree and dim?
- Does Euclidean improve materially beyond 500 epochs? (If yes, Stage-5's Euclidean numbers were ALSO under-trained and the whole Stage-5 table is doubly invalid — important to know.)
- Is the Poincaré advantage at LOW dim (dim 8, and ideally we also test dim 2 on `bfs_s50`) preserved at convergence? Low-dim is where hyperbolic should shine.
- Confirm the final config to PORT: learnable-global-α + which stopping rule + what epoch cap.

## Task 5 — Report + figures
- `outputs/fairness_check_report.md`: convergence table (Task 2), the three-rule comparison (Task 3), the verdict (Task 4), and the recommended port config.
- Figures under `outputs/figures/fairness/`: convergence curves per (subtree, dim); a summary bar of P-vs-E converged MAP under each rule.
- Tag runs `experiment="fairness_check"`.

## Tests (`tests/test_fairness_check.py`, fast)
- `epochs_to_converge` helper: on a synthetic decreasing-then-flat curve, returns the correct plateau epoch (hand-checked).
- Euclidean-with-learnable-α runs and produces finite metrics on a tiny tree (confirms the objective is truly geometry-agnostic).
- A tiny end-to-end: bfs-like toy tree, both geometries, few epochs, both finite; assert both use the same objective/scale handling.
- Do NOT run the full 8000-epoch sweeps in tests.

---

## Definition of Done — checklist
- [ ] `src/diagnostics/fairness_check.py` added; protected code untouched; BOTH geometries use identical learnable-α objective.
- [ ] Full convergence curves (8000 epochs, early-stop off) for 3 subtrees × 2 dims × 2 geometries × 3 seeds.
- [ ] Convergence table: epochs_to_converge + plateau distortion/MAP per geometry.
- [ ] Three stopping rules (R1/R2/R3) evaluated; conclusions' stability across rules reported.
- [ ] Explicit verdict: does Euclidean improve past 500 epochs? Does Poincaré win at convergence, esp. low dim?
- [ ] Recommended PORT config stated (objective + stopping rule + cap).
- [ ] `outputs/fairness_check_report.md` + convergence figures generated.
- [ ] `pytest tests/test_fairness_check.py` passes.

## Report back to the human (MOST IMPORTANT)
1. **Does Euclidean improve past 500 epochs?** (If yes, Stage-5's Euclidean was also under-trained — critical.) Give converged vs 500-epoch Euclidean numbers.
2. **Convergence table:** epochs_to_converge for P vs E on each subtree/dim — how much more does Poincaré really need?
3. **At true convergence:** P vs E distortion/MAP per subtree/dim. Does hyperbolic win at low dim? Match at high dim?
4. **Stopping-rule stability:** do R1/R2/R3 give the same P-vs-E conclusion? If any rule flips it, say so.
5. **Recommended port config** (objective + stopping rule + epoch cap) for the fair Stage-5 rerun.
Do NOT over-claim. This decides whether the Stage-5 rerun is fair; we commit only after seeing these numbers.
```
