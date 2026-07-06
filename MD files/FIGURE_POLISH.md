# Figure Polish — Regenerate All Paper Figures (Publication Quality)

## Context (read first)
Project: **hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet).** We are writing the paper (ACL/EACL format, two-column). The experiment figures were auto-generated during the experiment stages and have code-style titles and axis labels (e.g. `distortion_avg_scaled_mean vs dim`, `map_mean`, `degree_cv`) that look unpolished in a paper.

**This task regenerates ALL paper figures from the existing aggregated result CSVs with consistent, publication-quality styling.** No re-training, no new experiments — this is pure re-plotting from saved data. Do NOT modify any trainer/geometry/metrics/experiment code. New code goes in `src/figures/make_paper_figures.py`.

## Data sources (already on disk)
- `outputs/exp_final_aggregated.csv` — Q1 & Q2 data (per subtree × geometry × dim: distortion_avg_scaled mean/std, map mean/std; plus each subtree's depth, avg_branching, degree_cv, root).
- `outputs/exp_final_report.md` / the per-subtree Q2 table — advantage (P−E MAP at dim 8) per subtree with depth, avg_branching, degree_cv, root. (If the advantage isn't directly in a CSV, compute it from `exp_final_aggregated.csv`: for each subtree, adv = poincare MAP@dim8 − euclidean MAP@dim8.)
- `outputs/q3_aggregated.csv` and `outputs/q3_lambda_aggregated.csv` — Q3 ρ-vs-dim and ρ-vs-λ data.
- Q3 generality-radius data (depth vs embedding norm) — from the saved Q3 Poincaré dim128 run's coords + the tree; if a saved CSV exists use it, else recompute from the saved coords.

If any expected CSV is missing, STOP and report which one — do not fabricate.

## Global styling (apply to EVERY figure — consistency matters)
Create one shared style setup used by all plots:
- **No titles on the plots** (the LaTeX caption serves as the title). Remove all `plt.title(...)`.
- **Human-readable axis labels** (no underscores / code names):
  - `distortion_avg_scaled_mean` → "Average scaled distortion"
  - `map_mean` → "MAP"
  - `dimension` → "Embedding dimension"
  - `degree_cv` → "Degree irregularity ($\\mathrm{deg}_{cv}$)" (or plain "Degree irregularity")
  - `depth` → "Subtree depth"
  - `avg_branching` → "Average branching factor"
  - `Spearman ρ vs HyperLex` → "Spearman $\\rho$ (HyperLex)"
  - `lambda` → "$\\lambda$"
- **Fonts:** base font size ~11–12, axis labels ~12, legend ~10, tick labels ~10. Readable when the figure is scaled to one column (~3.2 in wide).
- **Colors (colorblind-safe, consistent across ALL figures):** Poincaré = a blue (e.g. `#0072B2`), Euclidean = a vermillion/orange-red (e.g. `#D55E00`). For the 3 roots in Q2 scatters use a colorblind-safe qualitative triple (e.g. `#009E73` green, `#E69F00` orange, `#CC79A7` purple) — and keep the SAME root→color mapping in both Q2 panels.
- **Lines/markers:** line width ~2, marker size ~5–6. Light grid (`alpha=0.3`).
- **Legend:** clean, no frame or light frame, placed where it doesn't overlap data.
- **Layout:** `tight_layout()`; save at `dpi=200`.
- **Save BOTH** `.pdf` (vector, for the paper) and `.png` (preview) for each figure, into `outputs/figures/paper/`.
- Single-column figures sized ~ (3.4, 2.6) inches; if a figure is meant to span both columns, size ~ (7.0, 2.8).

## Figures to produce (exact filenames — these match the LaTeX)

### Q1 (two separate single-column panels, each its own file)
1. `figQ1a_distortion_vs_dim.pdf` — x=dimension (log2 scale, ticks 2,4,…,128), y=Average scaled distortion. Two lines: Poincaré (blue) vs Euclidean (red), mean over subtrees, with std shaded band. No title.
2. `figQ1b_map_vs_dim.pdf` — same x, y=MAP. Same two lines + bands. No title.

### Q2 (two single-column scatter panels)
3. `figQ2_adv_vs_branching.pdf` — **NEW, needed.** x=Average branching factor, y=Hyperbolic advantage (MAP gap at $d{=}8$). Scatter, points colored by root (consistent mapping), with a fitted dashed trend line. Annotate the Pearson $r$ (=+0.64) in a corner text box (compute it from the data to confirm; report the computed value).
4. `figQ2a_adv_vs_depth.pdf` — x=Subtree depth, y=Hyperbolic advantage. Scatter colored by root + dashed trend. Annotate Pearson $r$ (≈−0.61, compute to confirm).
   (Also regenerate `figQ2b_adv_vs_degcv.pdf` — x=Degree irregularity — same style, for the appendix.)

### Q3 (single-column panels)
5. `figQ3_rho_vs_dim.pdf` — x=Embedding dimension (log2), y=Spearman $\\rho$ (HyperLex), Poincaré vs Euclidean (best-λ each), with bands. No title.
6. `figQ3_rho_vs_lambda.pdf` — x=$\\lambda$ (log scale), y=Spearman $\\rho$, Poincaré vs Euclidean at a representative dim (e.g. 128); show that ρ≈0 at λ=0 and peaks at an interior λ. No title.
7. `figQ3_generality_radius.pdf` — x=Tree depth, y=Poincaré norm $\\|x\\|$, scatter (small alpha for density). Annotate Spearman ρ (≈0.735). No title.

## Tasks
1. `src/figures/make_paper_figures.py` with a shared `set_style()` + one function per figure, reading the CSVs above.
2. Recompute and PRINT the key stats as it goes (so we can verify against the paper text): Q2 Pearson r for branching / depth / degree_cv; Q3 λ=0 ρ and best-λ ρ per geometry; generality-radius ρ. These MUST match the numbers already in the paper (branching +0.64, depth −0.61, degcv +0.53, etc.) — if any differs, STOP and report the discrepancy (it would mean a paper-text error to fix).
3. CLI: `python -m src.figures.make_paper_figures` regenerates everything into `outputs/figures/paper/`.
4. A quick `tests/test_paper_figures.py`: the style helper runs; each figure function produces a non-empty PDF file; the recomputed Q2 branching r is within 0.02 of 0.64 (guards the headline number).

## Definition of Done
- [ ] All 7 figures (+ the appendix degcv one) saved as PDF+PNG in `outputs/figures/paper/`, clean styling, no code-y titles, readable labels.
- [ ] Consistent colors (Poincaré blue / Euclidean red; fixed root palette) across all figures.
- [ ] The NEW `figQ2_adv_vs_branching.pdf` exists.
- [ ] Recomputed stats printed and MATCH the paper (branching +0.64, depth −0.61, degcv +0.53, Q3 λ=0 ρ≈0, generality ρ≈0.735); any mismatch reported, not silently changed.
- [ ] `pytest tests/test_paper_figures.py` passes.

## Report back to the human
1. Confirm all 7 (+1 appendix) figures generated in `outputs/figures/paper/`, with filenames.
2. Paste the recomputed key stats (Q2 r values, Q3 λ=0 and best-λ ρ, generality ρ) and confirm they match the paper text — flag any that don't.
3. Note anything that looked off in the data while replotting.
```
