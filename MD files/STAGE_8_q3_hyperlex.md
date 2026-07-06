# Stage 8 — Q3: HyperLex Downstream (does geometry capture human entailment?)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet).**

Q1 (efficiency) and Q2 (branching-scaled advantage) are done — intrinsic geometric results on WordNet subtrees, using the FAIR trainer (learnable-α + R2 convergence stopping). This stage adds the **downstream NLP experiment (Q3)**: do the hyperbolic embeddings capture *human-judged lexical entailment* better than Euclidean? This bridges "hyperbolic embeds trees better" → "…in a way that improves a real language task," making the paper genuinely NLP, not only geometry.

**Benchmark:** HyperLex — 2,616 word pairs with GRADED human ratings of hyponymy–hypernymy (lexical entailment: "how much is X a type of Y", 0–10). This is a *correlation* task (Spearman ρ between our geometry-derived score and human ratings), not classification.

**Do NOT modify** `geometry.py`/`metrics.py`. Reuse the production trainer (learnable-α + R2) unchanged. New code in `src/experiments/exp_q3_hyperlex.py`.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active. Reuse trainer, geometry, data (WordNet), tracking. `scipy.stats` for Spearman/Pearson.
- HyperLex data: download the HyperLex dataset (the standard `hyperlex-data` / nouns TSV with columns like WORD1, WORD2, POS, and the average rating). If the download domain isn't reachable, STOP and tell the human to place the file at `data/hyperlex/hyperlex-nouns.txt` and report the expected format — do not fabricate data.

---

## Task 0 — COVERAGE CHECK FIRST (cheap, before any embedding)
Before embedding anything, determine how many HyperLex pairs we can actually evaluate:
- Load HyperLex; keep NOUN pairs (POS=N) — HyperLex is mostly nouns; document the count.
- Map each HyperLex word to WordNet noun synset(s) (via `wordnet.synsets(word, pos='n')`). A pair is "coverable" by a WordNet region R if BOTH words have at least one synset in R.
- Report: total HyperLex noun pairs; how many map to WordNet at all; and (crucially) how many would be covered by candidate regions:
  - the full WordNet noun hierarchy under `entity.n.01`,
  - a few large mid-level roots (e.g. `physical_entity.n.01`, `abstraction.n.06`, `object.n.01`).
- **Decision rule (report it):** pick the SMALLEST region that covers a healthy fraction (aim ≥60–70%) of HyperLex noun pairs, to keep embedding tractable. If even `entity.n.01` coverage is low, report the number and STOP for us to reconsider (maybe lemma-matching issues — try matching on lemmas of all synsets, and note multi-word/OOV losses).
- Save `outputs/q3_coverage_report.md` with these numbers.

## Task 1 — Build the evaluation region + embed it
- Build the chosen WordNet noun region as a tree (reuse `dag_to_tree`), report its size + structural stats (depth, branching, degree_cv) so we know what we embedded. (This region is larger than the Q1/Q2 subtrees — expect more nodes; note training cost.)
- Embed it with the FAIR trainer, BOTH geometries, at a few dims: {8, 16, 32, 64, 128}, 3 seeds. (Include low-ish dims because Q1/Q2 say hyperbolic's edge is at low dim — Q3 should test whether that edge transfers to the task.) Tag `experiment="q3_hyperlex"`.
- For each HyperLex-covered pair, we'll need each word's embedding: if a word has multiple synsets in the region, use the synset that gives the best (min-distance) match, OR average — pick one, document it (min-distance is standard for lexical tasks).

## Task 2 — Two entailment scores from the geometry (try BOTH)
For a HyperLex pair (w1, w2) asking "is w1 a type of w2":

**Score A — distance-based (symmetric baseline):**
`score_A = -d_G(x_{w1}, x_{w2})`  (closer → higher entailment). Simple, standard.

**Score B — distance + hierarchical direction (uses hyperbolic structure):**
In these embeddings, general concepts sit closer to the origin, specific ones farther out (verify this holds — see Task 3). Entailment "w1 is-a w2" implies w2 is MORE GENERAL → w2 should be closer to origin (smaller norm) than w1, and they should be close. Define:
`score_B = -d_G(x_{w1}, x_{w2}) + λ * (‖x_{w1}‖ - ‖x_{w2}‖)`
(the norm-difference term rewards w2 being nearer the origin than w1; λ a small weight, try a few values on a dev split). For Euclidean, use distance-from-centroid as the "generality" proxy in place of Poincaré norm, so the SAME score form is applied to both geometries (fairness — both get the directional term).

Report Spearman ρ (and Pearson) between each score and HyperLex human ratings, for each geometry × dim × score.

## Task 3 — Sanity: does the generality-radius assumption hold?
- Check that in the learned embeddings, tree-depth correlates with embedding norm (general/shallow nodes near origin, specific/deep nodes near rim) — report the correlation for Poincaré (and the centroid-distance analog for Euclidean). This justifies Score B's direction term and is itself a nice interpretability result for the paper. If it does NOT hold, report that (Score B may then not help — an honest finding).

## Task 4 — The headline comparison
- **Matched-dimension:** at each dim, Poincaré ρ vs Euclidean ρ (best of Score A/B for each). Does hyperbolic capture human entailment better at matched dim, especially low dim?
- **Matched-quality-different-dim (the efficiency transfer):** find the Euclidean dim whose ρ matches Poincaré's ρ at dim 8 — does the Q1 "hyperbolic needs fewer dims" story transfer to the DOWNSTREAM task? This is the strongest possible Q3 result.
- Report which score (A or B) won for each geometry, and whether the hierarchical-direction term (B) helped hyperbolic more than Euclidean (would show the *geometry specifically*, not just proximity, matters).

## Task 5 — Report + figures
- `outputs/q3_hyperlex_report.md`: coverage numbers; embedded-region stats; ρ table (geometry × dim × score); the matched-dimension and matched-quality verdicts; the generality-radius sanity result; honest notes (multi-synset handling, OOV losses, λ tuning).
- Figures under `outputs/figures/q3/`: (i) ρ vs dim, Poincaré vs Euclidean; (ii) scatter of best-score vs human ratings for the best config; (iii) tree-depth vs embedding-norm (the generality-radius plot).
- Everything tracked; numbers auto-pulled, nothing hand-typed.

## Tests (`tests/test_q3.py`, fast)
- HyperLex loader parses the file → correct #pairs, expected columns.
- Word→synset→region coverage logic: on a tiny hand-built region + fake pairs, correctly identifies covered vs uncovered pairs (hand-checked).
- Score A and Score B compute finite values on a tiny embedding; Score B's direction term has the correct sign (a clear hypernym pair scores higher than its reverse).
- Spearman helper matches `scipy.stats.spearmanr` on a small hand case.
- Do NOT run the full embedding/eval in tests.

---

## Definition of Done — checklist
- [ ] Coverage checked FIRST; region chosen to cover ≥~60–70% of HyperLex noun pairs (or stopped + reported if impossible).
- [ ] Evaluation region embedded, both geometries, dims {8,16,32,64,128}, 3 seeds, fair trainer.
- [ ] Both entailment scores (A distance, B distance+direction) implemented for BOTH geometries.
- [ ] Generality-radius assumption checked (depth vs norm correlation).
- [ ] Matched-dimension ρ comparison + matched-quality-different-dim (efficiency transfer) reported.
- [ ] `outputs/q3_hyperlex_report.md` + figures generated.
- [ ] `pytest tests/test_q3.py` passes.

## Report back to the human
1. **Coverage:** total HyperLex noun pairs, how many covered, which region chosen, its size/stats.
2. **Headline ρ:** Poincaré vs Euclidean Spearman ρ at each dim (best score each). Does hyperbolic win, especially at low dim?
3. **Efficiency transfer:** does Euclidean need more dims to match Poincaré's dim-8 ρ? (The Q1 story transferring downstream.)
4. **Which score won**, and did the hierarchical-direction term (B) help hyperbolic more than Euclidean?
5. **Generality-radius:** does depth correlate with embedding norm? (interpretability result)
6. Paste `outputs/q3_hyperlex_report.md`.
Do NOT over-claim; report numbers, we interpret together.
```
