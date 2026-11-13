# [EACL 2027] Mind the Scale: Fair Comparison Reveals When Hyperbolic Geometry Helps Linguistic Hierarchies

<!--
📄 Paper: [ACL Anthology](https://aclanthology.org/XXXX)
-->

## What this repository shows

We compare Poincaré and Euclidean embeddings of WordNet noun hierarchies under a controlled, fair protocol.

- **A silent failure.** Training hyperbolic embeddings to reproduce tree distances at a fixed scale fails without any error, which makes Euclidean space look better at every dimension.
- **A simple fix.** Learning one global scale (equivalent to learning the curvature) with a stopping rule shared by both geometries reverses the result. Hyperbolic embeddings then need about half the dimensions, with the largest gains at low dimension and on widely branching hierarchies.
- **No downstream transfer.** On lexical entailment (HyperLex, WBLESS, BIBLESS), embedding distance carries almost no signal in either geometry. What matters is how well an embedding encodes generality, and Euclidean embeddings do this at least as well.

## Repository structure

```
configs/            Training configuration (default.yaml)
src/
  data.py           WordNet loading, DAG-to-tree reduction, subtree sampling
  trainer.py        Training loop, learnable scale, stopping rule
  experiments/      Experiment scripts (e.g. exp_final.py)
  figures/          Figure-generation script (make_paper_figures.py)
outputs/
  resubmit_csvs/    Per-seed results behind every table and figure
```

## Setup

```bash
git clone https://github.com/Labib1610/hyperbolic-dim-study.git
cd hyperbolic-dim-study
pip install -r requirements.txt

# PyTorch is deliberately NOT in requirements.txt: install it separately from the
# CUDA 12.8 (cu128) channel (used here for an RTX 5060 Ti / Blackwell GPU).
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

All experiments run on a single 16 GB consumer GPU. A subtree embedding trains in seconds to minutes.

## Data

- **WordNet** noun hierarchy (English).
- **HyperLex** for graded lexical entailment.
- **WBLESS** and **BIBLESS** for entailment detection and direction.

See the paper's appendix for how the WordNet regions and the 18 subtrees are built.

## Reproducing the results

The runner reads `configs/default.yaml` directly (the path is hardcoded, so there is no `--config` flag) and is driven by action flags. Run the three stages in order:

```bash
# 1. Build the 18 subtrees under data/subtrees/ (+ summary.csv).
python -m src.experiments.exp_final --build-subtrees

# 2. Run the full grid: 18 subtrees × 7 dims × 5 seeds × 2 geometries (long).
python -m src.experiments.exp_final --run

# 3. Aggregate results, write figures, and print the report.
python -m src.experiments.exp_final --analyze
```

Every run uses the same convergence-based stopping rule for both geometries (patience 500, relative improvement 0.5%, cap 8,000 epochs), five seeds per configuration, and learning rate 0.01.

## Results

All per-seed results are in `outputs/resubmit_csvs/`, so every number in the paper can be checked without retraining.

## Citation

If you use this code, please cite the paper. Will provide later.

<!--
```bibtex
@inproceedings{sayeedi-2027-mind,
  title     = {Mind the Scale: Fair Comparison Reveals When Hyperbolic Geometry Helps Linguistic Hierarchies},
  author    = {Sayeedi, Nurul Labib},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  url       = {https://aclanthology.org/XXXX}
}
```
-->

## Contact

Nurul Labib Sayeedi: nsayeedi2410045@bsds.uiu.ac.bd