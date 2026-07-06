"""Experiment Q3 — HyperLex downstream (does geometry capture human entailment?).

HyperLex is 2,163 noun pairs with graded human ratings (0-10) of lexical
entailment ("to what degree is X a type of Y?"). Q3 asks whether the hyperbolic
embeddings capture these human judgements better than Euclidean — a Spearman
correlation task.

This file currently implements **Task 0 (coverage check)**: how many HyperLex noun
pairs can we actually evaluate given a chosen WordNet noun region? Embedding /
scoring (Tasks 1-4) are added after the coverage is confirmed acceptable.

No edits to geometry/metrics; reuses the production trainer + WordNet data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.data import build_wordnet_noun_graph, dag_to_tree, ensure_wordnet, save_subtree

HYPERLEX_NOUNS = "data/hyperlex/nouns-verbs/hyperlex-nouns.txt"

# Candidate WordNet noun regions to consider for embedding (root synset names).
CANDIDATE_REGIONS = [
    "entity.n.01",           # full noun hierarchy (largest)
    "physical_entity.n.01",
    "abstraction.n.06",
    "object.n.01",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_hyperlex(path: str = HYPERLEX_NOUNS) -> pd.DataFrame:
    """Load a HyperLex file into a DataFrame with word1, word2, pos, rating.

    File is whitespace-separated with a header:
    ``WORD1 WORD2 POS TYPE AVG_SCORE AVG_SCORE_0_10 STD SCORES...``. We keep the
    0-10 average rating (``AVG_SCORE_0_10``) as the human score.
    """
    rows = []
    with open(path) as f:
        header = f.readline().split()
        assert header[:6] == ["WORD1", "WORD2", "POS", "TYPE", "AVG_SCORE", "AVG_SCORE_0_10"], header
        for line in f:
            toks = line.split()
            if len(toks) < 6:
                continue
            rows.append({"word1": toks[0], "word2": toks[1], "pos": toks[2],
                         "type": toks[3], "rating": float(toks[5])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Word -> WordNet noun synset -> region membership
# ---------------------------------------------------------------------------

_ancestor_cache: dict[str, set[str]] = {}


def _ancestors_inclusive(syn) -> set[str]:
    """All hypernym ancestors of ``syn`` (by name), including ``syn`` itself."""
    name = syn.name()
    cached = _ancestor_cache.get(name)
    if cached is None:
        cached = {a.name() for a in syn.closure(lambda s: s.hypernyms())}
        cached.add(name)
        _ancestor_cache[name] = cached
    return cached


def word_noun_synsets(word: str):
    from nltk.corpus import wordnet as wn
    return wn.synsets(word, pos="n")


def word_in_region(word: str, region_name: str) -> bool:
    """True if any noun synset of ``word`` lies in the region rooted at ``region_name``."""
    for syn in word_noun_synsets(word):
        if region_name in _ancestors_inclusive(syn):
            return True
    return False


def word_maps_to_wordnet(word: str) -> bool:
    return len(word_noun_synsets(word)) > 0


# ---------------------------------------------------------------------------
# Task 0 — coverage check
# ---------------------------------------------------------------------------


def _region_size(root: str) -> int:
    """Number of noun synsets in the region rooted at ``root`` (downward closure)."""
    return build_wordnet_noun_graph(root).number_of_nodes()


def coverage_check(hyperlex_path=HYPERLEX_NOUNS, regions=CANDIDATE_REGIONS,
                   out_path="outputs/q3_coverage_report.md", target=0.60) -> dict:
    ensure_wordnet()
    df = load_hyperlex(hyperlex_path)
    n_pairs = len(df)

    words = pd.unique(df[["word1", "word2"]].values.ravel())
    oov = [w for w in words if not word_maps_to_wordnet(w)]
    mappable_mask = df["word1"].map(word_maps_to_wordnet) & df["word2"].map(word_maps_to_wordnet)
    n_mappable = int(mappable_mask.sum())

    region_stats = []
    for root in regions:
        try:
            size = _region_size(root)
        except Exception as exc:  # noqa: BLE001
            size = -1
            print(f"  (region size for {root} failed: {exc})")
        covered = df.apply(
            lambda r: word_in_region(r["word1"], root) and word_in_region(r["word2"], root), axis=1
        ).sum()
        region_stats.append({"region": root, "size_nodes": size, "covered": int(covered),
                             "coverage_frac": covered / n_pairs})
        print(f"  {root:22s} size={size:>7} covered={int(covered):>5}/{n_pairs} "
              f"({covered / n_pairs:.1%})")

    # Decision: smallest region (fewest nodes) meeting the target coverage.
    eligible = [r for r in region_stats if r["coverage_frac"] >= target and r["size_nodes"] > 0]
    chosen = min(eligible, key=lambda r: r["size_nodes"]) if eligible else None

    # --- write report ---
    L = ["# Q3 HyperLex — Task 0 coverage report", "",
         f"HyperLex noun pairs: **{n_pairs}**",
         f"Unique words: {len(words)}; out-of-WordNet (no noun synset): **{len(oov)}**",
         f"Pairs where BOTH words map to WordNet nouns: **{n_mappable}** "
         f"({n_mappable / n_pairs:.1%})", "",
         "## Coverage by candidate region", "",
         "| region | size (nodes) | covered pairs | coverage |", "|---|---|---|---|"]
    for r in region_stats:
        L.append(f"| {r['region']} | {r['size_nodes']} | {r['covered']}/{n_pairs} | {r['coverage_frac']:.1%} |")
    L += ["", f"Target coverage: ≥{target:.0%}."]
    if chosen:
        L.append(f"**Chosen region (smallest meeting target): `{chosen['region']}` "
                 f"({chosen['size_nodes']} nodes, {chosen['coverage_frac']:.1%} coverage).**")
    else:
        L.append("**No candidate region meets the target — STOP and reconsider "
                 "(see coverage numbers above).**")
    if oov[:20]:
        L += ["", f"Sample OOV words: {', '.join(sorted(oov)[:20])}"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L))

    return {"n_pairs": n_pairs, "n_mappable": n_mappable, "n_oov": len(oov),
            "regions": region_stats, "chosen": chosen}


# ---------------------------------------------------------------------------
# Option-3 induced subtree: HyperLex-covered synsets + ALL ancestors to root
# ---------------------------------------------------------------------------


def build_hyperlex_induced_tree(hyperlex_path=HYPERLEX_NOUNS, root="entity.n.01"):
    """Subtree induced by HyperLex-covered synsets + all their ancestors to ``root``.

    Uses the canonical ``dag_to_tree`` tree, then takes the union of root-paths from
    every covered synset. All intermediate ancestors are kept intact (no chain
    collapsing), so tree-distances between HyperLex words are identical to the full
    entity tree. Returns ``(undirected_tree, covered_synset_names)``.
    """
    ensure_wordnet()
    tree_full = dag_to_tree(build_wordnet_noun_graph(root), root)
    df = load_hyperlex(hyperlex_path)
    words = set(df["word1"]).union(df["word2"])
    covered = {syn.name() for w in words for syn in word_noun_synsets(w)
               if syn.name() in tree_full}

    induced: set[str] = set()
    for name in covered:
        node = name
        while node is not None and node not in induced:
            induced.add(node)
            succ = list(tree_full.successors(node))  # child->parent: the parent
            node = succ[0] if succ else None

    sub = nx.Graph(tree_full.subgraph(induced).to_undirected())
    sub.graph["root"] = root
    return sub, covered


def save_hyperlex_induced_tree(name="hyperlex_entity", data_dir="data", root="entity.n.01"):
    """Build and persist the induced subtree under ``data/subtrees/<name>/``."""
    tree, covered = build_hyperlex_induced_tree(root=root)
    path = Path(data_dir) / "subtrees" / name
    save_subtree(tree, root, str(path))
    return tree, covered, path


# ---------------------------------------------------------------------------
# Task 1 — grid runner (embed the induced region, both geometries)
# ---------------------------------------------------------------------------

Q3_TAG = "q3_hyperlex"
Q3_DIMS = [8, 16, 32, 64, 128]
Q3_SEEDS = [0, 1, 2]
Q3_SUBTREE = "hyperlex_entity"
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]


def _completed_q3(df) -> set:
    done = set()
    req = ["train.geometry", "train.dim", "train.seed", "epochs_ran"]
    if df.empty or any(c not in df.columns for c in req) or "experiment" not in df.columns:
        return done
    for _, r in df.iterrows():
        if r.get("experiment") != Q3_TAG or pd.isna(r["epochs_ran"]):
            continue
        done.add((r["train.geometry"], int(r["train.dim"]), int(r["train.seed"])))
    return done


def run_grid_q3(dims=Q3_DIMS, seeds=Q3_SEEDS, subtree=Q3_SUBTREE,
                base_dir="outputs/runs", results_dir="outputs", data_dir="data",
                val_subsample=10000):
    """Embed the induced region with the fair trainer (large_tree), both geometries.

    Does NOT call evaluate_run — all-pairs metrics are infeasible at ~8k nodes; Q3
    scoring uses coords directly on the ~2153 covered HyperLex pairs.
    """
    from src.config import load_config
    from src.tracking import load_results
    from src.trainer import train_embedding

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config("configs/default.yaml")
    done = _completed_q3(load_results(results_dir / "results.csv"))
    fail_log = results_dir / "q3_failures.log"

    combos = [(g, d, s) for g in ("poincare", "euclidean") for d in dims for s in seeds]
    launched = skipped = 0
    failures = []
    import copy
    import time
    start = time.time()
    for i, (geom, dim, seed) in enumerate(combos, 1):
        if (geom, dim, seed) in done:
            skipped += 1
            continue
        try:
            cfg = copy.deepcopy(base_cfg)
            cfg["subtree"] = subtree
            cfg["seed"] = seed
            cfg.setdefault("paths", {})["data_dir"] = data_dir
            cfg["train"].update({
                "geometry": geom, "dim": dim, "loss": "distortion", "seed": seed,
                "optimizer": "radam" if geom == "poincare" else "adam",
                "large_tree": True, "val_subsample": val_subsample,
            })
            cfg["experiment"] = Q3_TAG
            run = train_embedding(subtree, cfg, base_dir=base_dir, results_dir=str(results_dir))
            done.add((geom, dim, seed))
            launched += 1
            print(f"  [{i}/{len(combos)}] {geom} dim{dim} seed{seed} -> {run.dir.name} "
                  f"(alpha={run.metrics['alpha']:.3f}, {time.time()-start:.0f}s elapsed)", flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"({geom},{dim},{seed}): {type(exc).__name__}: {exc}"
            failures.append(msg)
            with open(fail_log, "a") as f:
                f.write(msg + "\n")
    print(f"Q3 grid done: {launched} launched, {skipped} skipped, {len(failures)} failed, "
          f"{time.time()-start:.0f}s.")
    return {"launched": launched, "skipped": skipped, "failed": len(failures), "failures": failures}


# ---------------------------------------------------------------------------
# Task 2/3 — scoring + generality-radius
# ---------------------------------------------------------------------------

_word_idx_cache: dict = {}


def _word_indices(word, node_index):
    key = (word, id(node_index))
    hit = _word_idx_cache.get(key)
    if hit is None:
        hit = [node_index[s.name()] for s in word_noun_synsets(word) if s.name() in node_index]
        _word_idx_cache[key] = hit
    return hit


def covered_pairs(df, node_index) -> pd.DataFrame:
    """HyperLex pairs where BOTH words have >=1 synset in the embedded region."""
    mask = df.apply(lambda r: bool(_word_indices(r["word1"], node_index))
                    and bool(_word_indices(r["word2"], node_index)), axis=1)
    return df[mask].reset_index(drop=True)


def score_pairs(coords, geometry, node_index, cov_df):
    """Per covered pair: min-distance synset match -> (d_min, gen_diff, rating).

    ``gen`` (generality proxy) = Poincaré norm, or Euclidean distance-from-centroid,
    so both geometries use the same Score-B directional term.
    """
    import torch

    from src.geometry import EuclideanGeometry, PoincareBall

    geom = PoincareBall() if geometry == "poincare" else EuclideanGeometry()
    X = torch.tensor(np.asarray(coords), dtype=torch.float64)
    if geometry == "poincare":
        gen = X.norm(dim=-1)
    else:
        gen = (X - X.mean(0)).norm(dim=-1)

    d_min, gen_diff, rating = [], [], []
    for _, r in cov_df.iterrows():
        a = _word_indices(r["word1"], node_index)
        b = _word_indices(r["word2"], node_index)
        Xa, Xb = X[a], X[b]
        A2 = Xa.unsqueeze(1).expand(len(a), len(b), -1)
        B2 = Xb.unsqueeze(0).expand(len(a), len(b), -1)
        D = geom.dist(A2, B2).reshape(-1)
        mi = int(D.argmin())
        ia, ib = a[mi // len(b)], b[mi % len(b)]
        d_min.append(float(D.min()))
        gen_diff.append(float(gen[ia] - gen[ib]))
        rating.append(float(r["rating"]))
    return pd.DataFrame({"d_min": d_min, "gen_diff": gen_diff, "rating": rating})


def score_b(d_min, gen_diff, lam):
    """Score B = -d_G + lam * (gen(w1) - gen(w2)); higher => more entailment."""
    return -np.asarray(d_min) + lam * np.asarray(gen_diff)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def pearson(x, y) -> float:
    from scipy.stats import pearsonr
    return float(pearsonr(x, y)[0])


def eval_run_scores(scored, dev_mask, lambdas=LAMBDAS):
    """Score A (test ρ) and Score B (λ tuned on dev, reported on test)."""
    rating = scored["rating"].values
    test = ~dev_mask
    sA = -scored["d_min"].values
    rho_A = spearman(sA[test], rating[test])
    best_lam, best_dev = 0.0, -2.0
    for lam in lambdas:
        sB = score_b(scored["d_min"].values, scored["gen_diff"].values, lam)
        dev_rho = spearman(sB[dev_mask], rating[dev_mask])
        if dev_rho > best_dev:
            best_dev, best_lam = dev_rho, lam
    sB = score_b(scored["d_min"].values, scored["gen_diff"].values, best_lam)
    rho_B = spearman(sB[test], rating[test])
    return {"rho_A": rho_A, "rho_B": rho_B, "lambda": best_lam,
            "pearson_A": pearson(sA[test], rating[test])}


def generality_radius_rho(coords, geometry, depths_arr) -> float:
    """Spearman(tree-depth, embedding norm) — general/shallow near origin?"""
    import numpy as _np
    X = _np.asarray(coords, dtype=float)
    if geometry == "poincare":
        gen = _np.linalg.norm(X, axis=1)
    else:
        gen = _np.linalg.norm(X - X.mean(0), axis=1)
    return spearman(depths_arr, gen)


# ---------------------------------------------------------------------------
# Task 4/5 — analysis, figures, report
# ---------------------------------------------------------------------------


def _load_region(data_dir, subtree=Q3_SUBTREE):
    from src.data import load_subtree
    tree, meta = load_subtree(f"{data_dir}/subtrees/{subtree}")
    root = tree.graph.get("root") or meta.get("root")
    nodes = sorted(tree.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    depths = nx.single_source_shortest_path_length(tree, root)
    depths_arr = np.array([depths[n] for n in nodes], dtype=float)
    return node_index, depths_arr


def analyze_q3(results_dir="outputs", data_dir="data", subtree=Q3_SUBTREE):
    from src.tracking import load_results

    node_index, depths_arr = _load_region(data_dir, subtree)
    df = load_hyperlex()
    cov = covered_pairs(df, node_index)
    # fixed dev/test split of covered pairs
    rng = np.random.RandomState(0)
    dev_mask = rng.rand(len(cov)) < 0.5

    res = load_results(Path(results_dir) / "results.csv")
    res = res[res.get("experiment") == Q3_TAG] if "experiment" in res.columns else res
    rows = []
    scored_cache = {}
    for _, r in res.iterrows():
        rid = r["run_id"]
        cpath = Path(results_dir) / "runs" / rid / "artifacts" / "coords.npy"
        if not cpath.exists():
            continue
        coords = np.load(cpath)
        geom = r["train.geometry"]
        dim = int(r["train.dim"])
        seed = int(r["train.seed"])
        scored = score_pairs(coords, geom, node_index, cov)
        ev = eval_run_scores(scored, dev_mask)
        gr = generality_radius_rho(coords, geom, depths_arr)
        rows.append({"geometry": geom, "dim": dim, "seed": seed, "n_covered": len(cov),
                     **ev, "gen_radius_rho": gr})
        scored_cache[(geom, dim, seed)] = scored
    per_run = pd.DataFrame(rows)
    per_run.to_csv(Path(results_dir) / "q3_scores.csv", index=False)

    agg = (per_run.groupby(["geometry", "dim"])
           .agg(rho_A=("rho_A", "mean"), rho_A_std=("rho_A", "std"),
                rho_B=("rho_B", "mean"), rho_B_std=("rho_B", "std"),
                lambda_mean=("lambda", "mean"), gen_radius_rho=("gen_radius_rho", "mean"),
                n=("rho_A", "count"))
           .reset_index())
    agg["rho_best"] = agg[["rho_A", "rho_B"]].max(axis=1)
    agg.to_csv(Path(results_dir) / "q3_aggregated.csv", index=False)
    return per_run, agg, cov, scored_cache, depths_arr


def make_figures_q3(agg, per_run, cov, scored_cache, depths_arr,
                    results_dir="outputs", outdir="outputs/figures/q3"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    colors = {"poincare": "tab:blue", "euclidean": "tab:red"}
    dims = sorted(agg["dim"].unique())

    # (i) rho vs dim
    fig, ax = plt.subplots(figsize=(6, 4))
    for geom in ("poincare", "euclidean"):
        s = agg[agg.geometry == geom].sort_values("dim")
        ax.plot(s["dim"], s["rho_best"], "-o", color=colors[geom], label=f"{geom} (best A/B)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dimension"); ax.set_ylabel("Spearman ρ vs HyperLex")
    ax.set_title("Q3: entailment ρ vs dim"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "q3_rho_vs_dim.png", dpi=150, bbox_inches="tight")
    fig.savefig(outdir / "q3_rho_vs_dim.pdf", bbox_inches="tight"); plt.close(fig)

    # (ii) best-config scatter: best score vs human rating
    best = agg.loc[agg["rho_best"].idxmax()]
    key = None
    for (g, d, s) in scored_cache:
        if g == best["geometry"] and d == int(best["dim"]):
            key = (g, d, s); break
    if key:
        sc = scored_cache[key]
        lam = float(per_run[(per_run.geometry == key[0]) & (per_run.dim == key[1])]["lambda"].mean())
        score = score_b(sc["d_min"].values, sc["gen_diff"].values, lam)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sc["rating"].values, score, s=8, alpha=0.4)
        ax.set_xlabel("HyperLex human rating (0-10)")
        ax.set_ylabel(f"best score ({key[0]} dim{key[1]}, λ={lam:g})")
        ax.set_title(f"Q3: best-config score vs human rating (ρ={best['rho_best']:.3f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / "q3_score_vs_rating.png", dpi=150, bbox_inches="tight")
        fig.savefig(outdir / "q3_score_vs_rating.pdf", bbox_inches="tight"); plt.close(fig)

    # (iii) generality-radius: depth vs norm for a representative Poincaré run
    import numpy as _np
    prow = agg[agg.geometry == "poincare"].sort_values("dim").iloc[-1]
    # reload a poincare coords at that dim (seed 0)
    from src.tracking import load_results
    res = load_results(Path(results_dir) / "results.csv")
    res = res[(res.get("experiment") == Q3_TAG) & (res["train.geometry"] == "poincare")
              & (res["train.dim"] == int(prow["dim"])) & (res["train.seed"] == 0)]
    if not res.empty:
        coords = _np.load(Path(results_dir) / "runs" / res.iloc[0]["run_id"] / "artifacts" / "coords.npy")
        norms = _np.linalg.norm(coords, axis=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(depths_arr, norms, s=5, alpha=0.3)
        ax.set_xlabel("tree depth (from entity.n.01)"); ax.set_ylabel("Poincaré norm ‖x‖")
        ax.set_title(f"Q3: generality-radius (ρ={prow['gen_radius_rho']:.3f}, poincare dim{int(prow['dim'])})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(outdir / "q3_generality_radius.png", dpi=150, bbox_inches="tight")
        fig.savefig(outdir / "q3_generality_radius.pdf", bbox_inches="tight"); plt.close(fig)


def generate_report_q3(agg, per_run, cov, path="outputs/q3_hyperlex_report.md"):
    dims = sorted(agg["dim"].unique())

    def gv(geom, dim, col):
        s = agg[(agg.geometry == geom) & (agg.dim == dim)]
        return float(s.iloc[0][col]) if not s.empty else float("nan")

    L = ["# Q3 HyperLex — downstream lexical entailment", "",
         f"Embedded region: entity-induced tree (8291 nodes, 99.5% coverage). "
         f"Covered pairs scored: **{len(cov)}**. Score A = -d_G; Score B = -d_G + λ·(gen(w1)-gen(w2)), "
         f"λ tuned on a dev split, ρ reported on the held-out test split.", "",
         "## ρ table (geometry × dim, best score bolded)", "",
         "| dim | P ρ_A | P ρ_B (λ) | E ρ_A | E ρ_B (λ) |", "|---|---|---|---|---|"]
    for d in dims:
        L.append(f"| {d} | {gv('poincare', d, 'rho_A'):.3f} | {gv('poincare', d, 'rho_B'):.3f} "
                 f"({gv('poincare', d, 'lambda_mean'):.2f}) | {gv('euclidean', d, 'rho_A'):.3f} "
                 f"| {gv('euclidean', d, 'rho_B'):.3f} ({gv('euclidean', d, 'lambda_mean'):.2f}) |")
    L.append("")

    # matched-dimension
    L += ["## Matched-dimension verdict (best of A/B)", "", "| dim | Poincaré ρ | Euclidean ρ | winner |",
          "|---|---|---|---|"]
    for d in dims:
        p, e = gv("poincare", d, "rho_best"), gv("euclidean", d, "rho_best")
        L.append(f"| {d} | {p:.3f} | {e:.3f} | {'P' if p > e else 'E'} |")
    L.append("")

    # efficiency transfer
    p8 = gv("poincare", 8, "rho_best")
    e_match = next((d for d in dims if gv("euclidean", d, "rho_best") >= p8), None)
    L += ["## Efficiency transfer", "",
          f"- Poincaré ρ at dim 8 = **{p8:.3f}**.",
          f"- Smallest Euclidean dim reaching that ρ: **{e_match if e_match else '>'+str(max(dims))}**.",
          ""]

    # which score won + direction term
    p_b_better = sum(gv("poincare", d, "rho_B") > gv("poincare", d, "rho_A") for d in dims)
    e_b_better = sum(gv("euclidean", d, "rho_B") > gv("euclidean", d, "rho_A") for d in dims)
    L += ["## Score A vs B", "",
          f"- Score B (direction term) beat Score A on {p_b_better}/{len(dims)} dims for Poincaré, "
          f"{e_b_better}/{len(dims)} for Euclidean.", ""]

    # generality radius
    L += ["## Generality-radius sanity (depth vs norm)", "", "| geometry | mean ρ(depth, norm) |",
          "|---|---|"]
    for geom in ("poincare", "euclidean"):
        L.append(f"| {geom} | {agg[agg.geometry == geom]['gen_radius_rho'].mean():.3f} |")
    L.append("")

    Path(path).write_text("\n".join(L))
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Q3 HyperLex experiment.")
    p.add_argument("--coverage", action="store_true", help="Run the Task-0 coverage check.")
    p.add_argument("--build-induced", action="store_true", help="Build+save the induced subtree.")
    p.add_argument("--run", action="store_true", help="Run the embedding grid.")
    p.add_argument("--analyze", action="store_true", help="Score + figures + report.")
    p.add_argument("--target", type=float, default=0.60)
    args = p.parse_args(argv)

    if args.build_induced:
        print("Building HyperLex-induced subtree (entity.n.01)...")
        tree, covered, path = save_hyperlex_induced_tree()
        print(f"induced tree: {tree.number_of_nodes()} nodes, saved to {path}")
        return 0

    if args.run:
        print("Running Q3 embedding grid...")
        run_grid_q3()

    if args.analyze:
        print("Analyzing Q3 runs...")
        per_run, agg, cov, scored_cache, depths_arr = analyze_q3()
        make_figures_q3(agg, per_run, cov, scored_cache, depths_arr)
        print(generate_report_q3(agg, per_run, cov))
        return 0
    if args.coverage:
        print("Running HyperLex coverage check...")
        res = coverage_check(target=args.target)
        print(f"\nMappable pairs: {res['n_mappable']}/{res['n_pairs']}")
        if res["chosen"]:
            print(f"Chosen region: {res['chosen']['region']} "
                  f"({res['chosen']['size_nodes']} nodes, {res['chosen']['coverage_frac']:.1%})")
        else:
            print("No region met the target — see outputs/q3_coverage_report.md")
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
