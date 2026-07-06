# Stage 1 — Data Layer (WordNet hierarchy + subtree sampler)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies (WordNet), as a function of hierarchy structure (δ-hyperbolicity).**

Stage 0 (env + skeleton) is complete and the smoke test passed. This stage builds the **data layer only**: load the WordNet noun hierarchy, represent it as a tree/graph, compute tree-distances, and — critically — provide a **subtree sampler that can carve out subtrees with controllable structure** (size, depth, branching). The subtree sampler is the engine for our novel experiment (Q2), which studies how embedding quality depends on tree structure, so it must be flexible and well-tested.

**Do NOT implement geometry, embeddings, training, or metrics here.** Only data.

## Environment
- Repo: `hyperbolic-dim-study/`, package `src`, venv active.
- Libraries available: `nltk`, `networkx`, `numpy`, `scipy`. Use `networkx` for graph structure and shortest-path (tree) distances.

---

## Objective / Definition of Done
Implement `src/data.py` such that:
1. It can download + load the WordNet noun `is-a` (hypernym) hierarchy via NLTK.
2. It builds a directed graph (child → parent) and an undirected version for distances.
3. It computes tree-distance between any two nodes.
4. It samples **subtrees with controllable structure** and reports structural statistics for each.
5. A CLI produces a set of subtrees saved to disk plus a summary table.
All checklist items at the bottom pass, and `pytest tests/test_data.py` passes against the tiny hand-checkable tree.

---

## Task 1 — NLTK WordNet bootstrap
- Function `ensure_wordnet()` that downloads the `wordnet` (and `omw-1.4` if needed) corpora via `nltk.download(...)` if not already present, idempotently.
- Call it lazily inside loaders so tests can trigger it.

## Task 2 — Build the noun hypernym graph
Function `build_wordnet_noun_graph(root_synset_name: str | None = None) -> networkx.DiGraph`:
- Nodes = WordNet noun **synsets** (use synset name strings like `dog.n.01` as node ids).
- For each noun synset, add an edge **child → parent** for each hypernym (use `synset.hypernyms()`).
- WordNet noun hypernymy has a effective root `entity.n.01`. If `root_synset_name` is None, build from the full noun hierarchy (this is large; that's fine — we mostly use subtrees). If provided, build only the subgraph reachable *downward* from that root (all hyponyms transitively), which is the common way to get a clean single-rooted tree.
- Return the DiGraph. Also expose `to_undirected_tree(graph)` returning an undirected `networkx.Graph`.

**Important structural note:** WordNet noun hypernymy is *almost* a tree but has some nodes with multiple parents (it's a DAG). For this project we need clean **trees**. Provide `dag_to_tree(graph, root)` that converts to a tree by keeping, for each node, only the parent on the shortest path to the root (break ties deterministically by sorted synset name). Document this choice in a docstring — it is a deliberate simplification so tree-distance is well-defined.

## Task 3 — Tree distance
Function `tree_distance(tree_undirected, u, v) -> int`:
- Number of edges on the unique path between `u` and `v` in the undirected tree.
- Use `networkx.shortest_path_length`.
Function `all_pairs_tree_distances(tree_undirected) -> dict | np.ndarray`:
- For small subtrees (≤ a few thousand nodes) compute all-pairs distances (via BFS from each node). Return in a form indexed by node id (e.g., dict-of-dicts, plus a node-index mapping and a dense matrix for speed). Document memory: only used on subtrees, not the full graph.

## Task 4 — Structure-controlled subtree sampler (the important one)
Function `sample_subtree(graph, target_size: int, mode: str, seed: int, root_synset_name: str | None = None) -> networkx.Graph` returning an undirected tree with roughly `target_size` nodes.

Support these `mode`s, all deterministic given `seed`:
- `"bfs"`: pick a root, BFS outward until `target_size` nodes are collected → produces a broad, shallow subtree.
- `"deep"`: bias sampling toward following long hypernym chains → produces a deep, narrow subtree.
- `"balanced"`: try to keep branching roughly uniform per level.
- `"random_root_bfs"`: pick a random synset as root, then BFS.

The point of these modes is to produce subtrees that **span a range of structural properties** (depth, branching factor, and later δ-hyperbolicity), so downstream experiments can study structure-dependence. Prioritize getting varied structure over exact size.

## Task 5 — Structural statistics
Function `subtree_stats(tree_undirected, root) -> dict` returning:
- `num_nodes`, `num_edges`
- `depth` (max distance from root to any leaf)
- `avg_branching` (mean number of children over internal nodes)
- `max_branching`
- `num_leaves`
- `diameter` (longest shortest-path in the tree)
(δ-hyperbolicity is intentionally NOT computed here — it lives in the metrics stage. Leave a `# TODO Stage 4: delta-hyperbolicity` note.)

## Task 6 — Persistence
- Function `save_subtree(tree, root, path)` and `load_subtree(path)` using a simple, human-inspectable format: an edge list (`.edges` text file, `child parent` per line) plus a small JSON sidecar with `root` and the `subtree_stats` dict.
- Deterministic ordering (sorted) so files are reproducible/diffable.

## Task 7 — CLI
`python -m src.data --make-subtrees` should:
- Build the graph (default root `entity.n.01` or a documented smaller root like `animal.n.01` for speed — pick `animal.n.01` as the DEFAULT dev root to keep it fast; make it a `--root` flag).
- Sample a small **suite** of subtrees across modes and a few target sizes (e.g. sizes {50, 100, 200} × modes {bfs, deep, balanced}), seeds fixed.
- Save each to `data/subtrees/<name>/`.
- Print and save (`data/subtrees/summary.csv`) a table: subtree name, mode, target size, and all `subtree_stats`.
Add `--root`, `--sizes`, `--modes`, `--seed`, `--out` flags with sensible defaults.

## Task 8 — Tests (`tests/test_data.py`)
Build a **tiny hand-checkable tree** in the test (do NOT need WordNet for these):
```
        A
       / \
      B   C
     / \
    D   E
```
Assert, using your undirected-tree + tree_distance functions on this hand-built graph:
- `tree_distance(A,B) == 1`
- `tree_distance(A,D) == 2`
- `tree_distance(D,E) == 2`
- `tree_distance(D,C) == 3`
- `subtree_stats` returns `num_nodes == 5`, `depth == 2`, `num_leaves == 3` (D, E, C), `max_branching == 2`.
Add ONE integration test that (a) calls `ensure_wordnet()`, (b) builds from root `animal.n.01`, (c) samples a size-50 bfs subtree, (d) asserts it is a tree (edges == nodes − 1, connected). Mark it so it can be skipped if offline (`pytest.mark.skipif` on a network check), but it should run when online.

---

## Definition of Done — checklist
- [ ] `src/data.py` implements all functions above with docstrings.
- [ ] `dag_to_tree` documented as a deliberate simplification.
- [ ] The tiny-tree unit tests all pass with the exact hand-computed distances.
- [ ] `python -m src.data --make-subtrees` produces `data/subtrees/*/` and `summary.csv`.
- [ ] `summary.csv` shows subtrees with **varied** depth and branching (not all identical) — this is essential for later experiments.
- [ ] Files are deterministic given the seed (re-running produces identical `.edges`).
- [ ] `pytest tests/test_data.py` passes (online integration test included).

## Report back to the human
Report: (1) the `summary.csv` contents (paste the table), (2) the min/max range of `depth` and `avg_branching` across the sampled subtrees, (3) any WordNet nodes-with-multiple-parents count you observed before `dag_to_tree`, and (4) confirmation all tests pass.
**Why (2) matters:** we NEED structural variety across subtrees for the Q2 experiment. If depth/branching barely vary, tell the human — we will adjust the sampler before proceeding.
```
