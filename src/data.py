"""Data layer: WordNet noun hierarchy + structure-controlled subtree sampler.

This module builds the WordNet noun `is-a` (hypernym) hierarchy as a graph,
converts it into a clean single-rooted tree, computes tree distances, and — most
importantly for this project — samples **subtrees with controllable structure**
(broad/shallow, deep/narrow, balanced). Those subtrees are the substrate for the
Q2 experiment, which studies how embedding quality depends on tree structure
(depth, branching, and later δ-hyperbolicity).

No geometry, embeddings, training, or metrics live here — data only.

Graph convention
----------------
The directed graph stores edges as **child → parent** (i.e. an edge points from a
synset to its hypernym). Therefore, in a DiGraph ``g``:
    * ``g.successors(n)``   -> the parents (hypernyms) of ``n``
    * ``g.predecessors(n)`` -> the children (hyponyms) of ``n``
The tree root (e.g. ``entity.n.01`` or ``animal.n.01``) is the unique node with
no parents inside the graph (out-degree 0).
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
from random import Random

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Task 1 — NLTK WordNet bootstrap
# ---------------------------------------------------------------------------

_WORDNET_READY = False


def ensure_wordnet() -> None:
    """Idempotently ensure the WordNet (and OMW) corpora are available.

    Downloads via ``nltk.download`` only if the corpora cannot be found, so it
    is safe to call repeatedly and cheap on the happy path.
    """
    global _WORDNET_READY
    if _WORDNET_READY:
        return

    import nltk

    def _have(resource: str, path: str) -> bool:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            return False

    if not _have("wordnet", "corpora/wordnet"):
        nltk.download("wordnet", quiet=True)
    if not _have("omw-1.4", "corpora/omw-1.4"):
        # OMW is needed by some WordNet operations on newer NLTK versions.
        nltk.download("omw-1.4", quiet=True)

    _WORDNET_READY = True


# ---------------------------------------------------------------------------
# Task 2 — Build the noun hypernym graph
# ---------------------------------------------------------------------------


def build_wordnet_noun_graph(root_synset_name: str | None = None) -> nx.DiGraph:
    """Build the WordNet noun hypernym graph as a DiGraph of child → parent edges.

    Nodes are synset name strings (e.g. ``"dog.n.01"``). For each noun synset an
    edge is added from the synset to each of its hypernyms.

    If ``root_synset_name`` is ``None`` the full noun hierarchy is built (large,
    ~74k nodes — that's fine, we mostly operate on subtrees). If a root is given,
    only the subgraph reachable **downward** from that root (all transitive
    hyponyms) is built, and cross-links to hypernyms outside that subtree are
    dropped so the result is a clean single-rooted region.

    Note: WordNet noun hypernymy is *almost* a tree but is really a DAG (some
    nodes have multiple hypernyms). Use :func:`dag_to_tree` to obtain a proper
    tree. The returned graph exposes ``g.graph["root"]`` (the requested root, or
    ``None`` for the full hierarchy).
    """
    ensure_wordnet()
    from nltk.corpus import wordnet as wn

    g = nx.DiGraph()

    if root_synset_name is None:
        for syn in wn.all_synsets("n"):
            name = syn.name()
            g.add_node(name)
            for hyper in syn.hypernyms():
                g.add_edge(name, hyper.name())
        g.graph["root"] = None
        return g

    # Rooted build: collect the transitive hyponym closure of the root.
    root = wn.synset(root_synset_name)
    descendants: set[str] = set()
    stack = [root]
    while stack:
        syn = stack.pop()
        if syn.name() in descendants:
            continue
        descendants.add(syn.name())
        for hypo in syn.hyponyms():
            if hypo.name() not in descendants:
                stack.append(hypo)

    for name in descendants:
        g.add_node(name)
        for hyper in wn.synset(name).hypernyms():
            # Keep only edges that stay within the rooted subtree.
            if hyper.name() in descendants:
                g.add_edge(name, hyper.name())

    g.graph["root"] = root_synset_name
    return g


def to_undirected_tree(graph: nx.Graph) -> nx.Graph:
    """Return an undirected copy of ``graph`` (for symmetric path distances).

    Graph-level attributes such as ``root`` are preserved by NetworkX.
    """
    return graph.to_undirected()


def dag_to_tree(graph: nx.DiGraph, root: str) -> nx.DiGraph:
    """Collapse a child → parent DAG into a proper tree rooted at ``root``.

    **Deliberate simplification.** WordNet noun hypernymy is a DAG: a handful of
    synsets have more than one hypernym. Tree-distance is only well-defined on a
    tree, so we convert the DAG to a tree by keeping, for each node, exactly one
    parent — the parent lying on a shortest path to ``root``. Ties (parents at
    equal distance) are broken deterministically by sorted synset name. This
    discards a small number of secondary parent edges; that is an intentional,
    documented modelling choice for this study, not a bug.

    Returns a new child → parent DiGraph with ``graph["root"] == root``.
    """
    undirected = graph.to_undirected()
    dist_to_root = nx.single_source_shortest_path_length(undirected, root)

    tree = nx.DiGraph()
    tree.add_nodes_from(graph.nodes())
    for node in graph.nodes():
        if node == root:
            continue
        candidates = [p for p in graph.successors(node) if p in dist_to_root]
        if not candidates:
            continue
        # Prefer the parent closest to the root; tie-break by name.
        candidates.sort(key=lambda p: (dist_to_root[p], p))
        tree.add_edge(node, candidates[0])

    tree.graph["root"] = root
    return tree


# ---------------------------------------------------------------------------
# Task 3 — Tree distance
# ---------------------------------------------------------------------------


def tree_distance(tree_undirected: nx.Graph, u: str, v: str) -> int:
    """Number of edges on the unique path between ``u`` and ``v`` in a tree."""
    return nx.shortest_path_length(tree_undirected, u, v)


def all_pairs_tree_distances(tree_undirected: nx.Graph) -> dict:
    """Compute all-pairs shortest-path (tree) distances via BFS from each node.

    Returns a dict with:
        * ``"nodes"``  — list of node ids in a fixed (sorted) order,
        * ``"index"``  — mapping node id -> row/col index,
        * ``"matrix"`` — dense ``np.ndarray`` (int32) of pairwise distances,
        * ``"dists"``  — dict-of-dicts ``{u: {v: distance}}``.

    Memory note: this materialises an ``N x N`` matrix, so it is intended for
    **subtrees** (up to a few thousand nodes), not the full WordNet graph.
    """
    nodes = sorted(tree_undirected.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.int32)
    dists: dict[str, dict[str, int]] = {}

    for src in nodes:
        lengths = nx.single_source_shortest_path_length(tree_undirected, src)
        dists[src] = lengths
        i = index[src]
        for dst, d in lengths.items():
            matrix[i, index[dst]] = d

    return {"nodes": nodes, "index": index, "matrix": matrix, "dists": dists}


# ---------------------------------------------------------------------------
# Task 4 — Structure-controlled subtree sampler
# ---------------------------------------------------------------------------


def _grow_subtree(
    graph: nx.DiGraph,
    root: str,
    target_size: int,
    rng: Random,
    policy: str,
    max_children: int | None,
) -> nx.Graph:
    """Grow an undirected subtree downward from ``root`` toward ``target_size``.

    Nodes are added one child at a time from an active frontier of nodes that
    still have unexpanded children. A node stays in the frontier until it either
    runs out of children or reaches ``max_children`` (``None`` = no cap), so —
    unlike a plain capped BFS/DFS — capping shapes branching *without* starving
    growth. ``policy`` selects which frontier node to expand next:

        * ``"shallow"`` — expand the shallowest node first (breadth-first fill).
          With ``max_children=None`` this is broad/shallow; with a small cap it
          fills level by level with roughly uniform branching (balanced).
        * ``"deep"``    — expand the deepest node first, so the tree dives down
          long hypernym chains before widening (deep/narrow).

    Children are the graph's predecessors (hyponyms); each node attaches to
    exactly one parent, so the result is always a tree.

    Relaxation: the WordNet subtree has bottleneck "gateway" nodes (most of the
    hierarchy hangs off a few high-degree children), so a hard branching cap can
    cut off access to the bulk of the tree. To still reach ``target_size``,
    nodes that hit the cap but still have children are held in a reserve and
    re-activated (their cap budget reset) only once the frontier would otherwise
    empty. Thus the cap shapes the tree first, and only widens under pressure.
    """
    tree = nx.Graph()
    tree.add_node(root)
    depth = {root: 0}
    remaining: dict[str, list[str]] = {}
    added: dict[str, int] = {}
    counter = itertools.count()
    heap: list[tuple[int, int, str]] = []
    reserve: list[str] = []

    def _unexpanded_children(node: str) -> list[str]:
        children = [c for c in graph.predecessors(node) if c not in tree]
        children.sort()
        rng.shuffle(children)
        return children

    def _push(node: str) -> None:
        key = -depth[node] if policy == "deep" else depth[node]
        heapq.heappush(heap, (key, next(counter), node))

    def _activate(node: str) -> None:
        remaining[node] = _unexpanded_children(node)
        added[node] = 0
        if remaining[node]:
            _push(node)

    _activate(root)

    while tree.number_of_nodes() < target_size:
        if not heap:
            if not reserve:
                break  # whole reachable subtree exhausted
            # Relax: give capped-but-unexhausted nodes a fresh cap budget.
            for node in reserve:
                if remaining[node]:
                    added[node] = 0
                    _push(node)
            reserve = []
            continue

        _key, _tie, node = heapq.heappop(heap)
        rem = remaining[node]
        if not rem:
            continue
        child = rem.pop()
        tree.add_edge(node, child)
        depth[child] = depth[node] + 1
        added[node] += 1
        if rem:
            if max_children is None or added[node] < max_children:
                _push(node)
            else:
                reserve.append(node)  # capped, but keep for possible relaxation
        _activate(child)

    tree.graph["root"] = root
    return tree


def sample_subtree(
    graph: nx.DiGraph,
    target_size: int,
    mode: str,
    seed: int,
    root_synset_name: str | None = None,
) -> nx.Graph:
    """Sample an undirected subtree of roughly ``target_size`` nodes.

    All modes are deterministic given ``seed``. The point of the modes is to span
    a range of structural properties (depth, branching), so exact size is
    secondary to structural variety.

    Modes:
        * ``"bfs"``             — BFS from ``root``, take all children -> broad, shallow.
        * ``"deep"``            — DFS with a small branching cap -> deep, narrow.
        * ``"balanced"``        — BFS with a moderate branching cap -> uniform branching.
        * ``"random_root_bfs"`` — pick a random synset as root, then BFS.

    ``root_synset_name`` selects the starting root for the non-random modes; if
    omitted it defaults to ``graph.graph["root"]``. The returned tree carries its
    root in ``tree.graph["root"]``.
    """
    rng = Random(seed)

    if mode == "random_root_bfs":
        root = rng.choice(sorted(graph.nodes()))
        return _grow_subtree(graph, root, target_size, rng, policy="shallow", max_children=None)

    root = root_synset_name or graph.graph.get("root")
    if root is None:
        raise ValueError("root_synset_name is required (or graph must carry graph['root']).")
    if root not in graph:
        raise ValueError(f"root {root!r} is not a node in the graph.")

    if mode == "bfs":
        return _grow_subtree(graph, root, target_size, rng, policy="shallow", max_children=None)
    if mode == "balanced":
        return _grow_subtree(graph, root, target_size, rng, policy="shallow", max_children=3)
    if mode == "deep":
        return _grow_subtree(graph, root, target_size, rng, policy="deep", max_children=2)

    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Task 5 — Structural statistics
# ---------------------------------------------------------------------------


def subtree_stats(tree_undirected: nx.Graph, root: str) -> dict:
    """Compute structural statistics of a rooted undirected tree.

    Returns ``num_nodes``, ``num_edges``, ``depth`` (max root-to-leaf distance),
    ``avg_branching`` (mean children over internal nodes), ``max_branching``,
    ``num_leaves``, and ``diameter`` (longest shortest path in the tree).
    """
    # TODO Stage 4: delta-hyperbolicity (lives in the metrics stage, not here).
    num_nodes = tree_undirected.number_of_nodes()
    num_edges = tree_undirected.number_of_edges()

    depths = nx.single_source_shortest_path_length(tree_undirected, root)
    depth = max(depths.values()) if depths else 0

    # Orient the tree from the root to count children per node.
    child_count = {n: 0 for n in tree_undirected.nodes()}
    for parent, _child in nx.bfs_edges(tree_undirected, root):
        child_count[parent] += 1

    internal = [c for c in child_count.values() if c > 0]
    avg_branching = float(np.mean(internal)) if internal else 0.0
    max_branching = max(child_count.values()) if child_count else 0
    num_leaves = sum(1 for c in child_count.values() if c == 0)
    diameter = nx.diameter(tree_undirected) if num_nodes > 1 else 0

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "depth": depth,
        "avg_branching": avg_branching,
        "max_branching": max_branching,
        "num_leaves": num_leaves,
        "diameter": diameter,
    }


# ---------------------------------------------------------------------------
# Task 6 — Persistence
# ---------------------------------------------------------------------------


def save_subtree(tree: nx.Graph, root: str, path: str) -> None:
    """Persist a subtree to ``path/`` as a human-inspectable edge list + JSON.

    Writes ``edges.txt`` (``child parent`` per line, sorted for determinism) and
    ``meta.json`` (root + :func:`subtree_stats`). Edge orientation is recovered
    from the root: the endpoint closer to the root is the parent.
    """
    os.makedirs(path, exist_ok=True)
    depths = nx.single_source_shortest_path_length(tree, root)

    lines = []
    for u, v in tree.edges():
        if depths[u] < depths[v]:
            parent, child = u, v
        elif depths[v] < depths[u]:
            parent, child = v, u
        else:  # should not happen in a tree; keep deterministic anyway
            parent, child = sorted((u, v))
        lines.append((child, parent))
    lines.sort()

    with open(os.path.join(path, "edges.txt"), "w") as f:
        for child, parent in lines:
            f.write(f"{child} {parent}\n")

    meta = {"root": root, "stats": subtree_stats(tree, root)}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_subtree(path: str) -> tuple[nx.Graph, dict]:
    """Load a subtree saved by :func:`save_subtree`.

    Returns ``(tree, meta)`` where ``tree`` is an undirected graph with
    ``tree.graph["root"]`` set, and ``meta`` is the JSON sidecar dict.
    """
    tree = nx.Graph()
    with open(os.path.join(path, "edges.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            child, parent = line.split()
            tree.add_edge(child, parent)

    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)

    root = meta.get("root")
    tree.graph["root"] = root
    if root is not None and root not in tree:
        tree.add_node(root)  # single-node / isolated-root safety
    return tree, meta


# ---------------------------------------------------------------------------
# Task 7 — CLI
# ---------------------------------------------------------------------------

# Deterministic per-(mode) seed offsets so re-runs are reproducible.
_MODE_OFFSET = {"bfs": 0, "deep": 1, "balanced": 2, "random_root_bfs": 3}


def _make_subtrees(args: argparse.Namespace) -> None:
    import pandas as pd

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"Building WordNet noun graph rooted at {args.root!r} ...")
    raw = build_wordnet_noun_graph(args.root)
    multi_parent = sum(1 for n in raw.nodes() if raw.out_degree(n) > 1)
    print(
        f"  raw graph: {raw.number_of_nodes()} nodes, {raw.number_of_edges()} edges, "
        f"{multi_parent} nodes with multiple parents (DAG links dropped by dag_to_tree)."
    )

    tree_graph = dag_to_tree(raw, args.root)
    print(
        f"  tree graph: {tree_graph.number_of_nodes()} nodes, "
        f"{tree_graph.number_of_edges()} edges."
    )

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for mode in modes:
        for size in sizes:
            sub_seed = args.seed * 1000 + _MODE_OFFSET.get(mode, 9) * 100 + size
            tree = sample_subtree(
                tree_graph, target_size=size, mode=mode, seed=sub_seed,
                root_synset_name=args.root,
            )
            root = tree.graph["root"]
            stats = subtree_stats(tree, root)
            name = f"{mode}_s{size}"
            save_subtree(tree, root, os.path.join(out_dir, name))
            row = {"name": name, "mode": mode, "target_size": size, "root": root}
            row.update(stats)
            rows.append(row)
            print(
                f"  {name:20s} nodes={stats['num_nodes']:4d} depth={stats['depth']:3d} "
                f"avg_branch={stats['avg_branching']:.2f} max_branch={stats['max_branching']:3d} "
                f"leaves={stats['num_leaves']:4d} diameter={stats['diameter']:3d}"
            )

    df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nWrote {len(rows)} subtrees and summary -> {summary_path}")
    print(
        f"depth range: [{df['depth'].min()}, {df['depth'].max()}]  |  "
        f"avg_branching range: [{df['avg_branching'].min():.2f}, {df['avg_branching'].max():.2f}]"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WordNet subtree data layer.")
    p.add_argument("--make-subtrees", action="store_true", help="Sample a suite of subtrees.")
    p.add_argument("--root", default="animal.n.01", help="Root synset (default: animal.n.01 for speed).")
    p.add_argument("--sizes", default="50,100,200", help="Comma-separated target sizes.")
    p.add_argument("--modes", default="bfs,deep,balanced", help="Comma-separated sampler modes.")
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--out", default="data/subtrees", help="Output directory.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.make_subtrees:
        _make_subtrees(args)
    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
