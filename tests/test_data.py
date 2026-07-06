"""Tests for the data layer: tiny hand-checkable tree + one online integration test."""

import socket

import networkx as nx
import pytest

from src import data


def _build_tiny_tree():
    """Hand-built tree:

            A
           / \\
          B   C
         / \\
        D   E
    """
    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")])
    return g


def test_tree_distances_hand_checked():
    g = _build_tiny_tree()
    assert data.tree_distance(g, "A", "B") == 1
    assert data.tree_distance(g, "A", "D") == 2
    assert data.tree_distance(g, "D", "E") == 2
    assert data.tree_distance(g, "D", "C") == 3


def test_subtree_stats_hand_checked():
    g = _build_tiny_tree()
    stats = data.subtree_stats(g, "A")
    assert stats["num_nodes"] == 5
    assert stats["num_edges"] == 4
    assert stats["depth"] == 2
    assert stats["num_leaves"] == 3  # D, E, C
    assert stats["max_branching"] == 2


def test_all_pairs_distances_hand_checked():
    g = _build_tiny_tree()
    apsp = data.all_pairs_tree_distances(g)
    idx = apsp["index"]
    m = apsp["matrix"]
    assert m[idx["D"], idx["C"]] == 3
    assert m[idx["A"], idx["A"]] == 0
    assert apsp["dists"]["D"]["E"] == 2


def test_save_load_roundtrip(tmp_path):
    g = _build_tiny_tree()
    path = tmp_path / "tiny"
    data.save_subtree(g, "A", str(path))
    tree, meta = data.load_subtree(str(path))
    assert meta["root"] == "A"
    assert tree.number_of_nodes() == 5
    assert data.tree_distance(tree, "D", "C") == 3


def _online(host="wordnetweb.princeton.edu", timeout=3) -> bool:
    # Cheap connectivity probe; the corpus may already be cached locally, but we
    # skip the integration test if there's no network AND no cache.
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
        return True  # already cached, no network needed
    except LookupError:
        pass
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online(), reason="offline and WordNet not cached")
def test_wordnet_integration_bfs_subtree():
    data.ensure_wordnet()
    raw = data.build_wordnet_noun_graph("animal.n.01")
    tree_graph = data.dag_to_tree(raw, "animal.n.01")
    tree = data.sample_subtree(
        tree_graph, target_size=50, mode="bfs", seed=0, root_synset_name="animal.n.01"
    )
    n = tree.number_of_nodes()
    # A tree has exactly n-1 edges and is connected.
    assert tree.number_of_edges() == n - 1
    assert nx.is_connected(tree)
    assert n <= 50
