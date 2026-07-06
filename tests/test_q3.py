"""Fast tests for Q3 HyperLex Task-0 (loader + coverage logic). No embedding."""

import os

import pytest

from src.experiments import exp_q3_hyperlex as q3

HAS_HYPERLEX = os.path.exists(q3.HYPERLEX_NOUNS)


@pytest.mark.skipif(not HAS_HYPERLEX, reason="HyperLex file not present")
def test_loader_parses():
    df = q3.load_hyperlex()
    assert len(df) == 2163  # HyperLex noun subset
    assert set(["word1", "word2", "pos", "rating"]).issubset(df.columns)
    assert (df["pos"] == "N").all()
    assert df["rating"].between(0, 10).all()
    # spot-check a known pair exists
    assert ((df["word1"] == "mason") & (df["word2"] == "worker")).any()


def test_word_maps_to_wordnet():
    assert q3.word_maps_to_wordnet("dog")
    assert not q3.word_maps_to_wordnet("zzzznotaword")


def test_region_membership_handchecked():
    # dog.n.01 sits under carnivore/animal/entity but not under plant.
    assert q3.word_in_region("dog", "entity.n.01")
    assert q3.word_in_region("dog", "animal.n.01")
    assert not q3.word_in_region("dog", "plant.n.02")
    # a plant word is under plant/entity, not animal.
    assert q3.word_in_region("oak", "entity.n.01")
    assert not q3.word_in_region("oak", "animal.n.01")


def test_ancestors_include_self():
    from nltk.corpus import wordnet as wn

    syn = wn.synset("dog.n.01")
    anc = q3._ancestors_inclusive(syn)
    assert "dog.n.01" in anc
    assert "entity.n.01" in anc


def test_score_b_direction_sign():
    # hypernym (general) near origin, hyponym (specific) far out.
    # "hyponym is-a hypernym": gen(w1=hyponym) > gen(w2=hypernym) -> gen_diff>0 -> higher score.
    d = 1.0
    gen_hyponym, gen_hypernym = 0.9, 0.2
    correct = q3.score_b([d], [gen_hyponym - gen_hypernym], lam=1.0)[0]   # w1=hyponym, w2=hypernym
    reverse = q3.score_b([d], [gen_hypernym - gen_hyponym], lam=1.0)[0]   # w1=hypernym, w2=hyponym
    assert correct > reverse  # the true hypernymy direction scores higher


def test_score_pairs_finite_tiny():
    import networkx as nx
    import numpy as np

    # tiny region of 3 nodes with a hand embedding; fake HyperLex pair.
    node_index = {"a.n.01": 0, "b.n.01": 1, "c.n.01": 2}
    coords = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.1]])
    import pandas as pd

    # monkeypatch word->synset to map words to our fake synsets
    orig = q3.word_noun_synsets

    class _S:
        def __init__(self, n): self._n = n
        def name(self): return self._n

    q3.word_noun_synsets = lambda w: {"x": [_S("a.n.01")], "y": [_S("b.n.01")]}.get(w, [])
    q3._word_idx_cache.clear()
    try:
        cov = pd.DataFrame([{"word1": "x", "word2": "y", "rating": 5.0}])
        scored = q3.score_pairs(coords, "poincare", node_index, cov)
        assert len(scored) == 1
        assert np.isfinite(scored["d_min"]).all()
        assert np.isfinite(scored["gen_diff"]).all()
    finally:
        q3.word_noun_synsets = orig
        q3._word_idx_cache.clear()


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    x = [1, 2, 3, 4, 5]
    y = [2, 1, 4, 3, 6]
    assert abs(q3.spearman(x, y) - spearmanr(x, y).statistic) < 1e-12
