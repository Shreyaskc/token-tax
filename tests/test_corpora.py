import pytest

from tokentax import corpora


def test_load_toy_corpus_aligned_lengths():
    corpus = corpora.load_toy_corpus()
    lengths = {len(sents) for sents in corpus.values()}
    assert len(lengths) == 1  # every language has the same sentence count


def test_load_toy_corpus_has_english():
    corpus = corpora.load_toy_corpus()
    assert "eng_Latn" in corpus


def test_load_toy_corpus_returns_copy():
    a = corpora.load_toy_corpus()
    a["eng_Latn"].append("mutated")
    b = corpora.load_toy_corpus()
    assert "mutated" not in b["eng_Latn"]


def test_load_flores200_requires_explicit_languages():
    with pytest.raises(ValueError):
        corpora.load_flores200(languages=None)
