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


def test_flores200_languages_are_unique_and_include_english():
    assert len(corpora.FLORES200_LANGUAGES) == len(set(corpora.FLORES200_LANGUAGES)) == 204
    assert "eng_Latn" in corpora.FLORES200_LANGUAGES


def test_load_opus100_pair_unmapped_language_raises():
    with pytest.raises(KeyError):
        corpora.load_opus100_pair("not_a_real_lang")


def test_load_bible_corpus_pair_unmapped_language_raises():
    with pytest.raises(KeyError):
        corpora.load_bible_corpus_pair("not_a_real_lang")
