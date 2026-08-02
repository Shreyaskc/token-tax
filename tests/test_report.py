import pytest

from tokentax import corpora, report


def test_premium_report_basic(fake_tokenizer):
    lang = ["a b c d", "a b c d e f", "a b c d e f g h"]
    en = ["a b", "a b c", "a b c d"]
    r = report.premium_report(
        fake_tokenizer, lang, en, language="test_lang", n_resamples=499, random_state=0
    )
    assert r.tokenizer == "fake-whitespace"
    assert r.language == "test_lang"
    assert r.lower <= r.estimate <= r.upper
    assert r.n == 3


def test_premium_table_toy_corpus(fake_tokenizer):
    corpus = corpora.load_toy_corpus()
    reports = report.premium_table(fake_tokenizer, corpus, n_resamples=199, random_state=0)
    languages = {r.language for r in reports}
    assert languages == set(corpus) - {"eng_Latn"}
    for r in reports:
        assert r.lower <= r.estimate <= r.upper


def test_premium_table_missing_english_key_raises(fake_tokenizer):
    corpus = {"tam_Taml": ["a b"]}
    with pytest.raises(KeyError):
        report.premium_table(fake_tokenizer, corpus)
