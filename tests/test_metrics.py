import pytest

from tokentax import metrics


def test_token_counts(fake_tokenizer):
    counts = metrics.token_counts(fake_tokenizer, ["a b c", "a b"])
    assert list(counts) == [3, 2]


def test_premium_ratios(fake_tokenizer):
    lang = ["a b c d", "a b c d e f"]
    en = ["a b", "a b c"]
    ratios = metrics.premium_ratios(fake_tokenizer, lang, en)
    assert list(ratios) == [2.0, 2.0]


def test_premium_ratios_mismatched_length_raises(fake_tokenizer):
    with pytest.raises(ValueError):
        metrics.premium_ratios(fake_tokenizer, ["a"], ["a", "b"])


def test_premium_ratios_zero_english_tokens_raises(fake_tokenizer):
    with pytest.raises(ValueError):
        metrics.premium_ratios(fake_tokenizer, ["a b"], [""])


def test_bytes_per_token(fake_tokenizer):
    value = metrics.bytes_per_token(fake_tokenizer, ["ab cd"])
    assert value == pytest.approx(5 / 2)


def test_chars_per_token(fake_tokenizer):
    value = metrics.chars_per_token(fake_tokenizer, ["ab cd"])
    assert value == pytest.approx(5 / 2)


def test_bytes_per_token_zero_tokens_raises(fake_tokenizer):
    with pytest.raises(ValueError):
        metrics.bytes_per_token(fake_tokenizer, [""])


def test_effective_context_window():
    assert metrics.effective_context_window(1000, 2.0) == 500


def test_effective_context_window_invalid():
    with pytest.raises(ValueError):
        metrics.effective_context_window(1000, 0)
