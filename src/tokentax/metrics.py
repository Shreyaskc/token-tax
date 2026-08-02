"""Core tokenizer-cost metrics, computed per (tokenizer, language) pair."""
from __future__ import annotations

from typing import List

import numpy as np

from .registry import Tokenizer


def token_counts(tokenizer: Tokenizer, sentences: List[str]) -> np.ndarray:
    return np.array([tokenizer.count(s) for s in sentences], dtype=np.int64)


def premium_ratios(
    tokenizer: Tokenizer, sentences_lang: List[str], sentences_en: List[str]
) -> np.ndarray:
    """Per-sentence token-count ratio vs. the aligned English sentence.

    Requires `sentences_lang[i]` and `sentences_en[i]` to be translations of
    the same content (e.g. aligned FLORES rows) — the ratio is meaningless on
    independently sampled monolingual text.
    """
    if len(sentences_lang) != len(sentences_en):
        raise ValueError("parallel corpora must be the same length (aligned sentence pairs)")
    lang_counts = token_counts(tokenizer, sentences_lang)
    en_counts = token_counts(tokenizer, sentences_en)
    if np.any(en_counts == 0):
        raise ValueError("encountered a zero-token English reference sentence")
    return lang_counts / en_counts


def bytes_per_token(tokenizer: Tokenizer, sentences: List[str]) -> float:
    counts = token_counts(tokenizer, sentences)
    total_tokens = int(counts.sum())
    if total_tokens == 0:
        raise ValueError("zero total tokens across corpus")
    total_bytes = sum(len(s.encode("utf-8")) for s in sentences)
    return total_bytes / total_tokens


def chars_per_token(tokenizer: Tokenizer, sentences: List[str]) -> float:
    counts = token_counts(tokenizer, sentences)
    total_tokens = int(counts.sum())
    if total_tokens == 0:
        raise ValueError("zero total tokens across corpus")
    total_chars = sum(len(s) for s in sentences)
    return total_chars / total_tokens


def effective_context_window(context_tokens: int, tokens_per_word: float) -> float:
    """How many *words* of this language fit in a `context_tokens`-token window,
    given the language's average tokens-per-word (e.g. from `chars_per_token`
    combined with average word length).
    """
    if tokens_per_word <= 0:
        raise ValueError("tokens_per_word must be positive")
    return context_tokens / tokens_per_word
