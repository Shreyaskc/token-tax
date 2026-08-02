"""Turns raw per-sentence premium ratios into a publication-ready estimate
with a confidence interval, via `evalci` — this portfolio's own statistics
library. Every reported number in tokentax carries a CI, computed here rather
than re-implemented, per the portfolio's non-negotiables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import evalci

from . import metrics
from .registry import Tokenizer


@dataclass
class PremiumReport:
    tokenizer: str
    language: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    n: int

    def __repr__(self):
        return (
            f"PremiumReport(tokenizer={self.tokenizer!r}, language={self.language!r}, "
            f"premium={self.estimate:.3f}, {int(self.confidence * 100)}% CI="
            f"[{self.lower:.3f}, {self.upper:.3f}], n={self.n})"
        )


def premium_report(
    tokenizer: Tokenizer,
    sentences_lang: List[str],
    sentences_en: List[str],
    language: str,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    random_state: Optional[int] = None,
) -> PremiumReport:
    """Bootstrap CI on the mean per-sentence premium ratio for one
    (tokenizer, language) pair, computed over aligned parallel sentences.
    """
    ratios = metrics.premium_ratios(tokenizer, sentences_lang, sentences_en)
    result = evalci.ci(
        ratios,
        method="bootstrap",
        confidence=confidence,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    return PremiumReport(
        tokenizer=tokenizer.spec.name,
        language=language,
        estimate=result.estimate,
        lower=result.lower,
        upper=result.upper,
        confidence=result.confidence,
        n=result.n,
    )


def premium_table(
    tokenizer: Tokenizer,
    corpus: Dict[str, List[str]],
    english_key: str = "eng_Latn",
    **ci_kwargs,
) -> List[PremiumReport]:
    """One `PremiumReport` per non-English language present in `corpus`."""
    if english_key not in corpus:
        raise KeyError(f"corpus is missing the English reference language {english_key!r}")
    english = corpus[english_key]
    return [
        premium_report(tokenizer, sentences, english, language=lang, **ci_kwargs)
        for lang, sentences in corpus.items()
        if lang != english_key
    ]
