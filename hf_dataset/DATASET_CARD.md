---
license: cc0-1.0
language:
  - en
  - multilingual
tags:
  - tokenization
  - multilingual-nlp
  - llm-evaluation
  - benchmark
  - fairness
pretty_name: tokentax Results v1
size_categories:
  - 1M<n<10M
---

# tokentax-results-v1

Measurements of the "token tax" — the tokenization premium, i.e. how many
more tokens a language costs relative to English for the same content —
across 8 LLM tokenizers and 203 FLORES-200 languages, with confidence
intervals computed via [`evalci`](https://pypi.org/project/evalci/).

Produced by [tokentax](https://github.com/Shreyaskc/token-tax) (commit
[`0e5e24d`](https://github.com/Shreyaskc/token-tax/commit/0e5e24dbf142cc4b1a685c9a289db5f862fcccf2)).
Fully re-runnable: `scripts/run_flores_sweep.py` →
`scripts/summarize_with_evalci.py` → `scripts/run_opus_domain_check.py` in
that repo regenerate every file here from scratch.

## No original sentence text is redistributed

This dataset contains only token counts, premium ratios, and confidence
intervals — **not** the source sentences themselves. Premium ratios were
computed against FLORES-200 (`facebook/flores`, gated, CC-BY-SA-4.0),
OPUS-100 (`Helsinki-NLP/opus-100`), and a Bible-translation corpus
(`davidstap/biblenlp-corpus-mmteb`); none of that text is copied here. See
LICENSE for the license of the derived statistics vs. the source corpora.

## Files

### `flores200_summary.parquet`

The primary results table: long-format `(tokenizer, tokenizer_version,
language, corpus, metric, value)`, one row per metric per (tokenizer,
language) pair, from the full FLORES-200 devtest sweep (8 tokenizers ×
203 non-English languages).

| column | description |
|---|---|
| `tokenizer` | one of `gpt-4o`, `gpt-4`, `gpt-2`, `llama-3`, `qwen2.5`, `deepseek-v3`, `mistral`, `gemma-2` (Claude excluded from this release, see Limitations) |
| `tokenizer_version` | pinned Hugging Face revision, or `None` for tiktoken-backed tokenizers |
| `language` | FLORES-200 code, e.g. `tam_Taml` |
| `corpus` | `flores200_devtest` |
| `metric` | `premium_ratio_estimate`, `premium_ratio_ci_lower`, `premium_ratio_ci_upper`, or `n_sentences` |
| `value` | the metric's value |

`premium_ratio` is `tokens(language sentence) / tokens(aligned English
sentence)`, mean over 1012 aligned FLORES-200 devtest sentences; CIs are a
95% bootstrap interval via `evalci.ci(method="bootstrap")`.

### `flores200_raw.parquet`

Raw per-sentence premium ratios underlying the summary above — 1.64M rows,
one per (tokenizer, language, sentence). Kept so anyone can recompute a
different CI method, subset languages, or check for outlier sentences
without re-tokenizing FLORES-200 themselves.

| column | description |
|---|---|
| `tokenizer` | tokenizer name |
| `tokenizer_revision` | pinned HF revision |
| `language` | FLORES-200 code |
| `corpus` | `flores200_devtest` |
| `sentence_idx` | position in the devtest split (0–1011) |
| `premium_ratio` | tokens(this sentence) / tokens(aligned English sentence) |

### `opus_domain_check.parquet`

Domain-robustness check: for the 30 languages with the highest mean premium
in `flores200_summary`, compares the FLORES-200 estimate against a second,
non-encyclopedic corpus (OPUS-100 mixed-domain text and/or a Bible-corpus
religious-register text), where one is available.

| column | description |
|---|---|
| `language` | FLORES-200 code |
| `domain` | `opus100` or `bible` |
| `tokenizer` | tokenizer name |
| `flores_premium` | the FLORES-200 devtest estimate for this (tokenizer, language) |
| `domain_premium` | the estimate on this domain's own aligned English pair |
| `domain_ci_lower`, `domain_ci_upper` | 95% bootstrap CI on `domain_premium` |
| `n_sentences` | sentence count used for this domain's estimate |
| `rel_diff` | `\|domain_premium - flores_premium\| / flores_premium` |
| `flag` | `DIVERGES` if `rel_diff` exceeds 35%, else `ok` |

## Coverage and known limitations

- **Claude is excluded** from this release. Its tokenizer has no downloadable
  artifact (only Anthropic's token-count API), so including it would require
  every user of this dataset to hold an API key just to reproduce it. See the
  source repo's `registry.py` to re-add it.
- **`llama-3` resolves to the `NousResearch/Meta-Llama-3-8B` mirror**, not
  `meta-llama/Meta-Llama-3-8B` — the official repo requires Meta's manual
  license approval rather than an instant click-through.
- **Domain-robustness coverage is partial by construction.** Of the 30
  highest-premium languages, 9 (Shan, Santali, Dzongkha, Tamasheq, Central
  Atlas Tamazight, Lao, Tigrinya, Manipuri, Kabiyè) have no modern, ungated,
  non-loading-script parallel corpus available on Hugging Face for a second
  domain at all. That gap is reported explicitly in `opus_domain_check`
  (absent rows), not silently smoothed over — the languages with the
  highest token tax also tend to have the least data to cross-validate it.
  Where a second domain IS available, most (language, tokenizer, domain)
  triples replicate the FLORES estimate within ~14% (median relative
  difference across 264 comparisons); a handful of outliers exceed 35%
  (notably Kannada on OPUS-100, and Uyghur/Sanskrit on the Bible corpus,
  likely corpus-specific sampling artifacts, not a general finding).
- **Reproduces Petrov et al. 2023** ("Language Model Tokenizers Introduce
  Unfairness Between Languages," NeurIPS 2023) within 1.1% max relative error
  on GPT-2/GPT-4 across 5 languages — see the source repo's
  `scripts/validate_against_petrov2023.py`.

## Citation

An arXiv paper is planned; until it's out, cite the software/dataset
directly:

```bibtex
@misc{chandrahas2026tokentax,
  title  = {tokentax: The Tokenizer Cost Penalty Across Languages},
  author = {Chandrahas, Shreyas K},
  year   = {2026},
  url    = {https://github.com/Shreyaskc/token-tax}
}
```

## Versioning

This is `v1`, built from FLORES-200 devtest with the tokenizers listed above.
Re-runs after new tokenizer releases will ship as `tokentax-results-v2`, etc.,
each independently versioned per the source repo's release policy.
