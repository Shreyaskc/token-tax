# tokentax — The Tokenizer Cost Penalty Across Languages

[![PyPI](https://img.shields.io/pypi/v/tokentax)](https://pypi.org/project/tokentax/)
[![tests](https://github.com/Shreyaskc/token-tax/actions/workflows/tests.yml/badge.svg)](https://github.com/Shreyaskc/token-tax/actions/workflows/tests.yml)

**Class:** Dataset + measurement tool
**Citation anchor:** arXiv/ACL paper + versioned dataset DOI

`tokentax` quantifies the "token tax": the same sentence costs 1× tokens in
English but often 2–8× in Tamil, Amharic, or Burmese, because LLM tokenizers
are trained on English-heavy corpora. It computes tokenization premium,
effective context-window shrinkage, and API cost multipliers for major
tokenizers across 100+ languages on parallel corpora, with confidence
intervals via [`evalci`](https://pypi.org/project/evalci/).

```python
>>> import tokentax
>>> tok = tokentax.load_tokenizer("gpt-4o")
>>> corpus = tokentax.corpora.load_toy_corpus()
>>> tokentax.premium_report(tok, corpus["tam_Taml"], corpus["eng_Latn"], language="tam_Taml")
PremiumReport(tokenizer='gpt-4o', language='tam_Taml', premium=1.964, 95% CI=[1.673, 2.256], n=4)
```

## Status

Phase 1 (core pipeline) and Phase 2 (full FLORES-200 sweep) are done. The
pipeline is validated: it reproduces Petrov et al. 2023's published premium
ratios (GPT-2/GPT-4 tokenizers, five languages) within 1.1% — see
`scripts/validate_against_petrov2023.py`. The full sweep covers 8 tokenizers
(gpt-4o, gpt-4, gpt-2, llama-3, qwen2.5, deepseek-v3, mistral, gemma-2 — Claude
excluded, see below) × 203 non-English FLORES-200 languages, 1.64M raw
per-sentence rows, summarized with `evalci`-backed CIs in
`data/results/tokentax_summary_devtest.csv`. Newer tokenizers show a markedly
lower mean premium (gpt-4o/gemma-2 ≈ 2.1×) than legacy ones (gpt-2 ≈ 4.5×) —
the "are newer tokenizers fairer" trend the paper will explore.

The OPUS domain-robustness check (`scripts/run_opus_domain_check.py`) is also
done, for the top 30 languages by mean FLORES premium: 21/30 have a mapped
second-domain corpus (`Helsinki-NLP/opus-100` and/or
`davidstap/biblenlp-corpus-mmteb`, religious register); the other 9 (Shan,
Santali, Dzongkha, Tamasheq, Central Atlas Tamazight, Lao, Tigrinya, Manipuri,
Kabiyè) have no modern, ungated, non-loading-script parallel corpus available
for a second domain at all — that gap is itself a finding (the languages with
the highest token tax also have the least data to cross-validate it). Across
264 (language, domain, tokenizer) comparisons, median relative difference
from the FLORES estimate is 14% — the premium mostly replicates across
register, with some corpus-specific outliers worth a caveat in the paper
(Kannada on OPUS-100 diverges up to 106% for gpt-4o; Uyghur and Sanskrit's
Bible-corpus pairings diverge 44–72%, likely small/idiosyncratic samples in
those specific files). Results in `data/results/opus_domain_check.csv`.

**Phase 3 (results dataset + explorer) is published:**
[`shreyaskc/tokentax-results-v1`](https://huggingface.co/datasets/shreyaskc/tokentax-results-v1)
on Hugging Face — `flores200_summary.parquet` (the CI-backed long-format
table), `flores200_raw.parquet` (1.64M per-sentence rows), and
`opus_domain_check.parquet`, under CC0 for the derived statistics (no source
sentence text is redistributed). Re-publish with
`scripts/publish_hf_dataset.py`.

The [tokentax explorer](https://huggingface.co/spaces/shreyaskc/tokentax-explorer)
(heatmap + "what does N tokens buy in your language" calculator) is live as
a **static** Space — Hugging Face requires a PRO subscription to host
Gradio/Docker Spaces even on the free CPU tier, so `hf_space/` is a
Plotly.js + vanilla-JS page reading a bundled `data.json`/`pricing.json`
snapshot rather than a live Hub read. Rebuild with
`scripts/build_static_explorer.py` after a new results version, then
redeploy with `scripts/publish_hf_space.py`. A functionally identical
Gradio version (live Hub reads, no rebuild step) sits unpublished at
`scripts/gradio_explorer_prototype/` for if the account upgrades to PRO.

**Released on PyPI:** [`tokentax`](https://pypi.org/project/tokentax/) 0.1.0.
Not yet released: DOI, arXiv preprint. See `PLANNING.md` for the full
implementation brief and phased plan (paper, remaining release checklist).

Two registry notes from the real run: `llama-3` resolves to the
`NousResearch/Meta-Llama-3-8B` mirror, not `meta-llama/Meta-Llama-3-8B`,
because the official repo requires Meta's manual license approval rather
than an instant click-through; `mistral` turned out not to be gated at all
(needs `protobuf` installed, not a license). Claude is excluded from the
sweep by choice, not necessity — see `registry.py` if you want to re-add it
with `ANTHROPIC_API_KEY` set.

## Install

```bash
pip install tokentax
```

To hack on the library itself (and run the test suite):

```bash
git clone https://github.com/Shreyaskc/token-tax.git
cd token-tax
pip install -e ".[test,corpora]"
pytest tests/
```

Requires Python ≥3.9. Core runtime deps: numpy, pandas, pyyaml, evalci,
tiktoken, transformers, sentencepiece, huggingface_hub. Optional extras:
`corpora` (Hugging Face `datasets`, for pulling FLORES-200) and `claude`
(the `anthropic` client, for Claude's token-count API).

### One-time setup for gated resources

Several tokenizers (Llama 3, Mistral, Gemma) and the FLORES-200 corpus
(`facebook/flores`) are gated on Hugging Face — a free, auto-approved license
click, not a manual review:

1. Visit the dataset/model page while logged in and click "Agree and access
   repository".
2. `huggingface-cli login` (or set `HF_TOKEN`).

Claude has no downloadable tokenizer; `tokentax` uses Anthropic's free
token-count API instead, which needs `ANTHROPIC_API_KEY`.

## Usage

### List registered tokenizers

```bash
tokentax list-tokenizers
```

### Premium for one language vs. English

```bash
tokentax premium gpt-4o tam_Taml --corpus toy   # bundled toy corpus, no network
tokentax premium gpt-4o tam_Taml --corpus flores200   # full FLORES-200 (needs the gate above)
```

### Library API

```python
import tokentax

tok = tokentax.load_tokenizer("gpt-4o")

# per-sentence metrics
tokentax.metrics.premium_ratios(tok, tamil_sentences, english_sentences)
tokentax.metrics.bytes_per_token(tok, sentences)
tokentax.metrics.chars_per_token(tok, sentences)
tokentax.metrics.effective_context_window(context_tokens=128_000, tokens_per_word=2.4)

# CI-backed report for one (tokenizer, language) pair, or a whole corpus
tokentax.premium_report(tok, tamil_sentences, english_sentences, language="tam_Taml")
tokentax.premium_table(tok, tokentax.corpora.load_toy_corpus())

# $/token pricing (placeholder figures — see pricing.yaml)
tokentax.pricing.estimate_cost("gpt-4o", n_tokens=1_000_000)
```

## What's validated, and how

- Confidence intervals are computed by `evalci.ci(method="bootstrap")`, not
  reimplemented — the same statistically-validated routine used across this
  portfolio's benchmarks.
- Premium ratios are computed on FLORES-200's aligned parallel sentences
  (same meaning across languages), not independent monolingual corpora —
  the only methodologically defensible basis for a cross-language ratio.
- `pricing.yaml` ships with `verified: false`; `tokentax.pricing.load_pricing()`
  warns until it's checked against live provider pricing and flipped to true.
  Do not cite a dollar figure from an unverified snapshot.

## Release checklist

CITATION.cff (done) → full FLORES-200 run + validation against Petrov et al.
2023 (done) → HF dataset + Space (done) → PyPI (done) → arXiv → Papers with
Code → Zenodo → seed emails to multilingual-NLP and AI-policy researchers →
workshop poster.
