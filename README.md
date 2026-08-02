# tokentax — The Tokenizer Cost Penalty Across Languages

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

Phase 1 (core pipeline) is implemented and tested: tokenizer registry
(tiktoken, Hugging Face, and Anthropic's token-count API), premium/bytes-per-
token/context-window metrics, and confidence intervals wired through
`evalci`. Not yet released — no PyPI package, DOI, or arXiv preprint. See
`PLANNING.md` for the full implementation brief and phased plan (FLORES-200
full run, HF dataset + explorer, paper).

## Install

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
2023 → HF dataset + Space → PyPI → arXiv → Papers with Code → Zenodo →
seed emails to multilingual-NLP and AI-policy researchers → workshop poster.
