"""Phase 2: run every registered tokenizer (except Claude — dropped from this
sweep, see README) across every FLORES-200 language, computing per-sentence
premium ratios against English. Saves raw per-sentence records, not just
aggregates, so results are independently recomputable (portfolio non-negotiable).

Requires an authenticated Hugging Face session for facebook/flores and for
the gated tokenizers (llama-3, mistral, gemma-2):
    huggingface-cli login

Resumable: each (tokenizer, language) pair's raw ratios are cached to
data/results/raw/{tokenizer}__{language}.parquet; re-running skips pairs
already on disk. Delete a file to force recomputation of just that pair.

Usage:
    python scripts/run_flores_sweep.py
    python scripts/run_flores_sweep.py --languages-limit 5          # smoke test
    python scripts/run_flores_sweep.py --tokenizers gpt-4o,gpt-4    # subset
    python scripts/run_flores_sweep.py --split dev                  # override split
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokentax import corpora, registry
from tokentax.metrics import premium_ratios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
RAW_DIR = RESULTS_DIR / "raw"

DEFAULT_TOKENIZERS = [name for name in registry.available() if name != "claude"]


def run_pair(tokenizer_name: str, lang: str, corpus, split: str) -> pd.DataFrame:
    out_path = RAW_DIR / f"{tokenizer_name}__{lang}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    tok = registry.load(tokenizer_name)
    ratios = premium_ratios(tok, corpus[lang], corpus["eng_Latn"])
    df = pd.DataFrame(
        {
            "tokenizer": tokenizer_name,
            "tokenizer_revision": registry.REGISTRY[tokenizer_name].revision,
            "language": lang,
            "corpus": f"flores200_{split}",
            "sentence_idx": range(len(ratios)),
            "premium_ratio": ratios,
        }
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--tokenizers", default=",".join(DEFAULT_TOKENIZERS))
    parser.add_argument(
        "--languages-limit", type=int, default=None,
        help="cap the number of non-English languages, for a smoke test",
    )
    args = parser.parse_args(argv)

    tokenizer_names = args.tokenizers.split(",")
    languages = list(corpora.FLORES200_LANGUAGES)
    if args.languages_limit:
        non_english = [l for l in languages if l != "eng_Latn"][: args.languages_limit]
        languages = non_english + ["eng_Latn"]
    if "eng_Latn" not in languages:
        languages.append("eng_Latn")

    print(f"Pulling FLORES-200 ({args.split}) for {len(languages)} languages...")
    corpus = corpora.load_flores200(languages=languages, split=args.split)

    frames = []
    errors = []
    for tokenizer_name in tokenizer_names:
        for lang in languages:
            if lang == "eng_Latn":
                continue
            try:
                frames.append(run_pair(tokenizer_name, lang, corpus, args.split))
                print(f"done  {tokenizer_name:12s} {lang}")
            except Exception as e:  # noqa: BLE001 - log and keep the sweep going
                print(f"ERROR {tokenizer_name:12s} {lang}: {e}", file=sys.stderr)
                errors.append((tokenizer_name, lang, str(e)))

    if not frames:
        print("no results produced", file=sys.stderr)
        return 1

    raw = pd.concat(frames, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"tokentax_raw_{args.split}.parquet"
    raw.to_parquet(raw_path)
    print(f"\nwrote {len(raw)} raw rows to {raw_path}")

    if errors:
        print(f"\n{len(errors)} (tokenizer, language) pair(s) failed — see stderr above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
