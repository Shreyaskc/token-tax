"""Reproduce Petrov et al. 2023's premium ratios on FLORES-200 devtest as a
validation gate for tokentax's own pipeline (see README's Phase 2 plan).

Reference numbers in tests/fixtures/petrov2023_reference.csv are derived
from the paper's own published per-language token totals:
https://github.com/AleksandarPetrov/tokenization-fairness/blob/main/assets/tokenization_lengths_validated.csv

Requires an authenticated Hugging Face session (facebook/flores is gated):
    huggingface-cli login

Usage:
    python scripts/validate_against_petrov2023.py [--tolerance 0.05]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokentax import corpora, registry
from tokentax.metrics import token_counts

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "petrov2023_reference.csv"


def load_reference():
    with open(FIXTURE) as f:
        lines = [line for line in f if not line.startswith("#") and line.strip()]
    return list(csv.DictReader(lines))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="fractional tolerance vs. the published premium ratio (default 5%%; "
        "some drift vs. the 2023 numbers is expected from FLORES dataset revisions)",
    )
    parser.add_argument("--split", default="devtest", help="FLORES split Petrov et al. used")
    args = parser.parse_args(argv)

    reference = load_reference()
    languages = sorted({row["flores_code"] for row in reference} | {"eng_Latn"})

    print(f"Pulling FLORES-200 ({args.split}) for {languages}...")
    corpus = corpora.load_flores200(languages=languages, split=args.split)

    failures = []
    for row in reference:
        lang = row["flores_code"]
        tok_name = row["tokentax_tokenizer"]
        expected = float(row["premium_ratio"])

        tok = registry.load(tok_name)
        lang_total = int(token_counts(tok, corpus[lang]).sum())
        en_total = int(token_counts(tok, corpus["eng_Latn"]).sum())
        actual = lang_total / en_total
        rel_err = abs(actual - expected) / expected
        status = "OK" if rel_err <= args.tolerance else "FAIL"
        if status == "FAIL":
            failures.append((lang, tok_name))
        print(
            f"{status:4s} {row['petrov_language']:20s} {tok_name:8s} "
            f"expected={expected:.3f} actual={actual:.3f} rel_err={rel_err:.1%}"
        )

    if failures:
        print(f"\n{len(failures)} validation check(s) failed: {failures}")
        return 1
    print(f"\nAll {len(reference)} validation checks passed within {args.tolerance:.0%} tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
