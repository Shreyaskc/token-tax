"""Turn the raw per-sentence premium ratios from run_flores_sweep.py into a
CI-backed long-format summary table via evalci — the deliverable shape from
the README: (tokenizer, tokenizer_version, language, corpus, metric, value).

Usage:
    python scripts/summarize_with_evalci.py [--split devtest] [--n-resamples 9999]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokentax.report import premium_report_from_ratios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--n-resamples", type=int, default=9999)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    raw_path = RESULTS_DIR / f"tokentax_raw_{args.split}.parquet"
    if not raw_path.exists():
        print(f"error: {raw_path} not found; run scripts/run_flores_sweep.py first", file=sys.stderr)
        return 1
    raw = pd.read_parquet(raw_path)

    rows = []
    for (tokenizer, language), group in raw.groupby(["tokenizer", "language"]):
        report = premium_report_from_ratios(
            group["premium_ratio"].to_numpy(),
            tokenizer_name=tokenizer,
            language=language,
            confidence=args.confidence,
            n_resamples=args.n_resamples,
            random_state=args.random_state,
        )
        tokenizer_revision = group["tokenizer_revision"].iloc[0]
        corpus = group["corpus"].iloc[0]
        for metric, value in [
            ("premium_ratio_estimate", report.estimate),
            ("premium_ratio_ci_lower", report.lower),
            ("premium_ratio_ci_upper", report.upper),
            ("n_sentences", report.n),
        ]:
            rows.append(
                {
                    "tokenizer": tokenizer,
                    "tokenizer_version": tokenizer_revision,
                    "language": language,
                    "corpus": corpus,
                    "metric": metric,
                    "value": value,
                }
            )
        print(f"{tokenizer:12s} {language:10s} {report}")

    summary = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"tokentax_summary_{args.split}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nwrote {len(summary)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
