"""Phase 2 domain-robustness check: does the FLORES-200 premium ratio for the
most heavily-taxed languages replicate on a different register of text?

Takes the top-N languages by mean premium ratio from the FLORES-200 sweep
summary (data/results/tokentax_summary_devtest.csv — run run_flores_sweep.py
and summarize_with_evalci.py first) and, for each language that has a mapped
entry in OPUS100_LANG_MAP and/or BIBLE_LANG_MAP, computes the premium ratio
on that corpus's own aligned English pair and compares it to the FLORES
estimate.

Coverage is necessarily partial: several of the most-taxed FLORES languages
(often exactly the ones with the highest premiums) have no modern, ungated,
non-loading-script parallel corpus available for a second domain. Those are
reported as "no domain data available", not silently skipped.

Usage:
    python scripts/run_opus_domain_check.py [--top-n 30] [--tolerance 0.35]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokentax import corpora, registry
from tokentax.metrics import premium_ratios
from tokentax.report import premium_report_from_ratios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
FLORES_SUMMARY = RESULTS_DIR / "tokentax_summary_devtest.csv"

TOKENIZERS = [name for name in registry.available() if name != "claude"]


def top_languages_by_premium(n: int) -> list:
    df = pd.read_csv(FLORES_SUMMARY)
    est = df[df["metric"] == "premium_ratio_estimate"]
    mean_premium = est.groupby("language")["value"].mean().sort_values(ascending=False)
    return mean_premium.head(n).index.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--tolerance", type=float, default=0.35,
        help="flag a language/tokenizer as domain-sensitive if |domain_premium - "
        "flores_premium| / flores_premium exceeds this (default 35%%: different "
        "registers, small samples, and no shared corpus id all add real variance)",
    )
    parser.add_argument("--n-resamples", type=int, default=1999)
    args = parser.parse_args(argv)

    if not FLORES_SUMMARY.exists():
        print(f"error: {FLORES_SUMMARY} not found; run the FLORES-200 sweep first", file=sys.stderr)
        return 1

    languages = top_languages_by_premium(args.top_n)
    flores_df = pd.read_csv(FLORES_SUMMARY)
    flores_est = flores_df[flores_df["metric"] == "premium_ratio_estimate"].set_index(
        ["tokenizer", "language"]
    )["value"]

    rows = []
    no_coverage = []
    for lang in languages:
        domains = {}
        if lang in corpora.OPUS100_LANG_MAP:
            domains["opus100"] = corpora.load_opus100_pair
        if lang in corpora.BIBLE_LANG_MAP:
            domains["bible"] = corpora.load_bible_corpus_pair
        if not domains:
            no_coverage.append(lang)
            print(f"SKIP  {lang:10s} — no domain-robustness corpus available")
            continue

        for domain_name, loader in domains.items():
            try:
                lang_sentences, en_sentences = loader(lang)
            except Exception as e:  # noqa: BLE001 - log and keep going
                print(f"ERROR {lang:10s} {domain_name}: {e}", file=sys.stderr)
                continue

            for tok_name in TOKENIZERS:
                tok = registry.load(tok_name)
                ratios = premium_ratios(tok, lang_sentences, en_sentences)
                report = premium_report_from_ratios(
                    ratios, tokenizer_name=tok_name, language=lang,
                    n_resamples=args.n_resamples, random_state=0,
                )
                flores_value = flores_est.get((tok_name, lang))
                rel_diff = (
                    abs(report.estimate - flores_value) / flores_value
                    if flores_value else None
                )
                flag = "DIVERGES" if (rel_diff is not None and rel_diff > args.tolerance) else "ok"
                rows.append({
                    "language": lang, "domain": domain_name, "tokenizer": tok_name,
                    "flores_premium": flores_value, "domain_premium": report.estimate,
                    "domain_ci_lower": report.lower, "domain_ci_upper": report.upper,
                    "n_sentences": report.n, "rel_diff": rel_diff, "flag": flag,
                })
                print(
                    f"{flag:9s} {lang:10s} {domain_name:8s} {tok_name:12s} "
                    f"flores={flores_value:.3f} domain={report.estimate:.3f} "
                    f"({'n/a' if rel_diff is None else f'{rel_diff:.1%}'})"
                )

    if not rows:
        print("no domain-robustness comparisons produced", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "opus_domain_check.csv"
    out.to_csv(out_path, index=False)

    print(f"\nwrote {len(out)} rows to {out_path}")
    print(f"\n{len(no_coverage)}/{len(languages)} top languages have NO domain-robustness "
          f"corpus available: {no_coverage}")
    diverging = out[out["flag"] == "DIVERGES"]
    print(f"{len(diverging)}/{len(out)} (language, domain, tokenizer) checks diverge "
          f"by more than {args.tolerance:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
