"""`tokentax premium <tokenizer> <language>` — CLI entry point."""
import sys
import argparse

from . import corpora, registry
from .report import premium_report


def build_parser():
    parser = argparse.ArgumentParser(prog="tokentax")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tokenizers", help="List registered tokenizers")

    premium_p = sub.add_parser(
        "premium", help="Report the token premium for one language vs. English"
    )
    premium_p.add_argument("tokenizer", choices=registry.available())
    premium_p.add_argument("language", help="FLORES language code, e.g. tam_Taml")
    premium_p.add_argument("--corpus", choices=["toy", "flores200"], default="toy")
    premium_p.add_argument("--english-key", default="eng_Latn")
    premium_p.add_argument("--confidence", type=float, default=0.95)
    premium_p.add_argument("--n-resamples", type=int, default=9999)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tokenizers":
        for name in registry.available():
            spec = registry.REGISTRY[name]
            gated = " (gated)" if spec.gated else ""
            print(f"{name}{gated} — {spec.backend}:{spec.source}")
        return 0

    if args.command == "premium":
        if args.corpus == "toy":
            corpus = corpora.load_toy_corpus()
        else:
            corpus = corpora.load_flores200(languages=[args.english_key, args.language])

        if args.language not in corpus:
            print(f"error: language {args.language!r} not in corpus", file=sys.stderr)
            return 1
        if args.english_key not in corpus:
            print(f"error: english reference {args.english_key!r} not in corpus", file=sys.stderr)
            return 1

        tokenizer = registry.load(args.tokenizer)
        report = premium_report(
            tokenizer,
            corpus[args.language],
            corpus[args.english_key],
            language=args.language,
            confidence=args.confidence,
            n_resamples=args.n_resamples,
        )
        print(report)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
