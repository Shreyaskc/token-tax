"""Phase 3: build the static-HTML explorer's data files.

Hugging Face Spaces require a PRO subscription to host Gradio/Docker apps
(even on the free CPU tier) — only Static Spaces are free. This builds a
self-contained static page instead: hf_space/index.html (Plotly.js from CDN
+ vanilla JS) reads hf_space/data.json and hf_space/pricing.json, both
generated here from the local sweep results and pricing.yaml.

Unlike the Gradio prototype (scripts/gradio_explorer_prototype/), this is a
snapshot as of build time, not a live Hub read — re-run this script (then
scripts/publish_hf_space.py) after publishing a new results dataset version.

Usage:
    python scripts/build_static_explorer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
HF_SPACE_DIR = ROOT / "hf_space"
PRICING_PATH = ROOT / "src" / "tokentax" / "pricing.yaml"


def build_data_json() -> dict:
    summary_path = RESULTS_DIR / "tokentax_summary_devtest.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found — run the FLORES-200 sweep first")

    df = pd.read_csv(summary_path)
    tokenizers = sorted(df["tokenizer"].unique())

    languages: dict = {}
    for (tokenizer, language), group in df.groupby(["tokenizer", "language"]):
        values = group.set_index("metric")["value"]
        languages.setdefault(language, {})[tokenizer] = {
            "estimate": values["premium_ratio_estimate"],
            "lower": values["premium_ratio_ci_lower"],
            "upper": values["premium_ratio_ci_upper"],
            "n": int(values["n_sentences"]),
        }

    return {"tokenizers": tokenizers, "languages": languages}


def main(argv=None):
    HF_SPACE_DIR.mkdir(parents=True, exist_ok=True)

    data = build_data_json()
    (HF_SPACE_DIR / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {HF_SPACE_DIR / 'data.json'} ({len(data['languages'])} languages)")

    with open(PRICING_PATH) as f:
        pricing = yaml.safe_load(f)
    (HF_SPACE_DIR / "pricing.json").write_text(json.dumps(pricing, indent=2))
    print(f"wrote {HF_SPACE_DIR / 'pricing.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
