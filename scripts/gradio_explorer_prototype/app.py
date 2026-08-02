"""tokentax explorer: a heatmap of the tokenizer cost penalty across
languages, and a "what does N tokens buy in your language" calculator.

Reads shreyaskc/tokentax-results-v1 directly from the Hub (hf:// path) —
this Space has no bundled copy of the results, so it always reflects
whatever version of the dataset is currently published. pricing.yaml is
bundled here as a static snapshot (same file as the source repo's
src/tokentax/pricing.yaml) since pulling in the full tokentax package just
for its pricing module would drag in tiktoken/transformers/evalci as
Space dependencies for no benefit.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import yaml

DATASET_REPO = "shreyaskc/tokentax-results-v1"
SUMMARY_PATH = f"hf://datasets/{DATASET_REPO}/flores200_summary.parquet"
GITHUB_URL = "https://github.com/Shreyaskc/token-tax"

# Friendly names for the languages this project has specifically studied
# (the top-30-by-premium set from the Phase 2 domain-robustness check).
# Everything else falls back to its raw FLORES-200 code.
LANGUAGE_NAMES = {
    "shn_Mymr": "Shan", "sat_Olck": "Santali", "dzo_Tibt": "Dzongkha",
    "ory_Orya": "Odia", "mya_Mymr": "Burmese", "bod_Tibt": "Tibetan",
    "taq_Tfng": "Tamasheq", "lao_Laoo": "Lao", "tzm_Tfng": "Central Atlas Tamazight",
    "khm_Khmr": "Khmer", "mal_Mlym": "Malayalam", "sin_Sinh": "Sinhala",
    "kan_Knda": "Kannada", "kat_Geor": "Georgian", "tel_Telu": "Telugu",
    "guj_Gujr": "Gujarati", "pan_Guru": "Punjabi", "tam_Taml": "Tamil",
    "tir_Ethi": "Tigrinya", "amh_Ethi": "Amharic", "hye_Armn": "Armenian",
    "mni_Beng": "Manipuri", "asm_Beng": "Assamese", "ben_Beng": "Bengali",
    "uig_Arab": "Uyghur", "ydd_Hebr": "Yiddish", "kbp_Latn": "Kabiye",
    "mar_Deva": "Marathi", "san_Deva": "Sanskrit", "ckb_Arab": "Central Kurdish",
    "eng_Latn": "English",
}


def _label(code: str) -> str:
    name = LANGUAGE_NAMES.get(code)
    return f"{code} — {name}" if name else code


def _display_name(code: str) -> str:
    """Just the human name for use inline in prose; falls back to the code."""
    return LANGUAGE_NAMES.get(code, code)


def _unlabel(label: str) -> str:
    return label.split(" — ")[0]


def load_summary() -> pd.DataFrame:
    return pd.read_parquet(SUMMARY_PATH)


def load_pricing() -> dict:
    with open(Path(__file__).resolve().parent / "pricing.yaml") as f:
        return yaml.safe_load(f)


SUMMARY = load_summary()
PRICING = load_pricing()
ESTIMATES = SUMMARY[SUMMARY["metric"] == "premium_ratio_estimate"]
CI_LOWER = SUMMARY[SUMMARY["metric"] == "premium_ratio_ci_lower"].set_index(["tokenizer", "language"])["value"]
CI_UPPER = SUMMARY[SUMMARY["metric"] == "premium_ratio_ci_upper"].set_index(["tokenizer", "language"])["value"]

ALL_TOKENIZERS = sorted(ESTIMATES["tokenizer"].unique())
CALCULATOR_TOKENIZERS = sorted(set(ALL_TOKENIZERS) & set(PRICING["models"]))
LANGUAGE_CHOICES = [_label(c) for c in sorted(ESTIMATES["language"].unique())]


def build_heatmap(top_n: int) -> go.Figure:
    mean_premium = ESTIMATES.groupby("language")["value"].mean().sort_values(ascending=False)
    languages = mean_premium.head(top_n).index.tolist()
    pivot = ESTIMATES[ESTIMATES["language"].isin(languages)].pivot(
        index="language", columns="tokenizer", values="value"
    ).reindex(languages)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[_label(l) for l in pivot.index],
            colorscale="OrRd",
            colorbar=dict(title="premium"),
            hovertemplate="%{y}<br>%{x}<br>premium=%{z:.2f}×<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Token premium vs. English — top {top_n} languages by mean premium",
        xaxis_title="tokenizer", yaxis_title="language",
        height=max(400, 24 * len(languages)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def calculate(tokenizer: str, language_label: str, n_tokens: float) -> str:
    language = _unlabel(language_label)
    name = _display_name(language)
    key = (tokenizer, language)
    if key not in CI_LOWER.index:
        return f"No data for `{tokenizer}` × {name}."

    estimate = ESTIMATES.set_index(["tokenizer", "language"]).loc[key, "value"]
    lower, upper = CI_LOWER.loc[key], CI_UPPER.loc[key]

    n_tokens = int(n_tokens)
    price_per_token = PRICING["models"][tokenizer]["input_per_million_usd"] / 1_000_000
    cost_this_language = n_tokens * price_per_token
    cost_english_equivalent = cost_this_language / estimate

    verified_note = (
        "" if PRICING.get("verified") else
        "\n\n⚠️ Pricing is an unverified placeholder snapshot — see the source repo's "
        "`pricing.yaml`. Treat dollar figures as illustrative, not citable."
    )

    return (
        f"### {name} on `{tokenizer}`\n\n"
        f"**Premium ratio:** {estimate:.2f}× (95% CI [{lower:.2f}, {upper:.2f}]) — "
        f"the same content costs {estimate:.2f}× as many tokens as English.\n\n"
        f"**{n_tokens:,} tokens of {name} text** costs an estimated "
        f"**${cost_this_language:,.2f}** at `{tokenizer}`'s input-token price "
        f"(${PRICING['models'][tokenizer]['input_per_million_usd']:.2f} / 1M tokens).\n\n"
        f"The *same content*, written in English, would cost roughly "
        f"**${cost_english_equivalent:,.2f}** — {name} speakers pay a "
        f"**{estimate:.2f}× tax** for identical meaning."
        f"{verified_note}"
    )


with gr.Blocks(title="tokentax explorer") as demo:
    gr.Markdown(
        f"# tokentax explorer\n"
        f"The tokenizer cost penalty across languages — data from "
        f"[`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO}), "
        f"code at [{GITHUB_URL}]({GITHUB_URL})."
    )

    with gr.Tab("Heatmap"):
        top_n_slider = gr.Slider(
            minimum=10, maximum=len(ESTIMATES["language"].unique()), value=40, step=5,
            label="Show top N languages by mean premium",
        )
        heatmap_plot = gr.Plot(value=build_heatmap(40))
        top_n_slider.change(build_heatmap, inputs=top_n_slider, outputs=heatmap_plot)

    with gr.Tab("Cost calculator"):
        gr.Markdown("What does N tokens buy in your language, vs. English?")
        with gr.Row():
            tokenizer_dd = gr.Dropdown(CALCULATOR_TOKENIZERS, value=CALCULATOR_TOKENIZERS[0], label="Tokenizer")
            language_dd = gr.Dropdown(LANGUAGE_CHOICES, value=_label("tam_Taml"), label="Language")
            tokens_input = gr.Number(value=1_000_000, label="Number of tokens", precision=0)
        calc_button = gr.Button("Calculate")
        calc_output = gr.Markdown()
        calc_button.click(calculate, inputs=[tokenizer_dd, language_dd, tokens_input], outputs=calc_output)

    gr.Markdown(
        "Premium ratios are computed on FLORES-200 devtest (aligned parallel sentences), "
        "with 95% bootstrap confidence intervals via "
        "[`evalci`](https://pypi.org/project/evalci/). "
        "Claude is excluded (no downloadable tokenizer). See the GitHub repo for "
        "methodology, validation against Petrov et al. 2023, and the OPUS "
        "domain-robustness check."
    )


if __name__ == "__main__":
    demo.launch()
