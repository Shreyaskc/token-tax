"""Generates both paper figures from the published sweep results.
Run from the repo root: python paper/scripts/make_figures.py
"""
import matplotlib.pyplot as plt
import pandas as pd

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#d1495b"

# Approximate public release year of the model family associated with each
# tokenizer vocabulary (not necessarily the exact date the vocabulary itself
# was frozen) -- see paper Section 5 for the caveat.
RELEASE_YEAR = {
    "gpt-2": 2019, "gpt-4": 2022, "mistral": 2024, "llama-3": 2024,
    "qwen2.5": 2024, "gemma-2": 2024, "gpt-4o": 2024, "deepseek-v3": 2024,
}

LANGUAGE_NAMES = {
    "shn_Mymr": "Shan", "sat_Olck": "Santali", "dzo_Tibt": "Dzongkha",
    "ory_Orya": "Odia", "mya_Mymr": "Burmese", "bod_Tibt": "Tibetan",
    "taq_Tfng": "Tamasheq", "lao_Laoo": "Lao", "tzm_Tfng": "Tamazight",
    "khm_Khmr": "Khmer", "mal_Mlym": "Malayalam", "sin_Sinh": "Sinhala",
    "kan_Knda": "Kannada", "kat_Geor": "Georgian", "tel_Telu": "Telugu",
    "pan_Guru": "Punjabi", "tir_Ethi": "Tigrinya", "amh_Ethi": "Amharic",
    "guj_Gujr": "Gujarati", "tam_Taml": "Tamil", "hye_Armn": "Armenian",
    "mni_Beng": "Manipuri", "asm_Beng": "Assamese", "ben_Beng": "Bengali",
    "uig_Arab": "Uyghur", "ydd_Hebr": "Yiddish", "kbp_Latn": "Kabiye",
    "mar_Deva": "Marathi", "san_Deva": "Sanskrit", "ckb_Arab": "Central Kurdish",
}


def _assert_names_known(codes):
    """Fail loudly rather than silently plotting raw FLORES codes."""
    missing = [c for c in codes if c not in LANGUAGE_NAMES]
    if missing:
        raise KeyError(f"no display name for {missing}; add them to LANGUAGE_NAMES")


def load_summary():
    return pd.read_csv("data/results/tokentax_summary_devtest.csv")


def fig_tokenizer_trend(est: pd.DataFrame):
    mean_premium = est.groupby("tokenizer")["value"].mean().sort_values()
    years = [RELEASE_YEAR[t] for t in mean_premium.index]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    order = sorted(range(len(mean_premium)), key=lambda i: (years[i], mean_premium.iloc[i]))
    names = [mean_premium.index[i] for i in order]
    values = [mean_premium.iloc[i] for i in order]
    colors = [BLUE if years[i] <= 2019 else (INK_MUTED if years[i] <= 2022 else RED) for i in order]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({RELEASE_YEAR[n]})" for n in names], color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("mean token premium vs. English (203 FLORES-200 languages)", color=INK_PRIMARY)
    ax.set_title("Newer tokenizer vocabularies show a lower mean premium", color=INK_PRIMARY, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_PRIMARY)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v + 0.05, i, f"{v:.2f}×", va="center", fontsize=8, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig("paper/figures/tokenizer_trend.pdf")
    plt.close(fig)


def fig_top_languages(est: pd.DataFrame):
    # GPT-2 (2019) is a historical anchor for the trend figure only; a figure
    # about what languages are taxed *now* must not average it in.
    current = est[est["tokenizer"] != "gpt-2"]
    mean_premium = current.groupby("language")["value"].mean().sort_values(ascending=False).head(15)
    _assert_names_known(mean_premium.index)
    names = [LANGUAGE_NAMES[c] for c in mean_premium.index]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    y = range(len(names))
    ax.barh(list(y)[::-1], mean_premium.values, color=RED, edgecolor="none")
    ax.set_yticks(list(y)[::-1])
    ax.set_yticklabels(names, color=INK_PRIMARY, fontsize=9)
    ax.set_xlabel("mean token premium vs. English (7 current tokenizers, excl. GPT-2)", color=INK_PRIMARY)
    ax.set_title("15 highest-premium languages in FLORES-200", color=INK_PRIMARY, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_PRIMARY)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(mean_premium.values):
        ax.text(v + 0.1, len(names) - 1 - i, f"{v:.1f}×", va="center", fontsize=8, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig("paper/figures/top_languages.pdf")
    plt.close(fig)


# Vocabulary sizes read from the pinned tokenizer revisions (see registry.py);
# tiktoken encodings report n_vocab, HF tokenizers report len(tokenizer).
VOCAB_SIZE = {
    "mistral": 32768, "gpt-2": 50257, "gpt-4": 100277, "llama-3": 128256,
    "deepseek-v3": 128815, "qwen2.5": 151665, "gpt-4o": 200019, "gemma-2": 256000,
}


def fig_vocab_vs_premium(est: pd.DataFrame):
    mean_premium = est.groupby("tokenizer")["value"].mean()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, premium in mean_premium.items():
        vocab = VOCAB_SIZE[name]
        year = RELEASE_YEAR[name]
        color = BLUE if year <= 2019 else (INK_MUTED if year <= 2022 else RED)
        ax.scatter(vocab, premium, s=70, color=color, zorder=3)
        ax.annotate(name, (vocab, premium), textcoords="offset points",
                    xytext=(7, 3), fontsize=8, color=INK_PRIMARY)

    ax.set_xscale("log")
    ax.set_xlabel("vocabulary size (log scale)", color=INK_PRIMARY)
    ax.set_ylabel("mean token premium vs. English", color=INK_PRIMARY)
    ax.set_title("Larger vocabularies track a lower token premium", color=INK_PRIMARY, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_PRIMARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("paper/figures/vocab_vs_premium.pdf")
    plt.close(fig)


def main():
    df = load_summary()
    est = df[df["metric"] == "premium_ratio_estimate"]
    fig_tokenizer_trend(est)
    fig_top_languages(est)
    fig_vocab_vs_premium(est)
    print("wrote tokenizer_trend.pdf, top_languages.pdf, vocab_vs_premium.pdf to paper/figures/")


if __name__ == "__main__":
    main()
