"""Parallel-corpus loaders.

The primary corpus is FLORES-200 (`facebook/flores`, config "all"): human
translations of the same ~2000 sentences into 200+ languages, aligned by a
shared `id`. Premium ratios are computed on these aligned sentences, not on
independent monolingual text, because only a shared-meaning pair makes
"language X costs N times as many tokens as English for the same content" a
defensible claim.

`facebook/flores` (and its FLORES-200 successor, `openlanguagedata/flores_plus`)
are both *gated* on Hugging Face — a free, auto-approved license click, not a
manual review. One-time setup:

    1. Visit https://huggingface.co/datasets/facebook/flores and click
       "Agree and access repository" while logged in.
    2. `huggingface-cli login` locally (or set `HF_TOKEN`).

Column names below (`id`, `sentence`) follow FLORES's long-standing public
convention and are now verified against a live pull (2026-08-08): the schema
is `{id, URL, domain, topic, has_image, has_hyperlink, sentence}`.

For the Phase 2 domain-robustness check, `load_opus100` and
`load_bible_corpus` pull a second, non-FLORES-domain parallel corpus for
languages where one exists — FLORES is encyclopedic/news-register text, and
a premium ratio that only replicates on that one register is a weaker claim
than one that holds on subtitles/religious text too. Neither source covers
every language: see PLANNING.md's domain-robustness notes for the coverage
gap on very low-resource scripts (fewer accessible parallel corpora is
itself part of the token-tax story).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

FLORES_REPO = "facebook/flores"
_SENTENCE_COLS = ("sentence", "text", "translation")
_ID_COLS = ("id", "URL")

# The full set of 204 FLORES-200 language configs, read from the repo's
# (ungated) file listing on 2026-08-08 — `data/language/{code}/` directory
# names, not the gated sentence content itself. Verify this list still
# matches `facebook/flores` if HF adds/renames languages before a real run.
FLORES200_LANGUAGES = [
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn",
    "ajp_Arab", "aka_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab",
    "arb_Latn", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn",
    "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn",
    "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab",
    "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn",
    "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn",
    "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek",
    "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn",
    "fij_Latn", "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn",
    "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr",
    "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn",
    "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn",
    "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn",
    "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn",
    "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl",
    "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "kor_Hang",
    "lao_Laoo", "lij_Latn", "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn",
    "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn",
    "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Arab",
    "min_Latn", "mkd_Cyrl", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn",
    "mya_Mymr", "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn",
    "nus_Latn", "nya_Latn", "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru",
    "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn",
    "prs_Arab", "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn",
    "san_Deva", "sat_Olck", "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn",
    "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
    "spa_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn",
    "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl",
    "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn",
    "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn",
    "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn",
    "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr",
    "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn",
]


def _normalize(df, lang: str):
    sentence_col = next((c for c in _SENTENCE_COLS if c in df.columns), None)
    if sentence_col is None:
        raise ValueError(
            f"couldn't find a sentence column for {lang!r} in {list(df.columns)}; "
            f"update tokentax.corpora._SENTENCE_COLS to match"
        )
    return df.sort_values(by=[c for c in _ID_COLS if c in df.columns] or df.columns[0]) \
        [sentence_col].tolist()


def load_flores200(
    languages: Optional[List[str]] = None,
    split: str = "dev",
    cache_dir: Path = CACHE_DIR,
) -> Dict[str, List[str]]:
    """Load FLORES-200 parallel sentences: `{language_code: [sentence, ...]}`.

    `languages` are FLORES codes like `"eng_Latn"`, `"tam_Taml"`. Each
    language's pull is cached to `cache_dir/flores200_{split}_{lang}.parquet`
    so a full ~200-language run is resumable after a crash.
    """
    if languages is None:
        raise ValueError("pass an explicit list of FLORES language codes (e.g. ['eng_Latn', 'tam_Taml'])")

    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[str]] = {}
    for lang in languages:
        cache_path = cache_dir / f"flores200_{split}_{lang}.parquet"
        if cache_path.exists():
            import pandas as pd

            df = pd.read_parquet(cache_path)
        else:
            try:
                ds = load_dataset(FLORES_REPO, lang, split=split)
            except Exception as e:  # noqa: BLE001 - surface a fix-it message either way
                raise RuntimeError(
                    f"failed to load {FLORES_REPO!r} config {lang!r}: {e}\n"
                    "If this is a 401/gated error: accept the license at "
                    "https://huggingface.co/datasets/facebook/flores and run "
                    "`huggingface-cli login`."
                ) from e
            df = ds.to_pandas()
            df.to_parquet(cache_path)
        out[lang] = _normalize(df, lang)
    return out


# flores_code -> Helsinki-NLP/opus-100 config name. Coverage is partial: many
# very-low-resource FLORES languages (e.g. Shan, Santali, Tamasheq) have no
# OPUS-100 pairing at all. Only add an entry once you've confirmed the config
# exists in the dataset's file listing.
OPUS100_LANG_MAP = {
    "dzo_Tibt": "dz-en", "ory_Orya": "en-or", "mya_Mymr": "en-my", "khm_Khmr": "en-km",
    "mal_Mlym": "en-ml", "sin_Sinh": "en-si", "kan_Knda": "en-kn", "kat_Geor": "en-ka",
    "tel_Telu": "en-te", "guj_Gujr": "en-gu", "pan_Guru": "en-pa", "tam_Taml": "en-ta",
    "amh_Ethi": "am-en", "hye_Armn": "en-hy", "asm_Beng": "as-en", "ben_Beng": "bn-en",
    "uig_Arab": "en-ug", "ydd_Hebr": "en-yi", "mar_Deva": "en-mr",
}

# flores_code -> davidstap/biblenlp-corpus-mmteb language code (files are at
# data/eng-{code}/{split}.json). Also partial coverage, but reaches a couple
# of languages OPUS-100 doesn't (Sanskrit, Central Kurdish).
BIBLE_LANG_MAP = {
    "ory_Orya": "ory", "mya_Mymr": "mya", "mal_Mlym": "mal", "kan_Knda": "kan",
    "tel_Telu": "tel", "guj_Gujr": "guj", "pan_Guru": "pan", "tam_Taml": "tam",
    "asm_Beng": "asm", "ben_Beng": "ben", "uig_Arab": "uig", "mar_Deva": "mar",
    "san_Deva": "san", "ckb_Arab": "ckb",
}


def load_opus100_pair(flores_lang: str, split: str = "test", cache_dir: Path = CACHE_DIR):
    """(lang_sentences, english_sentences) aligned pair for one FLORES-coded
    language from `Helsinki-NLP/opus-100` — a mixed-domain (not encyclopedic)
    corpus, for the Phase 2 domain-robustness check.
    """
    try:
        config = OPUS100_LANG_MAP[flores_lang]
    except KeyError:
        raise KeyError(
            f"no OPUS-100 pairing for {flores_lang!r}; available: {sorted(OPUS100_LANG_MAP)}"
        ) from None

    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"opus100_{split}_{flores_lang}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        from datasets import load_dataset

        try:
            ds = load_dataset("Helsinki-NLP/opus-100", config, split=split)
        except ValueError as e:
            # a handful of opus-100 pairs (e.g. dz-en, en-hy) only ship a
            # "train" split, no "test" — fall back rather than fail the pair.
            if "Unknown split" not in str(e):
                raise
            ds = load_dataset("Helsinki-NLP/opus-100", config, split="train")
        df = pd.DataFrame([row["translation"] for row in ds])
        df.to_parquet(cache_path)

    left, right = config.split("-")
    other_code = right if left == "en" else left
    return df[other_code].tolist(), df["en"].tolist()


def load_bible_corpus_pair(flores_lang: str, split: str = "test", cache_dir: Path = CACHE_DIR):
    """(lang_sentences, english_sentences) aligned pair for one FLORES-coded
    language from `davidstap/biblenlp-corpus-mmteb` — religious-register text,
    for the Phase 2 domain-robustness check.
    """
    try:
        code = BIBLE_LANG_MAP[flores_lang]
    except KeyError:
        raise KeyError(
            f"no Bible-corpus pairing for {flores_lang!r}; available: {sorted(BIBLE_LANG_MAP)}"
        ) from None

    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"bible_{split}_{flores_lang}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="davidstap/biblenlp-corpus-mmteb",
            filename=f"data/eng-{code}/{split}.json",
            repo_type="dataset",
        )
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)
        df.to_parquet(cache_path)

    return df[code].tolist(), df["eng"].tolist()


# A tiny, hand-written parallel corpus for tests: no network, no gate, no
# ambiguity about column names. Not a substitute for FLORES in the real run.
TOY_CORPUS: Dict[str, List[str]] = {
    "eng_Latn": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Language models are trained on large amounts of text.",
        "The committee will reconvene next Tuesday afternoon.",
    ],
    "tam_Taml": [
        "விரைவான பழுப்பு நரி சோம்பேறி நாயின் மேல் குதிக்கிறது.",
        "அவள் கடற்கரையில் நத்தை ஓடுகளை விற்கிறாள்.",
        "மொழி மாதிரிகள் அதிக அளவு உரையில் பயிற்சி பெறுகின்றன.",
        "குழு அடுத்த செவ்வாய்க்கிழமை பிற்பகலில் மீண்டும் கூடும்.",
    ],
    "amh_Ethi": [
        "ፈጣኑ ቡናማ ቀበሮ ሰነፉን ውሻ ዘሎ ያልፋል።",
        "እሷ በባህር ዳርቻ ላይ የባህር ዛጎሎችን ትሸጣለች።",
        "የቋንቋ ሞዴሎች በብዙ ጽሑፍ ላይ የሰለጠኑ ናቸው።",
        "ኮሚቴው በሚቀጥለው ማክሰኞ ከሰዓት በኋላ እንደገና ይሰበሰባል።",
    ],
    "mya_Mymr": [
        "လျင်မြန်သောအညိုရောင်ခွေးအမဲသည် ပျင်းရိသောခွေးအပေါ်ကျော်တက်သည်။",
        "သူမသည် ပင်လယ်ကမ်းခြေတွင် ခရုအခွံများကို ရောင်းသည်။",
        "ဘာသာစကားပုံစံများသည် ကြီးမားသောစာသားများပေါ်တွင် လေ့ကျင့်ထားသည်။",
        "ကော်မတီသည် လာမည့်အင်္ဂါနေ့ မွန်းလွဲပိုင်းတွင် ပြန်လည်စုစည်းမည်။",
    ],
}


def load_toy_corpus() -> Dict[str, List[str]]:
    return {k: list(v) for k, v in TOY_CORPUS.items()}
