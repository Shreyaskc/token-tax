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
convention; they are unverified against a live pull in this environment
because the gate blocks unauthenticated schema inspection. `load_flores200`
fails loudly with the columns it actually saw if this assumption is wrong —
fix `_SENTENCE_COLS`/`_ID_COLS` below once you've pulled it once.
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
