"""$/token pricing lookups, backed by the versioned snapshot in pricing.yaml."""
from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Dict

import yaml

PRICING_PATH = Path(__file__).resolve().parent / "pricing.yaml"


@functools.lru_cache(maxsize=1)
def load_pricing() -> Dict:
    with open(PRICING_PATH) as f:
        data = yaml.safe_load(f)
    if not data.get("verified", False):
        warnings.warn(
            "tokentax pricing.yaml is unverified (placeholder figures); "
            "do not publish dollar-cost results until it's confirmed against "
            "live provider pricing and `verified: true` is set.",
            stacklevel=2,
        )
    return data


def price_per_token(model_key: str, kind: str = "input") -> float:
    pricing = load_pricing()
    try:
        entry = pricing["models"][model_key]
    except KeyError:
        raise KeyError(
            f"no pricing entry for {model_key!r}; available: {sorted(pricing['models'])}"
        ) from None
    per_million = entry[f"{kind}_per_million_usd"]
    return per_million / 1_000_000


def estimate_cost(model_key: str, n_tokens: int, kind: str = "input") -> float:
    return n_tokens * price_per_token(model_key, kind=kind)
