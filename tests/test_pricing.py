import pytest

from tokentax import pricing


def test_price_per_token_known_model():
    price = pricing.price_per_token("gpt-4o")
    assert price == pytest.approx(2.50 / 1_000_000)


def test_price_per_token_unknown_model_raises():
    with pytest.raises(KeyError):
        pricing.price_per_token("not-a-model")


def test_estimate_cost():
    cost = pricing.estimate_cost("gpt-4o", 1_000_000)
    assert cost == pytest.approx(2.50)


def test_load_pricing_warns_when_unverified():
    pricing.load_pricing.cache_clear()
    with pytest.warns(UserWarning, match="unverified"):
        pricing.load_pricing()
