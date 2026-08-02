import pytest

from tokentax import registry


def test_available_matches_registry_keys():
    assert registry.available() == sorted(registry.REGISTRY)


def test_every_spec_has_required_fields():
    for name, spec in registry.REGISTRY.items():
        assert spec.name == name
        assert spec.backend in {"tiktoken", "huggingface", "anthropic"}
        assert spec.source


def test_load_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        registry.load("not-a-real-tokenizer")


def test_load_tiktoken_gpt4():
    tok = registry.load("gpt-4")
    assert tok.count("hello world") > 0


def test_load_is_cached():
    a = registry.load("gpt-4")
    b = registry.load("gpt-4")
    assert a is b
