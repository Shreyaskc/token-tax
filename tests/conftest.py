import pytest

from tokentax.registry import Tokenizer, TokenizerSpec


class _WhitespaceEncoder:
    """Splits on whitespace: deterministic and network-free, so metric/report
    math can be tested without depending on a real tokenizer backend."""

    def __call__(self, text):
        return text.split()


@pytest.fixture
def fake_tokenizer():
    spec = TokenizerSpec(name="fake-whitespace", family="test", backend="test", source="n/a")
    return Tokenizer(spec, _WhitespaceEncoder())
