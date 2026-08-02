"""Tokenizer registry: one entry per tokenizer under study, with a pinned
source so results are reproducible. Loading is lazy — importing this module
does not download or import any tokenizer backend.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    family: str
    backend: str  # "tiktoken" | "huggingface" | "anthropic"
    source: str  # tiktoken encoding name, or an HF repo id, or "anthropic-api"
    # Pinned HF revision (commit SHA once frozen for a results run; "main" until then).
    revision: Optional[str] = None
    gated: bool = False  # requires accepting a license on huggingface.co before use
    notes: str = ""


REGISTRY: "dict[str, TokenizerSpec]" = {
    "gpt-4o": TokenizerSpec(
        name="gpt-4o", family="openai", backend="tiktoken", source="o200k_base",
        notes="o200k_base; the same vocabulary underlies OpenAI's current GPT-4o/GPT-5-class models",
    ),
    "gpt-4": TokenizerSpec(
        name="gpt-4", family="openai", backend="tiktoken", source="cl100k_base",
        notes="legacy vocabulary, kept for the Petrov et al. 2023 reproduction check",
    ),
    "gpt-2": TokenizerSpec(
        name="gpt-2", family="openai", backend="tiktoken", source="gpt2",
        notes="legacy vocabulary, kept for the Petrov et al. 2023 reproduction check",
    ),
    "llama-3": TokenizerSpec(
        name="llama-3", family="meta", backend="huggingface",
        source="NousResearch/Meta-Llama-3-8B", revision="main", gated=False,
        notes=(
            "the canonical meta-llama/Meta-Llama-3-8B requires Meta's manual "
            "license approval (not auto-approved like this portfolio's other "
            "gated resources); using the NousResearch mirror, a widely-used "
            "ungated re-upload of the same tokenizer, instead"
        ),
    ),
    "qwen2.5": TokenizerSpec(
        name="qwen2.5", family="alibaba", backend="huggingface",
        source="Qwen/Qwen2.5-7B", revision="main",
    ),
    "deepseek-v3": TokenizerSpec(
        name="deepseek-v3", family="deepseek", backend="huggingface",
        source="deepseek-ai/DeepSeek-V3", revision="main",
    ),
    "mistral": TokenizerSpec(
        name="mistral", family="mistral", backend="huggingface",
        source="mistralai/Mistral-7B-v0.3", revision="main", gated=False,
        notes="verified ungated as of 2026-08-08; requires protobuf installed for tokenizer conversion",
    ),
    "gemma-2": TokenizerSpec(
        name="gemma-2", family="google", backend="huggingface",
        source="google/gemma-2-9b", revision="main", gated=True,
        notes=(
            "gated on Hugging Face. Gemini has no public tokenizer artifact; "
            "gemma-2 is used as an open SentencePiece proxy for it and MUST be "
            "reported as an approximation, never silently as 'gemini' in results."
        ),
    ),
    "claude": TokenizerSpec(
        name="claude", family="anthropic", backend="anthropic", source="anthropic-api",
        notes=(
            "Claude has no downloadable tokenizer. Uses Anthropic's free "
            "token-count API (Messages.count_tokens) — a metadata call, not a "
            "billed generation. Requires ANTHROPIC_API_KEY; results are cached "
            "to disk since this backend needs network access."
        ),
    ),
}


class Tokenizer:
    """Uniform interface over every backend: `.encode(text) -> list[int]`."""

    def __init__(self, spec: TokenizerSpec, encode_fn: Callable[[str], List[int]]):
        self.spec = spec
        self._encode_fn = encode_fn

    def encode(self, text: str) -> List[int]:
        return self._encode_fn(text)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def __repr__(self):
        return f"Tokenizer({self.spec.name!r}, backend={self.spec.backend!r})"


def _load_tiktoken(spec: TokenizerSpec) -> Tokenizer:
    import tiktoken

    enc = tiktoken.get_encoding(spec.source)
    return Tokenizer(spec, lambda text: enc.encode(text))


def _load_huggingface(spec: TokenizerSpec) -> Tokenizer:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.source, revision=spec.revision)
    return Tokenizer(spec, lambda text: tok.encode(text, add_special_tokens=False))


def _load_anthropic(spec: TokenizerSpec) -> Tokenizer:
    import anthropic

    client = anthropic.Anthropic()
    model = "claude-opus-4-1-20250805"

    def encode(text: str) -> List[int]:
        n = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        ).input_tokens
        # The API returns a count, not token IDs; a dummy list keeps the
        # uniform `len(encode(text))` interface used everywhere else.
        return [0] * n

    return Tokenizer(spec, encode)


_LOADERS = {
    "tiktoken": _load_tiktoken,
    "huggingface": _load_huggingface,
    "anthropic": _load_anthropic,
}


@functools.lru_cache(maxsize=None)
def load(name: str) -> Tokenizer:
    """Load (and cache) the tokenizer registered under `name`."""
    try:
        spec = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown tokenizer {name!r}; available: {sorted(REGISTRY)}"
        ) from None
    return _LOADERS[spec.backend](spec)


def available() -> "list[str]":
    return sorted(REGISTRY)
