"""Provider selection.

Resolution order for `provider: auto` in assumptions.yaml:
  1. Anthropic, if a key is configured and the SDK is importable.
  2. OpenAI-compatible, same conditions.
  3. MockProvider — always available.

Keys are read from st.secrets when Streamlit is running, otherwise from the
environment. Nothing here ever logs a key.
"""
from __future__ import annotations

import os
from typing import Any

from supplyiq.agent.base import AgentProvider, ProviderUnavailable
from supplyiq.agent.mock import MockProvider
from supplyiq.config import load_assumptions

KEY_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def read_secret(name: str) -> str | None:
    """st.secrets first (Streamlit Cloud), then the environment. Never raises."""
    try:
        import streamlit as st  # imported lazily so the core package stays UI-free

        # Touching st.secrets with no secrets.toml raises; treat that as absent.
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            return value or None
    except Exception:
        pass
    value = (os.environ.get(name) or "").strip()
    return value or None


def _llm_provider(kind: str, a: dict[str, Any]) -> AgentProvider:
    from supplyiq.agent.llm import LLMProvider

    return LLMProvider(
        kind=kind,
        model=str(a["agent"]["model"]),
        max_output_tokens=int(a["agent"]["max_output_tokens"]),
        temperature=float(a["agent"]["temperature"]),
        api_key=read_secret(KEY_NAMES[kind]),
    )


def available_providers(a: dict[str, Any] | None = None) -> list[str]:
    """Names of the providers that could serve a request right now."""
    a = a or load_assumptions()
    out = []
    for kind in ("anthropic", "openai"):
        try:
            if _llm_provider(kind, a).available():
                out.append(kind)
        except Exception:
            pass
    out.append("mock")
    return out


def get_provider(a: dict[str, Any] | None = None, force: str | None = None) -> AgentProvider:
    """Return the provider to use. Falls back to the mock rather than failing."""
    a = a or load_assumptions()
    choice = (force or str(a["agent"]["provider"])).lower()

    if choice == "mock":
        return MockProvider()

    if choice in ("anthropic", "openai"):
        provider = _llm_provider(choice, a)
        if not provider.available():
            raise ProviderUnavailable(
                f"provider '{choice}' was requested but no {KEY_NAMES[choice]} is configured"
            )
        return provider

    if choice != "auto":
        raise ProviderUnavailable(
            f"unknown agent.provider {choice!r}; expected auto, mock, anthropic or openai"
        )

    for kind in ("anthropic", "openai"):
        try:
            provider = _llm_provider(kind, a)
            if provider.available():
                return provider
        except Exception:
            continue
    return MockProvider()


def provider_status(a: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small dict for the UI badge. Reports presence of a key, never its value."""
    a = a or load_assumptions()
    provider = get_provider(a)
    return {
        "configured": str(a["agent"]["provider"]),
        "active": provider.name,
        "is_mock": provider.is_mock,
        "model": None if provider.is_mock else str(a["agent"]["model"]),
        "available": available_providers(a),
        "badge": (
            "Demo mode: rule-based explanations"
            if provider.is_mock
            else f"LLM explanations via {provider.name}"
        ),
        "keys_present": {
            kind: read_secret(env) is not None for kind, env in KEY_NAMES.items()
        },
    }
