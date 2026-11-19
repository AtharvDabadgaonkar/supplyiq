"""The planning agent: a provider interface plus a rule-based mock fallback.

The hard rule this package exists to enforce: **the LLM never produces a
number.** Recommendations, costs and service impacts are computed by the
simulator (Phase 3/4) and handed to the provider as a read-only payload. A
provider's only job is to turn that payload into a short plain-English
explanation for the planner.

With no API key configured, `get_provider()` returns the MockProvider, so the
app, the tests and the Streamlit Cloud deployment all run unchanged.
"""
from __future__ import annotations

from supplyiq.agent.base import (
    AgentProvider,
    Explanation,
    ProviderUnavailable,
    Recommendation,
    NUMERIC_FIELDS,
)
from supplyiq.agent.mock import MockProvider
from supplyiq.agent.registry import available_providers, get_provider, provider_status

__all__ = [
    "AgentProvider",
    "Explanation",
    "MockProvider",
    "NUMERIC_FIELDS",
    "ProviderUnavailable",
    "Recommendation",
    "available_providers",
    "get_provider",
    "provider_status",
]
