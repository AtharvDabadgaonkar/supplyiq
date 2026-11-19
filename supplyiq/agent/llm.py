"""Optional LLM provider.

Deliberately uses urllib from the standard library rather than a vendor SDK, so
requirements.txt stays small for Streamlit Community Cloud and the app has no
hard dependency on a provider package.

The prompt is constrained to prose, and whatever comes back is run through
`agent.guard`. If the model invents a number, the explanation is rejected and
the rule-based one is returned with `fallback_reason` set, so the planner is
never shown a figure the simulator did not compute.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Literal

from supplyiq.agent.base import AgentProvider, Explanation, Recommendation
from supplyiq.agent.guard import unverified_numbers
from supplyiq.agent.mock import MockProvider

Kind = Literal["anthropic", "openai"]

ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
}

SYSTEM_PROMPT = (
    "You are a supply chain planning assistant writing a one-paragraph note for a "
    "demand or inventory planner.\n"
    "You are given a recommendation whose numbers were all computed by an inventory "
    "simulator. Explain, in plain English, what the recommendation does, what the "
    "service-versus-cost tradeoff is, and whether it looks worth approving.\n"
    "Hard rules:\n"
    "1. Do NOT calculate, estimate or invent any number. You may only restate figures "
    "that appear in the payload you are given.\n"
    "2. No bullet points, no headings. Three or four sentences of prose.\n"
    "3. Speak to a planner who knows the domain. No definitions, no filler.\n"
    "4. Do not claim anything about the data that is not in the payload."
)


class LLMProvider(AgentProvider):
    is_mock = False

    def __init__(
        self,
        kind: Kind,
        model: str,
        max_output_tokens: int = 400,
        temperature: float = 0.2,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.kind = kind
        self.name = kind
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        return bool(self._api_key)

    def explain(self, rec: Recommendation) -> Explanation:
        mock = MockProvider()
        if not self.available():
            return Explanation(
                text=mock.explain(rec).text,
                provider="mock",
                is_mock=True,
                fallback_reason=f"no API key configured for {self.kind}",
            )
        try:
            text = self._call(rec).strip()
        except Exception as exc:  # network, auth, rate limit, malformed response
            return Explanation(
                text=mock.explain(rec).text,
                provider="mock",
                is_mock=True,
                fallback_reason=f"{self.kind} call failed: {type(exc).__name__}",
            )

        bad = unverified_numbers(text, rec)
        if bad:
            return Explanation(
                text=mock.explain(rec).text,
                provider="mock",
                is_mock=True,
                fallback_reason=(
                    "rejected LLM text: it contained "
                    f"{len(bad)} number(s) the simulator did not produce "
                    f"({', '.join(f.raw for f in bad[:3])})"
                ),
            )
        return Explanation(text=text, provider=self.kind, model=self.model, is_mock=False)

    # ------------------------------------------------------------------ #
    def _payload(self, rec: Recommendation) -> tuple[dict[str, Any], dict[str, str]]:
        user = (
            "Recommendation payload (every number here is simulator output):\n"
            + json.dumps(rec.as_payload(), indent=2, default=str)
        )
        if self.kind == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user}],
            }
            headers = {
                "content-type": "application/json",
                "x-api-key": self._api_key or "",
                "anthropic-version": "2023-06-01",
            }
        else:
            body = {
                "model": self.model,
                "max_completion_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
            headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key or ''}",
            }
        return body, headers

    def _call(self, rec: Recommendation) -> str:
        body, headers = self._payload(rec)
        req = urllib.request.Request(
            ENDPOINTS[self.kind],
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return self._extract(data)

    @staticmethod
    def _extract_anthropic(data: dict[str, Any]) -> str:
        blocks = data.get("content") or []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    @staticmethod
    def _extract_openai(data: dict[str, Any]) -> str:
        return data["choices"][0]["message"]["content"]

    def _extract(self, data: dict[str, Any]) -> str:
        text = (
            self._extract_anthropic(data)
            if self.kind == "anthropic"
            else self._extract_openai(data)
        )
        if not text.strip():
            raise ValueError("provider returned an empty message")
        return text
