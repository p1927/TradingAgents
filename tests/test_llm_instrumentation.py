"""Tier 0 LLM instrumentation must attach to pydantic-based chat models.

Regression cover for
.claude/backlog/items/2026-09-07-minimax-adapter-missing-invoke-kills-debate-prefetch.md:
`_instrument_llm_client` did `llm.invoke = invoke`, but LangChain chat models are
pydantic models and pydantic v2 rejects assignment to an undeclared field:

    ValueError: "MinimaxChatOpenAI" object has no field "invoke"

The caller wraps the debate in a broad `except Exception` and logs at WARNING, so
this never surfaced as a crash — it silently skipped the agent's entire bootstrap
debate, for every provider, not only MiniMax.
"""
from __future__ import annotations

import pytest

from tradingagents.llm_clients.factory import _instrument_llm_client
from tradingagents.llm_clients.openai_client import MinimaxChatOpenAI


def _make_llm():
    llm = MinimaxChatOpenAI(
        model="MiniMax-M3", api_key="test-key", base_url="https://api.minimax.io/v1"
    )
    object.__setattr__(llm, "invoke", lambda *a, **k: "ORIGINAL")
    return llm


class _Client:
    model = "MiniMax-M3"

    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm

    def get_provider_name(self):
        return "minimax"


def test_a_pydantic_chat_model_can_be_instrumented():
    """The assignment must not raise — this is the exact original failure."""
    llm = _make_llm()
    instrumented = _instrument_llm_client(_Client(llm)).get_llm()
    assert getattr(instrumented, "_trade_obs_wrapped", False) is True


def test_the_wrapper_passes_the_underlying_result_through():
    llm = _make_llm()
    instrumented = _instrument_llm_client(_Client(llm)).get_llm()
    assert instrumented.invoke("prompt") == "ORIGINAL"


def test_instrumenting_twice_does_not_double_wrap():
    client = _instrument_llm_client(_Client(_make_llm()))
    first = client.get_llm()
    assert client.get_llm() is first
    assert _instrument_llm_client(client) is client


def test_plain_assignment_would_still_fail(monkeypatch):
    """Documents WHY object.__setattr__ is required, so it is not 'simplified' away."""
    llm = MinimaxChatOpenAI(
        model="MiniMax-M3", api_key="test-key", base_url="https://api.minimax.io/v1"
    )
    with pytest.raises(ValueError, match="has no field"):
        llm.invoke = lambda *a, **k: None


def test_importerror_inside_the_model_call_is_not_retried():
    """An ImportError from the model call itself must propagate once, not re-invoke the model."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ImportError("lazy provider import failed")

    llm = _make_llm()
    object.__setattr__(llm, "invoke", boom)
    instrumented = _instrument_llm_client(_Client(llm)).get_llm()
    with pytest.raises(ImportError):
        instrumented.invoke("prompt")
    assert len(calls) == 1


def test_minimax_debate_client_carries_trades_account_pace():
    """Inside Trade the debate's MiniMax chat model takes a slot on the one account pace (D271);
    another host is left untouched."""
    pytest.importorskip("trade_integrations")
    from tradingagents.llm_clients.openai_client import OpenAIClient

    def hooks(client):
        return client.root_client._client.event_hooks["request"]

    llm = OpenAIClient("MiniMax-M3", "https://api.minimax.io/v1", provider="minimax", api_key="k").get_llm()
    assert any(h.__name__ == "pace_httpx_request" for h in hooks(llm))
    other = OpenAIClient("gpt-4o", "http://localhost:9/v1", provider="openai", api_key="k").get_llm()
    assert not any(h.__name__ == "pace_httpx_request" for h in hooks(other))
