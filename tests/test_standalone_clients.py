"""Upstream clients work without importing the containing Trade monorepo."""

import builtins
from types import SimpleNamespace
import sys


def test_av_request_has_no_trade_dependency(monkeypatch):
    from tradingagents.dataflows import alpha_vantage_common as av

    original = builtins.__import__

    def no_trade(name, *args, **kwargs):
        if name.startswith("trade_integrations"):
            raise AssertionError("standalone client imported Trade")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trade)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fixture")
    response = SimpleNamespace(text="timestamp,close\n2026-01-01,1", raise_for_status=lambda: None)
    calls = []
    monkeypatch.setattr(av.requests, "get", lambda *a, **kw: calls.append(kw) or response)
    assert av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"}).startswith("timestamp")
    assert calls[0]["params"]["apikey"] == "fixture"
    assert calls[0]["timeout"] == av.REQUEST_TIMEOUT


def test_factory_has_no_trade_dependency(monkeypatch):
    from tradingagents.llm_clients.factory import create_llm_client

    client = object()
    monkeypatch.setitem(
        sys.modules,
        "tradingagents.llm_clients.openai_client",
        SimpleNamespace(OpenAIClient=lambda *a, **kw: client, is_openai_compatible=lambda _: True),
    )
    original = builtins.__import__

    def no_trade(name, *args, **kwargs):
        if name.startswith("trade_integrations"):
            raise AssertionError("standalone factory imported Trade")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trade)
    assert create_llm_client("openai", "fixture") is client
