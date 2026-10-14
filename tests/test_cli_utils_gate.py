"""CLI menu gates the D419 provider list before a key is prompted/persisted; the ticker normaliser fails loud."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import cli.utils as cli_utils


def test_ensure_api_key_refuses_disallowed_provider_before_prompting(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_utils, "_allowed_provider_keys", lambda: {"minimax", "nvidia"})
    with patch.object(cli_utils.questionary, "password") as ask:
        with pytest.raises(ValueError, match="not allowed"):
            cli_utils.ensure_api_key("openai")
    ask.assert_not_called()
    assert not (tmp_path / ".env").exists()


def test_provider_menu_lists_only_allowed(monkeypatch):
    monkeypatch.setattr(cli_utils, "_allowed_provider_keys", lambda: {"minimax"})
    seen = {}

    def fake_select(_msg, choices, **_kw):
        seen["keys"] = [c.value[0] for c in choices]
        return type("Q", (), {"ask": staticmethod(lambda: choices[0].value)})()

    monkeypatch.setattr(cli_utils.questionary, "select", fake_select)
    assert cli_utils.select_llm_provider()[0] == "minimax"
    assert seen["keys"] == ["minimax"]


def test_normalize_only_import_error_falls_back(monkeypatch):
    import tradingagents.dataflows.symbol_utils as su

    monkeypatch.setattr(su, "normalize_symbol", lambda t: (_ for _ in ()).throw(RuntimeError("broken")))
    with pytest.raises(RuntimeError, match="broken"):
        cli_utils.normalize_ticker_symbol(" aapl ")
    monkeypatch.setattr(su, "normalize_symbol", lambda t: t.strip().upper() + "!")
    assert cli_utils.normalize_ticker_symbol(" aapl ") == "AAPL!"
