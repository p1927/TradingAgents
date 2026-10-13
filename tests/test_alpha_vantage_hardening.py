"""Alpha Vantage request hardening.

Regressions for #990 (no request timeout -> can hang), #991 (invalid-key
responses mislabeled as rate limits and silently treated as transient), and
#1115 (fundamentals look-ahead filter never ran because the payload is a JSON
string, not a dict).
"""
import json

import pytest

# _make_api_request routes through Trade's tiered_api, so these tests can only
# run inside the Trade monorepo (where Trade's CI runs them); the fork's own CI
# skips them instead of failing on an import it can never satisfy.
pytest.importorskip("trade_integrations")

import tradingagents.dataflows.alpha_vantage_common as av  # noqa: E402
import tradingagents.dataflows.alpha_vantage_fundamentals as avf


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patched_get(body, capture=None):
    def fake_get(url, params=None, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(body)
    return fake_get


@pytest.mark.unit
def test_request_passes_timeout(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADE_STACK_HUB_DIR", str(tmp_path))
    captured = {}
    monkeypatch.setattr(av.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0", captured))
    av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert captured.get("timeout") == av.REQUEST_TIMEOUT  # #990


@pytest.mark.unit
def test_rate_limit_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADE_STACK_HUB_DIR", str(tmp_path))
    body = '{"Information": "Our standard API rate limit is 25 requests per day. ... your API key ..."}'
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_invalid_key_not_mislabeled_as_rate_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADE_STACK_HUB_DIR", str(tmp_path))
    # AV's invalid-key notice mentions "API key"; it must NOT be treated as a
    # (transient) rate limit, but surface as a real configuration error (#991).
    body = ('{"Information": "the parameter apikey is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(av.AlphaVantageRateLimitError):  # sanity: rate-limit path still distinct
        monkeypatch.setattr(av.requests, "get", _patched_get('{"Note": "API call frequency is 5 calls per minute."}'))
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


_FUNDAMENTALS_JSON = json.dumps({
    "symbol": "AAPL",
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "totalAssets": "1"},   # future -> must drop
        {"fiscalDateEnding": "2023-12-31", "totalAssets": "2"},   # past   -> must keep
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2024-06-30", "totalAssets": "3"},   # future -> must drop
        {"fiscalDateEnding": "2023-09-30", "totalAssets": "4"},   # past   -> must keep
    ],
})


@pytest.mark.unit
def test_fundamentals_look_ahead_filter_runs_on_json_string(monkeypatch):
    # #1115: the payload arrives as a JSON *string*; the old dict-only guard let
    # future-dated fiscal periods leak into historical runs.
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: _FUNDAMENTALS_JSON)
    out = avf.get_balance_sheet("AAPL", curr_date="2024-01-01")
    assert isinstance(out, str)  # callers still receive a str
    parsed = json.loads(out)
    assert [r["fiscalDateEnding"] for r in parsed["annualReports"]] == ["2023-12-31"]
    assert [r["fiscalDateEnding"] for r in parsed["quarterlyReports"]] == ["2023-09-30"]


@pytest.mark.unit
def test_fundamentals_no_curr_date_passes_through(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: _FUNDAMENTALS_JSON)
    assert avf.get_income_statement("AAPL") == _FUNDAMENTALS_JSON


@pytest.mark.unit
def test_fundamentals_non_json_body_unchanged(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    assert avf.get_cashflow("AAPL", curr_date="2024-01-01") == "not-json"


@pytest.mark.unit
def test_every_function_declares_max_age_hours(monkeypatch, tmp_path):
    """D381: the watcher stage failed with `tiered_fetch() missing max_age_hours`; each function
    the dataflows call must have a declared freshness and reach tiered_fetch with it."""
    import re
    from pathlib import Path

    called = set()
    for f in Path(av.__file__).parent.glob("alpha_vantage_*.py"):
        called |= set(re.findall(r'_make_api_request\(\s*"([A-Z_]+)"', f.read_text()))
    assert called and called <= set(av._MAX_AGE_HOURS)

    monkeypatch.setenv("TRADE_STACK_HUB_DIR", str(tmp_path))
    monkeypatch.setattr(av.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0"))
    seen = {}
    import trade_integrations.tiered_api as ta
    real = ta.tiered_fetch
    monkeypatch.setattr(ta, "tiered_fetch", lambda *a, **k: seen.update(k) or real(*a, **k))
    av._make_api_request("OVERVIEW", {"symbol": "AAPL"})
    assert "max_age_hours" in seen


@pytest.mark.unit
def test_missing_key_is_the_vendor_unavailable_error(monkeypatch, tmp_path):
    """No key -> AlphaVantageNotConfiguredError (router skips to the next vendor), not a raw
    tiered_api error, and get_api_key names the same env vars the registry does."""
    monkeypatch.setenv("TRADE_STACK_HUB_DIR", str(tmp_path))
    for var in ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av.get_api_key()
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("OVERVIEW", {"symbol": "AAPL"})
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "k2")
    assert av.get_api_key() == "k2"
