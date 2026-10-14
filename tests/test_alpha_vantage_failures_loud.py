"""Alpha Vantage failures must reach the vendor router, not come back as data.

A rate-limited indicator call used to return "Error retrieving ..." as if it were the indicator,
and a body that could not be date-filtered was returned unfiltered (rows after end_date leak into
a historical run).
"""
import pytest

import tradingagents.dataflows.alpha_vantage_indicator as avi
from tradingagents.dataflows.errors import VendorRateLimitError

# alpha_vantage_common imports without trade_integrations (that is lazy), so this runs standalone.
from tradingagents.dataflows.alpha_vantage_common import _filter_csv_by_date_range


@pytest.mark.unit
def test_rate_limited_indicator_raises_for_the_router(monkeypatch):
    def limited(*_a, **_k):
        raise VendorRateLimitError("daily cap")

    monkeypatch.setattr(avi, "_make_api_request", limited)
    with pytest.raises(VendorRateLimitError):
        avi.get_indicator("AAPL", "rsi", "2026-01-10", 5)


@pytest.mark.unit
def test_undatable_csv_body_is_an_error_not_unfiltered_data():
    with pytest.raises(ValueError, match="could not be date-filtered"):
        _filter_csv_by_date_range('{"Error Message": "Invalid API call"}', "2026-01-01", "2026-01-10")


@pytest.mark.unit
def test_dated_csv_is_still_filtered():
    csv = "timestamp,close\n2026-01-05,1\n2026-01-20,2\n"
    assert "2026-01-20" not in _filter_csv_by_date_range(csv, "2026-01-01", "2026-01-10")
