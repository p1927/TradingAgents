"""`in_window` upper bound is the next WALL-CLOCK midnight in the end bound's own zone (a zone-aware end), UTC for naive."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradingagents.dataflows.date_window import in_window

NY = ZoneInfo("America/New_York")
IST = ZoneInfo("Asia/Kolkata")


def test_ist_end_excludes_next_ist_day():
    end = datetime(2026, 3, 10, tzinfo=IST)
    assert in_window(datetime(2026, 3, 10, 18, 29, tzinfo=timezone.utc), end, end)  # 23:59 IST
    assert not in_window(datetime(2026, 3, 10, 18, 30, tzinfo=timezone.utc), end, end)  # 00:00 IST D+1


def test_spring_forward_day_is_23_hours():
    end = datetime(2026, 3, 8, tzinfo=NY)
    u = end.astimezone(timezone.utc)
    assert in_window(u + timedelta(hours=22, minutes=59), end, end)
    assert not in_window(u + timedelta(hours=23), end, end)


def test_naive_end_stays_utc_day():
    end = datetime(2026, 3, 10)
    assert in_window(datetime(2026, 3, 10, 23, 59, tzinfo=timezone.utc), end, end)
    assert not in_window(datetime(2026, 3, 11, tzinfo=timezone.utc), end, end)
