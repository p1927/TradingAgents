"""Shared look-ahead-safe date-window filtering for dated content.

News, StockTwits, and Reddit all pull recent items that must be trimmed to the
analysis window so a historical/backtest run never sees content published after
its as-of date. Centralizing the rule keeps every source consistent (#1126,
#1220): every timestamp is normalized to UTC, the upper bound is exclusive at
midnight after ``end`` (so an item stamped exactly then can't leak), and an
undated item is kept only when the window reaches the present (a live run), since
in a backtest we can't prove it isn't future.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; a naive value is assumed to be UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def in_window(pub_dt: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """Whether an item belongs in the half-open window ``[start, end + 1 day)``.

    The day is the bounds' own: a naive bound is a UTC day; a zone-aware one (Trade passes the market's
    midnight, e.g. IST) is a day of that zone, and the day is added on its WALL clock so a DST day is
    23/25 h. A replay of IST day D therefore never reads a headline stamped after 00:00 IST D+1.

    ``pub_dt`` None means undated: kept only when the window reaches the present.
    """
    end_exclusive = to_utc(end_dt + timedelta(days=1))
    if pub_dt is not None:
        return to_utc(start_dt) <= to_utc(pub_dt) < end_exclusive
    return end_exclusive >= datetime.now(timezone.utc)
