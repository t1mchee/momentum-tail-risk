"""Point-in-time discipline.

Every record in this project carries two timestamps:

``observed_at``
    When the fact *refers to*. A short-interest settlement date, a news article's
    publication time, a 13F report-period end.

``available_at``
    When we could first have *known* it. FINRA's dissemination datetime, a filing's
    EDGAR acceptance time, GDELT's crawl slot.

Features are built on ``available_at``, never ``observed_at``. This is the single
convention that makes the leakage story defensible, and it has to exist from the first
commit -- retrofitting point-in-time discipline onto a codebase never actually works,
because by then the violations are load-bearing.

The two traps this module is specifically designed to prevent, both flagged during data
review as the most likely sources of silent leakage:

1. Using FINRA short-interest *settlement* date instead of the *dissemination* date
   (~13 calendar days later, and the offset is not constant).
2. Using 13F / N-PORT *report period* end instead of the per-filing *acceptance* date
   (up to 45 days later, and highly variable across managers).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd

OBSERVED_AT = "observed_at"
AVAILABLE_AT = "available_at"

#: Everything is stored in UTC. Market-local reasoning happens at the edges, in the
#: loaders, where the timezone rules of each source are known.
TZ = "UTC"
NY = "America/New_York"


class LookaheadError(AssertionError):
    """Raised when a frame would let a feature see the future."""


# --------------------------------------------------------------------------------------
# Construction and validation
# --------------------------------------------------------------------------------------


def to_utc(values, *, tz: str = NY) -> pd.Series:
    """Coerce a date/datetime-like column to tz-aware UTC.

    Naive inputs are interpreted in ``tz`` (default US Eastern, since almost every
    source here is US market data) and then converted, rather than being silently
    assumed to already be UTC -- a mistake that shifts intraday alignment by hours.
    """
    s = pd.to_datetime(values, errors="coerce")
    if isinstance(s, pd.Timestamp):
        s = pd.Series([s])
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    return s.dt.tz_convert(TZ)


def validate(df: pd.DataFrame, *, name: str = "frame") -> pd.DataFrame:
    """Check that a frame obeys the point-in-time contract.

    Returns the frame unchanged so this can be used inline at the end of a loader.
    """
    missing = {OBSERVED_AT, AVAILABLE_AT} - set(df.columns)
    if missing:
        raise LookaheadError(f"{name}: missing point-in-time columns {sorted(missing)}")

    for col in (OBSERVED_AT, AVAILABLE_AT):
        if getattr(df[col].dt, "tz", None) is None:
            raise LookaheadError(f"{name}: {col} must be tz-aware (use pit.to_utc)")

    if df[AVAILABLE_AT].isna().any():
        raise LookaheadError(f"{name}: {AVAILABLE_AT} contains NaT; availability is unknown")

    bad = df[AVAILABLE_AT] < df[OBSERVED_AT]
    if bad.any():
        first = df.loc[bad].iloc[0]
        raise LookaheadError(
            f"{name}: {int(bad.sum())} rows are available before they were observed "
            f"(e.g. observed_at={first[OBSERVED_AT]}, available_at={first[AVAILABLE_AT]}). "
            "This is a time-travel bug, usually a timezone or lag-sign error."
        )
    return df


def stamp(
    df: pd.DataFrame,
    *,
    observed_at,
    available_at,
    tz: str = NY,
    name: str = "frame",
) -> pd.DataFrame:
    """Attach and validate point-in-time columns in one step.

    ``observed_at`` / ``available_at`` may each be a column name, a scalar, or a Series.
    """
    out = df.copy()
    out[OBSERVED_AT] = to_utc(
        df[observed_at] if isinstance(observed_at, str) else observed_at, tz=tz
    )
    out[AVAILABLE_AT] = to_utc(
        df[available_at] if isinstance(available_at, str) else available_at, tz=tz
    )
    return validate(out, name=name)


# --------------------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------------------


def as_of(df: pd.DataFrame, when: dt.datetime | pd.Timestamp | str) -> pd.DataFrame:
    """Everything knowable at ``when``. The workhorse of the whole project."""
    ts = pd.Timestamp(when)
    ts = ts.tz_localize(NY).tz_convert(TZ) if ts.tz is None else ts.tz_convert(TZ)
    return df.loc[df[AVAILABLE_AT] <= ts]


def latest_as_of(
    df: pd.DataFrame,
    when: dt.datetime | pd.Timestamp | str,
    *,
    by: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Most recent knowable record, optionally one per group.

    Sorts by ``observed_at`` rather than ``available_at`` so that a late-arriving
    *revision* of an older observation does not displace a newer observation.
    """
    visible = as_of(df, when).sort_values([OBSERVED_AT, AVAILABLE_AT])
    if by is None:
        return visible.tail(1)
    return visible.groupby(list([by] if isinstance(by, str) else by), as_index=False).tail(1)


def asof_align(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    value_cols: Sequence[str],
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Align an irregular point-in-time panel onto a trading calendar.

    Each calendar date receives the latest value whose ``available_at`` is at or before
    it. Uses a backward as-of join, which is the only merge direction that cannot leak.
    """
    left = pd.DataFrame({AVAILABLE_AT: pd.DatetimeIndex(calendar).tz_convert(TZ)})
    right = panel.sort_values(AVAILABLE_AT)[[AVAILABLE_AT, *value_cols]]
    merged = pd.merge_asof(
        left.sort_values(AVAILABLE_AT),
        right,
        on=AVAILABLE_AT,
        direction="backward",
        tolerance=tolerance,
    )
    return merged.set_index(AVAILABLE_AT)


# --------------------------------------------------------------------------------------
# Lag helpers
# --------------------------------------------------------------------------------------


def us_business_days(start, n: int) -> pd.Timestamp:
    """Add ``n`` US business days (holidays excluded) to a date."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.Timestamp(start) + n * bday


def next_open_after(when: pd.Timestamp, *, hour: int = 9, minute: int = 30) -> pd.Timestamp:
    """The next US equity market open strictly after ``when``.

    Used to convert "this became public at 4:15pm on a Tuesday" into "the first moment a
    position could actually have been changed on it". Conservative by construction: if a
    release lands during the session, we still wait for the following open rather than
    assuming intraday execution.
    """
    ts = when.tz_convert(NY) if when.tz is not None else when.tz_localize(NY)
    candidate = us_business_days(ts.normalize(), 1).tz_localize(None)
    candidate = pd.Timestamp(candidate).replace(hour=hour, minute=minute).tz_localize(NY)
    while candidate <= ts:
        candidate = pd.Timestamp(
            us_business_days(candidate.tz_localize(None).normalize(), 1)
        ).replace(hour=hour, minute=minute).tz_localize(NY)
    return candidate.tz_convert(TZ)


def assert_no_lookahead(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    name: str = "features",
) -> None:
    """Sanity check that features are strictly not contemporaneous with the target.

    Cheap, blunt, and worth running on every feature matrix before any model is fit:
    a feature index that reaches at or beyond the target index is the most common way a
    backtest quietly becomes a lie.
    """
    if features.index.max() >= target.index.max():
        raise LookaheadError(
            f"{name}: feature index reaches {features.index.max()} but target ends at "
            f"{target.index.max()}. Features must be strictly lagged relative to the target."
        )
