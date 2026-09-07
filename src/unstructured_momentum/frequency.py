"""Frequency and population assertions for calibrated instruments.

Three separate debugging sessions were spent on one species of bug: an instrument
calibrated on one population, applied silently to another, producing plausible output.

  1. the exposure extractor, validated on Item 7A market-risk disclosure, was fed Item 1A
     risk factors -- 10% schema rejections and 72% apparent disagreement, both artifacts
  2. cl.momentum's `len(dates) < 60` history guard means 60 TRADING DAYS on a daily panel
     and FIVE YEARS on a monthly one; it silently returned empty for every pre-2011 month
  3. cl.JUMP = 0.45 treats any move beyond 45% as a corporate action, which is right daily
     and wrong monthly; it erased BAC's real -53% and +73% through the 2009 crisis

None raised. All three produced numbers that looked reasonable. The defence is the one
already built for the extractor: an instrument asserts what it is being fed rather than
trusting the caller.
"""
from __future__ import annotations
import pandas as pd

DAILY, MONTHLY, MIXED = "daily", "monthly", "mixed"


class FrequencyMismatch(ValueError):
    """Raised when an instrument is handed data at a frequency it was not calibrated for."""


def infer(index: pd.DatetimeIndex) -> str:
    """Classify an index by its median spacing. `mixed` is its own answer, not a failure --
    the IWV panel is genuinely monthly to 2013 and daily after, and treating it as either
    one is how the split corrector went wrong."""
    if len(index) < 3:
        return MIXED
    gaps = pd.Series(index).diff().dt.days.dropna()
    med = gaps.median()
    # The tail matters more than the centre. A first version tested the median and the
    # 90th percentile, and classified the full IWV panel as `daily` -- 85 monthly dates
    # from 2006-2012 among 3,341 rows move neither statistic. That is precisely the panel
    # whose mixed frequency broke the split corrector, so the check has to see it.
    long_share = float((gaps > 20).mean())
    short_share = float((gaps <= 4).mean())
    if long_share > 0.01 and short_share > 0.01:
        return MIXED
    if med <= 4:
        return DAILY
    if 20 <= med <= 40:
        return MONTHLY
    return MIXED


def require(index: pd.DatetimeIndex, *, expect: str, instrument: str) -> None:
    """Refuse data whose frequency the instrument was not calibrated for."""
    got = infer(index)
    if got != expect:
        raise FrequencyMismatch(
            f"{instrument} is calibrated for {expect!r} data and was handed {got!r} "
            f"({len(index)} rows, {index.min().date()}..{index.max().date()}). "
            f"Its thresholds are frequency-dependent; re-calibrate or split the panel "
            f"before passing it."
        )
