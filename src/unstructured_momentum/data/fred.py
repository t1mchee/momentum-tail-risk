"""FRED macro series (no API key required).

The credit-spread problem
-------------------------
``BAMLH0A0HYM2`` -- the ICE BofA high-yield OAS, the series most people reach for -- is
**no longer freely available with full history**. Verified 2026-08-21: the CSV returns a
rolling 3-year window (795 rows starting 2023-08-21) regardless of ``cosd``/``coed``
parameters. The same cap applies to ``BAMLC0A0CM`` and ``BAMLH0A0HYM2EY``. It is an ICE
Data Indices licensing restriction, not a FRED-wide cap: ``DGS10``, ``T10Y2Y``,
``VIXCLS`` and ``NFCI`` all still return full history the same day.

A 3-year window is useless for a project whose entire subject is rare events -- it does
not even reach the 2022 episode. So the default credit proxy here is **``BAA10Y``**
(Moody's Baa corporate minus 10-Y Treasury, daily, from 1986), which is unrestricted and
spans every modern momentum crash. ``NFCI``/``NFCICREDIT`` (Chicago Fed, weekly, from
1971) is the secondary gauge and reaches back further still.

This substitution is a real cost and is reported as such: Baa-Treasury is investment
grade and duration-contaminated, so it is less sensitive to the leveraged-credit stress
that accompanies deleveraging than a high-yield OAS would be.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from .. import pit
from ..config import RAW, USER_AGENT

CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: series id -> (description, publication lag in business days)
SERIES: dict[str, tuple[str, int]] = {
    "BAA10Y": ("Moody's Baa minus 10Y Treasury (credit proxy; daily from 1986)", 1),
    "T10Y2Y": ("10Y-2Y term spread (daily from 1976)", 1),
    "T10Y3M": ("10Y-3M term spread", 1),
    "DGS10": ("10Y Treasury constant maturity", 1),
    "DGS2": ("2Y Treasury constant maturity", 1),
    "VIXCLS": ("VIX close (FRED mirror, no bot gate; from 1990)", 1),
    "NFCI": ("Chicago Fed National Financial Conditions Index (weekly, Wed)", 5),
    "NFCICREDIT": ("NFCI credit subindex (weekly)", 5),
    "STLFSI4": ("St. Louis Fed Financial Stress Index (weekly)", 5),
}

#: Restricted to a rolling 3-year window by ICE licensing. Kept here so that anyone who
#: reaches for the obvious series gets an explanation rather than a puzzling short frame.
RESTRICTED = {
    "BAMLH0A0HYM2": "ICE BofA HY OAS -- rolling 3y only since ~2025. Use BAA10Y.",
    "BAMLC0A0CM": "ICE BofA IG OAS -- rolling 3y only. Use BAA10Y.",
    "BAMLH0A0HYM2EY": "ICE BofA HY effective yield -- rolling 3y only.",
}


def fetch(*ids: str, refresh: bool = False) -> pd.DataFrame:
    """Download one or more FRED series into a single date-indexed frame."""
    for sid in ids:
        if sid in RESTRICTED:
            raise ValueError(f"{sid}: {RESTRICTED[sid]}")

    key = "_".join(sorted(ids))
    path = RAW / "fred" / f"{key}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    if refresh or not path.exists():
        resp = requests.get(
            CSV, params={"id": ",".join(ids)}, headers={"User-Agent": USER_AGENT}, timeout=90
        )
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
        text = resp.text
    else:
        text = path.read_text(encoding="utf-8")

    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # 'observation_date' on the current API
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    # FRED writes missing observations as empty strings and '.', neither of which pandas
    # types as NaN on its own.
    return df.replace(".", pd.NA).apply(pd.to_numeric, errors="coerce")


def panel(*, refresh: bool = False) -> pd.DataFrame:
    """Daily macro block: credit proxy, term spread, and financial conditions."""
    daily = fetch("BAA10Y", "T10Y2Y", "T10Y3M", refresh=refresh)
    weekly = fetch("NFCI", "NFCICREDIT", refresh=refresh)
    # Weekly series are forward-filled onto the daily grid, which is correct here: the
    # last published value genuinely is the current state of knowledge until the next.
    return daily.join(weekly.reindex(daily.index, method="ffill"), how="left")


def panel_pit(*, refresh: bool = False) -> pd.DataFrame:
    """`panel()` with point-in-time stamps.

    Uses the slowest constituent's lag (weekly NFCI, released Wednesdays covering the
    prior week) so that no column in a given row is ever ahead of its real release.
    Conservative by construction.
    """
    df = panel(refresh=refresh).reset_index()
    avail = df["date"].apply(lambda d: pit.us_business_days(d, 5))
    return pit.stamp(df, observed_at="date", available_at=avail, name="fred:panel")
