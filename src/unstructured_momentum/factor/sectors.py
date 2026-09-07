"""One sector taxonomy for a panel that carries two, and no placeholders ranked as values.

The iShares holdings panel spans a vendor change. Filings before roughly 2020 carry the Russell
scheme -- Financial Services, Technology, Producer Durables, Materials & Processing -- and later
ones carry GICS: Financials, Information Technology, Industrials, Materials. Both appear in the
same universe, so the same company changes sector without changing business, and any
sector-matched null over the panel silently compares one scheme's labels with another's.

Alongside them, 419 names carry `-`. That is a placeholder the vendor writes when it has no
label, and it was modal often enough in the winner leg to become `top_sector` in all 134 rows of
the conferred panel -- a placeholder ranked as a value, which is a species this project has
already paid for once with a `-` ranked as a stock.

The single-date work was unaffected because its labels are GICS throughout. Anything spanning
the panel is not.
"""

from __future__ import annotations

import pandas as pd

#: Russell scheme to GICS. Kept as an explicit table rather than a fuzzy match, because a
#: mapping this consequential should be readable and arguable rather than inferred.
RUSSELL_TO_GICS = {
    "Financial Services": "Financials",
    "Technology": "Information Technology",
    "Producer Durables": "Industrials",
    "Materials & Processing": "Materials",
    "Telecommunications": "Communication",
    "Other Energy": "Energy",
    "Product Durables": "Industrials",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Health Care": "Health Care",
    "Utilities": "Utilities",
}

#: Labels that are not sectors. `-` is the vendor's empty cell; `Other` is its residual bucket
#: and holds names from several real sectors, so treating it as one would make a null that
#: matches "Other" to "Other" look matched when it is not.
NOT_A_SECTOR = {"-", "", "Other", "n/a", "N/A", "None"}

#: Russell's "Autos & Transportation" spans GICS Consumer Discretionary and Industrials, and the
#: label alone cannot say which a given name belongs to. It is NOT mapped: guessing would put
#: roughly half of its 41 names in the wrong sector, and a sector-matched null built on a wrong
#: label is worse than one built on a missing label. Names carrying it are resolved from a later
#: filing where the vendor gives a GICS label, and dropped only if they never have one.
UNMAPPABLE = {"Autos & Transportation"}

#: The canonical set after mapping. Anything outside it is a mapping gap, not a sector.
GICS = {
    "Communication", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials",
    "Health Care", "Industrials", "Information Technology", "Materials", "Real Estate",
    "Utilities",
}

#: One inexactness, recorded rather than hidden. The Russell scheme had no Real Estate sector
#: and no Communication Services sector: REITs sat inside Financial Services and telecoms inside
#: Utilities. Mapping Financial Services to Financials therefore puts pre-switch REITs in
#: Financials, where the vendor had them, rather than inventing a Real Estate label the source
#: never carried. A sector-matched null is exact within a vintage and approximate across the
#: switch, and a claim that depends on the difference should not span it.
KNOWN_INEXACT = (
    "Russell had no Real Estate or Communication Services sector; pre-switch REITs map into "
    "Financials and pre-switch telecoms into Utilities, as the source had them."
)


def normalise(sectors: pd.Series) -> pd.Series:
    """Map both schemes onto GICS and turn non-sectors into NaN.

    NaN rather than a bucket: a name with no sector cannot be sector-matched, and giving it a
    label makes the null claim a match it does not have.
    """
    s = sectors.astype("string").str.strip()
    s = s.where(~s.isin(NOT_A_SECTOR))
    s = s.replace(RUSSELL_TO_GICS)
    return s.where(s.isin(GICS))


def resolve(panel_rows: pd.DataFrame, *, ticker: str = "ticker", sector: str = "sector",
            date: str = "as_of") -> pd.Series:
    """One sector per ticker, taken from the most recent filing that actually names one.

    The rule this replaces was first-occurrence-wins, and the earliest files are the least
    labelled: 3,435 of 7,475 names in the panel universe came back as "-" because a ticker first
    seen in 2006 kept the 2006 file's empty cell forever, even where its 2019 filing names a
    sector. Forty-six percent of the universe was unlabelled by a tie-break, not by the source.

    Latest rather than modal, because the vendor's later filings use the newer taxonomy and a
    modal rule would resurrect the Russell label for any name that lived longer under it.
    """
    d = panel_rows[[ticker, sector, date]].copy()
    d[sector] = d[sector].astype("string").str.strip()
    d = d[~d[sector].isin(NOT_A_SECTOR) & d[sector].notna()]
    if d.empty:
        return pd.Series(dtype="string")
    d = d.sort_values(date).drop_duplicates(ticker, keep="last")
    return normalise(d.set_index(ticker)[sector])


def audit(sectors: pd.Series) -> dict:
    """What normalising this series did, as numbers a caller can print or assert on."""
    raw = sectors.astype("string").str.strip()
    out = normalise(sectors)
    return {
        "n": int(len(raw)),
        "distinct_before": int(raw.nunique()),
        "distinct_after": int(out.nunique()),
        "placeholders": int(raw.isin(NOT_A_SECTOR).sum()),
        "russell_relabelled": int(raw.isin(
            [k for k, v in RUSSELL_TO_GICS.items() if k != v]).sum()),
        "unmapped": sorted(set(raw.dropna()) - set(NOT_A_SECTOR) - set(RUSSELL_TO_GICS) - GICS),
        "dropped": int(out.isna().sum() - raw.isna().sum()),
    }
