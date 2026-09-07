"""FINRA aggregate customer margin debt: the free leverage series that reaches 2007.

The positioning stack in this project cannot see leverage. Comomentum infers crowding from
returns, which is endogenous to the outcome it is used to condition; 13F is long-only,
quarterly and lagged 45 days. This series is none of those things, and it is the first
candidate conditioner here that is not computed from past returns.

Its limits are structural and are not fixable by better handling: it is market-level rather
than factor-level, and it measures customer margin at brokers, not the prime-brokerage leverage
that actually unwound in August 2007.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/margin/finra_margin_statistics.xlsx")
OUT = Path("data/processed/finra_margin.parquet")

#: FINRA disseminates roughly three to four weeks after the measurement month and the file
#: carries NO publication date. Availability is therefore month-end plus this many days. A
#: feature built on the measurement date would hold a leverage reading nobody could have seen.
PUBLICATION_LAG_DAYS = 25


def load(*, refresh: bool = False) -> pd.DataFrame:
    """Monthly margin debt with an availability stamp. Never returns the measurement date."""
    if OUT.exists() and not refresh:
        return pd.read_parquet(OUT)
    d = pd.read_excel(RAW)
    d.columns = ["ym", "debit_margin", "free_credit_cash", "free_credit_margin"]
    d["date"] = pd.PeriodIndex(d["ym"], freq="M").to_timestamp("M")
    d = d.sort_values("date").reset_index(drop=True)
    d["available_at"] = d["date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    d.to_parquet(OUT, index=False)
    return d


def features(*, window: int = 12) -> pd.DataFrame:
    """Trailing growth in margin debt, stamped by when it could have been read.

    Growth rather than level, because the level trends with the market and a level feature
    would be a slow proxy for prices. The z-score is trailing-standardised over an expanding
    window so it carries no information from its own future.
    """
    d = load().set_index("date")
    g = d["debit_margin"].pct_change(window)
    z = (g - g.expanding(36).mean()) / g.expanding(36).std()
    out = pd.DataFrame({"debit_margin": d["debit_margin"], "growth": g, "z_growth": z,
                        "available_at": d["available_at"]})
    return out.dropna(subset=["growth"])


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="margindebt")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--features", action="store_true")
    a = p.parse_args(argv)

    d = load(refresh=a.refresh)
    print(f"rows {len(d)}   {d['date'].min().date()} -> {d['date'].max().date()}")
    print(f"availability lag {PUBLICATION_LAG_DAYS}d (FINRA publishes no date in the file)")
    if a.features:
        f = features()
        print(f"\nfeature rows {len(f)}")
        print(f[["debit_margin", "growth", "z_growth"]].tail(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
