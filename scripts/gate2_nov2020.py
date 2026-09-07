"""Gate 2, deliverable 1 -- the November 2020 loser-leg extraction table, end to end.

Formation date 2020-10-31, the book carried into the 2020-11-09 vaccine reversal. Every stage
runs here except the structured extraction itself, which needs a model credential.

The pipeline is: point-in-time loser leg -> 8-K filings accepted in the window, keyed on
ACCEPTANCE time -> structured extraction of the external condition that determines survival
-> embed -> cluster -> entropy over cluster assignments. Winner leg is the contrast.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import eightk  # noqa: E402

FORMATION = pd.Timestamp("2020-10-31")
#: Filings are read over the quarter BEFORE formation. A filing accepted after the formation
#: date could not have informed the book, so acceptance time is the gate, not filing date.
WINDOW_START = pd.Timestamp("2020-08-01")


def leg_filings(tickers: set, ix: pd.DataFrame, acc: pd.Series) -> pd.DataFrame:
    lo = pd.Timestamp(WINDOW_START, tz="UTC")
    hi = pd.Timestamp(FORMATION, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    m = (acc >= lo) & (acc <= hi) & ix["ticker"].isin(tickers)
    out = ix[m].copy()
    out["accepted_at"] = acc[m]
    return out


def main() -> None:
    legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    if FORMATION not in legs:
        raise SystemExit(f"no leg panel entry at {FORMATION.date()}")
    losers = set(legs[FORMATION]["losers"])
    winners = set(legs[FORMATION]["winners"])

    ix = eightk.load_index()
    acc = pd.to_datetime(ix["accepted_at"], errors="coerce", utc=True)

    rows = []
    for name, tick in (("loser", losers), ("winner", winners)):
        f = leg_filings(tick, ix, acc)
        covered = f["ticker"].nunique()
        rows.append({"leg": name, "constituents": len(tick), "filings": len(f),
                     "names_with_a_filing": covered,
                     "coverage": round(covered / len(tick), 4),
                     "median_filings_per_covered_name":
                         float(f.groupby("ticker").size().median()) if len(f) else 0.0})
    C = pd.DataFrame(rows)
    print(f"FORMATION {FORMATION.date()}   filing window {WINDOW_START.date()} -> "
          f"{FORMATION.date()} (acceptance time)")
    print(C.to_string(index=False))

    lf = leg_filings(losers, ix, acc)
    print(f"\nLOSER-LEG FILINGS: {len(lf)} across {lf['ticker'].nunique()} names")
    it = lf["items"].fillna("").astype(str).str.split(",").explode().str.strip()
    it = it[it != ""]
    print("\n  most common 8-K items:")
    for k, v in it.value_counts().head(8).items():
        print(f"    {k:8s} {v:5d}")
    print(f"\n  document sizes: median {lf['n_chars'].median():,.0f} chars, "
          f"p90 {lf['n_chars'].quantile(0.9):,.0f}")

    Path("reports/gate2").mkdir(parents=True, exist_ok=True)
    lf.to_parquet("reports/gate2/nov2020_loser_filings.parquet")
    C.to_csv("reports/gate2/nov2020_coverage.csv", index=False)
    json.dump({"formation": str(FORMATION.date()),
               "window": [str(WINDOW_START.date()), str(FORMATION.date())],
               "coverage": C.to_dict("records")},
              open("reports/gate2/nov2020_coverage.json", "w"), indent=2)
    print("\nwrote reports/gate2/nov2020_{loser_filings.parquet,coverage.csv,coverage.json}")


if __name__ == "__main__":
    main()
