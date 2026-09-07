"""Unified feature panel: every pillar that survived its own validation, in one place.

Inclusion rule -- a pillar is here because it was TESTED and earned it, with the evidence
named. Nothing is included because it seemed reasonable.

  vol        trailing realised volatility. Explains 53% of forward quarterly severity on
             its own and beat every one of six candidate signals. The incumbent.
  tilt       conferred beta spread + sector tilt. 100th percentile against 200 cap-matched
             placebo books, clears in 104 of 128 months; computable same-day from holdings.
  disp       cross-construction dispersion across 16 builds. Survives vol AND |today's
             return| at t = -3.32, and absorbs the latter. Exploratory, unregistered.
  crowd      comomentum. Replicates out-of-sample at 3.06x crash-day frequency (published
             2.7x) and is worth zero basis points as a trading trigger -- both halves kept.
  skew       MTUM model-free implied skewness, 30d to 365d. The only FORWARD-LOOKING input
             here: an options surface cannot be stale, which is the defect that killed
             every return-based estimate of this book's exposure.
  bear       Daniel-Moskowitz bear state. Published benchmark. Fires in 2 of 98 design
             months, so it contributes almost nothing over 2013-2026 -- carried because a
             benchmark you drop when it underperforms is not a benchmark.

Point-in-time throughout: monthly inputs are forward-filled from the date they were first
available, never from the date they describe.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FAMILY = "data/processed/construction_family.parquet"
TILT = "data/processed/conferred_tilt.csv"
BOOKTILT = "data/processed/book_tilt.parquet"
CROWD = "data/processed/comomentum_iwv.parquet"
SKEW = "data/raw/impliedmoments/mtum_mfis.parquet"
FF = "/private/tmp/claude-502/-Users-Tim/10cf8f3c-b126-4a42-a4bb-696ed65be1f8/scratchpad/F-F_Research_Data_Factors_daily.csv"


#: Maximum age, in calendar days, that a carried-forward value may reach before it is set
#: missing. A forward-fill with no horizon is not point-in-time discipline, it is
#: fabrication: the first build of this panel carried a risk-neutral skew estimate from
#: Dec-2023 through Jul-2026 because the source series simply ended and ffill did not care.
#: Each limit is the publication cadence of its source plus slack.
MAX_STALE = {"monthly": 40, "daily": 5}


def _ffill(src: pd.Series, idx: pd.DatetimeIndex, kind: str) -> pd.Series:
    """Carry forward, then blank anything older than its source's cadence allows."""
    v = src.reindex(idx, method="ffill")
    stamp = pd.Series(src.index, index=src.index).reindex(idx, method="ffill")
    age = (pd.Series(idx, index=idx) - stamp).dt.days
    return v.where(age <= MAX_STALE[kind])


def build() -> pd.DataFrame:
    fam = pd.read_parquet(FAMILY)
    canon = fam["12m_dec_cap_broad"]
    F = pd.DataFrame(index=fam.index)
    F["wml"] = canon

    # --- vol: the incumbent -------------------------------------------------
    for w in (21, 63, 252):
        F[f"vol{w}"] = canon.rolling(w).std() * np.sqrt(252)
    F["vol_ratio"] = F.vol21 / F.vol252

    # --- dispersion: free from the family -----------------------------------
    F["disp"] = fam.max(axis=1) - fam.min(axis=1)
    F["disp63"] = F.disp.rolling(63).mean()

    # --- conferred tilt: monthly, PIT-forward-filled -------------------------
    t = pd.read_csv(TILT, parse_dates=["asof"]).set_index("asof").sort_index()
    for c in ("beta_spread", "d10y_spread", "hml_spread", "smb_spread"):
        F[c] = _ffill(t[c].dropna(), F.index, "monthly")
    bs = t["beta_spread"]
    F["tilt_pctile"] = _ffill(bs.expanding().apply(
        lambda s: (s.iloc[:-1] < s.iloc[-1]).mean() if len(s) > 1 else np.nan).dropna(),
        F.index, "monthly")

    bt = pd.read_parquet(BOOKTILT)
    axis = (bt.get("Information Technology", 0.0) - bt.get("Energy", 0.0)
            - bt.get("Financials", 0.0))
    F["sector_axis"] = _ffill(axis.dropna(), F.index, "monthly")
    F["sector_turnover"] = _ffill((bt.diff().abs().sum(axis=1) / 2).dropna(), F.index, "monthly")

    # --- crowding ------------------------------------------------------------
    cw = pd.read_parquet(CROWD)
    for c in ("comom_spread", "comom_avg"):
        if c in cw: F[c] = _ffill(cw[c].dropna(), F.index, "monthly")

    # --- options surface: the one forward-looking pillar ---------------------
    sk = pd.read_parquet(SKEW)
    sk["date"] = pd.to_datetime(sk["date"])
    sk = sk.sort_values("date").set_index("date")
    for c in ("mfis30", "mfis91", "mfis365"):
        if c in sk: F[c] = _ffill(sk[c].dropna(), F.index, "daily")
    # Term structure: near-dated minus far-dated. A steepening is the market paying up for
    # NEAR-term left tail specifically, which a level cannot show. Built from 30d/91d, not
    # 30d/365d -- the 365d tenor only starts 2023-03 and would have cut the slope to 28%
    # coverage while the 91d tenor runs from 2017.
    if {"mfis30", "mfis91"} <= set(sk.columns):
        F["mfis_slope"] = F.mfis30 - F.mfis91

    # --- bear state: the published benchmark --------------------------------
    ff = pd.read_csv(FF, skiprows=4)
    ff = ff[pd.to_numeric(ff.iloc[:, 0], errors="coerce").notna()]
    ff.index = pd.to_datetime(ff.iloc[:, 0].astype(int).astype(str), format="%Y%m%d")
    ff = ff.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    ff.columns = [c.strip() for c in ff.columns]
    # Gate 0 audit, item 2. This reindexed the market onto the feature calendar and filled
    # missing days with ZERO, which turns an absent observation into an observed flat day and
    # biases the two-year compounding toward zero exactly where data is thin. A missing return
    # must stay missing; the rolling product then yields NaN and the bear flag is NaN rather
    # than silently False, which is the difference between "not in a bear market" and "we do
    # not know". min_periods is explicit for the same reason.
    mkt = (ff["Mkt-RF"] / 100.0).reindex(F.index)
    compounded = (1 + mkt).rolling(504, min_periods=504).apply(np.prod, raw=True) - 1
    F["bear"] = (compounded < 0).astype(float).where(compounded.notna())
    return F


def coverage(F: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"non_null": F.notna().sum(), "share": F.notna().mean(),
                         "first": [F[c].first_valid_index() for c in F.columns],
                         "last": [F[c].last_valid_index() for c in F.columns]})
