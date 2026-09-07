"""Gate 1 -- the spine feature set, with honest first and last valid dates per column.

Four features. Two are computable over the full French archive; two need the IWV constituent
panel and are therefore bounded by it. The bound is NOT the text coverage limit and is not
2007: the IWV price panel carries ONE observation per month until 2012-12 and only becomes
daily in 2013-01, so anything requiring daily constituent returns -- comomentum above all --
starts in 2013 and cannot reach the 2007 quake or the 2009 rebound.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.config import SEALED_START  # noqa: E402
from unstructured_momentum.data import french, ishares  # noqa: E402
from unstructured_momentum.factor import legs  # noqa: E402

TD = 252


def slow_fast_turning_point(wml: pd.Series, fast: int = 21, slow: int = 126) -> pd.Series:
    """Distance between a fast and a slow trend in the factor, in trailing-vol units.

    A turning point is where the fast trend crosses the slow one. Scaling by trailing
    volatility makes the crossing comparable across a century in which the factor's own
    volatility moves by an order of magnitude.
    """
    f = wml.rolling(fast).mean()
    s = wml.rolling(slow).mean()
    v = wml.rolling(slow).std()
    return ((f - s) / v).rename("slow_fast")


def cross_sectional_dispersion(deciles: pd.DataFrame) -> pd.Series:
    """Daily dispersion across the ten momentum-sorted portfolios.

    Computed on French's decile portfolios rather than on single names, which is what makes it
    reach 1926. It measures how far apart the sorted portfolios are moving, which is the
    quantity a crowded sort compresses and a reversal blows out.
    """
    return deciles.std(axis=1).rename("xs_dispersion")


def factor_autocorr_breakdown(ind: pd.DataFrame, window: int = 252,
                              step: int = 21, k: int = 3) -> pd.Series:
    """Autocorrelation of the high-eigenvalue factors of the industry cross-section.

    Rolling correlation matrix of the industry portfolios, projected onto its leading k
    eigenvectors; the statistic is the mean lag-1 autocorrelation of those factor series. A
    breakdown -- autocorrelation collapsing toward zero or turning negative -- is the signature
    of a market that has stopped trending and started reversing.
    """
    X = ind.dropna(how="any")
    out = {}
    idx = X.index
    for i in range(window, len(X), step):
        w = X.iloc[i - window:i]
        C = np.corrcoef(w.T.to_numpy())
        C = np.nan_to_num(C, nan=0.0)
        vals, vecs = np.linalg.eigh(C)
        top = vecs[:, -k:]
        f = w.to_numpy() @ top
        acs = [float(pd.Series(f[:, j]).autocorr(lag=1)) for j in range(f.shape[1])]
        out[idx[i]] = float(np.mean(acs))
    return pd.Series(out).rename("factor_autocorr")


def comomentum(px: pd.DataFrame, legs_panel: dict, window: int = 252) -> pd.Series:
    """Lou-Polk comomentum: mean pairwise correlation of loser-leg residual returns.

    Needs DAILY constituent returns, which is why it cannot reach 2007.
    """
    rets = px.sort_index().pct_change()
    rets = rets.mask(rets.abs() > 0.5)
    mkt = rets.median(axis=1)
    out = {}
    for as_of in sorted(legs_panel):
        if as_of.date() >= SEALED_START:
            continue
        w = rets.loc[:as_of].iloc[-window:]
        if len(w) < window // 2:
            continue
        names = [n for n in legs_panel[as_of]["losers"] if n in w.columns]
        sub = w[names].dropna(axis=1, thresh=int(0.8 * len(w)))
        if sub.shape[1] < 20:
            continue
        m = mkt.reindex(sub.index)
        resid = sub.apply(lambda c: c - np.polyval(np.polyfit(m[c.notna() & m.notna()],
                                                              c[c.notna() & m.notna()], 1),
                                                   m), axis=0)
        C = resid.corr().to_numpy()
        iu = np.triu_indices_from(C, k=1)
        out[as_of] = float(np.nanmean(C[iu]))
    return pd.Series(out).rename("comomentum")


def main() -> None:
    wml = french.momentum()
    wml = wml[wml.index.date < SEALED_START]
    dec = french.load("mom_10_daily")
    dec = dec[[c for c in dec.columns if c.lower() not in ("hi 10", "lo 10")]] if False else dec
    ind = french.load("industry_49_daily")
    ind = ind.replace([-99.99, -999], np.nan)
    dec = dec.replace([-99.99, -999], np.nan)
    dec = dec[dec.index.date < SEALED_START]
    ind = ind[ind.index.date < SEALED_START]

    feats = {
        "slow_fast": slow_fast_turning_point(wml),
        "xs_dispersion": cross_sectional_dispersion(dec),
        "factor_autocorr": factor_autocorr_breakdown(ind),
    }

    import pickle
    lp = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    px = ishares.price_panel("IWV")
    px = px[legs.valid_tickers(px.columns)]
    feats["comomentum"] = comomentum(px, lp)

    rows = []
    for name, s in feats.items():
        s = s.dropna()
        rows.append({"feature": name, "first_valid": str(s.index.min().date()),
                     "last_valid": str(s.index.max().date()), "n_obs": int(len(s)),
                     "grain": "daily" if len(s) > 5000 else "monthly-ish"})
    T = pd.DataFrame(rows)
    print("GATE 1 FEATURE TABLE")
    print(T.to_string(index=False))
    print("\nEpisode reach (15 registered episodes):")
    from unstructured_momentum.events.registry import EPISODES
    for name, s in feats.items():
        s = s.dropna()
        seen = sum(1 for e in EPISODES
                   if s.index.min() <= pd.Timestamp(e.trough) <= s.index.max())
        print(f"  {name:18s} {seen:2d} of {len(EPISODES)}")
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    pd.concat(feats.values(), axis=1).to_parquet("data/processed/gate1_features.parquet")
    T.to_csv("reports/gate1_feature_table.csv", index=False)
    print("\nwrote data/processed/gate1_features.parquet and reports/gate1_feature_table.csv")


if __name__ == "__main__":
    main()
