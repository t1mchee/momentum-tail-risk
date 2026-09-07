"""The EIGHT reference series of memo section 3.1, and the conferred spread on each.

The memo names eight reference series and GAP-01's pass condition is "percentile >= 95 on at
least one". The implemented placebo tested ONE series, market beta, so the memo described a
test the code did not run. This builds the eight, computed the way the memo specifies:

  * each series estimated ONE AT A TIME, univariate, not jointly -- estimating them together
    would introduce collinearity between correlated references and the memo declines that
  * a fixed window, set in advance
  * the spread is winner-leg loading minus loser-leg loading

Written as a separate script so the existing single-series artifact, which the 105-of-137
result rests on, is left untouched and remains reproducible.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import french  # noqa: E402
from unstructured_momentum.factor import cleanlegs as cl  # noqa: E402

OUT = Path("data/processed/reference_panel.csv")
WINDOW_MONTHS = 12


def fred(name: str) -> pd.Series:
    d = pd.read_csv(f"data/raw/fred/{name}.csv")
    d.columns = ["date", name]
    s = pd.to_numeric(d[name], errors="coerce")
    return s.set_axis(pd.to_datetime(d["date"])).dropna()


def reference_series(px: pd.DataFrame) -> pd.DataFrame:
    """The eight, as daily CHANGES so every column is a return-like quantity."""
    mkt = french.market()["Mkt-RF"]                      # 1. broad US equity return
    d10y = fred("DGS10").diff()                          # 2. change in the 10-year yield
    wti = fred("DCOILWTICO").pct_change()                # 3. front-month WTI
    usd = fred("DTWEXBGS").pct_change()                  # 4. a dollar index
    ig = fred("BAA10Y").diff()                           # 5. an investment-grade credit spread

    # 6. the ten largest index names minus the equal-weighted index: a tradeable spread that
    #    lets a bet on index concentration be expressed, and the one series on this list that
    #    is not available from a vendor and has to be built from the holdings.
    top_minus_eq = _concentration_spread(px)

    base = pd.concat([mkt.rename("mkt"), d10y.rename("d10y"), wti.rename("wti"),
                      usd.rename("usd"), ig.rename("ig"),
                      top_minus_eq.rename("concentration")], axis=1)

    # 7-8. the first two principal components of the block above, estimated ONCE on a window
    #      ending before the panel starts, so the rotation is fixed rather than refitted.
    fit = base.loc[:"2013-12-31"].dropna()
    Z = (fit - fit.mean()) / fit.std()
    _, _, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    proj = ((base - fit.mean()) / fit.std()).dropna()
    pcs = pd.DataFrame(proj.to_numpy() @ Vt[:2].T, index=proj.index, columns=["pc1", "pc2"])
    return pd.concat([base, pcs], axis=1).dropna()


def _concentration_spread(px: pd.DataFrame) -> pd.Series:
    """Top-ten names by index weight, equally weighted, minus the equal-weighted index."""
    w = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "weight_pct"])
        w[f] = d.pivot_table(index="as_of", columns="ticker", values="weight_pct", aggfunc="last")
    W = pd.concat(w.values()).sort_index()
    W = W[~W.index.duplicated(keep="last")]
    R = px.pct_change().mask(lambda r: r.abs() > 0.5)
    common = R.index.intersection(W.index)
    out = {}
    for d in common:
        row = W.loc[d].dropna()
        if len(row) < 100:
            continue
        top = row.nlargest(10).index
        r = R.loc[d]
        t, e = r.reindex(top).mean(), r.mean()
        if t == t and e == e:
            out[d] = t - e
    return pd.Series(out).sort_index()


def main() -> None:
    px = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price"])
        d = d[d.price.notna() & (d.price > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
    P = pd.concat(px.values()).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)

    X = reference_series(P)
    print(f"reference series: {list(X.columns)}")
    print(f"  span {X.index.min().date()} -> {X.index.max().date()}, {len(X):,} days")

    rows = []
    for asof in pd.date_range("2014-01-31", "2025-12-31", freq="ME"):
        mom = cl.momentum(P, asof, min_obs=60)
        mom = mom[(mom > -0.95) & (mom < 5.0)]
        if len(mom) < 300:
            continue
        win, los = cl.legs(mom)
        m = (R.index > asof - pd.DateOffset(months=WINDOW_MONTHS)) & (R.index <= asof)
        rw = R.loc[m, [t for t in win if t in R.columns]].mean(axis=1)
        rl = R.loc[m, [t for t in los if t in R.columns]].mean(axis=1)
        row = {"asof": str(asof.date()), "n_leg": len(los)}
        for c in X.columns:
            # UNIVARIATE, one series at a time, as the memo specifies.
            d = pd.concat([rw.rename("w"), rl.rename("l"), X[c].rename("x")], axis=1).dropna()
            if len(d) < 40:
                row[f"{c}_spread"] = np.nan
                continue
            A = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
            bw = np.linalg.lstsq(A, d["w"].to_numpy(), rcond=None)[0][1]
            bl = np.linalg.lstsq(A, d["l"].to_numpy(), rcond=None)[0][1]
            row[f"{c}_spread"] = float(bw - bl)
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(D)} month-ends x {len([c for c in D.columns if c.endswith('_spread')])} series")
    print(D.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
