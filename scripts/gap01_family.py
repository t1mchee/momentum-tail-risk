"""GAP-01 as the memo specifies it: eight reference series, threshold on the FAMILY MAXIMUM.

The registered pass condition is "percentile >= 95 on at least one reference series". Against
eight series that rule fires 1 - 0.95^8 = 33.7 percent of the time under the null, not 5. The
fix is not to deflate the existing single-series number -- that number came from a one-series
test and is sound on its own terms -- but to run the eight-series test with its threshold set
on the distribution of the MAXIMUM percentile across the family, which makes the family-wise
rate 5 percent by construction. That construction is the one section 7.2 already uses for
paging, applied here to the concentration reading.

For each month-end: draw N size-matched placebo books, compute the conferred spread on each of
the eight series for the real book and every placebo, take each book's MAXIMUM percentile across
the family, and ask where the real book's maximum sits in the null distribution of maxima.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import french  # noqa: E402
from unstructured_momentum.factor import cleanlegs as cl, placebo as pb  # noqa: E402

sys.path.insert(0, "scripts")
from build_reference_panel import reference_series  # noqa: E402

N_DRAWS = 200
OUT = Path("reports/gap01_family.json")


def spreads(rw: pd.Series, rl: pd.Series, X: pd.DataFrame) -> np.ndarray:
    """Univariate loading spread on each reference series, one at a time."""
    out = []
    for c in X.columns:
        d = pd.concat([rw.rename("w"), rl.rename("l"), X[c].rename("x")],
                      axis=1, sort=False).dropna()
        if len(d) < 40:
            out.append(np.nan); continue
        A = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
        bw = np.linalg.lstsq(A, d["w"].to_numpy(), rcond=None)[0][1]
        bl = np.linalg.lstsq(A, d["l"].to_numpy(), rcond=None)[0][1]
        out.append(float(bw - bl))
    return np.asarray(out)


def main() -> None:
    px = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price", "weight_pct"])
        d = d[d.price.notna() & (d.price > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
    P = pd.concat(px.values()).sort_index(); P = P[~P.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)
    X = reference_series(P)
    caps_all = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "weight_pct"])
        caps_all[f] = d.pivot_table(index="as_of", columns="ticker",
                                    values="weight_pct", aggfunc="last")
    CAP = pd.concat(caps_all.values()).sort_index()
    CAP = CAP[~CAP.index.duplicated(keep="last")]

    rng = np.random.default_rng(20260902)
    rows = []
    dates = [d for d in pd.date_range("2014-01-31", "2022-12-31", freq="ME")]
    for i, asof in enumerate(dates, 1):
        mom = cl.momentum(P, asof, min_obs=60)
        mom = mom[(mom > -0.95) & (mom < 5.0)]
        if len(mom) < 300:
            continue
        win, los = cl.legs(mom)
        m = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        Rw = R.loc[m]
        real = spreads(Rw[[t for t in win if t in Rw.columns]].mean(axis=1),
                       Rw[[t for t in los if t in Rw.columns]].mean(axis=1), X)

        capd = CAP.loc[:asof]
        if not len(capd):
            continue
        caps = capd.iloc[-1].dropna()
        book = pb.Book(list(win), list(los), family="real", meta={})
        null = []
        for s in range(N_DRAWS):
            b = pb.random_matched(P, asof, book, caps, rng, seed=s)
            if b is None:
                continue
            nw = [t for t in b.winners if t in Rw.columns]
            nl = [t for t in b.losers if t in Rw.columns]
            if len(nw) < 20 or len(nl) < 20:
                continue
            null.append(spreads(Rw[nw].mean(axis=1), Rw[nl].mean(axis=1), X))
        if len(null) < 100:
            continue
        NU = np.abs(np.vstack(null))          # (n_placebo, n_series)
        RE = np.abs(real)                      # (n_series,)
        n = NU.shape[0]

        # A PERCENTILE saturates. With 200 draws its resolution is half a point, and the real
        # book exceeds every placebo on at least one of eight series on essentially every date,
        # so max-percentile pins at 100 while the null's max caps at 99.5 -- the first version
        # of this test cleared 108 of 108, which is a ceiling effect and not a result. The
        # statistic is therefore the standardised distance, which has continuous support.
        mu, sd = NU.mean(axis=0), NU.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)
        z_real = (RE - mu) / sd

        # Leave-one-out moments for the placebos, so the null is standardised the way the real
        # book is rather than by moments that contain it -- including a placebo in its own
        # mean shrinks its z and would make the threshold too easy to clear.
        s1, s2 = NU.sum(axis=0), (NU ** 2).sum(axis=0)
        mu_loo = (s1 - NU) / (n - 1)
        var_loo = ((s2 - NU ** 2) - (n - 1) * mu_loo ** 2) / (n - 2)
        sd_loo = np.sqrt(np.clip(var_loo, 1e-24, None))
        z_null = (NU - mu_loo) / sd_loo

        max_real = float(np.nanmax(z_real))
        max_null = np.nanmax(z_null, axis=1)
        thresh = float(np.nanpercentile(max_null, 95))
        pct_real = np.array([(NU[:, j] < RE[j]).mean() * 100 for j in range(NU.shape[1])])
        rows.append({"asof": str(asof.date()),
                     "max_z_real": max_real,
                     "family_threshold_95": thresh,
                     "clears_family": bool(max_real >= thresh),
                     "clears_naive_any95": bool((pct_real >= 95).any()),
                     "argmax_series": list(X.columns)[int(np.nanargmax(z_real))],
                     "per_series_z": {c: float(v) for c, v in zip(X.columns, z_real)}})
        if i % 12 == 0:
            print(f"  {i}/{len(dates)}", flush=True)

    D = pd.DataFrame(rows)
    n = len(D)
    print(f"\nGAP-01, EIGHT SERIES, {n} month-ends, {N_DRAWS} matched placebos each\n")
    print(f"  naive rule, percentile >= 95 on ANY series : {D.clears_naive_any95.sum()} of {n} "
          f"= {D.clears_naive_any95.mean():.1%}")
    print(f"     its null rate is 1 - 0.95^8 = 33.7%, so this is the rule that overstates")
    print(f"  FAMILY-MAXIMUM rule, 5% by construction    : {D.clears_family.sum()} of {n} "
          f"= {D.clears_family.mean():.1%}")
    print(f"\n  max z of the real book: median {D.max_z_real.median():.2f}  "
          f"threshold median {D.family_threshold_95.median():.2f}")
    print(f"\n  which series carries the maximum, share of dates:")
    for k, v in D.argmax_series.value_counts().items():
        print(f"    {k:14s} {v:3d}  ({v/n:.1%})")
    D.to_csv("reports/gap01_family.csv", index=False)
    json.dump({"n_dates": n, "n_draws": N_DRAWS,
               "clears_family": int(D.clears_family.sum()),
               "clears_naive": int(D.clears_naive_any95.sum()),
               "family_rate": float(D.clears_family.mean()),
               "naive_rate": float(D.clears_naive_any95.mean())}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT} and reports/gap01_family.csv")


if __name__ == "__main__":
    main()
