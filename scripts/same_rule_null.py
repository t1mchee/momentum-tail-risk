"""exp-079 -- the 12-1 sort against sorts on other trailing windows.

The random-book null is saturated, so it cannot say whether the 12-1 book is concentrated or
whether return sorts in general are. This builds the same-rule null the design already names:
a 6-1 sort and a 24-13 sort on the same dates, same universe, same eight reference series.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "scripts")

from unstructured_momentum.config import SEALED_START  # noqa: E402
from unstructured_momentum.factor import cleanlegs as cl, placebo as pb  # noqa: E402
from build_reference_panel import reference_series  # noqa: E402
from gap01_family import spreads  # noqa: E402

SORTS = {"12-1": (12, 1), "6-1": (6, 1), "24-13": (24, 13)}
N_DRAWS = 120


def main() -> None:
    px = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price"])
        d = d[d.price.notna() & (d.price > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
    P = pd.concat(px.values()).sort_index(); P = P[~P.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)
    X = reference_series(P)
    caps = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "weight_pct"])
        caps[f] = d.pivot_table(index="as_of", columns="ticker",
                                values="weight_pct", aggfunc="last")
    CAP = pd.concat(caps.values()).sort_index(); CAP = CAP[~CAP.index.duplicated(keep="last")]

    rng = np.random.default_rng(20260903)
    rows = []
    dates = [d for d in pd.date_range("2014-01-31", "2022-12-31", freq="ME")]
    for i, asof in enumerate(dates, 1):
        m = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        Rw = R.loc[m]
        capd = CAP.loc[:asof]
        if not len(capd):
            continue
        cps = capd.iloc[-1].dropna()

        # One shared placebo pool per date, so the three sorts are standardised identically.
        base = cl.momentum(P, asof, months_back=12, skip=1, min_obs=60)
        base = base[(base > -0.95) & (base < 5.0)]
        if len(base) < 300:
            continue
        bw, bl_ = cl.legs(base)
        null = []
        for s in range(N_DRAWS):
            b = pb.random_matched(P, asof, pb.Book(list(bw), list(bl_), family="r", meta={}),
                                 cps, rng, seed=s)
            if b is None:
                continue
            nw = [t for t in b.winners if t in Rw.columns]
            nl = [t for t in b.losers if t in Rw.columns]
            if len(nw) < 20 or len(nl) < 20:
                continue
            null.append(spreads(Rw[nw].mean(axis=1), Rw[nl].mean(axis=1), X))
        if len(null) < 60:
            continue
        NU = np.abs(np.vstack(null))
        mu, sd = NU.mean(axis=0), NU.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)

        row = {"asof": str(asof.date())}
        for name, (mb, sk) in SORTS.items():
            mom = cl.momentum(P, asof, months_back=mb, skip=sk, min_obs=60)
            mom = mom[(mom > -0.95) & (mom < 5.0)]
            if len(mom) < 300:
                row[f"z_{name}"] = np.nan; row[f"arg_{name}"] = None
                continue
            w, l = cl.legs(mom)
            z = (np.abs(spreads(Rw[[t for t in w if t in Rw.columns]].mean(axis=1),
                                Rw[[t for t in l if t in Rw.columns]].mean(axis=1), X)) - mu) / sd
            row[f"z_{name}"] = float(np.nanmax(z))
            row[f"arg_{name}"] = list(X.columns)[int(np.nanargmax(z))]
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(dates)}", flush=True)

    D = pd.DataFrame(rows).dropna(subset=[f"z_{k}" for k in SORTS])
    n = len(D)
    print(f"\nSAME-RULE NULL, {n} month-ends, {N_DRAWS} shared placebos per date\n")
    print("  max standardised concentration, median across dates:")
    for k in SORTS:
        print(f"    {k:6s} {D[f'z_{k}'].median():7.2f}")
    wins = (D["z_12-1"] > D[["z_6-1", "z_24-13"]].max(axis=1)).mean()
    print(f"\n  12-1 is the MOST concentrated of the three on {wins:.1%} of dates "
          f"(a three-way tie is 33.3%)")
    p = stats.binomtest(int((D['z_12-1'] > D[['z_6-1','z_24-13']].max(axis=1)).sum()),
                        n, 1/3, alternative="greater").pvalue
    print(f"  sign test against the tie rate: p = {p:.4f}")
    print("\n  which series carries each sort's maximum (top 3):")
    for k in SORTS:
        vc = D[f"arg_{k}"].value_counts(normalize=True).head(3)
        print(f"    {k:6s} " + "  ".join(f"{a} {b:.0%}" for a, b in vc.items()))
    agree = (D["arg_12-1"] == D["arg_6-1"]).mean()
    print(f"\n  12-1 and 6-1 pick the SAME series on {agree:.1%} of dates")
    D.to_csv("reports/same_rule_null.csv", index=False)
    json.dump({"n": n, "win_rate_12_1": float(wins), "sign_test_p": float(p),
               "median_z": {k: float(D[f"z_{k}"].median()) for k in SORTS}},
              open("reports/same_rule_null.json", "w"), indent=2)
    print("\nwrote reports/same_rule_null.{csv,json}")


if __name__ == "__main__":
    main()
