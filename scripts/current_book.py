"""What is the book long right now, and how concentrated is it?

The one question every reader asks that nothing else in this package answers. The holdout has
already been spent, so reading current data is what a live system would do rather than a breach.

Reports the latest formation date's conferred exposure on each of the eight reference series,
standardised against matched placebo books, and names the dominant axis with its sector and
holding composition.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "scripts")

from unstructured_momentum.factor import cleanlegs as cl, placebo as pb  # noqa: E402
from build_reference_panel import reference_series  # noqa: E402
from gap01_family import spreads  # noqa: E402

N_DRAWS = 200
LABEL = {"mkt": "broad equity beta", "d10y": "ten-year yield (duration)",
         "wti": "crude oil", "usd": "the dollar", "ig": "investment-grade credit",
         "concentration": "index concentration (top ten vs equal weight)",
         "pc1": "first cross-asset principal component",
         "pc2": "second cross-asset principal component"}


def main() -> None:
    px, sec, wt = {}, {}, {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price", "sector", "weight_pct"])
        d = d[d.price.notna() & (d.price > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
        wt[f] = d.pivot_table(index="as_of", columns="ticker", values="weight_pct", aggfunc="last")
        sec.update(d.drop_duplicates("ticker", keep="last").set_index("ticker")["sector"].to_dict())
    P = pd.concat(px.values()).sort_index(); P = P[~P.index.duplicated(keep="last")]
    W = pd.concat(wt.values()).sort_index(); W = W[~W.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)
    X = reference_series(P)

    asof = pd.Timestamp(P.index.max()) + pd.offsets.MonthEnd(0)
    asof = P.index[P.index <= asof].max()
    asof = pd.Timestamp(asof) - pd.offsets.MonthEnd(1) if asof.day < 20 else asof
    print(f"LATEST FORMATION DATE  {asof.date()}   (panel runs to {P.index.max().date()})\n")

    mom = cl.momentum(P, asof, min_obs=60)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    win, los = cl.legs(mom)
    m = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
    Rw = R.loc[m]
    real = spreads(Rw[[t for t in win if t in Rw.columns]].mean(axis=1),
                   Rw[[t for t in los if t in Rw.columns]].mean(axis=1), X)

    caps = W.loc[:asof].iloc[-1].dropna()
    rng = np.random.default_rng(20260903)
    null = []
    for s in range(N_DRAWS):
        b = pb.random_matched(P, asof, pb.Book(list(win), list(los), family="r", meta={}),
                              caps, rng, seed=s)
        if b is None:
            continue
        nw = [t for t in b.winners if t in Rw.columns]
        nl = [t for t in b.losers if t in Rw.columns]
        if len(nw) < 20 or len(nl) < 20:
            continue
        null.append(spreads(Rw[nw].mean(axis=1), Rw[nl].mean(axis=1), X))
    NU = np.abs(np.vstack(null)); mu, sd = NU.mean(axis=0), NU.std(axis=0, ddof=1)
    z = (np.abs(real) - mu) / np.where(sd > 0, sd, np.nan)

    order = np.argsort(-z)
    print("CONFERRED EXPOSURE, winner minus loser, standardised against "
          f"{len(null)} size-matched placebo books\n")
    print(f"  {'series':34s}{'spread':>10}{'z':>9}")
    for j in order:
        c = list(X.columns)[j]
        print(f"  {LABEL[c]:34s}{real[j]:+10.4f}{z[j]:9.1f}")
    top = list(X.columns)[order[0]]
    sign = "LONG" if real[order[0]] > 0 else "SHORT"
    print(f"\n  DOMINANT AXIS: the book is {sign} {LABEL[top]}  (z {z[order[0]]:.1f})")

    s_w = pd.Series({t: sec.get(t) for t in win}).dropna()
    s_l = pd.Series({t: sec.get(t) for t in los}).dropna()
    print(f"\n  winner leg {len(win)} names, {s_w.nunique()} sectors; "
          f"top: {', '.join(f'{k} {v/len(s_w):.0%}' for k, v in s_w.value_counts().head(3).items())}")
    print(f"  loser  leg {len(los)} names, {s_l.nunique()} sectors; "
          f"top: {', '.join(f'{k} {v/len(s_l):.0%}' for k, v in s_l.value_counts().head(3).items())}")
    w10 = W.loc[:asof].iloc[-1].reindex(win).dropna().nlargest(10)
    print(f"\n  ten largest winner-leg holdings by index weight:")
    print("    " + ", ".join(w10.index.tolist()))
    json.dump({"as_of": str(asof.date()), "dominant": top, "direction": sign,
               "z": {c: float(v) for c, v in zip(X.columns, z)},
               "spread": {c: float(v) for c, v in zip(X.columns, real)},
               "winner_top_sectors": s_w.value_counts().head(3).to_dict(),
               "loser_top_sectors": s_l.value_counts().head(3).to_dict(),
               "winner_top10": w10.index.tolist()},
              open("reports/current_book.json", "w"), indent=2)
    print("\nwrote reports/current_book.json")


if __name__ == "__main__":
    main()
