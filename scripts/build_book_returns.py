"""The reconstructed book's daily return series -- the thing several experiments were blocked on.

The book is built at month-ends and the design says it is "rebuilt monthly and measured daily",
so a daily return series is not missing data: it is the month-end membership held fixed between
rebalances and marked every day. That is what this builds, from `leg_members.pkl` and the
holdings panel, and it is what the tail model needs to run on the object the page is about
rather than on the published factor.

Three things are handled explicitly because getting any of them wrong is silent.

**Splits.** An uncorrected reverse split reads as a return of several hundred percent, and this
project has logged that fault once already. Prices go through the same split correction the
factor construction uses, and daily moves beyond the jump threshold are masked rather than
believed.

**Names that stop being priced.** A constituent can be acquired or delisted mid-month. Its
weight is dropped and the surviving weights are renormalised **within the leg on that day**, and
the share of leg weight that went missing is recorded per day. That is a choice, not a neutral
default: a delisting is often a large realised move that the panel cannot see, and on the short
leg the sign matters. The dropped-weight series is written out so a reader can see how much of
any day's return rests on it.

**Weights are point-in-time and are NOT the published weight column.** iShares reports
`weight_pct` to two decimal places, so every holding under 0.005 percent of the fund rounds to
exactly zero -- 1,184 of 2,683 names on a typical date. Dropping zero-weight names, which is the
obvious thing to do, silently removes the smallest sixty percent of the loser leg: precisely the
distressed micro-caps the short half is about, and precisely the bias this project already
documents running the other way. The weight here is `quantity * price`, the actual market value
held, which is unrounded and positive for every one of those names. It is then renormalised
within the leg so each side sums to one and the book is dollar-neutral. Logged as trp-91.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.factor import cleanlegs as cl  # noqa: E402

OUT = Path("data/processed/book_returns.parquet")
META = Path("reports/poc/book_returns_meta.json")
FIRST_YEAR = 2013


def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    px, wt = {}, {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < FIRST_YEAR:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price", "quantity"])
        d = d[d["price"].notna() & (d["price"] > 0) & d["quantity"].notna()]
        d["mv"] = d["quantity"] * d["price"]        # unrounded; weight_pct floors at 0.00
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
        wt[f] = d.pivot_table(index="as_of", columns="ticker", values="mv", aggfunc="last")
    P = pd.concat(px.values()).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    W = pd.concat(wt.values()).sort_index()
    W = W[~W.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)
    return R, W


def main() -> None:
    legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    R, W = panel()
    months = sorted(legs)
    print(f"panel {R.index.min().date()} to {R.index.max().date()}, {R.shape[1]:,} tickers")
    print(f"membership at {len(months)} month-ends\n")

    rows = []
    for i, t in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else R.index.max()
        win = [x for x in legs[t]["winners"] if x in R.columns]
        los = [x for x in legs[t]["losers"] if x in R.columns]
        # The membership month-end need not be a date the holdings panel carries: a file dated
        # the 30th for a month ending the 31st is common, and requiring an exact match dropped
        # 43 of 144 formations, three or four in every year, without any of them being a data
        # gap. Take the latest panel date at or before the formation instead, and refuse it if
        # it is more than a week stale.
        prior = W.index[W.index <= t]
        if not len(prior) or len(win) < 20 or len(los) < 20:
            continue
        w_at = prior[-1]
        if (t - w_at).days > 7:
            continue
        w0 = W.loc[w_at]

        def weights(names: list[str]) -> pd.Series:
            v = w0.reindex(names).astype(float)
            v = v[v > 0]
            return v / v.sum() if v.sum() > 0 else v

        ww, wl = weights(win), weights(los)
        if not len(ww) or not len(wl):
            continue
        days = R.index[(R.index > t) & (R.index <= end)]
        for day in days:
            rw, rl = R.loc[day, ww.index], R.loc[day, wl.index]
            # Renormalise within the leg over names priced that day; record what was dropped.
            mw, ml = rw.notna(), rl.notna()
            kept_w = ww[mw].sum(); kept_l = wl[ml].sum()
            if kept_w < 0.5 or kept_l < 0.5:
                continue
            ret_w = float((ww[mw] * rw[mw]).sum() / kept_w)
            ret_l = float((wl[ml] * rl[ml]).sum() / kept_l)
            rows.append({"date": day, "formation": t,
                         "r_winner": ret_w, "r_loser": ret_l, "r": ret_w - ret_l,
                         "weight_missing_winner": float(1 - kept_w),
                         "weight_missing_loser": float(1 - kept_l),
                         "n_winner": int(mw.sum()), "n_loser": int(ml.sum())})
        if i % 24 == 0:
            print(f"  {t.date()}  {len(ww)}/{len(wl)} names, {len(days)} days", flush=True)

    b = pd.DataFrame(rows).set_index("date").sort_index()
    b = b[~b.index.duplicated(keep="first")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b.to_parquet(OUT)

    ann = b["r"].std() * np.sqrt(252)
    meta = {
        "first": str(b.index.min().date()), "last": str(b.index.max().date()),
        "n_days": int(len(b)), "n_formations": int(b["formation"].nunique()),
        "ann_vol": float(ann),
        "median_weight_missing_winner": float(b["weight_missing_winner"].median()),
        "median_weight_missing_loser": float(b["weight_missing_loser"].median()),
        "p95_weight_missing_loser": float(b["weight_missing_loser"].quantile(0.95)),
        "note": ("weights are index weights at formation, renormalised within the leg over names "
                 "priced that day; the dropped share is recorded because a delisting is often a "
                 "large realised move the panel cannot see, and on the short leg the sign matters"),
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2))
    print(f"\n{len(b):,} daily book returns, {meta['first']} to {meta['last']}")
    print(f"  annualised vol {ann:.2%}")
    print(f"  median leg weight unpriced: winner {meta['median_weight_missing_winner']:.3%}, "
          f"loser {meta['median_weight_missing_loser']:.3%} (p95 {meta['p95_weight_missing_loser']:.2%})")

    # Agreement with the published factor, reported and never assumed.
    from unstructured_momentum.data import french
    w = french.momentum()
    j = pd.concat([b["r"].rename("book"), w.rename("french")], axis=1).dropna()
    m = j.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    print(f"\n  vs French WML: daily corr {j['book'].corr(j['french']):.3f}, "
          f"monthly corr {m['book'].corr(m['french']):.3f}, on {len(j):,} days")
    meta["corr_daily_vs_french"] = float(j["book"].corr(j["french"]))
    meta["corr_monthly_vs_french"] = float(m["book"].corr(m["french"]))
    META.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT} and {META}")


if __name__ == "__main__":
    main()
