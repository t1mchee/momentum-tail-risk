"""Books no sort constructed, so that the sort's effect can be measured against something.

The project's central finding is that the momentum sort MANUFACTURES the exposure it later gets
blamed for. That finding has never had a null. If any selection rule applied to the same
universe confers a comparable tilt, the claim is not about momentum, it is about selection.

Two downstream results inherit the same gap, and one of them says so in its own text: the naming
claim records that it "cannot separate naming what these companies ARE from naming what the sort
SELECTED, because a null matched on momentum rank is unbuildable when the leg IS the top
momentum decile". A placebo book dissolves that objection by being a book no sort constructed.

Everything here exists to make ONE comparison honest, so the real book and every placebo book go
through the SAME loading code -- `spread_for` -- rather than through two paths that are believed
to agree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..factor import cleanlegs as cl

CACHE = Path("data/interim/placebo")

#: Cap buckets for the matched family. Deciles over the universe, so a matched draw reproduces
#: the real book's size profile bucket by bucket rather than only in the mean -- momentum
#: winners are not a size-neutral sample and a mean-matched draw would still differ in shape.
N_CAP_BUCKETS = 10


@dataclass
class Book:
    """A pair of legs and the rule that produced them. The rule travels with the book."""

    winners: list[str]
    losers: list[str]
    family: str
    seed: int | None = None
    asof: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        h = hashlib.sha256(
            f"{self.family}|{self.seed}|{self.asof}|{'|'.join(sorted(self.winners))}"
            f"|{'|'.join(sorted(self.losers))}".encode()).hexdigest()[:16]
        return h


# ------------------------------------------------------------------ universe and caps

def caps_at(panel_rows: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Market value per ticker at the latest snapshot on or before `asof`.

    price times quantity, which is the index position rather than the company's full market
    capitalisation -- the panel holds a fund's holdings, not a share register. It is the size
    proxy the real book is itself built from, so it is the right one for matching even though it
    is not free float.
    """
    d = panel_rows[panel_rows["as_of"] <= asof]
    if d.empty:
        return pd.Series(dtype=float)
    d = d[d["as_of"] == d["as_of"].max()]
    v = pd.to_numeric(d["price"], errors="coerce") * pd.to_numeric(d["quantity"], errors="coerce")
    return v.set_axis(d["ticker"]).dropna().groupby(level=0).sum()


def real_book(P: pd.DataFrame, asof: pd.Timestamp, *, decile: float = 0.10,
              sectors: pd.Series | None = None) -> Book | None:
    """The book the 12-1 sort actually builds, through the same gate the panel builder uses."""
    mom = cl.momentum(P, asof, sectors=sectors)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    if len(mom) < 300:
        return None
    win, los = cl.legs(mom, decile)
    return Book(list(win), list(los), family="momentum_12_1", asof=asof,
                meta={"n_universe": int(len(mom))})


# ------------------------------------------------------------------ placebo families

def _eligible(P: pd.DataFrame, asof: pd.Timestamp, sectors: pd.Series | None) -> pd.Index:
    """Exactly the names the real sort could have chosen from, and no others.

    Drawing from a wider or narrower set than the sort saw is the quietest way to make a placebo
    incomparable: the difference would then be availability rather than selection.
    """
    mom = cl.momentum(P, asof, sectors=sectors)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    return mom.index


def random_free(P: pd.DataFrame, asof: pd.Timestamp, n_win: int, n_los: int,
                rng: np.random.Generator, *, sectors: pd.Series | None = None,
                seed: int | None = None, family: str = "random_free") -> Book | None:
    """Uniform draw from the eligible universe. No selection rule at all.

    A labelled DIAGNOSTIC rather than the primary null, because momentum winners are not a
    size-neutral sample of the universe: an unmatched comparison would let a size effect wear a
    selection effect's name.
    """
    pool = _eligible(P, asof, sectors)
    if len(pool) < n_win + n_los:
        return None
    pick = rng.choice(np.asarray(pool), size=n_win + n_los, replace=False)
    return Book(list(pick[:n_win]), list(pick[n_win:]), family=family, seed=seed,
                asof=asof, meta={"n_universe": int(len(pool))})


def random_matched(P: pd.DataFrame, asof: pd.Timestamp, target: Book, caps: pd.Series,
                   rng: np.random.Generator, *, sectors: pd.Series | None = None,
                   seed: int | None = None) -> Book | None:
    """Random draw stratified to reproduce the target book's cap profile, bucket by bucket.

    THE PRIMARY NULL. Each leg is filled from the same cap deciles, in the same counts, as the
    corresponding real leg -- so the placebo differs from the real book in WHICH names it holds
    and not in how large they are. Names in the real book are excluded from its own placebo,
    since a draw that could return the book itself is not a null for it.
    """
    # The pool is the intersection of what the sort could pick and what `caps` covers. Passing a
    # restricted caps series is therefore how a caller restricts the draw -- used by the
    # peer-network cell, which must draw only from names that HAVE a filing so that its placebo
    # legs match the real leg's post-filter size rather than being measured on fewer names.
    pool = _eligible(P, asof, sectors)
    c = caps.reindex(pool).dropna()
    if len(c) < 50:
        return None
    bucket = pd.qcut(c.rank(method="first"), N_CAP_BUCKETS, labels=False)
    out = {}
    for leg, names in (("w", target.winners), ("l", target.losers)):
        want = bucket.reindex([n for n in names if n in bucket.index]).dropna()
        picks = []
        for b, k in want.value_counts().items():
            avail = [n for n in bucket.index[bucket == b] if n not in set(names)]
            if len(avail) < k:
                # Not enough room in this bucket once the real names are removed. Taking the
                # nearest bucket would silently break the match the family exists to have, so
                # the shortfall is RECORDED and the draw is smaller instead.
                k = len(avail)
            if k:
                picks += list(rng.choice(avail, size=int(k), replace=False))
        out[leg] = picks
    # A one-leg target is legitimate: the comparator ran on a winner leg alone, so its placebo
    # is a winner-leg-shaped draw with the loser side left EMPTY rather than invented.
    if not out["w"] and not out["l"]:
        return None
    if target.winners and not out["w"]:
        return None
    if target.losers and not out["l"]:
        return None
    return Book(out["w"], out["l"], family="random_matched", seed=seed, asof=asof,
                meta={"n_universe": int(len(pool)),
                      "short_w": len(target.winners) - len(out["w"]),
                      "short_l": len(target.losers) - len(out["l"])})


def short_horizon(P: pd.DataFrame, asof: pd.Timestamp, *, decile: float = 0.10,
                  months_back: int = 1, skip: int = 0,
                  sectors: pd.Series | None = None) -> Book | None:
    """Selection on past returns over a DIFFERENT window. Still a sort, not the sort.

    The discriminating family. If a one-month rule confers comparable tilt, the finding is about
    selection on past returns rather than about the 12-1 sort specifically, and the claim's scope
    widens accordingly.
    """
    mom = cl.momentum(P, asof, months_back=months_back, skip=skip, sectors=sectors)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    if len(mom) < 300:
        return None
    win, los = cl.legs(mom, decile)
    return Book(list(win), list(los), family=f"short_horizon_{months_back}m", asof=asof,
                meta={"n_universe": int(len(mom))})


def by_characteristic(P: pd.DataFrame, asof: pd.Timestamp, caps: pd.Series, *,
                      decile: float = 0.10, sectors: pd.Series | None = None) -> Book | None:
    """Selection on something that is NOT a past return. The other discriminating family.

    If ranking on size confers a comparable tilt, then any sort confers one and the finding is
    about sorting; if it does not while the short-horizon rule does, the finding is specifically
    about selection on past returns.
    """
    pool = _eligible(P, asof, sectors)
    c = caps.reindex(pool).dropna()
    if len(c) < 300:
        return None
    win, los = cl.legs(c, decile)
    return Book(list(win), list(los), family="by_market_cap", asof=asof,
                meta={"n_universe": int(len(c))})


def reversed_momentum(book: Book) -> Book:
    """The legs swapped. A HARNESS CHECK, not a placebo family.

    Swapping legs negates the spread exactly, so on the registered ABSOLUTE statistic this is
    identical to the real book by construction -- a cell that cannot move and would therefore
    report as a perfect result, which is the failure mode trp-54 records. On the SIGNED spread it
    is a real check: the plumbing must negate to machine precision or the legs are crossed
    somewhere.
    """
    return Book(list(book.losers), list(book.winners), family="reversed_harness_check",
                asof=book.asof, meta=dict(book.meta))


# ------------------------------------------------------------------ the shared measurement

MIN_OBS = 40


def _loadings(r_leg: pd.Series, X: pd.DataFrame) -> dict:
    d = pd.concat([r_leg.rename("y"), X], axis=1).dropna()
    if len(d) < MIN_OBS:
        return {}
    A = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)
    return dict(zip(["alpha", *X.columns], beta))


def spread_for(book: Book, R: pd.DataFrame, X: pd.DataFrame, asof: pd.Timestamp,
               *, realised: bool = False) -> dict:
    """The loading spread for ONE book. Real and placebo both come through here.

    One function rather than two that are believed to agree, because the entire experiment is a
    comparison between a real book and placebo books and any difference in the measurement path
    would be indistinguishable from the effect.

    `realised=False` is the FORMATION window, the twelve months ending at the sort date: the
    registered primary, and what conferred_tilt.csv holds. `realised=True` is the three months
    AFTER, which is where the claim's headline figures live -- winners 0.554, losers 2.016 --
    and is a different estimand carrying a different availability stamp.
    """
    if realised:
        w = (R.index > asof) & (R.index <= asof + pd.DateOffset(months=3))
        avail = asof + pd.DateOffset(months=3)
    else:
        w = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        avail = asof
    cols_w = [t for t in book.winners if t in R.columns]
    cols_l = [t for t in book.losers if t in R.columns]
    if not cols_w or not cols_l:
        return {}
    lw = _loadings(R.loc[w, cols_w].mean(axis=1), X)
    ll = _loadings(R.loc[w, cols_l].mean(axis=1), X)
    if not lw or not ll:
        return {}
    out = {"asof": asof.normalize(), "family": book.family, "seed": book.seed,
           "estimand": "realised_after_formation" if realised else "conferred_in_formation",
           "available_at": pd.Timestamp(avail).normalize(),
           "n_w": len(cols_w), "n_l": len(cols_l), "book_key": book.key()}
    for k in X.columns:
        out[f"{k}_w"], out[f"{k}_l"] = lw[k], ll[k]
        out[f"{k}_spread"] = lw[k] - ll[k]
    out["beta_spread"] = out["Mkt-RF_spread"]
    out["abs_beta_spread"] = abs(out["Mkt-RF_spread"])
    return out


# ------------------------------------------------------------------ the construction gate

def ks_gate(real: Book, placebos: list[Book], caps: pd.Series, *,
            alpha: float = 0.05) -> dict:
    """Do the matched books actually match the real book's size profile?

    Declared as a gate in the registration and run BEFORE anything is scored. An unmatched
    comparison would let a size effect wear a selection effect's name, and the only way to know
    the stratification worked is to test it rather than to trust the code that wrote it.

    Reports the fraction of placebo books indistinguishable from the real book at `alpha`. A
    family that fails is not scored; a DIAGNOSTIC family is expected to fail and reports its
    distance instead of being gated on it.
    """
    from scipy import stats

    def prof(b: Book) -> np.ndarray:
        v = caps.reindex(b.winners + b.losers).dropna().to_numpy()
        return np.log(v[v > 0])

    ref = prof(real)
    rows = []
    for b in placebos:
        x = prof(b)
        if len(x) < 20 or len(ref) < 20:
            continue
        d, p = stats.ks_2samp(ref, x)
        rows.append({"family": b.family, "seed": b.seed, "ks": float(d), "p": float(p),
                     "passes": bool(p > alpha)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0, "by_family": {}}
    return {"n": int(len(df)),
            "by_family": {f: {"n": int(len(g)), "pass_rate": float(g.passes.mean()),
                              "median_ks": float(g.ks.median())}
                          for f, g in df.groupby("family")},
            "table": df}


def median_beta_spread(book: Book, R: pd.DataFrame, asof: pd.Timestamp, *,
                       realised: bool = True, min_obs: int = 20) -> dict:
    """The claim's own estimator: median per-name univariate beta against the panel's median.

    A SECOND estimator, kept because the claim's headline figures were produced by it and a
    placebo test of a headline should test the quantity the headline names. It differs from
    `spread_for` in three ways -- median of per-name betas rather than the beta of the leg
    portfolio, the panel's own cross-sectional median rather than the Fama-French market factor,
    and univariate rather than four-factor -- and on the 2019 formation window the two agree
    closely, at -0.428 against -0.359, so the estimator is not what separates the numbers.
    The WINDOW is: the same estimator reads -1.412 over the three months AFTER formation.

    It also runs where the four-factor regression cannot. The holdings panel is not daily
    throughout -- 104 snapshots in 2018 against 251 in 2023 -- and the post-formation window at
    2019 holds 31 of them, below what a four-factor fit needs. A univariate per-name fit at 20
    observations is the estimator that survives there, and saying so is better than reporting a
    quantity the data cannot support.
    """
    if realised:
        w = (R.index > asof) & (R.index <= asof + pd.DateOffset(months=3))
        avail = asof + pd.DateOffset(months=3)
    else:
        w = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        avail = asof
    r = R.loc[w]
    if r.empty:
        return {}
    mkt = r.median(axis=1)          # median, not mean: robust to any split jump that survived

    def med(names: list[str]) -> tuple[float, int]:
        out = []
        for t in names:
            if t not in r.columns:
                continue
            y = r[t]
            ok = y.notna() & mkt.notna()
            if ok.sum() < min_obs:
                continue
            out.append(np.polyfit(mkt[ok], y[ok], 1)[0])
        return (float(np.median(out)) if out else float("nan")), len(out)

    bw, nw = med(book.winners)
    bl, nl = med(book.losers)
    if not (nw and nl) or not np.isfinite(bw) or not np.isfinite(bl):
        return {}
    return {"asof": asof.normalize(), "family": book.family, "seed": book.seed,
            "estimand": ("median_beta_after_formation" if realised
                         else "median_beta_in_formation"),
            "available_at": pd.Timestamp(avail).normalize(),
            "n_w": nw, "n_l": nl, "n_obs": int(w.sum()), "book_key": book.key(),
            "beta_w": bw, "beta_l": bl, "beta_spread": bw - bl,
            "abs_beta_spread": abs(bw - bl)}


# ------------------------------------------------------------------ non-return characteristics

def trailing_vol(P: pd.DataFrame, asof: pd.Timestamp, *, months: int = 12) -> pd.Series:
    """Realised volatility of daily returns over the trailing window, per name.

    A non-return characteristic in the sense that matters here -- it is a second moment, not a
    return level -- but NOT a clean test against a beta-spread axis, because low-volatility names
    carry low betas almost by definition. It is included for the covariance-floor axis, where no
    such tautology exists, and excluded from the beta-axis reading.
    """
    w = (P.index > asof - pd.DateOffset(months=months)) & (P.index <= asof)
    r = P.loc[w].pct_change().where(lambda x: x.abs() < cl.JUMP)
    v = r.std()
    return v[v > 0].dropna()


def days_to_cover(asof: pd.Timestamp, root: str = "data/raw/shortinterest") -> pd.Series:
    """Days-to-cover from the most recent short-interest tape AVAILABLE at the date.

    Keyed on `available_at`, not on the settlement date. The tape settles bi-monthly and
    publishes about two weeks later, so using the settlement stamp would hand the sort two weeks
    of information it could not have had.
    """
    files = sorted(Path(root).glob("*.parquet"))
    best = None
    for f in files:
        d = pd.read_parquet(f, columns=["symbolCode", "daysToCoverQuantity", "available_at"])
        av = pd.to_datetime(d["available_at"]).max()
        if av <= asof and (best is None or av > best[0]):
            best = (av, d)
    if best is None:
        return pd.Series(dtype=float)
    d = best[1]
    s = pd.to_numeric(d["daysToCoverQuantity"], errors="coerce")
    return s.set_axis(d["symbolCode"].astype(str)).dropna().groupby(level=0).max()


def by_series(values: pd.Series, pool: pd.Index, *, k: int, top: bool = True,
              label: str = "sorted") -> Book | None:
    """A one-legged book: the k highest (or lowest) names of a characteristic within a pool."""
    v = values.reindex(pool).dropna()
    if len(v) < k:
        return None
    names = list(v.nlargest(k).index if top else v.nsmallest(k).index)
    return Book(names, [], family=label, meta={"n_pool": int(len(v))})


def covariance_floor(C: np.ndarray) -> float:
    """Mean pairwise correlation across ALL pairs in a book.

    The quantity the crowding-out mechanism is actually about. A grouping statistic measures
    what a partition adds OVER this, so a book with a high floor leaves any grouping less room
    -- and unlike a conferred beta spread, no sorting rule produces this by definition.
    """
    iu = np.triu_indices(len(C), 1)
    v = C[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")
