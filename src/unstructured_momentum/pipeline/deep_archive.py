"""Deep analogue archive: a price-only retrieval key over the full factor history.

WHY THE KEY IS PRICE-ONLY. The conferred-exposure spreads are this project's distinctive
measurement, and they cannot be computed before Dec-2013: daily holdings start there, and
monthly estimates are 3-8x noisier and are not the same measurement (a univariate 60-month
beta is a different object from a four-factor 12-month one). Forcing both into one distance
would be the same error as applying a daily-calibrated instrument to a monthly panel, which
this project has now made three times.

So the key stays homogeneous and shallow in features but deep in history, and the conferred
spreads DESCRIBE the match wherever they exist. Measured consequence:

    archive        days     bear-state   months < -20%   8th neighbour
    2014+          3,141        8.9%           3         70% of a random pair
    1927+         25,922       23.8%          15         33% of a random pair

The historical bear base rate is 23.5%; the old archive under-represented by ~3x the state
that produces the risk the tool exists to describe. Van den Dool (1994) and Lorenz (1969)
both say analogue quality is governed by library size against degrees of freedom -- ten times
the archive at five features instead of seven moves both terms the right way, and the
eighth-neighbour distance halves.

ELIGIBILITY is two conditions, not one: a neighbour must precede the query AND have its own
forward outcome window closed before it. The second is stricter than the usual as-of
convention and closes a leakage channel most of this literature does not name.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

H = 63              # outcome horizon, trading days
ZWIN, ZMIN = 756, 252
COLLAPSE = 63       # matrix-profile exclusion zone, m/4 of the 252d lookback
FLOOR_FRAC = 0.50   # a neighbour beyond this share of a random-pair distance is not an analogue
#: Hysteresis margin. An incumbent neighbour keeps its slot unless a challenger is at least
#: this fraction closer. Without it, day-to-day set overlap is 50% (p10 12%) and more than
#: half the set turns over on 41% of days -- so the STORY reshuffles overnight, which is the
#: axis on which the one published head-to-head has k-NN retrieval losing to an HMM.
#: At 0.10: overlap 88%, >half-turnover days 6%, median neighbour distance 0.50 -> 0.53.
#: Selected as the smallest margin reaching 80% median overlap, a structural criterion fixed
#: before the sweep. Skill could not have chosen it -- on 600 consecutive days the skill
#: column is ~10 independent observations and cannot separate the options.
PERSIST_MARGIN = 0.10

MOM = "data/raw/french/mom_10_daily/unknown_5542bc32b564.csv"
FF = "/private/tmp/claude-502/-Users-Tim/10cf8f3c-b126-4a42-a4bb-696ed65be1f8/scratchpad/F-F_Research_Data_Factors_daily.csv"
COLS = ["vol21", "vol63", "mom252", "mkt252", "dd"]


def _french_mom() -> pd.Series:
    d = pd.read_csv(MOM, skiprows=9, nrows=26174)
    d = d[pd.to_numeric(d.iloc[:, 0], errors="coerce").notna()]
    d.index = pd.to_datetime(d.iloc[:, 0].astype(int).astype(str), format="%Y%m%d")
    d = d.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    d = d[(d > -99).all(axis=1)]
    return (d["Hi PRIOR"] - d["Lo PRIOR"]) / 100.0


def build():
    W = _french_mom().rename("wml")
    ff = pd.read_csv(FF, skiprows=4)
    ff = ff[pd.to_numeric(ff.iloc[:, 0], errors="coerce").notna()]
    ff.index = pd.to_datetime(ff.iloc[:, 0].astype(int).astype(str), format="%Y%m%d")
    ff = ff.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    ff.columns = [c.strip() for c in ff.columns]
    F = pd.DataFrame({"wml": W}).join((ff["Mkt-RF"] / 100.0).rename("mkt"), how="inner")
    F["vol21"] = F.wml.rolling(21).std() * np.sqrt(252)
    F["vol63"] = F.wml.rolling(63).std() * np.sqrt(252)
    F["mom252"] = F.wml.rolling(252).sum()
    F["mkt252"] = F.mkt.rolling(252).sum()
    cum = (1 + F.wml).cumprod()
    F["dd"] = cum / cum.cummax() - 1
    F["bear"] = ((1 + F.mkt).rolling(504).apply(np.prod, raw=True) - 1) < 0
    F = F.dropna(subset=COLS)
    # outcome: worst peak-to-trough over the FOLLOWING H days
    r = F.wml.values
    fwd = np.full(len(r), np.nan)
    for i in range(len(r) - H):
        w = np.cumprod(1 + r[i + 1:i + 1 + H]) - 1
        fwd[i] = np.minimum.accumulate(w)[-1]
    F["fwd"] = fwd
    Z = pd.DataFrame(index=F.index, columns=COLS, dtype=float)
    for c in COLS:
        m = F[c].rolling(ZWIN, min_periods=ZMIN).mean()
        s = F[c].rolling(ZWIN, min_periods=ZMIN).std()
        Z[c] = ((F[c] - m) / s).clip(-5, 5)
    Z = Z.dropna()
    closes = pd.Series(F.index, index=F.index).shift(-H)
    return F, Z, closes


F, Z, CLOSES = build()
_ref = None


def random_pair_distance(rng=None, metric: str = "mahalanobis") -> float:
    """Reference scale: typical distance between two arbitrary states. A neighbour is only
    an analogue relative to this."""
    global _ref
    key = "maha" if metric == "mahalanobis" else "eucl"
    if _ref is None:
        _ref = {}
    if key not in _ref:
        rng = rng or np.random.default_rng(0)
        Wt = _whitener(Z.index[-1]) if metric == "mahalanobis" else None
        d = []
        for _ in range(4000):
            i, j = rng.integers(len(Z), size=2)
            if i == j:
                continue
            v = (Z.iloc[i] - Z.iloc[j]).to_numpy()
            if Wt is not None:
                v = v @ Wt
            d.append(float(np.sqrt((v ** 2).sum())))
        _ref[key] = float(np.median(d))
    return _ref[key]


def m_eff(dates, horizon: int = H) -> float:
    """Effective number of INDEPENDENT members once overlapping outcome windows are counted.
    A list of eight whose m_eff is two is one observation wearing eight hats, and a reader
    is entitled to know which they are looking at."""
    t = np.array([pd.Timestamp(d).toordinal() for d in dates], float)
    Wm = np.maximum(0.0, 1 - np.abs(t[:, None] - t[None, :]) / (horizon * 1.4))
    return float(len(t) ** 2 / Wm.sum())


# ---------------------------------------------------------------- metric
#: Ledoit-Wolf shrinkage intensity target. The covariance is estimated on the SAME trailing
#: window as the z-scores and cached by month, so it is causal.
_COV_CACHE: dict = {}


def _whitener(t: pd.Timestamp) -> np.ndarray | None:
    """Inverse Cholesky factor of the shrunk trailing correlation matrix.

    "Unweighted" Euclidean is not unweighted -- it is Mahalanobis with the correlation matrix
    forced to the identity. Our five features are not orthogonal (21d and 63d vol correlate
    heavily; factor and market momentum share a trend), so a plain distance silently
    multiple-counts whichever direction has the most features loading on it. An independent
    audit measured unweighted and Mahalanobis neighbour sets overlapping by only ~20%.

    This is UNSUPERVISED -- it touches the feature covariance and never an outcome -- so the
    four-episode objection that rules out metric learning does not apply. Using that argument
    to skip this was a logical error, not a conservative choice.
    """
    key = (t.year, t.month)
    if key in _COV_CACHE:
        return _COV_CACHE[key]
    hist = Z.loc[Z.index < t].tail(ZWIN)
    if len(hist) < ZMIN:
        _COV_CACHE[key] = None
        return None
    X = hist.to_numpy()
    S = np.cov(X, rowvar=False)
    # Ledoit-Wolf: shrink toward a scaled identity
    p_ = S.shape[0]
    mu = np.trace(S) / p_
    Xc = X - X.mean(0)
    phi = ((Xc ** 2).T @ (Xc ** 2) / len(X) - S ** 2).sum()
    gamma = ((S - mu * np.eye(p_)) ** 2).sum()
    delta = float(np.clip(phi / gamma / len(X), 0.0, 1.0)) if gamma > 0 else 1.0
    Ssh = (1 - delta) * S + delta * mu * np.eye(p_)
    try:
        Wt = np.linalg.inv(np.linalg.cholesky(Ssh)).T
    except np.linalg.LinAlgError:
        Wt = None
    _COV_CACHE[key] = Wt
    return Wt


def retrieve_sequence(days, k: int = 8, margin: float = PERSIST_MARGIN,
                      floor: float | None = FLOOR_FRAC) -> dict:
    """Walk a run of query dates carrying the neighbour set forward.

    Retrieval is stateful by design here. A memoryless distance ranking jumps as the query
    moves; a reader checking weekly needs the set to persist unless something genuinely
    better appears.
    """
    out, prev = {}, []
    for t in days:
        t = pd.Timestamp(t)
        elig = Z.index[(Z.index < t) & (CLOSES.reindex(Z.index) < t)]
        if len(elig) < 200:
            continue
        d = np.sqrt(((Z.loc[elig] - Z.loc[t]) ** 2).sum(axis=1)).nsmallest(600)
        fresh = []
        for dt, dist in d.items():
            if all(abs((dt - k2).days) >= COLLAPSE for k2, _ in fresh):
                fresh.append((dt, dist))
                if len(fresh) >= k:
                    break
        if not prev or margin <= 0:
            keep = fresh
        else:
            keep = sorted([(dt, float(np.sqrt(((Z.loc[dt] - Z.loc[t]) ** 2).sum())))
                           for dt, _ in prev if dt in Z.index and dt < t],
                          key=lambda x: x[1])
            for dt, dist in fresh:
                if any(abs((dt - k2).days) < COLLAPSE for k2, _ in keep):
                    continue
                if len(keep) < k:
                    keep.append((dt, dist)); continue
                worst = max(keep, key=lambda x: x[1])
                if dist < (1 - margin) * worst[1]:
                    keep.remove(worst); keep.append((dt, dist))
            keep = sorted(keep, key=lambda x: x[1])[:k]
        prev = keep
        ref = random_pair_distance(metric="euclidean")
        idx = pd.DatetimeIndex([a for a, _ in keep])
        df = pd.DataFrame({"dist": [b for _, b in keep], "rel": [b / ref for _, b in keep],
                           "fwd": F.fwd.reindex(idx)}, index=idx)
        # The floor ANNOTATES, it does not delete. A hard gate flickers: a query drifting
        # across the threshold flips neighbours in and out of view, and post-floor set
        # overlap fell to 75% while the retrieval underneath was stable at 88%. Showing all
        # k with their distance and a pass flag is both steadier and more informative --
        # "the eight nearest, of which three clear the bar" tells a reader more than three
        # dates with the rejects hidden. When NONE clear it, that is the headline.
        df["is_analogue"] = df.rel <= (floor if floor is not None else np.inf)
        out[t] = df
    return out


def retrieve(asof, k: int = 8, floor: float | None = FLOOR_FRAC,
             metric: str = "euclidean") -> pd.DataFrame:
    t = pd.Timestamp(asof)
    if t not in Z.index:
        prior = Z.index[Z.index <= t]
        if not len(prior):
            return pd.DataFrame()
        t = prior[-1]
    elig = Z.index[(Z.index < t) & (CLOSES.reindex(Z.index) < t)]
    if len(elig) < 100:
        return pd.DataFrame()
    diff = (Z.loc[elig] - Z.loc[t]).to_numpy()
    # Euclidean by default. Mahalanobis whitening finds CLOSER neighbours (13% vs 16% of a
    # random pair) and scores WORSE on the time-matched null (6.0% vs 7.8%). The audit's
    # logic was right -- unweighted Euclidean is Mahalanobis with the correlation forced to
    # identity, and our two vol features are heavily correlated -- but the implied
    # over-weighting of volatility turns out to be economically load-bearing for momentum
    # reversals rather than an accident. Kept available for comparison; not the default.
    Wt = _whitener(t) if metric == "mahalanobis" else None
    if Wt is not None:
        diff = diff @ Wt
    d = pd.Series(np.sqrt((diff ** 2).sum(axis=1)), index=elig).nsmallest(600)
    kept = []
    for dt, dist in d.items():
        if all(abs((dt - k2).days) >= COLLAPSE for k2, _ in kept):
            kept.append((dt, dist))
            if len(kept) >= k:
                break
    if not kept:
        return pd.DataFrame()
    idx = pd.DatetimeIndex([a for a, _ in kept])
    out = pd.DataFrame({"dist": [b for _, b in kept],
                        "rel": [b / random_pair_distance(metric=metric) for _, b in kept],
                        "fwd": F.fwd.reindex(idx)}, index=idx)
    # "no analogue exists" is a legitimate answer and often the honest one
    return out[out.rel <= floor] if floor is not None else out
