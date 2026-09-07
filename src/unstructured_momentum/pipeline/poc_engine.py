"""Analogue engine with a mechanism layer: deep retrieval, conferred-exposure annotation.

WHY THE MECHANISM FEATURES ANNOTATE RATHER THAN RETRIEVE. Putting the conferred-exposure block
INTO the retrieval key was tried and rejected on measurement. Distance to the 8th neighbour, as
a share of a random pair:

    5 price features, 1928-2026   25,671 days    41%
    5 price features, 2014+        3,141 days    70%
    14 features (mech + disp)      1,857 days    96%

At 96% the 8th "analogue" is a randomly chosen date carrying a similarity label. That is fatal
for a tool whose output a PM reads as "this resembles Nov-2022", and it is the one place where
the distributional skill metric and the actual user need come apart: skill is scored against a
time-matched null that is ALSO arbitrary dates, so a key can score well while returning matches
nobody should look at. Retrieval quality is the binding constraint here, not skill.

So retrieval stays deep, tight and price-only, and the conferred-exposure decomposition
(exp-045: the momentum sort MANUFACTURES factor loadings rather than discovering them) is
attached to each match. The PM gets genuinely similar dates AND what the book was betting on
each of them, instead of trading the first away for the second.

COVERAGE IS HONEST, NOT IMPUTED. Daily holdings begin Dec-2013, so matches before then carry no
bet decomposition. That is shown as absent rather than back-filled -- and because deep retrieval
mostly returns pre-2014 dates, typically only one match in eight carries a bet at all.

There is ALSO a hole inside the covered span: 2017 has 7% coverage and 2018 has 26%, against
100% for every other year. Absence there is a gap in the holdings panel, NOT a date that
predates the data, and the readout distinguishes the two -- the earlier version called a
2017 match "predates holdings data", which was simply false.

Hence TWO PANELS, answering two different questions rather than compromising between them:

  DEEP     5 price features over 1928-2026. "When did a state like this last occur?"
           8th neighbour ~41% of a random pair. Mostly no bet decomposition.
  MODERN   the same 5 features restricted to the holdings era. "Which recent book most
           resembles this one, and what was it betting?" 8th neighbour ~70% -- looser, and
           labelled as such -- but every match carries its conferred-exposure decomposition.

The modern panel is deliberately NOT the 14-feature key: adding the mechanism block to the
distance is what pushed neighbours to 96%. The mechanism data annotates in both panels; it
selects in neither.

KEY SELECTION IS ON ANALOGUE QUALITY, NOT SKILL. On the holdings span there are 7-19
independent 63-day blocks, at which the twCRPS skill column cannot rank configurations -- a
14-feature key measured 2.0% skill and "beat" the benchmark while an 18-feature key measured
21.8% and did not. Neighbour distance is a direct property of the key and needs no power.

DATA BOUNDARY. The mechanism panel ends when the holdings panel does, which is currently
2026-02-09 -- earlier than the price series. A readout is therefore as-of the mechanism cutoff,
and asking for a later date silently gives a stale bet decomposition unless it is flagged. This
module flags it.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from . import deep_archive as da

H = 63
ZWIN, ZMIN = 504, 252
COLLAPSE = 63

PRICE = ["vol21", "vol63", "mom252", "mkt252", "dd"]
#: The conferred-exposure block. These ARE the mechanism identification work: each is a spread
#: between the winner and loser legs, so a non-zero value means the sort has manufactured that
#: exposure. beta_spread > 0 means the book is a levered market bet; hml_spread < 0 means it is
#: short value; smb_spread < 0 means it is long large-caps; d10y_spread carries the duration bet.
MECH = ["beta_spread", "d10y_spread", "hml_spread", "smb_spread", "tilt_pctile"]
DISP = ["vol252", "vol_ratio", "disp", "disp63"]

#: Human-readable reading of each mechanism axis, used by the readout. Sign convention is
#: winners-minus-losers throughout, so positive always means "the long leg carries more of it".
MECH_LABEL = {
    "beta_spread":  ("market beta",     "levered long the market", "short the market"),
    "hml_spread":   ("value tilt",      "long value",              "long growth"),
    "smb_spread":   ("size tilt",       "long small caps",         "long large caps"),
    "d10y_spread":  ("duration",        "long duration",           "short duration"),
}

PANEL = "data/processed/feature_panel.parquet"
DEJAVU = "data/processed/dejavu_chronobert.parquet"


def severity(asof) -> dict | None:
    """The severity estimate, from VOL-SCALED CLIMATOLOGY -- not from the analogue set.

    Measured 2026-08-30 (docs/ANALOGUE_SYSTEM.md sec 7.6): scored on twCRPS over 6,269 dates
    against the identical outcome, vol-scaled climatology beats the analogue ensemble at k=8
    AND k=16, and beats it even when cut to eight randomly chosen members. Retrieval does
    extract real information -- it beats random selection at matched size -- but it is
    dominated by this one line of arithmetic.

    So the NUMBER comes from here and the analogue panels explain rather than forecast.
    Each past outcome is divided by the trailing volatility prevailing at the time and
    rescaled to today's, which is strictly as-of: both volatilities are trailing.
    """
    t = pd.Timestamp(asof)
    prior = da.F.index[(da.F.index < t)]
    fwd = da.F.fwd.reindex(prior).dropna()
    vol = da.F["vol21"].replace(0, np.nan)
    vt = vol.get(t, np.nan)
    if not np.isfinite(vt) or len(fwd) < 200:
        return None
    vs = vol.reindex(fwd.index)
    m = np.isfinite(vs) & (vs > 0)
    scaled = (fwd[m] / vs[m] * vt).to_numpy()
    uncond = fwd.to_numpy()
    return {"median": float(np.median(scaled)), "p25": float(np.percentile(scaled, 25)),
            "p10": float(np.percentile(scaled, 10)), "p05": float(np.percentile(scaled, 5)),
            "vol_pctile": float((vol.reindex(prior).dropna() <= vt).mean()),
            "uncond_median": float(np.median(uncond)), "n": int(m.sum())}


def rhyme(asof) -> pd.DataFrame:
    """Narrative rhyme from the News Deja Vu layer, if this date falls in an episode window.

    Coverage is 8 episode windows (2018-2025), not a daily series -- so this is absent for
    most dates by construction, and absent is shown rather than imputed.
    """
    try:
        D = pd.read_parquet(DEJAVU)
    except Exception:
        return pd.DataFrame()
    t = pd.Timestamp(asof)
    hit = D[(pd.to_datetime(D.day) == t) & (D["rank"] == 1)]
    return hit


def _zs(df: pd.DataFrame) -> pd.DataFrame:
    Z = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for c in df.columns:
        m = df[c].rolling(ZWIN, min_periods=ZMIN).mean()
        s = df[c].rolling(ZWIN, min_periods=ZMIN).std()
        Z[c] = ((df[c] - m) / s).clip(-5, 5)
    return Z


def build(blocks=("mech", "disp")):
    """Assemble the PoC key. `blocks` selects which feature families join the price core."""
    fp = pd.read_parquet(PANEL)
    cols = list(PRICE)
    raw = da.F[PRICE].copy()
    if "mech" in blocks:
        for c in MECH: raw[c] = fp[c].reindex(raw.index)
        cols += MECH
    if "disp" in blocks:
        for c in DISP: raw[c] = fp[c].reindex(raw.index)
        cols += DISP
    Z = _zs(raw[cols]).dropna()
    return Z, raw.loc[Z.index], fp


Z, RAW, FP = build()
#: Retrieval runs on the deep archive; this module supplies the mechanism layer over the top.
FPM = FP[MECH]
MECH_START = FPM.dropna().index.min()
MECH_END = FP[MECH].dropna().index.max()
#: Years inside the covered span where the bet decomposition is mostly absent. Surfaced in the
#: readout so a reader can tell a data gap from a date that predates the panel.
#: Bound at MECH_END, not the end of the feature panel. The panel runs past the holdings
#: data, so measuring to its end counted the trailing months as missing and reported the
#: final (partial) year as a data gap when it is simply where the holdings stop.
_cov = FPM.loc[MECH_START:MECH_END].notna().all(axis=1).groupby(lambda d: d.year).mean()
GAP_YEARS = sorted(int(y) for y, v in _cov.items() if v < 0.75)
CLOSES = pd.Series(Z.index, index=Z.index).shift(-H)


def random_pair_distance(n: int = 4000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed); a = Z.to_numpy()
    i, j = rng.integers(0, len(a), n), rng.integers(0, len(a), n)
    return float(np.median(np.sqrt(((a[i] - a[j]) ** 2).sum(1))))


RPD = random_pair_distance()


MODERN_FROM = pd.Timestamp("2013-12-02")


def _modern_retrieve(asof, k: int = 8) -> pd.DataFrame:
    """Same price key, archive restricted to the holdings era, so every match carries a bet."""
    t = pd.Timestamp(asof)
    ZM = da.Z[da.Z.index >= MODERN_FROM]
    if t not in ZM.index:
        prior = ZM.index[ZM.index <= t]
        if not len(prior): return pd.DataFrame()
        t = prior[-1]
    closes = pd.Series(ZM.index, index=ZM.index).shift(-H)
    elig = ZM.index[(ZM.index < t) & (closes.reindex(ZM.index) < t)]
    if len(elig) < 50: return pd.DataFrame()
    d = np.sqrt(((ZM.loc[elig] - ZM.loc[t]) ** 2).sum(axis=1)).nsmallest(400)
    kept = []
    for dt, dist in d.items():
        if all(abs((dt - o).days) >= COLLAPSE for o, _ in kept):
            kept.append((dt, dist))
            if len(kept) >= k: break
    if not kept: return pd.DataFrame()
    idx = pd.DatetimeIndex([a for a, _ in kept])
    rp = da.random_pair_distance(metric="euclidean")
    out = pd.DataFrame({"dist": [b for _, b in kept], "rel": [b / rp for _, b in kept],
                        "fwd": da.F.fwd.reindex(idx)}, index=idx)
    for c in MECH: out[c] = FPM[c].reindex(idx)
    out["has_mech"] = out[MECH].notna().all(axis=1)
    return out


def retrieve(asof, k: int = 8) -> pd.DataFrame:
    """Nearest prior states, with the eligibility rule of the deep archive: a neighbour must
    precede the query AND have its own 63-day outcome window closed before it."""
    out = da.retrieve(asof, k=k, floor=None)
    if out.empty: return out
    for c in MECH: out[c] = FPM[c].reindex(out.index)
    out["has_mech"] = out[MECH].notna().all(axis=1)
    return out


def mechanism_delta(asof, k: int = 8, modern: bool = False) -> pd.DataFrame:
    """For each retrieved analogue, how the book's manufactured exposure differed from today's.

    This is the interpretability payload: two dates can look alike on price dynamics while the
    sort has built opposite bets underneath them, and that difference is what a PM needs to see
    before treating an analogue as informative about the present.
    """
    t = pd.Timestamp(asof)
    hist = FPM.loc[:t].dropna()
    if hist.empty: return pd.DataFrame()
    now = hist.iloc[-1]
    nb = _modern_retrieve(t, k) if modern else retrieve(t, k)
    if nb.empty: return nb
    for c in MECH:
        nb[f"d_{c}"] = nb[c] - now[c]      # NaN where the match predates holdings data
    nb.attrs["now"] = now
    return nb


def _panel(nb, title, note) -> list:
    L = ["", title, f"  {note}",
         f"  {'date':<12}{'dist':>7}{'worst 63d':>11}   how the bet differed from today"]
    for dt, r in nb.iterrows():
        fwd = f"{r.fwd:+.1%}" if np.isfinite(r.fwd) else "  n/a"
        if not r.has_mech:
            desc = ("-- predates holdings data, no bet decomposition" if dt < MECH_START
                    else "-- holdings panel has a gap here, no bet decomposition")
        else:
            diffs = [f"{name} {r[f'd_{c}']:+.2f}"
                     for c, (name, _, _) in MECH_LABEL.items() if abs(r[f"d_{c}"]) > 0.25]
            desc = ", ".join(diffs) if diffs else "materially the same bet"
        L.append(f"  {dt:%Y-%m-%d}  {r.rel:>6.0%}{fwd:>11}   {desc}")
    # NO AGGREGATE HERE, DELIBERATELY. A median across the analogue outcomes reads as a
    # second severity estimate sitting under the first, and it is a worse one: on 6,269 dates
    # it loses to vol-scaling in every subset tested, and loses WORST on the dates where the
    # two disagree most (12.4% MAE vs 9.1%, winning 32% of the time). Printing it invites the
    # reader to split the difference between a measured-better number and a measured-worse
    # one. The per-date outcomes stay -- each is a fact about that date, and context is the
    # job of this panel -- but they are not summarised into a rival forecast.
    return L


def readout(asof, k: int = 8) -> str:
    """PM-facing text: what the book is betting, what it resembles, and how those differ."""
    t = pd.Timestamp(asof)
    deep = mechanism_delta(t, k, modern=False)
    mod = mechanism_delta(t, k, modern=True)
    if deep.empty: return f"{t:%Y-%m-%d}: no retrievable state."
    now = deep.attrs["now"]
    L = [f"MOMENTUM REVERSAL READOUT   AS OF {t:%Y-%m-%d}",
         f"horizon: worst cumulative drawdown over the next {H} trading days"]
    sv = severity(t)
    if sv:
        L += ["", "SEVERITY  (vol-scaled climatology -- see note)",
              f"  central     {sv['median']:+.1%}      unconditional would be {sv['uncond_median']:+.1%}",
              f"  1-in-4      {sv['p25']:+.1%}",
              f"  1-in-10     {sv['p10']:+.1%}",
              f"  1-in-20     {sv['p05']:+.1%}",
              f"  trailing volatility sits at the {sv['vol_pctile']:.0%} percentile of its history",
              "  This number does NOT come from the analogues below. Scored on 6,269 dates,",
              "  vol-scaled climatology beats the analogue ensemble as a severity estimator;",
              "  the analogues are here to say WHAT this resembles and WHY, not how bad."]
    if t > MECH_END:
        L.append(f"  ** holdings panel ends {MECH_END:%Y-%m-%d}; the bet below is carried "
                 f"forward and is STALE by {(t - MECH_END).days} days **")
    if GAP_YEARS:
        L.append(f"  note: the bet decomposition is largely missing for "
                 f"{', '.join(str(y) for y in GAP_YEARS)} -- a gap in the holdings panel, "
                 f"distinct from dates that predate it ({MECH_START:%Y-%m})")
    L += ["", "WHAT THE BOOK IS BETTING (conferred exposure, winners minus losers)",
          "  the momentum sort MANUFACTURES these loadings -- they are a by-product of ranking",
          "  on past returns, not a view anyone expressed"]
    for c, (name, pos, neg) in MECH_LABEL.items():
        v = now[c]
        L.append(f"  {name:<14}{v:+7.3f}   {pos if v > 0 else neg}")
    L.append(f"  overall tilt sits at the {now['tilt_pctile']:.0%} percentile of its own history")

    L += _panel(deep, "WHAT THIS RESEMBLES  (1928-2026, deepest archive)",
                "tight matches; most predate holdings data so carry no bet decomposition")
    if not mod.empty:
        L += _panel(mod, "NEAREST MODERN COMPARABLES  (2014+, holdings era)",
                    "looser matches -- a 12-year archive cannot match as closely -- but every "
                    "one carries its bet")
    rh = rhyme(t)
    if len(rh):
        L += ["", "NARRATIVE RHYME  (episode text, 8 windows 2018-2025)"]
        for _, r in rh.iterrows():
            L.append(f"  {r.leg:<8} reads most like {r.match_episode} "
                     f"{pd.Timestamp(r.match_day):%Y-%m-%d} (sim {r.sim:.3f})")
    L += ["", "The analogue outcome spreads above are DESCRIPTIVE -- what happened after "
              "similar states.", "Use the severity block for the number."]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    print(f"retrieval: {len(da.COLS)} price features, {len(da.Z):,} days, "
          f"{da.Z.index.min():%Y-%m-%d}..{da.Z.index.max():%Y-%m-%d}")
    print(f"mechanism layer: {len(FPM.dropna()):,} days, "
          f"{FPM.dropna().index.min():%Y-%m-%d}..{MECH_END:%Y-%m-%d}\n")
    print(readout(sys.argv[1] if len(sys.argv) > 1 else MECH_END))
