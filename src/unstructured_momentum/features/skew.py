"""Per-name option skew from captured chains, plus implied carry from put-call parity.

What this is for
----------------
The market-wide implied correlation index sat at the 26th percentile on the eve of the
September 2019 reversal while crowding inside the book's own winning side was at the 87th.
A measure averaged over the whole market cannot see crowding inside one portfolio. The
repair is to measure option prices on the book's *own holdings*, which is what the daily
capture exists to make possible.

There is no free history, so nothing here can be validated through time yet. What can be
validated today is a *cross-section*: within the losing side of the book on a single
snapshot, is protection against a sharp rise more expensive on the names that are most
heavily shorted? That test needs no history at all, and it is the one place in this project
where the sample is ninety companies rather than two dozen episodes.

Gates, fixed before anything was computed
-----------------------------------------
Declared at registration and implemented here unchanged:

* a quote is discarded if either side is zero or the ask is below the bid;
* interpolation uses only contracts with absolute delta in ``[0.10, 0.45]``, so the
  quarter-delta point is always interpolated and never extrapolated;
* expiries outside ``[21, 90]`` days are excluded;
* a company must supply usable quotes on *both* wings at the *same* expiry, or it is
  dropped whole rather than measured on one side.

Gating does not attrit at random. It removes the smaller and less liquid end of the losing
side, so every result here holds for the surviving subset and is reported with that subset
named. :func:`coverage` exists to make that impossible to omit.

Sign convention
---------------
The measure is **upside skew**: the implied volatility of the quarter-delta call minus that
of the quarter-delta put. Positive means the market charges more to insure a sharp *rise*
than a sharp fall, which is the squeeze-risk direction for a short position. This
orientation is chosen so the registered hypothesis predicts a positive correlation; the
conventional risk reversal is its negative.

The session date is read from inside the file
---------------------------------------------
A chain captured in the small hours of one morning holds the *previous* session. The folder
is named for the capture, not for the data. This module derives the session from the
underlying's last-trade timestamp and returns it, so nothing downstream can join a chain to
the wrong day's short-selling data.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RAW

#: OCC symbol: root, then YYMMDD, then C or P, then strike in thousandths.
_OCC = re.compile(r"^(?P<root>[A-Z0-9]+?)(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})"
                  r"(?P<cp>[CP])(?P<k>\d{8})$")

# ---- registered gates. Changing any of these invalidates the registration. -------------
TARGET_DELTA = 0.25
DELTA_LO, DELTA_HI = 0.10, 0.45
DTE_LO, DTE_HI = 21, 90

#: Short rate used only for the implied-carry calculation, which is not part of the
#: primary test. A constant is honest here: the carry estimate is dominated by the
#: American early-exercise premium, not by a few basis points of discounting.
RISK_FREE = 0.04


def _chain_dir(day: dt.date):
    return RAW / "cboe_options" / day.strftime("%Y%m%d")


def parse_occ(symbol: str) -> tuple[str, dt.date, str, float] | None:
    m = _OCC.match(symbol)
    if not m:
        return None
    return (m["root"], dt.date(2000 + int(m["y"]), int(m["m"]), int(m["d"])),
            m["cp"], int(m["k"]) / 1000.0)


@dataclass
class Chain:
    ticker: str
    session: dt.date | None       #: derived from the file, never from the folder name
    captured_at: str
    spot: float
    contracts: pd.DataFrame
    iv30: float = float("nan")


def load_chain(day: dt.date, ticker: str) -> Chain | None:
    """Read one captured chain. Returns ``None`` if the file is absent or unusable."""
    p = _chain_dir(day) / f"{ticker}.json.gz"
    if not p.exists():
        return None
    try:
        raw = json.loads(gzip.open(p).read())
    except (OSError, ValueError):
        return None
    d = raw.get("data") or {}
    lt = (d.get("last_trade_time") or "")[:10]
    try:
        session = dt.date.fromisoformat(lt) if lt else None
    except ValueError:
        session = None

    rows = []
    for o in d.get("options", []):
        parsed = parse_occ(o.get("option", ""))
        if not parsed:
            continue
        _, expiry, cp, strike = parsed
        rows.append({
            "expiry": expiry, "cp": cp, "strike": strike,
            "bid": o.get("bid"), "ask": o.get("ask"), "iv": o.get("iv"),
            "delta": o.get("delta"), "open_interest": o.get("open_interest"),
            "volume": o.get("volume"),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return Chain(ticker=ticker, session=session, captured_at=raw.get("timestamp", ""),
                 spot=float(d.get("current_price") or "nan"), contracts=df,
                 iv30=float(d.get("iv30") or "nan"))


def apply_gates(df: pd.DataFrame, session: dt.date) -> pd.DataFrame:
    """The registered liquidity gates, in the registered order."""
    out = df.copy()
    out["dte"] = [(e - session).days for e in out["expiry"]]
    ok_quote = (out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])
    ok_delta = out["delta"].abs().between(DELTA_LO, DELTA_HI)
    ok_dte = out["dte"].between(DTE_LO, DTE_HI)
    ok_iv = out["iv"] > 0
    return out[ok_quote & ok_delta & ok_dte & ok_iv]


def iv_at_delta(wing: pd.DataFrame, target: float = TARGET_DELTA) -> float:
    """Linear interpolation of implied volatility in absolute-delta space.

    Never extrapolates: the target must be bracketed by observed contracts, which the
    ``[0.10, 0.45]`` gate is there to make likely. Returns NaN when it is not bracketed,
    and that name is then dropped rather than approximated.
    """
    w = wing.dropna(subset=["delta", "iv"]).copy()
    if w.empty:
        return float("nan")
    w["ad"] = w["delta"].abs()
    w = w.groupby("ad", as_index=False)["iv"].mean().sort_values("ad")
    if w["ad"].min() > target or w["ad"].max() < target:
        return float("nan")
    return float(np.interp(target, w["ad"].to_numpy(), w["iv"].to_numpy()))


def _atm_iv(gated: pd.DataFrame, expiry: dt.date) -> float:
    """Implied volatility nearest the half-delta point, as a level control."""
    e = gated[gated["expiry"] == expiry]
    if e.empty:
        return float("nan")
    i = (e["delta"].abs() - 0.45).abs().idxmin()
    return float(e.loc[i, "iv"])


def implied_carry(chain: Chain, expiry: dt.date, *, r: float = RISK_FREE) -> float:
    """Continuous carry implied by put-call parity, from the strike nearest the money.

    ``C - P = S e^{-qT} - K e^{-rT}`` gives ``q``, the continuous rate the option market
    is charging to hold the stock. Two caveats, both recorded rather than buried:

    * ``q`` conflates the **dividend yield with the borrow cost**. It is not borrow on its
      own, and calling it that would be wrong.
    * These are **American** options, so parity holds as an inequality rather than an
      equality. Early-exercise value sits mostly in the put, which biases ``q`` upward,
      and most strongly on exactly the hard-to-borrow names of interest.

    Stored from day one so the control is available for every future snapshot rather than
    starting from whenever it is first needed. It is not used in the primary test.
    """
    df = chain.contracts
    e = df[(df["expiry"] == expiry) & (df["bid"] > 0) & (df["ask"] > 0)]
    if e.empty or not math.isfinite(chain.spot) or chain.spot <= 0:
        return float("nan")
    piv = e.pivot_table(index="strike", columns="cp", values="mid", aggfunc="mean")
    if not {"C", "P"}.issubset(piv.columns):
        return float("nan")
    piv = piv.dropna()
    if piv.empty:
        return float("nan")
    k = float(piv.index[np.argmin(np.abs(piv.index.to_numpy() - chain.spot))])
    c, p = float(piv.loc[k, "C"]), float(piv.loc[k, "P"])
    t = max((expiry - (chain.session or dt.date.today())).days, 1) / 365.0
    inner = (c - p + k * math.exp(-r * t)) / chain.spot
    if inner <= 0:
        return float("nan")
    return float(-math.log(inner) / t)


@dataclass
class NameSkew:
    ticker: str
    session: dt.date | None = None
    upside_skew: float = float("nan")     #: call IV minus put IV at the quarter delta
    butterfly: float = float("nan")       #: average wing IV minus the at-the-money level
    atm_iv: float = float("nan")
    carry: float = float("nan")
    n_expiries: int = 0
    expiries_used: list = field(default_factory=list)
    skew_30d: float = float("nan")        #: secondary, uncorrected: nearest 30 days only
    dropped: str = ""                     #: why, when it is dropped


def name_skew(day: dt.date, ticker: str) -> NameSkew:
    """Skew for one company, averaged over every expiry that passes the gates.

    Averaging across the whole registered ``[21, 90]`` day window is deliberate: picking a
    single expiry would add a free parameter the registration does not contain. The
    nearest-30-day variant is computed alongside and reported as an uncorrected secondary.
    """
    ch = load_chain(day, ticker)
    if ch is None:
        return NameSkew(ticker, dropped="no chain captured")
    if ch.session is None:
        return NameSkew(ticker, dropped="no session timestamp inside the file")

    g = apply_gates(ch.contracts, ch.session)
    if g.empty:
        return NameSkew(ticker, ch.session, dropped="no quotes survive the gates")

    per = []
    for expiry, blk in g.groupby("expiry"):
        calls, puts = blk[blk["cp"] == "C"], blk[blk["cp"] == "P"]
        if calls.empty or puts.empty:
            continue                                   # both wings required, same expiry
        ivc, ivp = iv_at_delta(calls), iv_at_delta(puts)
        if not (math.isfinite(ivc) and math.isfinite(ivp)):
            continue                                   # quarter delta not bracketed
        atm = _atm_iv(g, expiry)
        per.append({"expiry": expiry, "dte": int(blk["dte"].iloc[0]),
                    "skew": ivc - ivp, "fly": (ivc + ivp) / 2 - atm, "atm": atm})

    if not per:
        return NameSkew(ticker, ch.session,
                        dropped="no expiry has both wings bracketed at the quarter delta")

    p = pd.DataFrame(per)
    near = p.iloc[(p["dte"] - 30).abs().argmin()]
    return NameSkew(
        ticker=ticker, session=ch.session,
        upside_skew=float(p["skew"].mean()), butterfly=float(p["fly"].mean()),
        atm_iv=float(p["atm"].mean()), carry=implied_carry(ch, near["expiry"]),
        n_expiries=len(p), expiries_used=[str(e) for e in p["expiry"]],
        skew_30d=float(near["skew"]),
    )


def panel(day: dt.date, tickers: list[str]) -> pd.DataFrame:
    """Skew for a list of companies, with the reason recorded for every one that drops."""
    rows = []
    for t in tickers:
        s = name_skew(day, t)
        rows.append({
            "ticker": s.ticker, "session": s.session, "upside_skew": s.upside_skew,
            "butterfly": s.butterfly, "atm_iv": s.atm_iv, "carry": s.carry,
            "n_expiries": s.n_expiries, "skew_30d": s.skew_30d, "dropped": s.dropped,
        })
    return pd.DataFrame(rows).set_index("ticker")


def coverage(p: pd.DataFrame) -> pd.DataFrame:
    """Attrition, as a first-class output.

    Q4 established what an unstated coverage rate does to a null: a refutation resting on
    53 percent of the losing side is a materially weaker claim than one resting on all of
    it, and the difference has to be visible in the headline rather than in a footnote.
    Option gating attrits differently but it does attrit, and toward the larger and more
    liquid end. So the surviving count and the reasons for every drop are returned as a
    table, not printed as an aside.
    """
    reasons = p["dropped"].replace("", "kept")
    out = reasons.value_counts().rename("names").to_frame()
    out["share"] = (out["names"] / len(p)).round(3)
    return out


# ---------------------------------------------------------------------------------
# The pre-registered cross-sectional test
# ---------------------------------------------------------------------------------

def cross_section(day: dt.date, losers: list[str]) -> dict:
    """Registered primary test, plus the uncorrected secondaries, plus coverage.

    One test carries the inferential weight: the rank correlation between upside skew and
    the short-volume ratio across the losing side, raw and holding company size and
    volatility level constant. Everything else returned here is exploratory and labelled.
    """
    import numpy as _np
    from scipy import stats as _st

    from ..data import ishares as _ish
    from ..data import shortside as _ss

    p = panel(day, losers)
    cov = coverage(p)
    kept = p[p["dropped"] == ""].copy()
    session = kept["session"].mode().iloc[0]

    sv = _ss.short_volume(session).set_index("ticker")
    snap = _ish.fetch("IWV").positions.set_index("ticker")
    j = kept.join(sv[["short_ratio"]]).join(snap[["sector", "market_value"]])
    j = j.dropna(subset=["upside_skew", "short_ratio", "market_value", "sector"])
    j["logcap"] = _np.log(j["market_value"])

    def _partial(y, x, ctrls):
        rank = _st.rankdata
        Y, X = rank(y), rank(x)
        C = _np.column_stack([_np.ones(len(y))] + [rank(c) for c in ctrls])
        res = lambda v: v - C @ _np.linalg.lstsq(C, v, rcond=None)[0]  # noqa: E731
        return _st.pearsonr(res(X), res(Y))

    rho, pval = _st.spearmanr(j["upside_skew"], j["short_ratio"])
    prho, ppval = _partial(j["upside_skew"], j["short_ratio"], [j["logcap"], j["atm_iv"]])

    # Clustered on sector, not blocked on time: a single cross-section has no time
    # dimension to block over, and the dependence that remains is sectoral.
    rng = _np.random.default_rng(20260821)
    groups = [g.index.to_numpy() for _, g in j.groupby("sector")]
    draws = []
    for _ in range(4000):
        idx = _np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        s = j.loc[idx]
        if s["sector"].nunique() < 2 or len(s) < 12:
            continue
        draws.append(_st.spearmanr(s["upside_skew"], s["short_ratio"])[0])
    draws = _np.array(draws)

    return {
        "session": str(session), "n": int(len(j)),
        "coverage": cov, "dropped": list(p.index[p["dropped"] != ""]),
        "rho": float(rho), "p": float(pval),
        "partial_rho": float(prho), "partial_p": float(ppval),
        "ci": [float(x) for x in _np.percentile(draws, [2.5, 97.5])],
        "share_above_zero": float((draws > 0).mean()),
        "secondary_uncorrected": {
            lbl: [float(v) for v in _st.spearmanr(a, b)]
            for lbl, a, b in (
                ("skew vs log market cap", j["upside_skew"], j["logcap"]),
                ("skew vs at-the-money volatility", j["upside_skew"], j["atm_iv"]),
                ("skew vs implied carry", j["upside_skew"], j["carry"]),
                ("short ratio vs log market cap", j["short_ratio"], j["logcap"]),
                ("nearest-30-day skew vs short ratio", j["skew_30d"], j["short_ratio"]),
            )
        },
    }


def _main() -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cross-section", action="store_true")
    ap.add_argument("--day", default="2026-08-21", help="capture folder, not the session")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.day)
    legs = _json.load(open(RAW.parent / "interim" / f"capture_legs_{day:%Y%m%d}.json"))
    r = cross_section(day, legs["losers"])
    print(f"session {r['session']}   N = {r['n']} of {len(legs['losers'])} loser-leg names\n")
    print(r["coverage"].to_string(), "\n")
    print(f"PRIMARY  rho     = {r['rho']:+.4f}  (p = {r['p']:.3f})")
    print(f"PRIMARY  partial = {r['partial_rho']:+.4f}  (p = {r['partial_p']:.3f})"
          "   holding size and volatility constant")
    print(f"         sector-clustered 95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    print(f"\ndropped ({len(r['dropped'])}): {', '.join(r['dropped'])}")
    print("\nsecondary, uncorrected and exploratory:")
    for k, (rr, pp) in r["secondary_uncorrected"].items():
        print(f"   {k:36s} rho = {rr:+.4f}  p = {pp:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
