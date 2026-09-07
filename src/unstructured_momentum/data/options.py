"""Cboe delayed option chains -> momentum-basket implied correlation.

Why this is the most urgent thing in the project
------------------------------------------------
``https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json`` serves full
option chains -- bid/ask, ``iv``, ``open_interest``, ``volume``, and greeks -- free, with
no key. It also carries a precomputed ``iv30`` per underlying.

**It is a snapshot. There is no history.** Whatever is not captured is gone permanently.
Every day without a capture job is a day of forward-looking factor risk data that cannot
be bought at any price. Nothing else in this project has that property.

What it buys us
---------------
Every other positioning input here is backward-looking (returns, holdings) or lagged by
weeks (short interest, 13F, N-PORT). Options are the only *forward-looking, daily,
factor-level* risk measure available free.

The construct that matters is **implied correlation on the momentum basket**: MTUM's own
implied volatility against the weight-adjusted implied volatilities of its constituents.

.. math::

    \\rho_{impl} = \\frac{\\sigma_p^2 - \\sum_i w_i^2 \\sigma_i^2}
                        {(\\sum_i w_i \\sigma_i)^2 - \\sum_i w_i^2 \\sigma_i^2}

Momentum crashes are correlation events: the long leg stops behaving like 125 separate
companies and starts behaving like one crowded position. Implied correlation rising while
realised correlation stays low is the pre-crash signature, and it is *not* captured by
Cboe's COR1M index, which is computed on SPX rather than on the momentum basket.

Two things verified 2026-08-21 that shape the design
-----------------------------------------------------
* **MTUM's own options are too thin to trade off.** 1,452 contracts, many with ``iv: 0.0``
  on the wings and two- to three-digit volumes per strike. The headline ``iv30`` is
  usable; the surface is not. So basket vol comes from ``iv30``, and the constituent
  surfaces (NVDA alone has 3,228 contracts with live IV) do the heavy lifting.
* This is an **undocumented CDN endpoint** powering cboe.com's own quote pages, not a
  published API with terms. Low legal risk for research, real durability risk. Rate-limit
  politely and treat a shape change as expected rather than exceptional.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .. import pit
from ..config import RAW, USER_AGENT
from ..factor import legs as legs_mod
from . import ishares

CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

#: How many MTUM constituents to capture. The top 30 carry the large majority of weight;
#: past that, option liquidity thins out and adds noise rather than information.
TOP_N = 30


class ChainError(RuntimeError):
    """Response was not a usable option chain."""


class NoChain(ChainError):
    """Underlying has no listed options (404). Permanent, not transient."""


class RateLimited(requests.RequestException):
    """Cloudflare 429 / error 1015. Transient but needs a long, patient backoff."""


#: Minimum seconds between chain requests. Cboe sits behind Cloudflare and rate-limits
#: aggressively: capturing 128 names at 0.2s spacing returned HTTP 429 "error code: 1015"
#: for EVERY subsequent request, including the basket underlying, and killed the whole
#: capture. A rate-limited day is an unrecoverable gap, so this errs slow -- 128 names at
#: 1.5s is ~3 minutes, which costs nothing for a once-daily job.
MIN_REQUEST_INTERVAL = 1.5
_last_request: float = 0.0


def _throttle() -> None:
    global _last_request
    import time

    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


@dataclass
class Chain:
    symbol: str
    fetched_at: pd.Timestamp
    spot: float
    iv30: float | None
    n_contracts: int
    n_with_iv: int
    raw: dict

    @property
    def usable(self) -> bool:
        return self.iv30 is not None and self.iv30 > 0 and self.n_with_iv >= 20


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=15, max=120),
    retry=retry_if_exception_type((requests.ConnectionError, RateLimited)),
)
def fetch_chain(symbol: str, *, session: requests.Session | None = None) -> Chain:
    """Fetch one underlying's chain.

    Retries only on connection errors. A 404 means the underlying has no listed options,
    which is a permanent fact -- retrying wastes three round-trips to reach the same
    answer, and (as happened on the first both-leg capture) tenacity then raises
    ``RetryError``, which is NOT a ``RequestException`` and so escaped the caller's
    handler and aborted the entire day's capture. One un-optioned ticker must never cost
    a day of unrecoverable data.
    """
    s = session or requests.Session()
    _throttle()
    resp = s.get(
        CHAIN_URL.format(symbol=symbol.upper()),
        headers={"User-Agent": USER_AGENT},
        timeout=45,
    )
    if resp.status_code == 429:
        raise RateLimited(f"{symbol}: Cloudflare 429 (error 1015) -- back off")
    if resp.status_code == 404:
        raise NoChain(f"{symbol}: no listed options")
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data")
    if not data or "options" not in data:
        raise ChainError(f"{symbol}: response has no options payload")

    opts = data["options"]
    return Chain(
        symbol=symbol.upper(),
        fetched_at=pd.Timestamp.now(tz="UTC"),
        spot=float(data.get("current_price") or np.nan),
        iv30=float(data["iv30"]) / 100.0 if data.get("iv30") else None,
        n_contracts=len(opts),
        n_with_iv=sum(1 for c in opts if (c.get("iv") or 0) > 0),
        raw=payload,
    )


#: When the capture should run, and why it is stated rather than left to whoever schedules
#: it. The first two captures ran at 01:36 and therefore hold the *previous* session: the
#: folder said the 21st and the quotes were the 20th's close. Session is now read from
#: inside each file so no join can go wrong either way, but an accumulating forward-only
#: series inherits whatever convention it starts with, and a series that changes convention
#: halfway is worse than either choice. Fixed here: capture after the close, so the folder
#: date and the session date agree from this point on.
CAPTURE_AFTER_CLOSE_ET = "17:30"


def session_of(day: dt.date, ticker: str) -> dt.date | None:
    """The trading session a captured file actually holds, read from inside it.

    Never infer this from the folder name. The folder records when the job ran.
    """
    p = _raw_dir(day) / f"{ticker}.json.gz"
    if not p.exists():
        return None
    try:
        raw = json.loads(gzip.open(p).read())
    except (OSError, ValueError):
        return None
    lt = ((raw.get("data") or {}).get("last_trade_time") or "")[:10]
    try:
        return dt.date.fromisoformat(lt) if lt else None
    except ValueError:
        return None


def write_manifest(day: dt.date, universe: pd.DataFrame, captured: list[str]) -> dict:
    """Record what was *asked for* alongside what arrived.

    A capture that silently collects the wrong population looks like progress, which is how
    the first day's winner-leg-only run survived unnoticed until the test that needed the
    other leg was designed. In a forward-only capture that day cannot be re-collected, so
    the manifest is written every run and names the misses.
    """
    got = set(captured)
    asked = list(universe["ticker"])
    sessions = {}
    for t in captured[:400]:
        sd = session_of(day, t)
        if sd:
            sessions[str(sd)] = sessions.get(str(sd), 0) + 1
    m = {
        "capture_folder": day.strftime("%Y%m%d"),
        "session_counts": sessions,
        "session": max(sessions, key=sessions.get) if sessions else None,
        "asked": len(asked),
        "captured": len(got),
        "missing": sorted(set(asked) - got),
        "by_leg": universe.groupby("leg")["ticker"].count().to_dict(),
        "legs": {leg: sorted(g["ticker"]) for leg, g in universe.groupby("leg")},
    }
    (_raw_dir(day) / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def _raw_dir(day: dt.date):
    d = _folder(day)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _folder(day: dt.date):
    """Path of a capture folder WITHOUT creating it, for existence tests."""
    return RAW / "cboe_options" / f"{day:%Y%m%d}"


#: Permanent, append-only record of sessions that were never captured. The skew test
#: counts *verified* sessions, and it can only do that if the gaps are enumerated rather
#: than inferred from the folder sequence looking dense enough.
GAP_LEDGER = RAW / "cboe_options" / "gap_ledger.jsonl"


def _ledger_sessions() -> set[str]:
    """ISO dates of every session the ledger already accounts for."""
    if not GAP_LEDGER.exists():
        return set()
    out: set[str] = set()
    for ln in GAP_LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        out.update(e.get("sessions") or [])
        if e.get("session"):
            out.add(e["session"])
    return out


def log_gap(entry: dict) -> None:
    """Append one line to the gap ledger. Append-only on purpose: a gap, once real, stays
    on the record even after the job that caused it is fixed."""
    entry = {"recorded_at": pd.Timestamp.now(tz="UTC").isoformat(), **entry}
    GAP_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with GAP_LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def check_capture(day: dt.date, universe: pd.DataFrame | None = None) -> dict:
    """Post-capture assertions, run after the files are safely on disk.

    Four checks, each answering a failure that has actually happened here:

    a. **Folder date == session date** (read from inside the files, never the folder).
       The pre-open captures of Aug 2026 filed the prior session under the run date; left
       alone, the next after-close run would compute the same NY date and pour a second
       session into the same folder. Renamed to the session date when that is unambiguous
       (target folder absent), otherwise logged and left for ``session_of`` to sort out.
    b. **Asked vs captured**, with the misses named -- the winner-leg-only day survived
       precisely because nothing compared the two.
    c. **Both legs nonempty** -- same incident, stated as its own check.
    d. **Continuity**: a missing previous-business-day folder goes on the gap ledger,
       once, so the loss is a record instead of an absence.

    Results are merged into the folder's ``manifest.json`` under ``"checks"`` and printed
    for the launchd log. Never raises past its caller's data: run it after persistence.
    """
    folder = _folder(day)
    if not folder.exists():
        raise ChainError(f"no capture folder for {day}")
    captured = sorted(p.name[: -len(".json.gz")] for p in folder.glob("*.json.gz"))

    if universe is not None:
        manifest = write_manifest(day, universe, [t for t in captured if t != "MTUM"])
    else:
        mp = folder / "manifest.json"
        manifest = json.loads(mp.read_text()) if mp.exists() else {"capture_folder": folder.name}

    checks: dict = {}

    # (a) the session the files actually hold vs the folder date
    sessions: dict[str, int] = {}
    for t in captured:
        sd = session_of(day, t)
        if sd:
            sessions[str(sd)] = sessions.get(str(sd), 0) + 1
    modal = max(sessions, key=sessions.get) if sessions else None
    checks["session"] = modal
    if modal is None:
        checks["session_check"] = "FAIL: no file carries a readable last_trade_time"
    elif modal == str(day):
        checks["session_check"] = "ok: folder date matches the session inside the files"
    else:
        target = _folder(dt.date.fromisoformat(modal))
        if target.exists():
            checks["session_check"] = (
                f"MISMATCH: files hold session {modal} but {target.name} already exists; "
                f"left as {folder.name} -- joins must go through session_of()")
        else:
            folder.rename(target)
            folder, day = target, dt.date.fromisoformat(modal)
            checks["session_check"] = (
                f"renamed: files hold session {modal}, folder now {target.name}")

    # (b) asked vs captured, misses named -- counted as asked-minus-missing so a stray
    #     file from another run can never pad the numerator
    asked = manifest.get("asked")
    missing = manifest.get("missing", [])
    if asked is None:
        checks["coverage"] = "unknown: no manifest records what was asked for"
    else:
        checks["coverage"] = (f"{asked - len(missing)}/{asked} asked tickers captured"
                              + (f"; missing: {', '.join(missing)}" if missing else ""))

    # (c) both legs nonempty ON DISK -- asked counts would pass a day where every fetch
    #     of one leg failed, which is the winner-leg-only incident all over again
    leg_lists, got = manifest.get("legs", {}), set(captured)
    if not leg_lists:
        checks["legs"] = "unknown: no manifest records leg membership"
    else:
        have = {leg: sum(1 for t in names if t in got) for leg, names in leg_lists.items()}
        winners, losers = have.get("winner", 0), have.get("loser", 0)
        if winners > 0 and losers > 0:
            checks["legs"] = f"ok: {winners} winner / {losers} loser names captured"
        else:
            checks["legs"] = (f"FAIL: a leg is empty on disk (winner={winners}, "
                              f"loser={losers}); the cross-sectional test needs both")

    # (d) continuity against the previous business day, deduped against the ledger
    prev = np.busday_offset(day, -1, roll="backward").astype("datetime64[D]").item()
    if _folder(prev).exists():
        checks["continuity"] = f"ok: previous business day {prev} is on disk"
    elif str(prev) in _ledger_sessions():
        checks["continuity"] = f"gap at {prev} already on the ledger"
    else:
        log_gap({"kind": "missing_session", "session": str(prev),
                 "detected_checking": folder.name,
                 "note": "previous business day's capture folder absent; the session "
                         "cannot be re-fetched and is permanently lost"})
        checks["continuity"] = f"GAP: previous business day {prev} missing -> gap ledger"

    manifest["checks"] = checks
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for k, v in checks.items():
        print(f"capture check [{k}]: {v}")
    return checks


def persist_chain(c: Chain) -> None:
    """Store the full chain gzipped.

    The whole surface is kept, not just the summary, because we cannot know today which
    slice of it we will want in a year and the data is unrecoverable.
    """
    day = c.fetched_at.tz_convert(pit.NY).date()
    path = _raw_dir(day) / f"{c.symbol}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(c.raw, fh)


def basket_symbols(top_n: int = TOP_N) -> pd.DataFrame:
    """Latest MTUM constituents and weights, renormalised to the captured subset."""
    h = ishares.fetch("MTUM")
    top = h.positions.nlargest(top_n, "weight_pct")[["ticker", "weight_pct"]].copy()
    top["weight"] = top["weight_pct"] / top["weight_pct"].sum()
    return top.reset_index(drop=True)


def capture_universe(
    *, top_n: int = TOP_N, n_losers: int = 100, cap_floor_rank: int = 1000
) -> pd.DataFrame:
    """Names to capture: MTUM basket (winner side) AND the momentum LOSER leg.

    THIS IS THE POINT OF THE MODULE AND IT WAS MISSING. The original capture fetched MTUM
    plus its top 30 constituents -- every one a winner-leg name. The day-one
    cross-sectional test that makes this module validatable without history is a
    LOSER-LEG test (per-name risk reversal against short-side pressure), and it was
    therefore impossible to run. Unlike a parser bug this is unrecoverable: a day not
    captured is a day gone.

    Loser leg is drawn from the momentum sort inside the top ``cap_floor_rank`` names by
    index weight, matching the project's universe decision -- an unfiltered Russell 3000
    loser decile is 0.26% of index weight with no options to speak of.
    """
    rows = []

    for _, r in basket_symbols(top_n).iterrows():
        rows.append({"ticker": r["ticker"], "leg": "winner", "weight": float(r["weight"])})

    try:
        px = ishares.price_panel("IWV")
        px = px[legs_mod.valid_tickers(px.columns)]
        snap = ishares.fetch("IWV").positions.set_index("ticker")
        big = snap["weight_pct"].nlargest(cap_floor_rank).index      # cap floor by index weight
        px = px[[c for c in px.columns if c in set(big)]]
        lg = legs_mod.build(px.index.max(), prices=px)
        losers = lg["losers"].head(n_losers)
        for t in losers.index:
            rows.append({"ticker": t, "leg": "loser", "weight": float("nan")})
    except Exception:  # noqa: BLE001 - never let the loser leg abort the winner capture
        pass

    out = pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)
    return out


def implied_correlation(basket_iv: float, weights: np.ndarray, ivs: np.ndarray) -> float:
    """Implied correlation of a basket from its own IV and its constituents'.

    Returns NaN rather than a bogus number when the denominator collapses (which happens
    if only one constituent has a usable quote).
    """
    w, s = np.asarray(weights, float), np.asarray(ivs, float)
    ok = np.isfinite(s) & (s > 0)
    if ok.sum() < 5:
        return float("nan")
    w, s = w[ok] / w[ok].sum(), s[ok]

    weighted_sum_sq = float((w * s).sum() ** 2)
    sum_w2_s2 = float((w**2 * s**2).sum())
    denom = weighted_sum_sq - sum_w2_s2
    if denom <= 0:
        return float("nan")
    return (basket_iv**2 - sum_w2_s2) / denom


def capture(top_n: int = TOP_N, *, pause: float = 0.0) -> pd.DataFrame:
    """Daily job: snapshot MTUM plus its top constituents, and derive basket metrics.

    Appends one row per day to the accumulating summary. Idempotent per capture date.
    """
    import time

    universe = capture_universe(top_n=top_n)
    basket = universe[universe["leg"] == "winner"][["ticker", "weight"]].reset_index(drop=True)
    session = requests.Session()

    chains: dict[str, Chain] = {}
    skipped: list[str] = []
    for sym in ["MTUM", *universe["ticker"].tolist()]:
        try:
            c = fetch_chain(sym, session=session)
            persist_chain(c)
            chains[sym] = c
        except Exception:  # noqa: BLE001
            # A single unquoted or un-optioned name must NEVER abort the day's capture.
            # Deliberately broad: this data cannot be re-fetched tomorrow, so any
            # per-name failure is logged by omission and the loop continues.
            skipped.append(sym)
            continue
        time.sleep(pause)

    if "MTUM" not in chains:
        raise ChainError("failed to capture the basket underlying (MTUM); aborting")

    mt = chains["MTUM"]
    ivs = np.array([chains[t].iv30 if t in chains else np.nan for t in basket["ticker"]], float)
    rho = implied_correlation(mt.iv30, basket["weight"].to_numpy(), ivs)

    row = pd.Series(
        {
            "as_of": pd.Timestamp(mt.fetched_at.tz_convert(pit.NY).date()),
            "fetched_at": mt.fetched_at,
            "basket_iv30": mt.iv30,
            "avg_constituent_iv30": float(np.nansum(basket["weight"].to_numpy() * ivs)),
            "implied_correlation": rho,
            "n_constituents_quoted": int(np.isfinite(ivs).sum()),
            "n_captured": len(chains),
            "n_skipped": len(skipped),
            "n_losers_captured": int(
                sum(1 for t in universe[universe["leg"] == "loser"]["ticker"] if t in chains)
            ),
            "spot": mt.spot,
        }
    )

    sp = RAW / "cboe_options" / "basket_summary.parquet"
    new = pd.DataFrame([row])
    combined = pd.concat([pd.read_parquet(sp), new], ignore_index=True) if sp.exists() else new
    combined = (
        combined.drop_duplicates(subset=["as_of"], keep="last")
        .sort_values("as_of")
        .reset_index(drop=True)
    )
    combined.to_parquet(sp, index=False)

    try:
        check_capture(mt.fetched_at.tz_convert(pit.NY).date(), universe=universe)
    except Exception as exc:  # noqa: BLE001
        # The chains and the summary row are already on disk; a failed assertion must
        # report itself in the launchd log, not un-capture the day.
        print(f"capture checks failed to run: {type(exc).__name__}: {exc}")

    return new


def load_basket(*, refresh: bool = False) -> pd.DataFrame:
    """Accumulated basket implied-vol / implied-correlation series, PIT-stamped.

    ``available_at`` is the capture time. These are 15-minute-delayed quotes, so the
    observation is stamped at capture rather than at the market close it approximates.
    """
    sp = RAW / "cboe_options" / "basket_summary.parquet"
    if not sp.exists():
        raise FileNotFoundError(
            "No option captures yet. Run capture() daily -- this series has no history "
            "and cannot be backfilled at any price."
        )
    df = pd.read_parquet(sp).sort_values("as_of").copy()
    df["implied_minus_realised"] = np.nan  # filled by the feature layer against realised
    return pit.stamp(df, observed_at="as_of", available_at="fetched_at", name="cboe:options")
