"""End-to-end: one date in, one brief out, with every gap named.

The rule this module enforces is that a stage which cannot run says so, with a reason, and the
brief renders anyway. A monitor whose gaps are invisible is worse than one full of holes,
because the reader cannot tell "measured and benign" from "not measured". Every stage
therefore returns either a payload or a `could_not_measure` entry carrying the reason, and the
renderer prints both.

Stages are deliberately independent. Nothing here reaches into another stage's internals, so a
stage can be stubbed, replaced, or fail without taking the run down. That is what lets the
regression set be useful this early: the six dates exercise the seams even while several
stages are still returning reasons rather than numbers.
"""

from __future__ import annotations

import datetime as dt
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path("data/processed/briefs")


@dataclass
class Stage:
    name: str
    fn: object
    #: Stages that are known not to exist yet report a reason rather than a traceback.
    planned: bool = False


@dataclass
class RunResult:
    as_of: pd.Timestamp
    values: dict = field(default_factory=dict)
    could_not_measure: dict = field(default_factory=dict)
    generated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))

    @property
    def coverage(self) -> float:
        n = len(self.values) + len(self.could_not_measure)
        return len(self.values) / n if n else 0.0


# ------------------------------------------------------------------ stages

#: Annualised target volatility for the sizing rule. Barroso & Santa-Clara use 12%.
TARGET_VOL: float = 0.12


def stage_conferred(asof: pd.Timestamp) -> dict:
    """The tilt the sort just built. Leak-free: estimated on a window closing at as_of."""
    from ..factor import cleanlegs as cl
    from ..features import conferred

    cov = conferred.coverage()
    yrs = set(cov.query("usable").year)
    if asof.year not in yrs:
        raise RuntimeError(
            f"holdings panel is {cov.set_index('year').cadence.get(asof.year,'absent')} in "
            f"{asof.year}; a daily-return loading needs a daily panel")
    P, S = conferred.load_panel(range(asof.year - 1, asof.year + 1))
    P, _ = cl.correct_splits(P)
    if (P.index <= asof).sum() < 260:
        raise RuntimeError(f"only {(P.index <= asof).sum()} panel dates before {asof.date()}; "
                           "260 needed to estimate a formation-window loading")
    X = conferred._factors(conferred.ten_year_change())
    row = conferred.conferred_at(asof, P, S, X)
    if not row:
        raise RuntimeError("leg could not be formed at this date")
    panel_path = Path("data/processed/conferred_tilt.csv")
    pct = float("nan")
    if panel_path.exists():
        hist = pd.read_csv(panel_path)
        pct = conferred.percentile_of(row["conferred_Mkt-RF_spread"], hist, "beta_spread")
    # Where this month's book sits against PLACEBO books -- selections no sort made, matched to
    # its own cap profile. Read per date from the placebo panel rather than carried as a
    # remembered sentence, because the mechanism is a regularity and not a constant: it clears
    # its own placebo 95th in 104 of 128 months, and November 2020 sits at the 55th. A brief
    # that inherits the 2019 figure would report an 81 percent regularity as a certainty.
    pb_pct, pb_clears, pb_base = float("nan"), None, float("nan")
    pb_k = pb_n = None
    pb_neff = pb_lo = pb_hi = pb_autocorr = float("nan")
    pb_path = Path("data/processed/exp045/panel.csv")
    if pb_path.exists():
        pan = pd.read_csv(pb_path)
        pan["asof"] = pd.to_datetime(pan["asof"])
        pb_base = float(pan["clears_95"].mean())
        # A bare rate broke this project's own rule for months: print the count, the effective
        # sample after serial dependence, and an interval, all MEASURED here rather than
        # recalled. Months are adjacent and the indicator is persistent, so the naive binomial
        # interval is far too tight; n_eff uses the lag-1 variance inflation and the interval
        # is a circular block bootstrap at the same scale.
        _s = pan["clears_95"].astype(float)
        _x = _s.to_numpy()
        pb_k, pb_n = int(_x.sum()), int(len(_x))
        if pb_n > 10:
            # ONE estimator, the project's own. Newey-West style inflation over twelve lags,
            # keeping the positive autocorrelations. A second implementation here would print
            # an n_eff that disagrees with the registered one, which is the fault this brief
            # exists to avoid.
            _ac = [_s.autocorr(lag=k) for k in range(1, 13)]
            _vif = 1 + 2 * sum(a for a in _ac if a == a and a > 0)
            pb_neff = pb_n / _vif if _vif > 0 else float("nan")
            _rng = np.random.default_rng(20260831)
            _L, _nb = 12, int(np.ceil(pb_n / 12))
            _bs = []
            for _ in range(4000):
                _st = _rng.integers(0, pb_n, _nb)
                _idx = (_st[:, None] + np.arange(_L)[None, :]).ravel() % pb_n
                _bs.append(_x[_idx[:pb_n]].mean())
            pb_lo, pb_hi = (float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5)))
            pb_autocorr = _ac[0]
        near = pan[pan["asof"] <= asof]
        if len(near) and (asof - near["asof"].iloc[-1]).days <= 45:
            pb_pct = float(near["percentile"].iloc[-1])
            pb_clears = bool(near["clears_95"].iloc[-1])
    # The post-formation block is a DIFFERENT estimand on a different window, and it is labelled
    # rather than blended. It uses the median of per-name betas against the panel's own
    # cross-sectional median, because a four-factor fit cannot run on the three months after a
    # sort: the holdings panel is not daily throughout and that window can hold 31 snapshots
    # against a forty-observation minimum.
    post = {}
    try:
        from ..factor import placebo as pb
        # asof.year - 1 because momentum needs a twelve-month lookback; loading from
        # asof.year leaves the formation window half outside the panel and the book
        # silently fails to form.
        P2, S2 = conferred.load_panel(range(asof.year - 1, asof.year + 2))
        P2, _ = cl.correct_splits(P2)
        R2 = P2.pct_change().where(lambda r: r.abs() < cl.JUMP)
        bk = pb.real_book(P2, asof, sectors=S2)
        if bk is None:
            post = {"unavailable": "leg could not be formed on the loaded panel window"}
        else:
            post = pb.median_beta_spread(bk, R2, asof, realised=True)
            if not post:
                post = {"unavailable": "post-formation window has too few panel snapshots"}
    except Exception as exc:  # noqa: BLE001
        post = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return {"beta_spread": row["conferred_Mkt-RF_spread"],
            "beta_spread_pctile": pct,
            "post_beta_spread": post.get("beta_spread"),
            "post_n_obs": post.get("n_obs"),
            "post_available_at": (str(post["available_at"].date())
                                  if post.get("available_at") is not None
                                  and hasattr(post.get("available_at"), "date") else None),
            "post_unavailable": post.get("unavailable"),
            # Naming used to report "driver library pending" here. It is no longer pending: the
            # X-ray names the bet on its own book and `stage_named_bet` reads that artifact, so
            # the line moved to its own stage and carries the X-ray's book tag with it.
            "naming_stage": "named_bet",
            "placebo_pctile": pb_pct,
            "placebo_clears": pb_clears,
            "placebo_base_rate": pb_base,
            "placebo_k": pb_k, "placebo_n": pb_n, "placebo_neff": pb_neff,
            "placebo_ci": [pb_lo, pb_hi], "placebo_autocorr": pb_autocorr,
            "d10y_spread": row["conferred_d10y_spread"],
            "beta_winner": row["conferred_Mkt-RF_w"],
            "beta_loser": row["conferred_Mkt-RF_l"],
            "n_leg": row["n_leg"],
            "available_at": str(row["available_at"].date())}


def stage_composition(asof: pd.Timestamp) -> dict:
    """What the book is made of, from the partition that beats a composition-matched control."""
    runs = Path("data/processed/exp043_runs.csv")
    if not runs.exists():
        raise RuntimeError("no partition runs on disk; the naming stage has not been run")
    d = pd.read_csv(runs)
    d["book_ts"] = pd.to_datetime(d["book"])
    # A brief for 2019-09-06 reads the book formed at 2019-08-30. Matching on an exact date
    # string missed it, because the partition is computed per BOOK and a brief is per DAY.
    # Only books at or before the as-of date are eligible, so a later partition cannot leak in.
    prior = d[(d.arm == "leg") & (d.book_ts <= asof)]
    if prior.empty:
        raise RuntimeError(
            f"no partition at or before {asof.date()}; naming is currently available only for "
            "the two books the registered experiment covered, and is not computed on demand")
    book = prior["book_ts"].max()
    age = (asof - book).days
    if age > 45:
        raise RuntimeError(
            f"nearest partition is {book.date()}, {age} days stale; a book rebalances monthly "
            "so a partition older than 45 days describes different holdings")
    got = prior[prior["book_ts"] == book]
    top = got.largest_group.dropna().value_counts()
    return {"largest_group": top.index[0],
            "from_book": str(book.date()),
            "book_age_days": int(age),
            "agreement": f"{int(top.iloc[0])} of {len(got)} runs",
            "median_names": float(got.largest_n.median()),
            "median_sectors": float(got.largest_sectors.median())}


#: Printed in the block. The severity model's own audit found it wrong in both directions at
#: once -- too cautious on average, not cautious enough at crowded rotations -- so a block that
#: printed a VaR without that sentence would be quoting the average and hiding the state.
SEVERITY_EQUIVALENCE = (
    "A 5% VaR at horizon h is the loss exceeded on one day in twenty over h sessions. ES is the "
    "average loss GIVEN the VaR is breached, so it is always the worse number.")

#: Horizons the block renders. The module's own tuple is (5, 10, 21); horizon 1 is added here
#: because the block specifies it, and is computed by the same path.
BLOCK_HORIZONS = (1, 10, 21)
BLOCK_QUANTILES = (0.05, 0.01)


#: Rendering a brief refits quantile regressions roughly fourteen hundred times -- the expanding
#: backtest, the Shapley subsets, and a hundred-point scan per crash threshold -- which took the
#: golden-date tests from seconds to about eight minutes a date once the stage was wired. The
#: work is a pure function of (date, factor series, horizons, quantiles), so it is cached on
#: disk against a hash of exactly those. Nothing about the cache changes a number: a miss
#: computes, a hit returns what the miss computed.
SEVERITY_CACHE = Path("data/interim/severity_cache")


def _severity_key(asof: pd.Timestamp, mom: pd.Series, mkt: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(str(asof.date()).encode())
    for s_ in (mom, mkt):
        h.update(str(len(s_)).encode())
        h.update(str(s_.index[-1]).encode() if len(s_) else b"-")
        h.update(np.round(s_.to_numpy(dtype=float)[-2000:], 10).tobytes() if len(s_) else b"-")
    h.update(str((BLOCK_HORIZONS, BLOCK_QUANTILES)).encode())

    # The ESTIMATOR is part of the key, not just its inputs. Without this the cache is keyed
    # on data alone, so changing the model silently serves the old numbers forever -- which is
    # exactly what happened when the conditional-ES fix was applied and the page reprinted the
    # inverted values byte-for-byte. A cache that cannot notice a code change is a way of
    # publishing a result the code no longer produces.
    h.update(hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "model" / "severity.py").read_bytes()
    ).digest())
    return h.hexdigest()[:20]


def stage_severity(asof: pd.Timestamp) -> dict:
    """VaR and ES at the block's horizons, with per-state coverage MEASURED, never recalled."""
    import numpy as np

    from ..data import french
    from ..model import baselines, severity as sv

    mom = french.momentum().loc[:asof]
    mkt = french.market()["Mkt-RF"].loc[:asof]
    key = _severity_key(asof, mom, mkt)
    cached = SEVERITY_CACHE / f"{key}.json"
    if cached.exists():
        d = json.loads(cached.read_text())
        d["from_cache"] = True
        return d
    out: dict = {"equivalence": SEVERITY_EQUIVALENCE, "unavailable": {}, "estimates": []}

    for q in BLOCK_QUANTILES:
        try:
            cur = sv.current_severity(mom, mkt, as_of=asof, horizons=BLOCK_HORIZONS, q=q)
            for r in cur.to_dict("records"):
                out["estimates"].append(r)
        except Exception as exc:  # noqa: BLE001
            out["unavailable"][f"var_es_q{q}"] = f"{type(exc).__name__}: {exc}"

    # Per-state coverage, from the per-observation predictions the A1 refactor exposes. The
    # under-alarm annotation is computed here rather than quoted from the claim that first
    # measured it, so a block can never print a calibration that the current data does not show.
    try:
        obs = sv.expanding_backtest(mom, mkt, horizon=10, q=0.05, per_observation=True)
        design = sv.build_design(mom, mkt)
        st = design.reindex(obs["date"])
        bear = st["bear"].to_numpy()
        rows = []
        for label, mask in (("bear", bear > 0.5), ("calm", bear <= 0.5), ("all", bear == bear)):
            sub = obs[mask]
            if len(sub) < 50:
                continue
            ch = baselines.christoffersen(sub["y"].to_numpy(),
                                          sub["pred_full"].to_numpy(), 0.05)
            rows.append({"state": label, "n": int(len(sub)),
                         "breach_rate": float((sub["y"] < sub["pred_full"]).mean()),
                         "expected": 0.05, **{k: float(v) for k, v in ch.items()
                                              if isinstance(v, (int, float))}})
        out["coverage_by_state"] = rows
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["coverage_by_state"] = f"{type(exc).__name__}: {exc}"

    # Crash-probability inversion at the registered thresholds, each printed beside a base rate
    # that is exact by construction because the level IS a percentile of the same distribution.
    try:
        fwd = sv.forward_return(mom, 10).dropna()
        design = sv.build_design(mom, mkt)
        df = pd.concat([fwd.rename("y"), design], axis=1).dropna()
        train = df.loc[:asof]
        cur = design.loc[:asof].dropna().tail(1)
        cols = [c for c in design.columns if c in train.columns]
        rows = []
        for qq, level in sorted(sv.CRASH_THRESHOLD_10D.items(), reverse=True):
            # Invert: find the quantile whose fitted value sits at the threshold. Solved by a
            # scan over quantiles rather than in closed form, since the fit is a quantile
            # regression and has no analytic inverse.
            grid = np.linspace(0.005, 0.50, 100)
            preds = []
            for g in grid:
                Xtr = np.column_stack([np.ones(len(train))]
                                      + [train[c].to_numpy() for c in cols])
                b = sv._fit_quantile(Xtr, train["y"].to_numpy(), float(g))
                preds.append(float(np.concatenate([[1.0], cur[cols].to_numpy().ravel()]) @ b))
            preds = np.array(preds)
            below = grid[preds <= level]
            p_now = float(below.max()) if len(below) else float(grid.min())
            rows.append({"threshold_q": qq, "level": level, "conditional_prob": p_now,
                         "base_rate": qq,
                         "realised_base_rate": float((train["y"] <= level).mean())})
        out["crash_probability"] = {"rows": rows,
                                    "vintage": sv.CRASH_THRESHOLD_VINTAGE}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["crash_probability"] = f"{type(exc).__name__}: {exc}"
    if not out["estimates"]:
        raise RuntimeError("no severity estimate available: "
                           + "; ".join(f"{k}={v}" for k, v in out["unavailable"].items()))
    SEVERITY_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out, default=str))
    out["from_cache"] = False
    return out


#: Printed inside the channels block, from the claim that established it. The two mechanism
#: classes separate cleanly on state variables measured at the trough AND feeding that group
#: membership into the severity model makes its forecasts worse. A block that showed the
#: separation without the second half would read as a forecast.
CHANNEL_SCOPE = ("DESCRIPTIVE ONLY. These channels separate the mechanism classes after the "
                 "fact; group membership fed to the severity model made its forecasts worse.")


def stage_sizing(asof: pd.Timestamp) -> dict:
    """Barroso-Santa-Clara constant-volatility scaling: the one line that is a DECISION.

    Every other block on the page describes a state. This one says how much to hold. It is
    deliberately the simplest thing in the system -- 126-day realised vol against a 12 percent
    target, no conditioning, no model -- because the repeated finding of this project is that
    trailing volatility beats every elaborate alternative it was tested against, and the
    honest product of that finding is a sizing rule rather than another score.

    Computed on the [S] book, which exists at every date in the archive. The X-ray computes the
    same quantity on the [X] book, which is what actually gets sized; the two are different
    portfolios and the numbers are not interchangeable.
    """
    import numpy as np

    from ..data import french

    mom = french.momentum().loc[:asof]
    if len(mom) < 126:
        raise RuntimeError(f"only {len(mom)} observations before {asof.date()}; "
                           "the 126-day window cannot be formed")
    rv126 = float(mom.tail(126).std() * np.sqrt(252))
    prior = mom.iloc[:-10]
    rv_prior = (float(prior.tail(126).std() * np.sqrt(252))
                if len(prior) >= 126 else float("nan"))
    return {"rv126_ann": rv126,
            "target_vol": TARGET_VOL,
            "leverage": TARGET_VOL / rv126,
            "leverage_10d_ago": (TARGET_VOL / rv_prior) if rv_prior == rv_prior else None,
            "wml_21d": float(mom.tail(21).sum()),
            "wml_5d": float(mom.tail(5).sum()),
            "book": "french_wml"}


def stage_channels(asof: pd.Timestamp) -> dict:
    """Two-channel state as of the date, sliced point-in-time.

    The channels are rolling windows and causal by construction, but the frame is sliced to the
    as-of date anyway: a stage that relies on its inputs being causal rather than on cutting them
    is one refactor away from leaking.
    """
    from unstructured_momentum.features import channels

    st = channels.two_channel_state()
    st = st.loc[st.index <= asof]
    if st.empty:
        raise RuntimeError("no channel state at or before this date")
    row = st.iloc[-1]
    want = ["wml_asymmetry", "loser_beta_dn", "joint_z", "factor_dispersion", "factor_corr",
            "eig_concentration", "joint_stress_quiet", "bear", "mkt_var"]
    have = {k: float(row[k]) for k in want if k in row.index and row[k] == row[k]}
    hist = st.loc[st.index <= asof]
    pct = {k: float((hist[k] <= row[k]).mean()) for k in have}
    return {"as_of_row": str(st.index[-1].date()), "n_history": int(len(hist)),
            "values": have, "percentiles": pct, "scope": CHANNEL_SCOPE,
            "stale_days": int((asof - st.index[-1]).days)}


#: Comomentum measures how much the legs' residual returns move together. It is a positioning
#: DESCRIPTION and this project's own measurement of it found the economic magnitude small, so
#: the block prints that beside the percentile rather than letting a high reading imply a signal.
COMOM_SCOPE = ("Percentile is within this series' own history; the measured economic magnitude "
               "of the effect is small.")


def stage_crowding(asof: pd.Timestamp) -> dict:
    """Four positioning reads, each carrying its own availability. Absent inputs are reason-coded.

    Nothing here is blended into a single crowding number. They start at different dates, cover
    different parts of the book, and one of them starts after every golden date.
    """
    import gzip

    out: dict = {"components": {}, "unavailable": {}}

    # 1. Comomentum percentile within its own history.
    try:
        cm = pd.read_parquet("data/processed/comomentum_iwv.parquet")
        cm = cm.loc[cm.index <= asof]
        if cm.empty:
            out["unavailable"]["comomentum"] = "no observation at or before this date"
        else:
            row = cm.iloc[-1]
            out["components"]["comomentum"] = {
                "spread": float(row["comom_spread"]), "avg": float(row["comom_avg"]),
                "pctile_spread": float((cm["comom_spread"] <= row["comom_spread"]).mean()),
                "as_of_row": str(cm.index[-1].date()), "n_history": int(len(cm)),
                "scope": COMOM_SCOPE}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["comomentum"] = f"{type(exc).__name__}: {exc}"

    # 2. Short-volume change. Daily FINRA files, capture begins 2018-08.
    try:
        files = sorted(Path("data/raw/finra_shortvol").glob("*.txt.gz"))
        stamps = [pd.Timestamp(f.name[:8]) for f in files]
        usable = [(t, f) for t, f in zip(stamps, files) if t <= asof]
        if len(usable) < 22:
            out["unavailable"]["short_volume"] = (
                f"capture begins {stamps[0].date() if stamps else 'n/a'}; "
                f"{len(usable)} files at or before this date, need 22")
        else:
            def ratio(f):
                with gzip.open(f, "rt") as fh:
                    d = pd.read_csv(fh, sep="|")
                d = d[d.get("Market", "").astype(str).ne("")] if "Market" in d else d
                sv = pd.to_numeric(d.get("ShortVolume"), errors="coerce").sum()
                tv = pd.to_numeric(d.get("TotalVolume"), errors="coerce").sum()
                return float(sv / tv) if tv else float("nan")
            # Same snapshot-versus-calendar fault as the MTUM flow feature: counting FILES
            # rather than days means any upstream outage silently widens a "21-session"
            # change into however long the gap was. Found by sweeping for the pattern after
            # the MTUM instance, not by it failing.
            span_days = (usable[-1][0] - usable[-22][0]).days
            if span_days > 45:
                out["unavailable"]["short_volume"] = (
                    f"22 files span {span_days} calendar days, so a 21-session change cannot "
                    f"be formed here without crossing a capture gap")
            else:
                now, prior = ratio(usable[-1][1]), ratio(usable[-22][1])
                out["components"]["short_volume"] = {
                    "ratio": now, "ratio_21d_prior": prior, "change": now - prior,
                    "as_of_row": str(usable[-1][0].date()),
                    "window_calendar_days": span_days}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["short_volume"] = f"{type(exc).__name__}: {exc}"

    # 3. Option-skew snapshot. Capture started well after every pre-2023 golden date, so the
    #    caveat is the reading for those dates rather than a footnote under a number.
    try:
        bs = pd.read_parquet("data/raw/cboe_options/basket_summary.parquet")
        bs["as_of"] = pd.to_datetime(bs["as_of"])
        start = bs["as_of"].min()
        avail = bs[bs["as_of"] <= asof]
        if avail.empty:
            out["unavailable"]["skew"] = (
                f"option capture begins {start.date()}, after this date; no snapshot exists "
                "for it and none can be reconstructed")
        elif int(np.busday_count(avail["as_of"].iloc[-1].date(), asof.date())) > 1:
            # Staleness alarm. The capture scheduler has died silently before (2026-08-21..25,
            # sessions permanently lost); a stale row served as if current would hide exactly
            # that failure mode. One business day of lag is normal -- the capture stamps the
            # prior close -- so anything older moves the whole component into the gaps.
            last = avail["as_of"].iloc[-1]
            out["unavailable"]["skew"] = (
                f"options capture stale: last session {last.date()} "
                f"({int(np.busday_count(last.date(), asof.date()))} business days before "
                f"{asof.date()}); serving it as current would mask a dead capture job")
        else:
            r = avail.iloc[-1]
            out["components"]["skew"] = {
                "basket_iv30": float(r["basket_iv30"]),
                "avg_constituent_iv30": float(r["avg_constituent_iv30"]),
                "implied_correlation": float(r["implied_correlation"]),
                "n_quoted": int(r["n_constituents_quoted"]),
                "n_captured": int(r["n_captured"]), "n_skipped": int(r["n_skipped"]),
                "n_losers_captured": int(r["n_losers_captured"]),
                "capture_start": str(start.date()), "as_of_row": str(r["as_of"].date())}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["skew"] = f"{type(exc).__name__}: {exc}"

    # 4. MTUM flow change, from shares outstanding in the holdings files.
    try:
        files = sorted(Path("data/raw/ishares/MTUM").glob("MTUM_*.csv"))
        stamps = [pd.Timestamp(f.stem.split("_")[1]) for f in files]
        usable = [(t, f) for t, f in zip(stamps, files) if t <= asof]
        if len(usable) < 22:
            out["unavailable"]["mtum_flow"] = (
                f"{len(usable)} MTUM snapshots at or before this date, need 22")
        else:
            def shares(f):
                for ln in f.read_text(errors="replace").splitlines()[:12]:
                    if ln.lower().startswith("shares outstanding"):
                        return float(ln.split(",", 1)[1].strip().strip('"').replace(",", ""))
                return float("nan")
            # Gate 0/1 audit: MTUM has the same upstream outage as IWV, 2017-01 to 2017-06,
            # and this counted SNAPSHOTS rather than calendar days. Across the hole the 22nd
            # prior snapshot is seven months old, so a "21-day" flow change silently spans
            # seven. The window must be a calendar window or the feature is missing.
            span_days = (usable[-1][0] - usable[-22][0]).days
            if span_days > 45:
                out["unavailable"]["mtum_flow"] = (
                    f"22 snapshots span {span_days} calendar days, which crosses a holdings "
                    f"gap; a 21-session flow change cannot be formed here")
            else:
                now, prior = shares(usable[-1][1]), shares(usable[-22][1])
                out["components"]["mtum_flow"] = {
                    "shares_outstanding": now, "shares_21d_prior": prior,
                    "pct_change_21d": (now / prior - 1.0) if prior else float("nan"),
                    "as_of_row": str(usable[-1][0].date()),
                    "window_calendar_days": span_days}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["mtum_flow"] = f"{type(exc).__name__}: {exc}"

    if not out["components"]:
        raise RuntimeError("no crowding component available: "
                           + "; ".join(f"{k}={v}" for k, v in out["unavailable"].items()))
    return out


def stage_skew(asof: pd.Timestamp) -> dict:
    raise NotImplementedError("option-skew stage not yet wired into the pipeline")


#: Printed above the odds-conditioning panel. Every clause is load-bearing.
#:
#: The panel lives in the BRIEF rather than the X-ray for one reason: the X-ray's registered
#: scope forbids probability-of-crash numbers outright, and a crash-FREQUENCY ratio sits close
#: enough to one that putting it there would blur a boundary the X-ray exists to hold. The
#: brief already carries exceedance probabilities against base rates under the same
#: discipline, so this is the block it belongs beside.
ODDS_DISCLOSURE = (
    "DESCRIPTIVE CONDITIONING CONTEXT, NOT A FORECAST AND NOT A TIMING SIGNAL. These are three "
    "PUBLISHED indicators, not ours. Each was put through this project's own attack discipline "
    "in exp-055 -- autocorrelation-matched surrogate ranking on a century of factor data plus a "
    "post-publication direction check. TWO of the three survived it. The third, the bear-state "
    "by market-volatility indicator, FAILED on the post-publication direction check and is "
    "labelled FAILED in the table above -- note that it also carries the LARGEST ratio on this "
    "page, so the biggest number here is the least supported one. Our own saturation series was "
    "run through the identical attack, also FAILED, and is printed last as the scale. The ratio "
    "beside each is a HISTORICAL CRASH FREQUENCY RATIO measured over the full sample: how much "
    "more often a 1%-worst WML day fell within the following 21 calendar days when the "
    "indicator sat in its top quintile (armed, for the binary one) than when it did not. It is "
    "a statement about the past frequency of the tail in that state. It is NOT a probability "
    "for the next 10 days, NOT additive with the exceedance block above, and carries no claim "
    "about when. The state shown is point-in-time -- value and quintile cut both taken over "
    "history to this date -- while the ratio was measured on a full-sample cut, so the two are "
    "not the same slice and the block does not multiply them together.")

#: exp-055 built its indicators on French series that were already decimal and divided by 100
#: again. Rank cuts are indifferent to that; the bear-state SIGN is not, and it differs on 556
#: of 26,274 days. The recipe is reproduced as registered rather than silently repaired,
#: because the ratio on disk was measured under it -- but a reader is owed the sentence.
# This note used to say the block still ran on a double percent-to-decimal scaling that moved
# the bear-state flag on 556 of 26,274 days, and argued for keeping it so the printed state and
# the measured effect came from one recipe. That scaling was repaired on 2026-08-27 and every
# ratio re-measured under the corrected recipe, but the note was not updated -- so for four
# days the page disclosed a defect it no longer had. A stale disclosure is not a harmless
# excess of caution: it misdescribes the running code just as a stale result would, and it
# spends the reader's trust on the wrong thing.
ODDS_RECIPE_NOTE = (
    "Recipe frozen with exp-055. A double percent-to-decimal scaling in its French inputs, "
    "which left the quintile cuts unchanged but moved the bear-state flag on 556 of 26,274 "
    "days, was REPAIRED on 2026-08-27 and every ratio re-measured under the corrected recipe, "
    "so the state printed here and the effect printed beside it now come from one construction.")


def stage_odds_conditioning(asof: pd.Timestamp) -> dict:
    """Where the three surviving published conditioners stand today, beside what they measured.

    Book [S]: every indicator is scored against Ken French WML, which is the series exp-055
    attacked. Nothing here is blended, ranked or combined into a score.
    """
    from ..data import french
    from ..features import conditioners as cond

    out: dict = {"rows": [], "unavailable": {}, "disclosure": ODDS_DISCLOSURE,
                 "recipe_note": ODDS_RECIPE_NOTE}
    try:
        ratios = cond.measured_ratios()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"no measured ratios to condition on: {exc}") from exc

    idx = french.momentum().index
    for name in ("comomentum", "momentum_gap", "bear_vol", "ours_top_share_descriptive"):
        meas = ratios.get(name)
        if meas is None:
            out["unavailable"][name] = "no row for this indicator in the exp-055 eval"
            continue
        try:
            st = cond.state_at(name, asof, idx)
        except Exception as exc:  # noqa: BLE001
            out["unavailable"][name] = f"{type(exc).__name__}: {exc}"
            continue
        st |= {"ratio": meas.get("effect"), "survives": meas.get("survives"),
               "placebo_rank": meas.get("placebo_rank"),
               "post_publication_effect": meas.get("post_publication_effect"),
               "publication_year": meas.get("publication_year")}
        out["rows"].append(st)
    if not out["rows"]:
        raise RuntimeError("no conditioner state computable: "
                           + "; ".join(f"{k}={v}" for k, v in out["unavailable"].items()))
    return out


#: How far back the brief will reach for an X-ray. The X-ray's themes are built on a trailing
#: 21-day co-mention window, so an X-ray older than that window describes a different news
#: state; beyond it the line goes to could_not_measure rather than being served as current.
XRAY_MAX_STALE_DAYS = 21
XRAY_DIR = Path("reports/xray")


def stage_named_bet(asof: pd.Timestamp) -> dict:
    """The X-ray's named bet, read from its artifact. Never computed here, never invented.

    Book [X]: this is the month-end frozen book, NOT the book the conferred block above is
    measured on. The two are different selections and the brief says so beside the numbers.
    """
    if not XRAY_DIR.exists():
        raise RuntimeError(f"{XRAY_DIR} does not exist; the X-ray stage has never run")
    cands = []
    for p in sorted(XRAY_DIR.glob("*.json")):
        try:
            d = pd.Timestamp(p.stem)
        except ValueError:
            continue
        if d <= asof:
            cands.append((d, p))
    if not cands:
        raise RuntimeError(f"no X-ray artifact at or before {asof.date()} in {XRAY_DIR}")
    xdate, path = max(cands)
    stale = int((asof - xdate).days)
    if stale > XRAY_MAX_STALE_DAYS:
        raise RuntimeError(
            f"newest X-ray is {xdate.date()}, {stale} days before this brief; its themes are "
            f"built on a trailing 21-day co-mention window, so anything older describes a "
            f"different news state and is not served as current")
    x = json.loads(path.read_text())
    top = x.get("top_theme")
    if not top:
        raise RuntimeError(f"X-ray {xdate.date()} scored no theme, so there is no bet to name")
    return {"xray_date": str(xdate.date()), "stale_days": stale, "artifact": str(path),
            "book_as_of": x.get("book_as_of"), "gated": bool(top.get("gate")),
            "n_gated": x.get("n_gated"), "n_themes": x.get("n_themes"),
            "rotation_armed": x.get("rotation_armed"),
            "members": top.get("members", []), "n": top.get("n"), "side": top.get("side"),
            "story_beta": top.get("story_beta"), "load_pct": top.get("load_pct"),
            "saturation": top.get("saturation"), "sat_pct": top.get("sat_pct"),
            "crowd_top10": top.get("crowd_top10"), "crowd_pct": top.get("crowd_pct"),
            "crowd_quarter": top.get("crowd_quarter"),
            "top_filers": top.get("top_filers"),
            "bet_name": x.get("bet_name")}


def stage_drivers(asof: pd.Timestamp) -> dict:
    """What the severity estimate is made of, which names carry the tail, and what is scheduled.

    Three independent reads under one block. Each reports its own availability; none is folded
    into a headline, and the event-to-driver classification reports that it has no driver
    library rather than guessing at one.
    """
    from ..data import french
    from ..factor import cleanlegs as cl
    from ..features import conferred
    from ..model import severity as sv

    out: dict = {"unavailable": {}}

    # 1. Shapley over the severity model's own terms, exhaustive over subsets.
    try:
        mom = french.momentum().loc[:asof]
        mkt = french.market()["Mkt-RF"].loc[:asof]
        sh = sv.shapley_over_terms(mom, mkt, asof)
        out["shapley"] = {
            "terms": sh.to_dict("records"), "baseline": sh.attrs["baseline"],
            "full": sh.attrs["full"], "n_subsets": sh.attrs["n_subsets"],
            "reconstruction_error": sh.attrs["reconstruction_error"]}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["shapley"] = f"{type(exc).__name__}: {exc}"

    # 2. Component ES: which constituents carry the book's worst days.
    try:
        P, S = conferred.load_panel(range(asof.year - 1, asof.year + 1))
        P, _ = cl.correct_splits(P)
        m = cl.momentum(P, asof, sectors=S)
        m = m[(m > -0.95) & (m < 5.0)]
        win, _ = cl.legs(m)
        w = (P.index > asof - pd.DateOffset(months=6)) & (P.index <= asof)
        R = P.loc[w, [t for t in win if t in P.columns]].pct_change()
        R = R.where(R.abs() < cl.JUMP).dropna(axis=1, thresh=int(0.8 * w.sum()))
        book = R.mean(axis=1)
        cut = book.quantile(0.05)
        tail = R.loc[book <= cut]
        if tail.empty or R.shape[1] < 20:
            out["unavailable"]["component_es"] = "too few constituents or tail days in window"
        else:
            contrib = (tail.mean() / R.shape[1]).sort_values()
            out["component_es"] = {
                "n_names": int(R.shape[1]), "n_tail_days": int(len(tail)),
                "book_es": float(book[book <= cut].mean()), "cut": float(cut),
                "top": [{"ticker": t, "contribution": float(c)}
                        for t, c in contrib.head(10).items()]}
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["component_es"] = f"{type(exc).__name__}: {exc}"

    # 3. Forward 20-session catalyst calendar.
    horizon_end = asof + pd.Timedelta(days=30)
    try:
        f = pd.read_parquet("data/raw/fed/fomc_statement_dates.parquet")
        f["date"] = pd.to_datetime(f["date"])
        ahead = f[(f["date"] > asof) & (f["date"] <= horizon_end)]
        out["catalysts"] = [{"date": str(d.date()), "kind": "FOMC statement",
                             # No driver library exists to map an event class onto a named
                             # driver, so the classification reports its absence.
                             "driver": "pending: no named driver"}
                            for d in ahead["date"]]
    except Exception as exc:  # noqa: BLE001
        out["unavailable"]["fomc"] = f"{type(exc).__name__}: {exc}"
    # The earnings file records 8-K acceptance stamps, which are events that HAVE happened. A
    # forward calendar needs scheduled dates, and deriving expected windows from last year's
    # pattern would be inventing a design rather than reading one.
    out["unavailable"]["earnings_calendar"] = (
        "data/raw/edgar/earnings_calendar.parquet holds realised 8-K acceptance stamps, not "
        "scheduled future dates; no forward earnings calendar exists to read")
    return out


def stage_translation(asof: pd.Timestamp, values: dict) -> dict:
    """The operational read. Refuses rather than invents when its inputs are absent."""
    if "conferred" not in values:
        raise RuntimeError("no measured tilt to translate; translation without a measurement "
                           "would be the model writing a story from nothing")
    import os
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError("no model credential in this environment")
    from ..llm.translate import Translator, log_falsifier

    c = values["conferred"]
    tilt = {"conferred market-beta spread (winner minus loser)": f"{c['beta_spread']:+.3f}",
            "percentile in 134 months of history": (
                f"{c['beta_spread_pctile']:.0%}" if c["beta_spread_pctile"] == c["beta_spread_pctile"]
                else "unavailable"),
            "conferred 10y-yield-change spread": f"{c['d10y_spread']:+.4f}",
            "winner-leg beta / loser-leg beta": f"{c['beta_winner']:.3f} / {c['beta_loser']:.3f}",
            "how to read it": "negative spread = the sort has built a DEFENSIVE book"}
    comp = values.get("composition")
    composition = ([f"{comp['largest_group']} ({comp['agreement']}), "
                    f"median {comp['median_names']:.0f} names across "
                    f"{comp['median_sectors']:.0f} sectors"] if comp else [])
    channels = values.get("channels", {})
    t = Translator().generate(str(asof.date()), tilt, composition, channels)
    rec = log_falsifier(str(asof.date()), t,
                        {"iwv": "panel", "fred": "disk", "run_at": str(pd.Timestamp.now().date())})
    return {"driver": t.driver, "transmission": t.transmission,
            "hedge": t.hedge_instrument, "hedge_rationale": t.hedge_rationale,
            "catalysts": t.catalysts, "confidence": t.confidence,
            "falsifier": f"{rec['variable']} {rec['direction']} >= {rec['threshold']} "
                         f"within {rec['horizon_days']}d",
            "falsifier_refutes": rec["refutes"]}


STAGES = [
    Stage("conferred", stage_conferred),
    Stage("composition", stage_composition),
    Stage("named_bet", stage_named_bet),
    Stage("severity", stage_severity, planned=True),
    Stage("sizing", stage_sizing),
    Stage("channels", stage_channels),
    Stage("crowding", stage_crowding),
    Stage("odds_conditioning", stage_odds_conditioning),
    Stage("skew", stage_skew, planned=True),
    Stage("drivers", stage_drivers),
]


def run(as_of: str | dt.date | pd.Timestamp, *, translate: bool = True) -> RunResult:
    asof = pd.Timestamp(as_of).normalize()
    res = RunResult(as_of=asof)
    for st in STAGES:
        try:
            res.values[st.name] = st.fn(asof)
        except NotImplementedError as e:
            res.could_not_measure[st.name] = f"not built: {e}"
        except Exception as e:  # noqa: BLE001
            res.could_not_measure[st.name] = f"{type(e).__name__}: {e}"
            if not st.planned and "-v" in __import__("sys").argv:
                traceback.print_exc()
    if translate:
        try:
            res.values["translation"] = stage_translation(asof, res.values)
        except Exception as e:  # noqa: BLE001
            res.could_not_measure["translation"] = f"{type(e).__name__}: {e}"
    else:
        res.could_not_measure["translation"] = "skipped: --no-translate"
    return res
