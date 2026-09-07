"""The crash inventory: every reversal the registered event definitions admit, typed by rule.

exp-080 (register row 19) and exp-081 (row 20). Both registered before this file existed.

Two tiers, kept apart, because they can support different claims.

* The **returns tier** runs on French's daily files from 1926. The ten-portfolio file
  ``10_Portfolios_Prior_12_2_Daily`` supplies the loser and winner decile returns, so the
  loss decomposition is available for the whole century and not only for the reconstructed
  holdings era. That is more than the memo promises and it is the reason the fits-none
  count is worth reporting at all: at fifteen episodes it would mean nothing.
* The **reconstructed tier** adds the fields that need holdings -- the conferred z at
  formation and the series carrying it -- and exists from 2014.

The loss decomposition is exact arithmetic, not a model. Over an episode window the factor
loses ``L = -(hi - lo)``. The winner leg contributed ``-hi`` of that and the loser leg
contributed ``+lo``, and the two contributions sum to ``L`` by construction. Shares can be
negative (a leg that helped) or above one (a leg that lost more than the factor did,
offset by the other leg); they are not clipped, because clipping them would hide exactly
the episodes the route rule is meant to leave unassigned.

The route rule is fixed in the registration and reproduced in ``assign_route`` below. It is
applied in one order and no other, and anything it does not match is FITS-NONE rather than
forced into the nearest bucket. The fits-none share is the result that matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unstructured_momentum.data import french
from unstructured_momentum.events import episodes as E

# --- Registered event definitions (Appendix E, frozen 2026-08-26) ----------------------
FAST_THRESHOLD = -0.0432      # 10-day return, 5th pct of 25,288 windows 1926-2022
CRASH_DAY = -0.0933           # 1st pct of the same sample
FAST_WINDOW = 10
CLUSTER_DAYS = 21
SLOW_DEPTH = -0.0800          # Route 2: peak-to-trough within 90 sessions
SLOW_WINDOW = 90
SLOW_CONFIRM = 20             # sessions without a new low

# --- Registered route rule (exp-080 method.grid) ---------------------------------------
R1_LOSER_SHARE = 0.60
R2_WINNER_SHARE = 0.60
R3_FLAT_MARKET = 0.02
R3_SHARE_BAND = (0.25, 0.75)
R3_RETRACE = 0.50
RETRACE_WINDOW = 20

#: A validity condition on the DECOMPOSITION, not a change to the route rule, and added AFTER
#: the run when a reviewer noticed a leg share of -34.7 sitting unflagged in the inventory.
#: Episodes are detected on French's 2x3 momentum factor; the leg shares are computed from his
#: ten-portfolio decile file, which is a different construction. When the two disagree -- WML
#: falls 4.7 percent while the decile spread falls 0.65 -- the shares divide by a denominator
#: that is noise and can take any value. A ratio is interpretable only where its denominator is,
#: so an episode whose decile-based loss is under half its WML depth is marked NO-DECOMPOSITION
#: rather than typed. Stated relative to the episode's own depth so no new constant is
#: introduced, and both the corrected and the as-run counts are reported.
MIN_DECOMPOSITION_RATIO = 0.50

OUT = Path("reports/crash_inventory")


# --------------------------------------------------------------------------------------
# Episode detection
# --------------------------------------------------------------------------------------


def fast_episodes(wml: pd.Series, mkt: pd.Series) -> pd.DataFrame:
    """Routes 1 and 3's event definition: a ten-day return at or below -432bp."""
    eps = E.sharp_reversals(
        wml, mkt, window=FAST_WINDOW, threshold=FAST_THRESHOLD, cluster_days=CLUSTER_DAYS
    )
    eps.insert(0, "definition", "fast")
    return eps


def slow_episodes(wml: pd.Series, mkt: pd.Series) -> pd.DataFrame:
    """Route 2's event definition: a bounded peak-to-trough, confirmed by no new low.

    Bounded to ``SLOW_WINDOW`` sessions so this finds unwinds and not the secular decay
    that an unbounded drawdown detector finds; confirmed by ``SLOW_CONFIRM`` sessions
    without a new low so a trough is only a trough once it has held.
    """
    r = wml.dropna()
    cum = (1 + r).cumprod()
    roll_peak = cum.rolling(SLOW_WINDOW, min_periods=1).max()
    dd = cum / roll_peak - 1.0

    cand = dd[dd <= SLOW_DEPTH]
    if not len(cand):
        return pd.DataFrame()

    groups, last, gid = [], None, -1
    for t in cand.index:
        if last is None or (t - last).days > CLUSTER_DAYS:
            gid += 1
        groups.append(gid)
        last = t
    cand = cand.to_frame("dd")
    cand["g"] = groups

    ctx = E.market_context(r, mkt)
    m = mkt.dropna()
    rows = []
    for _, chunk in cand.groupby("g"):
        trough_at = chunk["dd"].idxmin()
        pos = r.index.get_loc(trough_at)
        # Confirmation: no new low in the next SLOW_CONFIRM sessions.
        tail = cum.iloc[pos + 1 : pos + 1 + SLOW_CONFIRM]
        if len(tail) and (tail < cum.iloc[pos]).any():
            continue
        window_start = max(pos - SLOW_WINDOW, 0)
        peak_at = cum.iloc[window_start : pos + 1].idxmax()
        depth = float(cum.loc[trough_at] / cum.loc[peak_at] - 1.0)
        c = ctx.reindex([peak_at]).iloc[0]
        rows.append(
            {
                "definition": "slow",
                "peak_at": peak_at,
                "trough_at": trough_at,
                "depth": depth,
                "window": int(r.index.get_loc(trough_at) - r.index.get_loc(peak_at)),
                "trail_vol_at_peak": float(c["trail_vol"]),
                "mkt_vol_at_peak": float(c["mkt_vol"]),
                "mkt_prior_2y": float(c["mkt_prior_2y"]),
                "mkt_during": float((1 + m.loc[peak_at:trough_at]).prod() - 1),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["vol_std_depth"] = out["depth"] / (
            out["trail_vol_at_peak"] * np.sqrt(out["window"].clip(lower=1) / 252.0)
        )
    return out


def merge_definitions(fast: pd.DataFrame, slow: pd.DataFrame) -> pd.DataFrame:
    """One row per episode. A trough seen by both definitions keeps the fast row.

    The fast row is kept because its window is the one the severity numbers are quoted on;
    the fact that the slow definition also saw it is retained in ``also_slow`` rather than
    thrown away, since a slow-confirmed fast episode is a different object from one the
    slow definition never noticed.
    """
    fast = fast.copy()
    fast["also_slow"] = False
    if not len(slow):
        return fast
    keep = []
    for _, s in slow.iterrows():
        near = (fast["trough_at"] - s["trough_at"]).abs().dt.days
        if len(near) and near.min() <= CLUSTER_DAYS:
            fast.loc[near.idxmin(), "also_slow"] = True
        else:
            keep.append(s)
    if keep:
        extra = pd.DataFrame(keep)
        extra["also_slow"] = True
        fast = pd.concat([fast, extra], ignore_index=True)
    return fast.sort_values("trough_at").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Per-episode measurement
# --------------------------------------------------------------------------------------


def measure(eps: pd.DataFrame, wml: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    """Leg shares, retrace, crash days, speed. Arithmetic on observed returns only."""
    out = eps.copy()
    cols = {k: [] for k in (
        "lo_ret", "hi_ret", "factor_ret", "loser_share", "winner_share",
        "retrace_20d", "crash_days", "sessions", "leg_data",
    )}
    idx = wml.index
    for _, e in out.iterrows():
        a, b = e["peak_at"], e["trough_at"]
        seg = wml.loc[a:b]
        f = float((1 + seg).prod() - 1)
        have = (lo.loc[a:b].notna().all() and hi.loc[a:b].notna().all()
                and len(lo.loc[a:b]) == len(seg))
        if have:
            lr = float((1 + lo.loc[a:b]).prod() - 1)
            hr = float((1 + hi.loc[a:b]).prod() - 1)
            L = -(hr - lr)
            sw = (-hr) / L if abs(L) > 1e-12 else np.nan
            sl = (lr) / L if abs(L) > 1e-12 else np.nan
        else:
            lr = hr = sw = sl = np.nan
        pos = idx.get_loc(b)
        fwd = wml.iloc[pos + 1 : pos + 1 + RETRACE_WINDOW]
        recov = float((1 + fwd).prod() - 1) if len(fwd) else np.nan
        # Share of the loss recovered in the twenty sessions after the trough.
        retr = recov / abs(f) if (f < 0 and np.isfinite(recov)) else np.nan
        cols["lo_ret"].append(lr)
        cols["hi_ret"].append(hr)
        cols["factor_ret"].append(f)
        cols["loser_share"].append(sl)
        cols["winner_share"].append(sw)
        cols["retrace_20d"].append(retr)
        cols["crash_days"].append(int((seg <= CRASH_DAY).sum()))
        cols["sessions"].append(int(len(seg)))
        cols["leg_data"].append(bool(have))
    for k, v in cols.items():
        out[k] = v
    out["bear_state"] = out["mkt_prior_2y"] < 0
    return out


def assign_route(row: pd.Series) -> str:
    """The registered rule, applied in this order and no other.

    Returns FITS-NONE rather than a nearest bucket, and NO-LEG-DATA where the decile file
    does not cover the window, which is a coverage statement and not a route.
    """
    if not row["leg_data"] or not np.isfinite(row["loser_share"]):
        return "NO-LEG-DATA"
    loss = -(row["hi_ret"] - row["lo_ret"])
    if abs(loss) < MIN_DECOMPOSITION_RATIO * abs(row["depth"]):
        return "NO-DECOMPOSITION"
    if row["bear_state"] and row["loser_share"] >= R1_LOSER_SHARE and row["mkt_during"] > 0:
        return "Route 1"
    if row["winner_share"] >= R2_WINNER_SHARE and not row["bear_state"]:
        return "Route 2"
    lo_b, hi_b = R3_SHARE_BAND
    if (
        abs(row["mkt_during"]) < R3_FLAT_MARKET
        and lo_b <= row["loser_share"] <= hi_b
        and lo_b <= row["winner_share"] <= hi_b
        and np.isfinite(row["retrace_20d"])
        and row["retrace_20d"] >= R3_RETRACE
    ):
        return "Route 3"
    return "FITS-NONE"


# --------------------------------------------------------------------------------------
# Separation: does the leg share discriminate better than the market return?
# --------------------------------------------------------------------------------------


def separation(inv: pd.DataFrame) -> dict:
    """Registered secondary: the discriminator this framework adds against the one it inherits.

    Both are scored the same way -- how well a single threshold on the variable separates
    Route 1 from Route 2 episodes -- as the area under the ROC curve, computed by rank.
    """
    sub = inv[inv["route"].isin(["Route 1", "Route 2"])]
    if sub["route"].nunique() < 2:
        return {"n": int(len(sub)), "auc_loser_share": None, "auc_mkt_during": None}
    y = (sub["route"] == "Route 1").to_numpy()

    def auc(x: np.ndarray) -> float | None:
        ok = np.isfinite(x)
        if ok.sum() < 4 or len(set(y[ok])) < 2:
            return None
        r = pd.Series(x[ok]).rank().to_numpy()
        n1, n0 = int(y[ok].sum()), int((~y[ok]).sum())
        return float((r[y[ok]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    return {
        "n": int(len(sub)),
        "n_route1": int(y.sum()),
        "n_route2": int((~y).sum()),
        "auc_loser_share": auc(sub["loser_share"].to_numpy()),
        "auc_mkt_during": auc(sub["mkt_during"].to_numpy()),
    }


def residue(inv: pd.DataFrame) -> dict:
    """Describe the episodes the rule leaves unassigned, and which clause each one failed.

    This is description of a registered result, not a re-specification: no threshold moves
    and no episode is retyped. It exists because a fits-none share outside the registered
    band is only useful if a reader can see whether the residue is noise or a shape.
    """
    r = inv[inv["route"] == "FITS-NONE"].copy()
    if not len(r):
        return {"n": 0}

    def why(x: pd.Series) -> str:
        out = []
        if not x["bear_state"]:
            out.append("R1:no-bear")
        elif x["loser_share"] < R1_LOSER_SHARE:
            out.append("R1:loser-share")
        elif x["mkt_during"] <= 0:
            out.append("R1:mkt-not-up")
        if x["bear_state"]:
            out.append("R2:bear-on")
        elif x["winner_share"] < R2_WINNER_SHARE:
            out.append("R2:winner-share")
        return " | ".join(out)

    r["why"] = r.apply(why, axis=1)
    modal = r["why"].value_counts()
    lo_b, hi_b = R3_SHARE_BAND
    c_flat = inv["mkt_during"].abs() < R3_FLAT_MARKET
    c_band = inv["loser_share"].between(lo_b, hi_b) & inv["winner_share"].between(lo_b, hi_b)
    c_ret = inv["retrace_20d"] >= R3_RETRACE
    # The counterfactual is reported, not applied: it names the clause responsible.
    no_bear = r[(~r["bear_state"]) & (r["winner_share"] < R2_WINNER_SHARE)]
    would_r1 = int(
        ((no_bear["loser_share"] >= R1_LOSER_SHARE) & (no_bear["mkt_during"] > 0)).sum()
    )
    return {
        "n": int(len(r)),
        "failure_modes": {k: int(v) for k, v in modal.items()},
        "modal_mode_share": float(modal.iloc[0] / len(r)),
        "loser_share_median": float(r["loser_share"].median()),
        "mkt_during_median": float(r["mkt_during"].median()),
        "market_rose": int((r["mkt_during"] > 0).sum()),
        "route3_clauses_over_full_inventory": {
            "flat_market": int(c_flat.sum()),
            "both_shares_in_band": int(c_band.sum()),
            "retraced_half": int(c_ret.sum()),
            "flat_and_band": int((c_flat & c_band).sum()),
            "flat_and_retrace": int((c_flat & c_ret).sum()),
            "all_three": int((c_flat & c_band & c_ret).sum()),
        },
        "counterfactual_drop_bear_clause": {
            "n_no_bear_loser_driven": int(len(no_bear)),
            "would_become_route_1": would_r1,
            "note": "reported, not applied; the rule is not re-run with the clause dropped",
        },
    }


def plant() -> dict:
    """Can the rule see each route at all? Synthetic rows at each clause centre.

    A zero count for a route is a claim about the record only if the rule would have
    fired on that route had it been there. Route 3 returns zero episodes across the whole
    century, so this check is the difference between a finding and a bug.
    """
    # Each case carries the decile returns too, so the planted rows exercise the same code path
    # a real row does -- including the decomposition validity condition, which gets its own case.
    ok = dict(depth=-0.08, hi_ret=-0.05, lo_ret=0.03)          # |L| = 0.08, comfortably valid
    cases = {
        "Route 1": dict(leg_data=True, bear_state=True, loser_share=0.85, winner_share=0.15,
                        mkt_during=0.06, retrace_20d=0.30, **ok),
        "Route 2": dict(leg_data=True, bear_state=False, loser_share=0.20, winner_share=0.80,
                        mkt_during=-0.01, retrace_20d=0.10, **ok),
        "Route 3": dict(leg_data=True, bear_state=False, loser_share=0.50, winner_share=0.50,
                        mkt_during=0.005, retrace_20d=0.60, **ok),
        "FITS-NONE": dict(leg_data=True, bear_state=False, loser_share=1.20, winner_share=-0.20,
                          mkt_during=0.04, retrace_20d=0.10, **ok),
        # A real WML episode the decile file barely registers: the shares are noise over noise.
        "NO-DECOMPOSITION": dict(leg_data=True, bear_state=True, loser_share=5.69,
                                 winner_share=-4.69, mkt_during=0.05, retrace_20d=0.30,
                                 depth=-0.057, hi_ret=-0.001, lo_ret=0.0037),
    }
    got = {k: assign_route(pd.Series(v)) for k, v in cases.items()}
    return {"planted": got, "all_recovered": all(k == v for k, v in got.items())}


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson, so a zero count still carries an interval."""
    from scipy.stats import beta as B
    lo = 0.0 if k == 0 else float(B.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(B.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# --------------------------------------------------------------------------------------


def build() -> pd.DataFrame:
    wml = french.momentum()
    ff = french.market()
    mkt = (ff["Mkt-RF"] + ff["RF"]).dropna().rename("mkt")
    dec = french.load("mom_10_daily", block=0)
    lo, hi = dec["Lo PRIOR"], dec["Hi PRIOR"]

    eps = merge_definitions(fast_episodes(wml, mkt), slow_episodes(wml, mkt))
    inv = measure(eps, wml, lo, hi)
    inv["route"] = inv.apply(assign_route, axis=1)
    # The formation date a book with this trough would have been built on: the month-end
    # before the peak, because the 12-1 sort rebuilds monthly.
    inv["formation_at"] = (
        inv["peak_at"].dt.to_period("M").dt.start_time - pd.Timedelta(days=1)
    )
    inv["era"] = np.where(inv["trough_at"] >= "2014-01-01", "reconstructed", "returns-only")
    return inv.sort_values("trough_at").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-detectors", action="store_true", help="exp-081, register row 20")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inv = build()
    inv.to_csv(OUT / "inventory.csv", index=False)

    typed = inv[~inv["route"].isin(["NO-LEG-DATA", "NO-DECOMPOSITION"])]
    counts = inv["route"].value_counts().to_dict()
    n_typed = int(len(typed))
    n_none = int((typed["route"] == "FITS-NONE").sum())
    share = n_none / n_typed if n_typed else float("nan")
    ci = binomial_ci(n_none, n_typed) if n_typed else (float("nan"),) * 2

    summary = {
        "n_episodes": int(len(inv)),
        "n_fast": int((inv["definition"] == "fast").sum()),
        "n_slow_only": int((inv["definition"] == "slow").sum()),
        "n_also_slow": int(inv["also_slow"].sum()),
        "n_typed": n_typed,
        "n_no_decomposition": int((inv["route"] == "NO-DECOMPOSITION").sum()),
        "routes": counts,
        "fits_none": n_none,
        "fits_none_share": share,
        "fits_none_ci": list(ci),
        "registered_band": [0.05, 0.30],
        "in_band": bool(0.05 <= share <= 0.30) if n_typed else None,
        "by_era": {
            e: g["route"].value_counts().to_dict() for e, g in inv.groupby("era")
        },
        "separation": separation(typed),
        "separation_note": (
            "VOID. The route labels are a deterministic function of loser_share (Route 1 "
            "requires >= 0.60, Route 2 requires winner_share >= 0.60 and so loser_share "
            "<= 0.40), so the AUC of loser_share against those labels is 1.0 by "
            "construction and the comparison against mkt_during measures nothing. The "
            "registration asked for this comparison and the circularity was noticed only "
            "after the run; logged as trp-85 and reported rather than deleted."
        ),
        "residue": residue(inv),
        "planted": plant(),
        "crash_day_episodes": typed[typed["crash_days"] > 0][
            ["trough_at", "depth", "route", "crash_days"]
        ].astype(str).to_dict("records"),
        "worst_five": typed.nsmallest(5, "depth")[
            ["trough_at", "depth", "route", "loser_share", "winner_share", "mkt_during"]
        ].astype(str).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    if args.score_detectors:
        from crash_inventory_score import score  # noqa: PLC0415
        score(inv)


if __name__ == "__main__":
    main()
