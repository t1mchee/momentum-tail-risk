"""exp-090. Is the book's index-concentration reading discovered, or mechanical?

Two questions, both SENSITIVITY ANALYSES rather than hypothesis tests. No p-value is claimed
for the difference between weightings: the two z values and the placebo band are printed side
by side and the reader draws the comparison.

Q1  Re-score the conferred z on all eight reference series under two weightings of the SAME
    names -- value weighting by index market value (quantity x price) and equal weighting --
    against the same matched placebo books on the same dates.

    The placebo books are matched on capitalisation PROFILE, not on realised weight
    concentration. Those are different constraints, and whether the matching already absorbs
    a concentration effect is exactly what this measures.

Q2  The leg membership used by scripts/current_book.py (via factor.cleanlegs) and the stored
    membership in data/processed/leg_members.pkl (via factor.legs.build) are built by
    different paths. Jaccard overlap per leg per date, how many names differ, and whether the
    printed top-ten holdings change.

Declared before the run, from reading the code rather than from any result:

  D1  scripts/current_book.py aggregates each leg with `.mean(axis=1)`. That is EQUAL
      weighting. The live z of 15.4 is therefore already an equal-weighted number, and the
      "(a) as now" arm below is a new construction rather than a reproduction of the live
      artifact. Both arms are reported; the arm that reproduces the live number is named.

  D2  current_book.py and gap01_family.py pass `weight_pct` as the `caps` series to
      placebo.random_matched. weight_pct is 0.00 for the smallest ~53% of the universe
      (trp-91), and random_matched buckets with `pd.qcut(c.rank(method="first"), 10)`, which
      breaks those ties by index order. The bottom five cap buckets are therefore an
      arbitrary alphabetical partition of the zero-weight names, not a cap match. The
      primary grid keeps weight_pct so the number is comparable to the live artifact; a
      SECONDARY grid repeats everything with caps = market value. Both were fixed before any
      z was seen.

Writes reports/poc/e6_summary.json, reports/poc/e6_by_date.csv, reports/poc/e6_membership.csv.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "scripts")

from unstructured_momentum.factor import cleanlegs as cl, placebo as pb  # noqa: E402
from build_reference_panel import reference_series  # noqa: E402

N_DRAWS = 500
N_DATES = 24                      # the latest formation date plus 23 further month-ends
SEED = 20260903                   # the seed the live artifact uses
MIN_OBS = 40                      # the regression's own minimum, as in gap01_family.spreads
OUT = Path("reports/poc")
PANEL = sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet"))
LEG_PICKLE = "data/processed/leg_members.pkl"

LABEL = {"mkt": "broad equity beta", "d10y": "ten-year yield (duration)",
         "wti": "crude oil", "usd": "the dollar", "ig": "investment-grade credit",
         "concentration": "index concentration (top ten vs equal weight)",
         "pc1": "first cross-asset principal component",
         "pc2": "second cross-asset principal component"}


# ------------------------------------------------------------------ data

def load_panels():
    px, mv, wp = {}, {}, {}
    for f in PANEL:
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price", "quantity", "weight_pct"])
        d = d[d.price.notna() & (d.price > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
        # trp-91: weight_pct rounds to 0.00 for the smallest ~53% of names. quantity*price is
        # the market value the fund actually holds and does not round those names away.
        d = d.assign(mv=pd.to_numeric(d.price, errors="coerce")
                     * pd.to_numeric(d.quantity, errors="coerce"))
        mv[f] = d.pivot_table(index="as_of", columns="ticker", values="mv", aggfunc="sum")
        wp[f] = d.pivot_table(index="as_of", columns="ticker", values="weight_pct",
                              aggfunc="last")

    def stack(dd):
        D = pd.concat(dd.values()).sort_index()
        return D[~D.index.duplicated(keep="last")]

    return stack(px), stack(mv), stack(wp)


def row_at(D: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    d = D.loc[:asof]
    return d.iloc[-1].dropna() if len(d) else pd.Series(dtype=float)


# ------------------------------------------------------------------ the measurement

class Scorer:
    """Univariate loading spread on each reference series, one at a time.

    Numerically identical to gap01_family.spreads -- same window, same per-series dropna,
    same intercept-plus-slope least squares -- but the design matrix for each series is
    factorised ONCE per date instead of once per book, because the spread of the slopes is
    a linear functional of the leg-return difference:

        bw - bl = pinv(A)[1] @ (y_w - y_l)

    which is the whole reason 500 books x 2 weightings x 8 series x 24 dates is affordable.
    """

    def __init__(self, Rw: pd.DataFrame, X: pd.DataFrame):
        self.dates = Rw.index
        self.cols = list(X.columns)
        Xw = X.reindex(self.dates)
        self.idx, self.slope_row, self.n = {}, {}, {}
        for c in self.cols:
            x = Xw[c].to_numpy(float)
            ok = np.isfinite(x)
            self.idx[c] = np.flatnonzero(ok)
            self.n[c] = int(ok.sum())
            if self.n[c] < MIN_OBS:
                self.slope_row[c] = None
                continue
            A = np.column_stack([np.ones(self.n[c]), x[ok]])
            self.slope_row[c] = np.linalg.pinv(A)[1]

    def spreads(self, y_w: np.ndarray, y_l: np.ndarray) -> np.ndarray:
        d = y_w - y_l
        out = np.full(len(self.cols), np.nan)
        for j, c in enumerate(self.cols):
            if self.slope_row[c] is None:
                continue
            i = self.idx[c]
            v = d[i]
            if np.isfinite(v).all():
                out[j] = float(self.slope_row[c] @ v)
            else:
                # A leg with no data on some day inside the window. Fall back to the exact
                # per-book dropna rather than pretending the fast path applies.
                m = np.isfinite(v)
                if m.sum() < MIN_OBS:
                    continue
                A = np.column_stack([np.ones(int(m.sum())), self._x(c)[m]])
                out[j] = float(np.linalg.lstsq(A, v[m], rcond=None)[0][1])
        return out

    def _x(self, c: str) -> np.ndarray:
        return self.Xcache[c]


class LegAggregator:
    """Leg return per day under a weighting, NaN-aware.

    Equal weighting is the mean over the names that traded that day, which is what
    `DataFrame.mean(axis=1)` does. Value weighting is the same average with market-value
    weights, renormalised each day over the names that traded, so a missing print reweights
    the leg instead of silently pulling it toward zero.
    """

    def __init__(self, Rw: pd.DataFrame):
        self.cols = {t: i for i, t in enumerate(Rw.columns)}
        A = Rw.to_numpy(float)
        self.M = np.isfinite(A)
        self.A = np.where(self.M, A, 0.0)

    def leg(self, names: list[str], w: np.ndarray | None) -> np.ndarray:
        j = np.array([self.cols[t] for t in names if t in self.cols], dtype=int)
        if len(j) == 0:
            return np.full(self.A.shape[0], np.nan)
        if w is None:
            w = np.ones(len(j))
        num = self.A[:, j] @ w
        den = self.M[:, j].astype(float) @ w
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def concentration(names: list[str], size: pd.Series) -> dict:
    v = size.reindex(names).dropna()
    v = v[v > 0]
    if v.empty:
        return {"n": 0, "eff_n": None, "top1": None, "top1_name": None, "top5": None}
    w = v / v.sum()
    return {"n": int(len(v)), "eff_n": float(1.0 / (w ** 2).sum()),
            "top1": float(w.max()), "top1_name": str(w.idxmax()),
            "top5": float(w.nlargest(5).sum())}


def latest_diagnostic(P, R, X, MV, WP, asof) -> dict:
    """Instrument check on the latest date, plus the one number Q1 turns on.

    Added AFTER the primary grid had run, and it changes no grid, threshold or estimator:
    it re-draws the same 500 books with the same seed and records descriptive statistics
    the per-date CSV does not carry.

    A z can fall for two different reasons and the summary table cannot tell them apart:
    the real loading shrank, or the null widened. So this records the real book's
    PERCENTILE inside the placebo distribution and the placebo distribution's own shape,
    and it records the realised weight concentration of the VALUE-WEIGHTED placebo legs
    against the real one -- which is the direct answer to whether matching on capitalisation
    profile already delivers a matched realised concentration, or only looks as though it
    should.
    """
    mom = cl.momentum(P, asof, min_obs=60)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    win, los = cl.legs(mom)
    m = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
    Rw = R.loc[m]
    agg, sc = LegAggregator(Rw), Scorer(Rw, X)
    sc.Xcache = {c: X.reindex(Rw.index)[c].to_numpy(float)[sc.idx[c]] for c in sc.cols}
    mv = row_at(MV, asof)
    vw, vl = mv.reindex(win).dropna(), mv.reindex(los).dropna()
    real = {"equal": sc.spreads(agg.leg(list(win), None), agg.leg(list(los), None)),
            "value": sc.spreads(agg.leg(list(vw.index), vw.to_numpy()),
                                agg.leg(list(vl.index), vl.to_numpy()))}
    book = pb.Book(list(win), list(los), family="r", meta={})
    rng = np.random.default_rng(SEED)
    null = {"equal": [], "value": []}
    eff_n = []
    for s in range(N_DRAWS):
        b = pb.random_matched(P, asof, book, row_at(WP, asof), rng, seed=s)
        if b is None:
            continue
        nw = [t for t in b.winners if t in Rw.columns]
        nl = [t for t in b.losers if t in Rw.columns]
        if len(nw) < 20 or len(nl) < 20:
            continue
        null["equal"].append(sc.spreads(agg.leg(nw, None), agg.leg(nl, None)))
        pw, pl = mv.reindex(nw).dropna(), mv.reindex(nl).dropna()
        if len(pw) < 20 or len(pl) < 20:
            continue
        null["value"].append(sc.spreads(agg.leg(list(pw.index), pw.to_numpy()),
                                        agg.leg(list(pl.index), pl.to_numpy())))
        q = pw / pw.sum()
        eff_n.append(float(1.0 / (q ** 2).sum()))
    j = sc.cols.index("concentration")
    out = {"eff_n_real_value_weighted_winner_leg": float(concentration(win, mv)["eff_n"]),
           "eff_n_placebo_value_weighted_winner_leg": {
               "median": float(np.median(eff_n)), "p5": float(np.percentile(eff_n, 5)),
               "p95": float(np.percentile(eff_n, 95))}}
    for k in ("equal", "value"):
        A = np.abs(np.vstack(null[k]))[:, j]
        r = abs(real[k][j])
        trimmed = np.sort(A)[:-5]
        out[k] = {"real_abs_spread": float(r),
                  "placebo_mean": float(A.mean()), "placebo_sd": float(A.std(ddof=1)),
                  "placebo_median": float(np.median(A)), "placebo_max": float(A.max()),
                  "real_percentile_in_placebo": float((A < r).mean() * 100),
                  "z": float((r - A.mean()) / A.std(ddof=1)),
                  "z_dropping_5_largest_placebos":
                      float((r - trimmed.mean()) / trimmed.std(ddof=1))}
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P, MV, WP = load_panels()
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)
    X = reference_series(P)
    stored = pickle.load(open(LEG_PICKLE, "rb"))
    print(f"panel {P.shape[0]} dates x {P.shape[1]} names -> {P.index.max().date()}")
    print(f"reference series {list(X.columns)}  {X.index.min().date()} -> "
          f"{X.index.max().date()}")
    print(f"stored legs {len(stored)} month-ends -> {max(stored).date()}\n")

    # The latest formation date, by exactly the rule current_book.py uses.
    asof = pd.Timestamp(P.index.max()) + pd.offsets.MonthEnd(0)
    asof = P.index[P.index <= asof].max()
    asof = pd.Timestamp(asof) - pd.offsets.MonthEnd(1) if asof.day < 20 else asof
    latest = pd.Timestamp(asof)
    dates = list(pd.date_range(end=latest, periods=N_DATES, freq="ME"))
    print(f"LATEST FORMATION DATE  {latest.date()}   grid {dates[0].date()} .. "
          f"{dates[-1].date()}  ({len(dates)} month-ends)\n")

    cap_grids = {"weight_pct": WP, "market_value": MV}
    rows, mrows, band_store = [], [], {}

    # `--report-only` re-renders the tables and the summary from the three per-date CSVs a
    # previous full run wrote, without redrawing the placebo books. It exists because the
    # reporting block below crashed AFTER the CSVs were written on the first run (a
    # `D.asof` that resolved to `DataFrame.asof`, the pandas method, rather than to the
    # column -- so the mask was the scalar False and every table came back empty). Nothing
    # is recomputed and nothing is re-tuned; the numbers are the ones already on disk.
    report_only = "--report-only" in sys.argv
    if report_only:
        D = pd.read_csv(OUT / "e6_by_date.csv")
        M = pd.read_csv(OUT / "e6_membership.csv")
        PR = pd.read_csv(OUT / "e6_profile.csv")
        print(f"--report-only: reusing {len(D)} scored rows from {OUT/'e6_by_date.csv'}\n")

    for asof in ([] if report_only else dates):
        mom = cl.momentum(P, asof, min_obs=60)
        mom = mom[(mom > -0.95) & (mom < 5.0)]
        if len(mom) < 300:
            print(f"{asof.date()}  skipped, universe {len(mom)}")
            continue
        win, los = cl.legs(mom)

        m = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        Rw = R.loc[m]
        agg = LegAggregator(Rw)
        sc = Scorer(Rw, X)
        sc.Xcache = {c: X.reindex(Rw.index)[c].to_numpy(float)[sc.idx[c]] for c in sc.cols}

        mv_row = row_at(MV, asof)
        wp_row = row_at(WP, asof)

        # ---- question 2: the two membership paths, on the same date
        st = stored.get(asof)
        if st is not None:
            for leg, a, b in (("winners", win, st["winners"]), ("losers", los, st["losers"])):
                sa, sb = set(a), set(b)
                t_a = list(wp_row.reindex(a).dropna().nlargest(10).index)
                t_b = list(wp_row.reindex(b).dropna().nlargest(10).index)
                t_a_mv = list(mv_row.reindex(a).dropna().nlargest(10).index)
                mrows.append({
                    "asof": str(asof.date()), "leg": leg,
                    "n_cleanlegs": len(sa), "n_stored": len(sb),
                    "jaccard": len(sa & sb) / max(len(sa | sb), 1),
                    "n_shared": len(sa & sb),
                    "n_cleanlegs_only": len(sa - sb), "n_stored_only": len(sb - sa),
                    "top10_cleanlegs": " ".join(t_a), "top10_stored": " ".join(t_b),
                    "top10_overlap": len(set(t_a) & set(t_b)),
                    "top10_identical": t_a == t_b,
                    "top10_cleanlegs_mv": " ".join(t_a_mv),
                    "top10_wp_vs_mv_overlap": len(set(t_a) & set(t_a_mv)),
                })

        # ---- question 1: the same names under two weightings
        w_win = mv_row.reindex(win).dropna()
        w_los = mv_row.reindex(los).dropna()
        real = {
            "equal": sc.spreads(agg.leg(win, None), agg.leg(los, None)),
            "value": sc.spreads(agg.leg(list(w_win.index), w_win.to_numpy()),
                                agg.leg(list(w_los.index), w_los.to_numpy())),
        }

        book = pb.Book(list(win), list(los), family="r", meta={})
        for cap_name, CAP in cap_grids.items():
            caps = row_at(CAP, asof)
            rng = np.random.default_rng(SEED)
            null = {"equal": [], "value": []}
            for s in range(N_DRAWS):
                b = pb.random_matched(P, asof, book, caps, rng, seed=s)
                if b is None:
                    continue
                nw = [t for t in b.winners if t in Rw.columns]
                nl = [t for t in b.losers if t in Rw.columns]
                if len(nw) < 20 or len(nl) < 20:
                    continue
                null["equal"].append(sc.spreads(agg.leg(nw, None), agg.leg(nl, None)))
                vw = mv_row.reindex(nw).dropna()
                vl = mv_row.reindex(nl).dropna()
                if len(vw) < 20 or len(vl) < 20:
                    null["value"].append(np.full(len(sc.cols), np.nan))
                    continue
                null["value"].append(sc.spreads(
                    agg.leg(list(vw.index), vw.to_numpy()),
                    agg.leg(list(vl.index), vl.to_numpy())))

            for wname in ("equal", "value"):
                NU = np.abs(np.vstack(null[wname]))
                mu = np.nanmean(NU, axis=0)
                sd = np.nanstd(NU, axis=0, ddof=1)
                lo = np.nanpercentile(NU, 5, axis=0)
                hi = np.nanpercentile(NU, 95, axis=0)
                z = (np.abs(real[wname]) - mu) / np.where(sd > 0, sd, np.nan)
                for j, c in enumerate(sc.cols):
                    rows.append({
                        "asof": str(asof.date()), "caps": cap_name, "weighting": wname,
                        "series": c, "spread": float(real[wname][j]),
                        "abs_spread": float(abs(real[wname][j])),
                        "z": float(z[j]),
                        "placebo_mean": float(mu[j]), "placebo_sd": float(sd[j]),
                        "placebo_p5": float(lo[j]), "placebo_p95": float(hi[j]),
                        "n_placebo": int(np.isfinite(NU[:, j]).sum()),
                        "n_obs": int(sc.n[c]),
                    })
                if asof == latest:
                    band_store[(cap_name, wname)] = (mu, sd, lo, hi)

        cw = concentration(win, mv_row)
        cl_ = concentration(los, mv_row)
        cw_st = concentration(st["winners"], mv_row) if st is not None else {}
        rows_note = {
            "asof": str(asof.date()),
            "n_universe": int(len(mom)), "n_leg": len(win),
            "win_eff_n": cw["eff_n"], "win_top1": cw["top1"],
            "win_top1_name": cw["top1_name"], "win_top5": cw["top5"],
            "los_eff_n": cl_["eff_n"], "los_top1": cl_["top1"], "los_top5": cl_["top5"],
            "stored_win_eff_n": cw_st.get("eff_n"), "stored_win_top1": cw_st.get("top1"),
            "stored_win_top1_name": cw_st.get("top1_name"),
            "stored_win_top5": cw_st.get("top5"),
            "caps_zero_frac_universe": float((row_at(WP, asof).reindex(mom.index) == 0).mean()),
            "caps_zero_frac_winleg": float((row_at(WP, asof).reindex(win) == 0).mean()),
        }
        band_store.setdefault("profile", []).append(rows_note)
        print(f"{asof.date()}  universe {len(mom):4d}  leg {len(win):3d}  "
              f"effN(win,MV) {cw['eff_n']:5.1f}  top1 {cw['top1_name']:<6s} "
              f"{cw['top1']:.3f}   [{time.time()-t0:5.0f}s]", flush=True)

    if not report_only:
        D = pd.DataFrame(rows)
        M = pd.DataFrame(mrows)
        PR = pd.DataFrame(band_store["profile"])
        D.to_csv(OUT / "e6_by_date.csv", index=False)
        M.to_csv(OUT / "e6_membership.csv", index=False)
        PR.to_csv(OUT / "e6_profile.csv", index=False)

    # ------------------------------------------------------------ printing
    for cap_name in cap_grids:
        d = D[(D["caps"] == cap_name) & (D["asof"] == str(latest.date()))]
        tag = "PRIMARY, as the live artifact passes it" if cap_name == "weight_pct" \
            else "SECONDARY, caps that do not round 53% of the universe to zero"
        print(f"\n{'='*94}\nLATEST FORMATION DATE {latest.date()}   caps = {cap_name}  ({tag})")
        print(f"{len(d[d.weighting=='equal'])} series, "
              f"{int(d.n_placebo.min())} usable placebo books\n")
        print(f"  {'series':34s}{'sprd(val)':>10}{'z(val)':>8}"
              f"{'sprd(eq)':>10}{'z(eq)':>8}   {'placebo |spread| band (p5-p95)':>34}")
        e = d[d.weighting == "equal"].set_index("series")
        v = d[d.weighting == "value"].set_index("series")
        for c in e.sort_values("z", ascending=False).index:
            print(f"  {LABEL[c]:34s}{v.spread[c]:+10.4f}{v.z[c]:8.1f}"
                  f"{e.spread[c]:+10.4f}{e.z[c]:8.1f}   "
                  f"val [{v.placebo_p5[c]:.4f},{v.placebo_p95[c]:.4f}]  "
                  f"eq [{e.placebo_p5[c]:.4f},{e.placebo_p95[c]:.4f}]")
        for wname, s in (("value", v), ("equal", e)):
            top = s.z.idxmax()
            print(f"  DOMINANT AXIS ({wname:5s} weighted): {LABEL[top]}  z {s.z[top]:.1f}")

    print(f"\n{'='*94}\nACROSS {PR.shape[0]} MONTH-ENDS -- median z by series and weighting\n")
    for cap_name in cap_grids:
        g = D[D.caps == cap_name].groupby(["series", "weighting"]).z.median().unstack()
        rk_v = g["value"].rank(ascending=False)
        rk_e = g["equal"].rank(ascending=False)
        print(f"  caps = {cap_name}")
        print(f"    {'series':34s}{'med z(val)':>11}{'rank':>6}"
              f"{'med z(eq)':>11}{'rank':>6}{'dates conc. ranks 1st':>24}")
        for c in g["equal"].sort_values(ascending=False).index:
            print(f"    {LABEL[c]:34s}{g['value'][c]:11.1f}{rk_v[c]:6.0f}"
                  f"{g['equal'][c]:11.1f}{rk_e[c]:6.0f}")
        for wname in ("value", "equal"):
            sub = D[(D.caps == cap_name) & (D.weighting == wname)]
            top = sub.loc[sub.groupby("asof").z.idxmax()].series.value_counts()
            print(f"    {wname:5s}-weighted top-ranked series by date: {top.to_dict()}")
        print()

    print(f"{'='*94}\nMEMBERSHIP: cleanlegs (current_book.py) vs stored leg_members.pkl\n")
    for leg, g in M.groupby("leg"):
        print(f"  {leg}: jaccard median {g.jaccard.median():.3f} "
              f"[{g.jaccard.min():.3f}, {g.jaccard.max():.3f}]   "
              f"differing names median {g.n_cleanlegs_only.median():.0f} of "
              f"{g.n_cleanlegs.median():.0f}   "
              f"top-ten identical on {int(g.top10_identical.sum())}/{len(g)} dates, "
              f"median overlap {g.top10_overlap.median():.0f}/10")
    lm = M[(M["asof"] == str(latest.date())) & (M["leg"] == "winners")].iloc[0]
    print(f"\n  at {latest.date()}, winner leg:")
    print(f"    cleanlegs top ten (what the dashboard prints): {lm.top10_cleanlegs}")
    print(f"    stored    top ten:                             {lm.top10_stored}")
    print(f"    overlap {lm.top10_overlap}/10")

    # ------------------------------------------------------------ summary
    lat = D[D["asof"] == str(latest.date())]

    def per_series(cap_name):
        e = lat[(lat["caps"] == cap_name) & (lat["weighting"] == "equal")].set_index("series")
        v = lat[(lat["caps"] == cap_name) & (lat["weighting"] == "value")].set_index("series")
        return {c: {"spread_value": float(v.spread[c]), "z_value": float(v.z[c]),
                    "spread_equal": float(e.spread[c]), "z_equal": float(e.z[c]),
                    "placebo_band_value_p5_p95": [float(v.placebo_p5[c]),
                                                  float(v.placebo_p95[c])],
                    "placebo_band_equal_p5_p95": [float(e.placebo_p5[c]),
                                                  float(e.placebo_p95[c])],
                    "rank_value": int(v.z.rank(ascending=False)[c]),
                    "rank_equal": int(e.z.rank(ascending=False)[c])}
                for c in e.index}

    summary = {
        "experiment": "exp-090",
        "kind": "sensitivity analysis, not a hypothesis test; no p-value is claimed for the "
                "difference between weightings",
        "latest_formation_date": str(latest.date()),
        "dates": [str(d.date()) for d in dates],
        "n_dates": int(PR.shape[0]),
        "n_draws_requested": N_DRAWS,
        "seed": SEED,
        "membership_path_for_q1": "factor.cleanlegs, the path scripts/current_book.py uses",
        "reference_panel_span": [str(X.index.min().date()), str(X.index.max().date())],
        "latest": {cap: per_series(cap) for cap in cap_grids},
        "median_z_across_dates": {
            cap: D[D.caps == cap].groupby(["series", "weighting"]).z.median()
            .unstack().to_dict() for cap in cap_grids},
        "ranking": {
            cap: {w: list(lat[(lat["caps"] == cap) & (lat["weighting"] == w)]
                          .sort_values("z", ascending=False).series)
                  for w in ("value", "equal")} for cap in cap_grids},
        "concentration_profile_latest": PR[PR["asof"] == str(latest.date())]
        .to_dict("records")[0],
        "latest_date_diagnostic": latest_diagnostic(P, R, X, MV, WP, latest),
        "membership": {
            "stored_pickle": LEG_PICKLE,
            "per_leg": {leg: {"jaccard_median": float(g.jaccard.median()),
                              "jaccard_min": float(g.jaccard.min()),
                              "jaccard_max": float(g.jaccard.max()),
                              "median_names_differing": float(g.n_cleanlegs_only.median()),
                              "median_leg_size": float(g.n_cleanlegs.median()),
                              "top10_identical_dates": int(g.top10_identical.sum()),
                              "top10_dates": int(len(g)),
                              "top10_overlap_median": float(g.top10_overlap.median())}
                        for leg, g in M.groupby("leg")},
            "latest_winner_top10_cleanlegs": lm.top10_cleanlegs.split(),
            "latest_winner_top10_stored": lm.top10_stored.split(),
        },
        "declared_before_run": {
            "D1": "current_book.py aggregates legs with .mean(axis=1), which is EQUAL "
                  "weighting. The live z of 15.4 is already the equal-weighted number.",
            "D2": "current_book.py passes weight_pct as caps to placebo.random_matched; it "
                  "is 0.00 for ~53% of the universe (trp-91) and pd.qcut breaks those ties "
                  "by index order, so the bottom five cap buckets are an arbitrary "
                  "alphabetical partition rather than a cap match. Repeated with market-value "
                  "caps as a secondary grid.",
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(summary, open(OUT / "e6_summary.json", "w"), indent=2, default=str)
    print(f"\nwrote {OUT/'e6_summary.json'}, {OUT/'e6_by_date.csv'} ({len(D)} rows), "
          f"{OUT/'e6_membership.csv'} ({len(M)} rows), {OUT/'e6_profile.csv'} "
          f"({len(PR)} rows)   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
