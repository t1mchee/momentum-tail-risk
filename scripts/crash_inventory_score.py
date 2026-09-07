"""exp-081, register row 20: do the route detectors, read at the formation date, recover the
route the episode turned out to be?

Registered with the inventory and before either ran, at a pass level of 0.67 agreement against
a one-in-three null. The registration assumed about twelve scorable episodes; the census
produced fewer, and the count is reported before the result.

Three detectors, one per route, each standardised on its own distribution so the argmax is
comparable across them:

* **Route 1** -- the loser leg's stated-condition rate, from the gated extraction.
* **Route 2** -- the winner-minus-loser sensitivity to the ten-year yield, from the reference
  panel, net of nothing here: the sector-net version needs a per-name industry adjustment that
  the shipped panel does not carry, and using the raw spread is the conservative choice because
  it can only add noise to Route 2's own detector.
* **Route 3** -- within-leg co-movement over the formation window against its own trailing
  median, from the holdings panel.

The standardisation is full-sample, not point-in-time. That is a real limitation and it is
stated here rather than in a footnote: it makes the comparison fair BETWEEN detectors on a
date, which is what an argmax needs, and it means each reading knows the distribution of its
own series across the whole panel. A production version standardises on the trailing window
only; the direction of the bias is toward agreement, so a null here is not explained by it.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.factor import cleanlegs as cl  # noqa: E402

OUT = Path("reports/crash_inventory")
GATE2 = Path("reports/gate2")
PASS_LEVEL = 0.67   # registered
NULL = 1 / 3        # registered
COMOVE_WINDOW = 21
COMOVE_TRAIL = 36   # months of prior readings the band is taken from


# --------------------------------------------------------------------------------------


def route1_readings() -> pd.Series:
    """Stated-condition rate per formation date, from every extraction parquet on disk."""
    rows = {}
    for f in sorted(glob.glob(str(GATE2 / "nov2020_extractions_loser_*.parquet"))):
        date = Path(f).stem.split("_")[-1]
        d = pd.read_parquet(f)
        ok = d[d["error"].isna()] if "error" in d else d
        if not len(ok):
            continue
        rows[pd.Timestamp(date)] = float(ok["states_a_condition"].fillna(False).mean())
    # The flagship date's parquet carries no date in its name.
    p = GATE2 / "nov2020_extractions.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        ok = d[d["error"].isna()] if "error" in d else d
        if len(ok):
            rows[pd.Timestamp("2020-10-31")] = float(
                ok["states_a_condition"].fillna(False).mean()
            )
    return pd.Series(rows).sort_index()


def route2_readings() -> pd.Series:
    """Winner-minus-loser loading on the ten-year yield, from the committed reference panel."""
    p = Path("data/processed/reference_panel.csv")
    d = pd.read_csv(p)
    d["asof"] = pd.to_datetime(d["asof"])
    return d.set_index("asof")["d10y_spread"].sort_index()


def route3_readings() -> pd.Series:
    """Within-loser-leg co-movement over the formation window, per month-end.

    Average pairwise correlation of constituent daily returns, computed as the mean off-diagonal
    of the correlation matrix. Computed on the same holdings panel the book is built from, so a
    date with no book has no reading rather than a zero.
    """
    px = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2012:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price"])
        d = d[d["price"].notna() & (d["price"] > 0)]
        px[f] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
    P = pd.concat(px.values()).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    P, _ = cl.correct_splits(P)
    R = P.pct_change().mask(lambda r: r.abs() > cl.JUMP)

    out = {}
    for asof in pd.date_range("2014-01-31", P.index.max(), freq="ME"):
        try:
            mom = cl.momentum(P, asof, min_obs=60)
        except Exception:  # noqa: BLE001
            continue
        mom = mom[(mom > -0.95) & (mom < 5.0)]
        if len(mom) < 300:
            continue
        _, los = cl.legs(mom)
        w = R.loc[(R.index <= asof)].tail(COMOVE_WINDOW)
        w = w[[t for t in los if t in w.columns]].dropna(axis=1, thresh=int(0.8 * len(w)))
        if w.shape[1] < 30:
            continue
        C = w.corr().to_numpy()
        iu = np.triu_indices_from(C, k=1)
        v = C[iu]
        out[asof] = float(np.nanmean(v))
    return pd.Series(out).sort_index()


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def _r3_excess(s: pd.Series) -> pd.Series:
    """Co-movement against its own trailing band, which is what Table 3 specifies."""
    trail = s.shift(1).rolling(COMOVE_TRAIL, min_periods=12).median()
    return (s - trail).dropna()


def _binom_p(k: int, n: int, p: float = NULL) -> float:
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))


def score(inv: pd.DataFrame) -> dict:
    typed = inv[(inv["era"] == "reconstructed") & (inv["route"].isin(["Route 1", "Route 2", "Route 3"]))]
    r1, r2, r3 = route1_readings(), route2_readings(), _r3_excess(route3_readings())
    z1, z2, z3 = _z(r1), _z(r2), _z(r3)

    rows, skipped = [], []
    for _, e in typed.iterrows():
        f = pd.Timestamp(e["formation_at"])
        have = {k: (s.get(f) if f in s.index else None) for k, s in
                (("Route 1", z1), ("Route 2", z2), ("Route 3", z3))}
        missing = [k for k, v in have.items() if v is None or not np.isfinite(v)]
        if missing:
            skipped.append({"formation_at": str(f.date()), "trough_at": str(e["trough_at"])[:10],
                            "route": e["route"], "missing": missing})
            continue
        pick = max(have, key=lambda k: have[k])
        rows.append({
            "formation_at": str(f.date()), "trough_at": str(e["trough_at"])[:10],
            "true_route": e["route"], "argmax": pick, "hit": pick == e["route"],
            "z_route1": round(have["Route 1"], 3), "z_route2": round(have["Route 2"], 3),
            "z_route3": round(have["Route 3"], 3),
        })

    n, k = len(rows), sum(r["hit"] for r in rows)
    res = {
        "n_reconstructed_episodes": int((inv["era"] == "reconstructed").sum()),
        "n_typed": int(len(typed)),
        "n_scorable": n,
        "hits": int(k),
        "agreement": (k / n) if n else None,
        "registered_pass_level": PASS_LEVEL,
        "null": NULL,
        "p_one_sided_exact": _binom_p(k, n) if n else None,
        "hits_needed_for_p05": next((h for h in range(n + 1) if _binom_p(h, n) < 0.05), None) if n else None,
        "classes_present_in_truth": sorted(typed["route"].unique().tolist()),
        "rows": rows,
        "skipped_for_missing_detector": skipped,
        "standardisation": "full-sample per detector; not point-in-time; bias runs toward agreement",
    }
    (OUT / "detector_scoring.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k2: v for k2, v in res.items() if k2 != "rows"}, indent=2))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    return res


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "inventory.csv", parse_dates=["peak_at", "trough_at", "formation_at"])
    score(inv)
