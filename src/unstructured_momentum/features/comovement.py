"""Within-group co-movement of a partitioned book, against sector-matched random groups.

The question a partition has to answer is not whether its group names sound plausible but
whether the names inside a group actually fall together. That is measurable, it is not
recorded in any document the model could have read, and it is the only thing that separates
a real shared exposure from a fluent description of one.

The null is the whole design. Random groups matched on group count, group sizes AND sector
composition already know the industry structure, so only structure BEYOND sector can
register. Three earlier experiments in this project died against a sector twin and this
measure is built so that this one can die the same way.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

IWV = Path("data/raw/ishares/IWV")

#: A daily move beyond this is a split in an unadjusted price series, not a return. iShares
#: reports its own close unadjusted, and this project has already had a 1-for-20 reverse
#: split read as +1,900 percent momentum.
MAX_ABS_DAILY = 0.60


def price_panel(start: str, end: str) -> pd.DataFrame:
    """Daily close panel from the index snapshots covering a window."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    frames = {}
    for f in sorted(glob.glob(str(IWV / "IWV_*.csv"))):
        stamp = pd.Timestamp(f.rsplit("_", 1)[-1][:8])
        if not (a <= stamp <= b):
            continue
        try:
            d = pd.read_csv(f, skiprows=9)
        except Exception:
            continue
        d.columns = [c.strip() for c in d.columns]
        if "Ticker" not in d.columns or "Price" not in d.columns:
            continue
        d = d[d.get("Asset Class", "Equity").astype(str).str.contains("Equity", na=False)]
        px = pd.to_numeric(d["Price"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        s = pd.Series(px.values, index=d["Ticker"].astype(str).values).dropna()
        frames[stamp] = s[~s.index.duplicated()]
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).T.sort_index()


def residual_returns(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Daily returns with the equal-weight market return projected out, per name."""
    px = panel.reindex(columns=[t for t in tickers if t in panel.columns]).dropna(axis=1, how="all")
    if px.shape[1] < 2:
        return pd.DataFrame()
    r = px.pct_change()
    r = r.mask(r.abs() > MAX_ABS_DAILY)
    mkt = panel.pct_change().mask(lambda x: x.abs() > MAX_ABS_DAILY).mean(axis=1)
    out = {}
    for c in r.columns:
        y = r[c]
        ok = y.notna() & mkt.notna()
        if ok.sum() < 20:
            continue
        beta = np.polyfit(mkt[ok], y[ok], 1)[0]
        out[c] = y - beta * mkt
    return pd.DataFrame(out)


def mean_within_group(resid: pd.DataFrame, labels: pd.Series, *, min_size: int = 4) -> float:
    """Size-weighted mean of within-group pairwise residual correlation."""
    tot_w, tot = 0.0, 0.0
    for g, members in labels.dropna().groupby(labels.dropna()):
        cols = [t for t in members.index if t in resid.columns]
        if len(cols) < min_size:
            continue
        c = resid[cols].corr()
        iu = np.triu_indices(len(cols), 1)
        v = c.to_numpy()[iu]
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        tot += v.mean() * len(cols)
        tot_w += len(cols)
    return float(tot / tot_w) if tot_w else float("nan")


def matched_random_labels(labels: pd.Series, sectors: pd.Series, rng) -> pd.Series:
    """Reassign group labels at random WITHIN sector, preserving every group's size.

    Shuffling within sector is what makes the null know the industry structure. A null that
    shuffled freely would be beaten by any partition that tracked sector at all, which is
    precisely the confound three earlier experiments here died on.
    """
    out = pd.Series(index=labels.index, dtype="float64")
    assigned = labels.dropna()
    for sec, idx in sectors.reindex(assigned.index).groupby(sectors.reindex(assigned.index)):
        vals = assigned.reindex(idx.index).to_numpy().copy()
        rng.shuffle(vals)
        out.loc[idx.index] = vals
    return out


def test_partition(resid: pd.DataFrame, labels: pd.Series, sectors: pd.Series,
                   *, n_perm: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    observed = mean_within_group(resid, labels)
    draws = []
    for _ in range(n_perm):
        d = mean_within_group(resid, matched_random_labels(labels, sectors, rng))
        if d == d:
            draws.append(d)
    draws = np.array(draws)
    return {
        "observed": observed,
        "null_mean": float(draws.mean()) if len(draws) else float("nan"),
        "null_sd": float(draws.std()) if len(draws) else float("nan"),
        "lift": float(observed - draws.mean()) if len(draws) else float("nan"),
        "perm_p": float((draws >= observed).mean()) if len(draws) else float("nan"),
        "n_groups": int(labels.dropna().nunique()),
        "n_assigned": int(labels.notna().sum()),
    }
