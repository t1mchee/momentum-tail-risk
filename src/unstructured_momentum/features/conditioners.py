"""The three published crash-frequency conditioners that survived this project's own attack.

exp-055, the placebo factory, turned the surrogate-attack discipline the project used on
itself onto the literature: Lou-Polk comomentum, the momentum gap, and the
Daniel-Moskowitz bear-state-by-volatility indicator, each scored for how much more often a
1-percent-worst WML day falls in the next 21 calendar days when the indicator is in its top
quintile (or armed, for the binary one). All three survived autocorrelation-matched
surrogate ranking on a century of factor data. Our own saturation series did not, and is
carried here for exactly that reason -- an indicator that failed the same attack is the only
honest scale for the three that passed.

This module holds the indicator recipes ONCE. `scripts/placebo_factory.py` computed
`reports/placebo_factory/eval.json` from them and now imports them from here, so the state a
brief prints today is built by the same code as the ratio printed beside it. The recipes are
FROZEN: the ratios in eval.json are only interpretable against the construction that
produced them, so nothing here may be "improved" without re-running exp-055.

Nothing in this module is a forecast. A ratio is a historical crash frequency in one state
divided by the frequency in the others, over the whole sample. It says the tail has been
fatter in that state; it does not say when.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Frozen with exp-055's registration. `TOP_Q` is the quintile cut; `HORIZON` the exposure
#: window in calendar days; `CRASH_Q` the crash definition.
TOP_Q = 0.80
HORIZON = 21
CRASH_Q = 0.01

EVAL_PATH = Path("reports/placebo_factory/eval.json")

#: Human-readable identity of each indicator, printed beside its state.
SOURCES = {
    "comomentum": "Lou & Polk comomentum, replicated on the IWV panel (published 2022)",
    "momentum_gap": "the momentum gap, portfolio-level proxy from the French decile files "
                    "(published 2022; disclosed approximation)",
    "bear_vol": "Daniel & Moskowitz bear state x high market volatility (published 2016)",
    "ours_top_share_descriptive": "OUR OWN top-theme attention saturation — the control that "
                                  "FAILED the same attack",
}


#: exp-055 v1 divided already-decimal French returns by 100 a second time. Rank cuts are
#: scale-invariant so the gap and comomentum arms were untouched, but the bear flag is a SIGN
#: test on a compounded 504-day return, and compounding is not scale-invariant: the flag
#: differed on 556 of 26,274 days. The v2 re-run (2026-08-27) corrects it at the source and
#: re-measures every ratio under the corrected recipe, so the state printed in the brief and
#: the effect printed beside it now come from the same construction.
SCALING_FIXED_AT = "2026-08-27"


def _dec(df_or_s):
    """French loaders already return decimals; nothing to rescale."""
    return df_or_s


# --------------------------------------------------------------------------------------
# The indicator series, each returned on the WML trading calendar
# --------------------------------------------------------------------------------------

def comomentum(index: pd.Index) -> pd.Series:
    como = pd.read_parquet("data/processed/comomentum_iwv.parquet").comom_avg
    return como.reindex(index, method="ffill").rename("comomentum")


def momentum_gap(index: pd.Index, decile_frame: pd.DataFrame | None = None) -> pd.Series:
    if decile_frame is None:
        from ..data import french
        decile_frame = french.load("mom_10_daily")
    dec = _dec(decile_frame)
    gap = ((1 + dec["Hi PRIOR"]).rolling(231).apply(np.prod, raw=True)
           - (1 + dec["Lo PRIOR"]).rolling(231).apply(np.prod, raw=True)).shift(21)
    return gap.reindex(index).rename("momentum_gap")


def bear_vol(index: pd.Index, market_frame: pd.DataFrame | None = None) -> pd.Series:
    if market_frame is None:
        from ..data import french
        market_frame = french.market()
    mkt = _dec(market_frame)
    mret = mkt["Mkt-RF"] + mkt["RF"]
    bear = ((1 + mret).rolling(504).apply(np.prod, raw=True) - 1) < 0
    vol = mkt["Mkt-RF"].rolling(126).std()
    return (bear & (vol > vol.median())).astype(float).reindex(index).rename("bear_vol")


def saturation(index: pd.Index) -> pd.Series:
    ours = pd.read_parquet("data/processed/state_series.parquet").top_share
    return ours.reindex(index).dropna().reindex(index).rename("top_share")


BUILDERS = {"comomentum": comomentum, "momentum_gap": momentum_gap,
            "bear_vol": bear_vol, "ours_top_share_descriptive": saturation}

#: Which indicators read as a binary armed/quiet rather than as a quintile.
BINARY = {"bear_vol"}


# --------------------------------------------------------------------------------------
# Point-in-time state
# --------------------------------------------------------------------------------------

def state_at(name: str, asof: pd.Timestamp, index: pd.Index) -> dict:
    """The indicator's state at ``asof``, from data at or before ``asof`` only.

    Deliberately different from how exp-055 scored the ratio: that used a FULL-SAMPLE
    quintile cut, because it was measuring a historical frequency and had the whole sample in
    hand. A monitor does not, so the cut here is taken over history to date. The two answers
    can differ, and the block says which one it is showing.
    """
    s = BUILDERS[name](index).dropna()
    s = s.loc[s.index <= asof]
    if s.empty:
        raise RuntimeError("no observation at or before this date")
    row_date = s.index[-1]
    value = float(s.iloc[-1])
    stale = int((pd.Timestamp(asof) - row_date).days)
    out = {"indicator": name, "value": value, "as_of_row": str(row_date.date()),
           "n_history": int(len(s)), "stale_days": stale, "source": SOURCES.get(name, "")}
    if name in BINARY:
        out |= {"kind": "binary", "armed": bool(value > 0.5),
                "state": "ARMED" if value > 0.5 else "quiet"}
    else:
        pct = float((s <= value).mean())
        cut = float(s.quantile(TOP_Q))
        out |= {"kind": "quintile", "pctile": pct,
                "quintile": int(min(5, np.floor(pct * 5) + 1)),
                "armed": bool(value >= cut), "top_quintile_cut": cut,
                "state": ("TOP QUINTILE" if value >= cut
                          else f"quintile {int(min(5, np.floor(pct * 5) + 1))} of 5")}
    return out


def measured_ratios(path: Path = EVAL_PATH) -> dict[str, dict]:
    """The frequency ratios exp-055 measured, read from disk rather than remembered."""
    import json
    if not path.exists():
        raise RuntimeError(f"{path} absent; exp-055 has not been run in this checkout")
    return {r["indicator"]: r for r in json.loads(path.read_text()) if "indicator" in r}
