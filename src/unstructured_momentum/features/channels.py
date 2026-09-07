"""The two-channel model: asymmetry and fragility.

The taxonomy this operationalises was validated at Stage 0: Aug 2007 is absent from the
momentum-only detector (5.97th pctile) and present in the joint channel (0.10th pctile),
while Sept 2019 is the mirror image (-6.03 sigma momentum, joint +0.59). Each channel is
blind to what the other catches, so a single-channel monitor is structurally incomplete.

**Asymmetry channel** — the Daniel-Moskowitz mechanism. After a market decline the
past-loser leg becomes high-beta and option-like, so WML carries a large conditional
negative beta and is destroyed by a rebound. Measured by the *difference* between
up-market and down-market leg betas, which is the quantity that actually creates the
optionality; the level of beta is not the risk, the asymmetry is.

**Fragility channel** — crowding. The key design choice is conditioning on **quiet market
days**. Joint factor stress on a violent market day is market beta; joint factor stress
while the market is calm is a positioning event, which is what Aug 2007 was. Without that
conditioning the fragility channel is a volatility proxy and adds nothing to Barroso.

Why the panic-state model needs this
------------------------------------
Christoffersen tests on the panic-state baseline give a breach rate of 0.028 against an
expected 0.050 (p_uc = 0.054): too conservative on average while being *under*-alarmed at
crowded rotations. Before Sept 2019 it predicted VaR5 of -3.48% against an unconditional
-4.86% because ``bear = 0``. It is wrong in both directions at different times, and the
fragility channel exists to cover the direction it misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import french
from ..factor import comovement, constructions

#: A "quiet" market day: absolute market return below this trailing percentile.
QUIET_PCTILE = 50.0


# --------------------------------------------------------------------------------------
# Asymmetry channel
# --------------------------------------------------------------------------------------


def leg_beta_asymmetry(weighting: str = "vw", window: int = 252) -> pd.DataFrame:
    """Up-market minus down-market beta, per leg and for the spread.

    Estimated on a rolling window by splitting on the sign of the market excess return.
    A *negative* ``wml_asymmetry`` means WML's beta is lower in up markets than down --
    the option-like short-leg signature that produces crash risk on a rebound.

    Uses only trailing data, so it is ex-ante estimable. Grundy-Martin's version used
    *realised future* betas and produced a strongly upward-biased, non-implementable
    backtest; the ex-ante version of that hedge failed outright. That is the canonical
    lookahead failure in this literature and the reason this is computed trailing-only.
    """
    lg = constructions.legs(weighting)
    out = pd.DataFrame(index=lg.index, dtype=float)

    up = lg["mkt"] > 0
    for name in ("winners", "losers"):
        b_up, b_dn = [], []
        for mask in (up, ~up):
            y = lg[name].where(mask)
            x = lg["mkt"].where(mask)
            cov = y.rolling(window, min_periods=60).cov(x)
            var = x.rolling(window, min_periods=60).var()
            (b_up if mask is up else b_dn).append(cov / var)
        out[f"{name}_beta_up"] = b_up[0]
        out[f"{name}_beta_dn"] = b_dn[0]
        out[f"{name}_asymmetry"] = out[f"{name}_beta_up"] - out[f"{name}_beta_dn"]

    out["wml_beta_up"] = out["winners_beta_up"] - out["losers_beta_up"]
    out["wml_beta_dn"] = out["winners_beta_dn"] - out["losers_beta_dn"]
    out["wml_asymmetry"] = out["wml_beta_up"] - out["wml_beta_dn"]
    return out


def asymmetry_channel(window: int = 252) -> pd.DataFrame:
    """State variables for the panic-rebound mechanism."""
    ff = french.market()
    mkt = ff["Mkt-RF"]
    wml = french.momentum()

    asym = leg_beta_asymmetry(window=window)
    cum2y = (1 + mkt).rolling(504).apply(np.prod, raw=True) - 1

    out = pd.DataFrame(index=wml.index)
    out["bear"] = (cum2y < 0).astype(float)
    out["mkt_var"] = mkt.rolling(126).var() * 252
    out["wml_vol"] = wml.rolling(126).std() * np.sqrt(252)
    out["wml_asymmetry"] = asym["wml_asymmetry"]
    out["loser_beta_dn"] = asym["losers_beta_dn"]
    # The interaction is the mechanism: a levered short leg matters when the market can
    # rebound hard, i.e. in a high-variance bear state.
    out["bear_x_var"] = out["bear"] * out["mkt_var"]
    out["bear_x_asym"] = out["bear"] * out["wml_asymmetry"]
    return out


# --------------------------------------------------------------------------------------
# Fragility channel
# --------------------------------------------------------------------------------------


def quiet_day_mask(market: pd.Series, window: int = 252, pctile: float = QUIET_PCTILE) -> pd.Series:
    """True where |market return| is below its trailing percentile."""
    absmkt = market.abs()
    thresh = absmkt.rolling(window, min_periods=60).quantile(pctile / 100.0)
    return (absmkt < thresh).rename("quiet")


def fragility_channel(window: int = 63) -> pd.DataFrame:
    """State variables for the crowding mechanism.

    ``joint_stress_quiet`` is the discriminating one: cross-factor stress accumulated only
    over calm market days. On violent days joint factor moves are market beta; on calm days
    they are positioning.
    """
    ff = french.market()
    mkt = ff["Mkt-RF"]
    wml = french.momentum()

    cm = comovement.panel()
    quiet = quiet_day_mask(mkt)

    out = pd.DataFrame(index=wml.index)
    out["joint_z"] = cm["joint_z"].reindex(wml.index)
    out["factor_dispersion"] = cm["factor_dispersion"].reindex(wml.index)
    out["factor_corr"] = cm["factor_avg_corr"].reindex(wml.index)

    # Joint stress restricted to quiet days, then averaged over the window.
    jq = out["joint_z"].where(quiet.reindex(out.index).fillna(False))
    out["joint_stress_quiet"] = jq.rolling(window, min_periods=15).mean()

    # Short-horizon own-autocorrelation: a crowded trade trends then snaps.
    out["wml_autocorr"] = wml.rolling(window, min_periods=30).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else np.nan, raw=False
    )

    # Eigenvalue concentration of the factor complex: how much of the long-short
    # complex's variance sits on one axis.
    fp = comovement.factor_panel().reindex(wml.index)
    out["eig_concentration"] = _rolling_pc1_share(fp, window=126)
    return out


def _rolling_pc1_share(panel: pd.DataFrame, window: int = 126) -> pd.Series:
    """Share of variance on the first principal component, rolling."""
    vals = []
    idx = panel.index
    arr = panel.to_numpy()
    for i in range(len(idx)):
        if i < window:
            vals.append(np.nan)
            continue
        w = arr[i - window : i]
        w = w[~np.isnan(w).any(axis=1)]
        if len(w) < window // 2:
            vals.append(np.nan)
            continue
        z = (w - w.mean(0)) / (w.std(0) + 1e-12)
        ev = np.linalg.svd(z, compute_uv=False) ** 2
        vals.append(float(ev[0] / ev.sum()))
    return pd.Series(vals, index=idx, name="eig_concentration")


# --------------------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------------------


def two_channel_state(window: int = 63) -> pd.DataFrame:
    """Both channels aligned on one index."""
    return asymmetry_channel().join(fragility_channel(window), how="outer")


def separation_table(episodes: pd.DataFrame, state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Gate (b): do the mechanism classes separate in two-channel space?

    Samples each channel's indicators at the episode trough and reports them by mechanism
    label. If ``A_panic_rebound`` and ``J_joint`` do not separate, the taxonomy dies here
    and the project reports a one-channel system — which is a finding, not a failure.
    """
    st = state if state is not None else two_channel_state()
    cols = ["bear", "wml_asymmetry", "loser_beta_dn", "joint_z", "joint_stress_quiet",
            "eig_concentration"]
    cols = [c for c in cols if c in st.columns]

    rows = []
    for _, e in episodes.iterrows():
        t = pd.Timestamp(e["trough"])
        sub = st.loc[:t]
        if not len(sub):
            continue
        rec = {"key": e["key"], "mechanism": e["mechanism"], "trough": t.date()}
        rec.update({c: float(sub[c].iloc[-1]) for c in cols})
        rows.append(rec)
    return pd.DataFrame(rows)
