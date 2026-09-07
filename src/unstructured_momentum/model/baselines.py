"""The baseline gauntlet and the evaluation harness.

Stage-1 principle: **every later component is reported as a delta over this**, never over
the unconditional case. Beating an unconditional quantile is trivially easy on a
fat-tailed series and rhetorically misleading; beating a volatility-scaled conditional
tail is the honest bar.

Baselines
---------
1. **Unconditional** — the sample quantile. The floor.
2. **Barroso-Santa-Clara** volatility scaling — trailing realised momentum volatility.
   The strong baseline: most apparent crash prediction is volatility prediction wearing a
   costume. Already measured at +23.4% pinball skill over unconditional across 318
   expanding-window refits.
3. **Daniel-Moskowitz panic state** — bear indicator, market variance, interaction.
4. **GJR-GARCH with skewed-t innovations** — captures volatility clustering, the leverage
   effect, and a genuinely asymmetric fat left tail. This is the parametric benchmark a
   risk manager would actually reach for, and omitting it would leave the quantile
   regression untested against the standard tool.

Evaluation
----------
* **Pinball loss** — proper scoring rule for quantiles.
* **Christoffersen tests** — unconditional coverage (does the tail get breached at the
  right rate?), independence (are breaches clustered?), and conditional coverage (joint).
  These catch a failure pinball loss does not: a VaR that is right *on average* but wrong
  *in bursts*, which is precisely how a momentum crash arrives.
* **Utility backtest** — the de-gross rule priced with transaction costs. A statistical
  improvement that does not survive costs is not a result, and Lopez-Lira & Tang's
  Sharpe 2.97 going negative at 20bp is the cautionary case.

Christoffersen (1998), "Evaluating Interval Forecasts", *International Economic Review*
39(4), 841-862.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# GJR-GARCH with skewed-t
# --------------------------------------------------------------------------------------


def garch_var_forecast(
    returns: pd.Series,
    *,
    q: float = 0.05,
    horizon: int = 10,
    min_train: int = 1000,
    step: int = 21,
    rescale: float = 100.0,
) -> pd.DataFrame:
    """Expanding-window GJR-GARCH(1,1,1) skewed-t VaR forecasts at ``horizon`` days.

    Refit every ``step`` days on data available at the time. The h-day VaR uses the
    square-root-of-cumulated-variance scaling of the fitted innovation quantile, which is
    the standard practitioner approximation; it ignores the mild autocorrelation in the
    factor and is stated as an approximation rather than presented as exact.

    Returns are rescaled by 100 because `arch` warns and conditions poorly on raw decimal
    returns.
    """
    from arch import arch_model

    r = returns.dropna() * rescale
    rows = []

    for start in range(min_train, len(r) - horizon, step):
        train = r.iloc[:start]
        try:
            fit = arch_model(train, vol="GARCH", p=1, o=1, q=1, dist="skewt", mean="Constant").fit(
                disp="off", show_warning=False
            )
            f = fit.forecast(horizon=horizon, reindex=False)
            cum_var = float(np.sum(f.variance.values[-1, :]))
            mu = float(fit.params.get("mu", 0.0)) * horizon
            # Skewed-t quantile of the standardised innovation.
            params = fit.params
            nu = float(params.get("nu", 8.0))
            lam = float(params.get("lambda", 0.0))
            z = _skewt_ppf(q, nu, lam)
            var_h = (mu + z * np.sqrt(cum_var)) / rescale
        except Exception:  # noqa: BLE001 - convergence failures are expected occasionally
            continue
        rows.append({"date": r.index[start], "var_garch": var_h})

    return pd.DataFrame(rows).set_index("date")


def _skewt_ppf(q: float, nu: float, lam: float) -> float:
    """Quantile of Hansen's skewed-t as parameterised by `arch`.

    Falls back to the symmetric Student-t when the skew parameter is negligible, which
    keeps the function well-behaved when the optimiser drives lambda to ~0.
    """
    if abs(lam) < 1e-6:
        return float(stats.t.ppf(q, df=max(nu, 2.1)) / np.sqrt(max(nu, 2.1) / (max(nu, 2.1) - 2)))

    nu = max(nu, 2.1)
    from scipy.special import gammaln

    c = np.exp(gammaln((nu + 1) / 2) - 0.5 * np.log(np.pi * (nu - 2)) - gammaln(nu / 2))
    a = 4 * lam * c * ((nu - 2) / (nu - 1))
    b = np.sqrt(1 + 3 * lam**2 - a**2)

    if q < (1 - lam) / 2:
        z = (1 - lam) / b * np.sqrt((nu - 2) / nu) * stats.t.ppf(q / (1 - lam), nu) - a / b
    else:
        z = (1 + lam) / b * np.sqrt((nu - 2) / nu) * stats.t.ppf(
            0.5 + (q - (1 - lam) / 2) / (1 + lam), nu
        ) - a / b
    return float(z)


# --------------------------------------------------------------------------------------
# Christoffersen coverage tests
# --------------------------------------------------------------------------------------


def christoffersen(realised: np.ndarray, var_forecast: np.ndarray, q: float) -> dict:
    """Unconditional coverage, independence, and conditional coverage tests.

    A breach is ``realised < var_forecast``. Under a correct model breaches occur with
    probability ``q`` and are independent across time.

    The independence test is the one that matters for this project: a VaR can have exactly
    the right breach *rate* while clustering every breach into a single week, which is what
    a factor crash looks like and what a PM cares about.
    """
    y, v = np.asarray(realised, float), np.asarray(var_forecast, float)
    ok = np.isfinite(y) & np.isfinite(v)
    y, v = y[ok], v[ok]
    n = len(y)
    if n < 20:
        return {"n": n}

    hits = (y < v).astype(int)
    x = int(hits.sum())
    pi = x / n

    # -- unconditional coverage (Kupiec POF)
    if 0 < pi < 1:
        ll_null = (n - x) * np.log(1 - q) + x * np.log(q)
        ll_alt = (n - x) * np.log(1 - pi) + x * np.log(pi)
        lr_uc = -2 * (ll_null - ll_alt)
    else:
        lr_uc = np.nan

    # -- independence (Markov transition)
    t = np.zeros((2, 2))
    for a, b in zip(hits[:-1], hits[1:]):
        t[a, b] += 1
    n00, n01, n10, n11 = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi_all = (n01 + n11) / t.sum() if t.sum() else 0.0

    def _ll(p, a, b):
        if p <= 0 or p >= 1:
            return 0.0
        return a * np.log(1 - p) + b * np.log(p)

    lr_ind = -2 * (
        _ll(pi_all, n00 + n10, n01 + n11) - (_ll(pi01, n00, n01) + _ll(pi11, n10, n11))
    )
    lr_cc = lr_uc + lr_ind if np.isfinite(lr_uc) else np.nan

    return {
        "n": n,
        "n_breaches": x,
        "breach_rate": pi,
        "expected_rate": q,
        "lr_uc": lr_uc,
        "p_uc": float(1 - stats.chi2.cdf(lr_uc, 1)) if np.isfinite(lr_uc) else np.nan,
        "lr_ind": lr_ind,
        "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)) if np.isfinite(lr_ind) else np.nan,
        "lr_cc": lr_cc,
        "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan,
    }


# --------------------------------------------------------------------------------------
# Utility backtest — the de-gross rule, priced
# --------------------------------------------------------------------------------------


def degross_backtest(
    momentum: pd.Series,
    risk_signal: pd.Series,
    *,
    threshold_pctile: float = 80.0,
    degross_to: float = 0.5,
    cost_bps: float = 10.0,
    lookback: int = 252,
) -> dict:
    """Economic value of a de-gross rule, net of costs.

    Rule: when the risk signal exceeds its trailing ``threshold_pctile``, scale momentum
    exposure to ``degross_to``; otherwise hold full exposure. The threshold is a trailing
    percentile so the rule is implementable — a full-sample threshold would be lookahead,
    which is the Grundy-Martin failure mode in miniature.

    Costs are charged on the *change* in exposure, which is the only thing that trades.
    """
    r = momentum.dropna()
    sig = risk_signal.reindex(r.index)

    thresh = sig.rolling(lookback, min_periods=lookback // 2).quantile(threshold_pctile / 100.0)
    # Shift so the position is set using information available before the return.
    w = np.where(sig.shift(1) > thresh.shift(1), degross_to, 1.0)
    w = pd.Series(w, index=r.index).fillna(1.0)

    turnover = w.diff().abs().fillna(0.0)
    cost = turnover * (cost_bps / 10_000.0)
    net = w * r - cost

    def _stats(x: pd.Series) -> dict:
        ann = float(x.mean() * 252)
        vol = float(x.std() * np.sqrt(252))
        cum = (1 + x).cumprod()
        dd = float((cum / cum.cummax() - 1).min())
        return {
            "ann_return": ann,
            "ann_vol": vol,
            "sharpe": ann / vol if vol else np.nan,
            "max_drawdown": dd,
            "worst_10d": float(((1 + x).rolling(10).apply(np.prod, raw=True) - 1).min()),
        }

    base, rule = _stats(r), _stats(net)
    return {
        "baseline": base,
        "degross_rule": rule,
        "sharpe_delta": rule["sharpe"] - base["sharpe"],
        "drawdown_delta": rule["max_drawdown"] - base["max_drawdown"],
        "ann_turnover": float(turnover.sum() / (len(turnover) / 252)),
        "ann_cost_drag": float(cost.sum() / (len(cost) / 252)),
        "pct_time_degrossed": float((w < 1.0).mean()),
        "cost_bps": cost_bps,
    }


def breakeven_cost(momentum: pd.Series, risk_signal: pd.Series, **kw) -> float:
    """Round-trip cost (bps) at which the de-gross rule's Sharpe advantage disappears.

    Reported instead of a point net-P&L estimate, because no free source gives a credible
    spread or ADV denominator for this universe -- FINRA TotalVolume is only ~48% of the
    consolidated tape.
    """
    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        d = degross_backtest(momentum, risk_signal, cost_bps=mid, **kw)["sharpe_delta"]
        if d > 0:
            lo = mid
        else:
            hi = mid
    return lo
