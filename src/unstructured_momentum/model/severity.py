"""Conditional severity of momentum reversals — the project's headline quantitative output.

Why severity and not probability
--------------------------------
Required element 6 of the brief asks for "risk horizon, estimated probability **and/or**
severity of reversal/crash risk". The "and/or" is load-bearing here.

*Probability* of a rare event is not answerable by this project. There are ~24 modern
episodes collapsing to 8-17 independent regimes, and against a ~15% base rate that
requires roughly a 2.2x lift to reach significance — a floor no dataset or method removes.

*Severity* is answerable, because it is estimated from **every day in the sample**, not
from the events. "Given today's state, how bad is the left tail of the next h days?" has
thousands of observations behind it. So the system reports a conditional tail, states the
horizon, and declines the probability with a stated reason. That is faithful to the brief
and statistically honest at the same time.

Method
------
Quantile regression of the h-day forward momentum return on lagged state variables gives
a conditional VaR directly, without assuming a distribution — which matters because the
whole subject is a fat left tail that a Gaussian would understate precisely when it counts.
Conditional expected shortfall is then the average realised return below the fitted
quantile, so ES inherits the quantile fit rather than a second parametric assumption.

Evaluation uses **pinball loss**, the proper scoring rule for quantiles. A proper rule is
essential here: the naive alternative (counting how often realised returns breach the
predicted VaR) is minimised by a model that simply predicts a very wide tail, and would
reward exactly the kind of uninformative caution this project is trying to avoid.

Baselines, in the order they must be beaten
-------------------------------------------
1. **Unconditional** — the sample quantile. Anything that cannot beat this is noise.
2. **Volatility-scaled** (Barroso-Santa-Clara) — trailing realised momentum volatility.
   This is the strong baseline: most of what looks like crash prediction is volatility
   prediction wearing a costume, and a state variable that beats (1) but not (2) has
   added nothing beyond a vol forecast.
3. **Panic-state** (Daniel-Moskowitz) — bear indicator, market variance, and their
   interaction. The published, free, ex-ante-estimable benchmark.

Anything the project's own state variables add is reported as an increment over (3), never
over (1), because beating the unconditional quantile is trivially easy and rhetorically
misleading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Forecast horizons in trading days. 10d is the headline; 5 and 21 expose the term
#: structure, which distinguishes an imminent unwind from a slow bleed.
HORIZONS: tuple[int, ...] = (5, 10, 21)

#: Left-tail quantiles to fit. 0.05 is the reported headline.
QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.10)

#: The registered crash threshold, FROZEN as a level rather than recomputed per call.
#:
#: Anchored to the unconditional distribution instead of to a round number, so the base rate
#: printed beside a crash probability is exact by construction rather than approximately right:
#: 5.00 percent and 1.00 percent of DESIGN-tier 10-day momentum returns fall at or below these,
#: because that is how they were defined.
#:
#: Computed once on 25,288 ten-day forward returns from 1926-11-03 to 2022-12-15, which is every
#: DESIGN-tier observation. Nothing from 2023 onward entered it.
#:
#: One number is worth stating plainly because it is milder than the informal "8 percent class"
#: this band is sometimes described by: the 5th percentile of a 10-day momentum return is only
#: -4.3 percent. An 8-percent-class move sits near the 1st percentile, not the 5th.
CRASH_THRESHOLD_10D = {0.05: -0.0432, 0.01: -0.0933}
CRASH_THRESHOLD_VINTAGE = "DESIGN tier through 2022-12-15, n=25,288, frozen 2026-08-26"


def forward_return(returns: pd.Series, horizon: int) -> pd.Series:
    """Cumulative return over the NEXT ``horizon`` days, aligned to the decision date.

    Shifted so that the value at date t is what happens *after* t. Getting this backwards
    is the single easiest way to manufacture a spectacular and entirely fake result.
    """
    fwd = (1.0 + returns).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return fwd.shift(-horizon).rename(f"fwd_{horizon}d")


def realised_vol(returns: pd.Series, window: int = 126) -> pd.Series:
    """Trailing annualised volatility — the Barroso-Santa-Clara state variable."""
    return (returns.rolling(window).std() * np.sqrt(252)).rename("realised_vol")


def panic_state(market: pd.Series, window_bear: int = 504, window_var: int = 126) -> pd.DataFrame:
    """Daniel-Moskowitz panic-state variables, computable from free data alone."""
    cum2y = (1.0 + market).rolling(window_bear).apply(np.prod, raw=True) - 1.0
    bear = (cum2y < 0).astype(float).rename("bear")
    mvar = (market.rolling(window_var).var() * 252).rename("mkt_var")
    out = pd.concat([bear, mvar], axis=1)
    out["bear_x_var"] = out["bear"] * out["mkt_var"]
    return out


def build_design(
    momentum: pd.Series,
    market: pd.Series,
    extra_state: pd.DataFrame | None = None,
    *,
    lag: int = 1,
) -> pd.DataFrame:
    """Assemble lagged state variables aligned to a decision date.

    Every state column is lagged by ``lag`` days. The forward return is *not* — it is
    already shifted into the future by `forward_return`. So a row reads: "standing here,
    knowing only these values, what does the next h days look like?"
    """
    state = pd.concat([realised_vol(momentum), panic_state(market)], axis=1)
    if extra_state is not None and len(extra_state):
        state = state.join(extra_state, how="left")
    return state.shift(lag)


# --------------------------------------------------------------------------------------
# Quantile fitting
# --------------------------------------------------------------------------------------


def _fit_quantile(X: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """Quantile regression via statsmodels, falling back to the unconditional quantile.

    The fallback matters: quantile regression can fail to converge on collinear or
    near-degenerate designs, and silently returning a garbage coefficient vector would be
    worse than returning an honest constant.
    """
    import statsmodels.api as sm

    try:
        res = sm.QuantReg(y, X).fit(q=q, max_iter=5000)
        beta = np.asarray(res.params, dtype=float)
        if not np.all(np.isfinite(beta)):
            raise ValueError("non-finite coefficients")
        return beta
    except Exception:  # noqa: BLE001
        beta = np.zeros(X.shape[1])
        beta[0] = float(np.quantile(y, q))
        return beta


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Proper scoring rule for a quantile forecast. Lower is better."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def expanding_backtest(
    momentum: pd.Series,
    market: pd.Series,
    *,
    horizon: int = 10,
    q: float = 0.05,
    extra_state: pd.DataFrame | None = None,
    min_train: int = 1000,
    step: int = 21,
    per_observation: bool = False,
) -> pd.DataFrame:
    """Expanding-window out-of-sample conditional VaR, versus three baselines.

    Refits every ``step`` days on data available at the time, so every prediction uses
    only its own past. This is the design Daniel & Moskowitz use to answer the
    data-snooping objection, and it is the only version of this exercise worth reporting:
    an in-sample quantile fit on a fat-tailed series will always look excellent.

    Overlapping forward windows make consecutive observations dependent, so the reported
    losses are point estimates and any inference on them needs a block bootstrap.

    ``per_observation=True`` returns one row per test observation -- date, realised forward
    return, and each spec's prediction -- instead of the block summary. Calibration needs
    per-observation predictions: a block's mean loss cannot be conditioned on state, and
    Christoffersen tests need the breach sequence itself. The default is unchanged, and
    `blocks_from_observations` reconstructs the block frame so the two paths can be compared
    rather than trusted.
    """
    fwd = forward_return(momentum, horizon)
    design = build_design(momentum, market, extra_state)

    df = pd.concat([fwd.rename("y"), design], axis=1).dropna()
    if len(df) < min_train + step:
        raise ValueError(f"need > {min_train + step} usable rows, have {len(df)}")

    state_cols = [c for c in df.columns if c != "y"]
    dm_cols = ["bear", "mkt_var", "bear_x_var"]
    vol_cols = ["realised_vol"]

    specs = {
        "unconditional": [],
        "vol_scaled": vol_cols,
        "panic_state": dm_cols,
        "full": state_cols,
    }

    rows, obs = [], []
    for start in range(min_train, len(df) - 1, step):
        train, test = df.iloc[:start], df.iloc[start : start + step]
        if not len(test):
            continue
        ytr = train["y"].to_numpy()

        rec = {"date": test.index[0], "n_train": len(train), "n_test": len(test)}
        per = {"block": test.index[0], "date": test.index, "y": test["y"].to_numpy(),
               "n_train": len(train)}
        for name, cols in specs.items():
            Xtr = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in cols])
            Xte = np.column_stack([np.ones(len(test))] + [test[c].to_numpy() for c in cols])
            beta = _fit_quantile(Xtr, ytr, q)
            pred = Xte @ beta
            rec[f"loss_{name}"] = pinball_loss(test["y"].to_numpy(), pred, q)
            rec[f"var_{name}"] = float(pred[0])
            per[f"pred_{name}"] = pred
        rows.append(rec)
        obs.append(pd.DataFrame(per))

    block = pd.DataFrame(rows).set_index("date")
    if not per_observation:
        return block
    o = pd.concat(obs, ignore_index=True)
    o.attrs["q"] = q
    o.attrs["horizon"] = horizon
    return o


def blocks_from_observations(obs: pd.DataFrame, q: float | None = None) -> pd.DataFrame:
    """Collapse per-observation predictions back to the block frame.

    Exists so the refactor can be CHECKED rather than believed. The published losses are means
    over blocks of a within-block mean pinball loss, which is not the same number as a mean over
    observations whenever blocks differ in length -- so the aggregation is reproduced exactly
    here, and `expanding_backtest` is asserted to return this frame bit-for-bit.
    """
    q = obs.attrs.get("q") if q is None else q
    specs = [c[len("pred_"):] for c in obs.columns if c.startswith("pred_")]
    rows = []
    for b, g in obs.groupby("block", sort=True):
        rec = {"date": b, "n_train": int(g["n_train"].iloc[0]), "n_test": len(g)}
        for name in specs:
            pred = g[f"pred_{name}"].to_numpy()
            rec[f"loss_{name}"] = pinball_loss(g["y"].to_numpy(), pred, q)
            rec[f"var_{name}"] = float(pred[0])
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Mean pinball loss per spec, and skill relative to the two meaningful baselines."""
    losses = {c.replace("loss_", ""): bt[c].mean() for c in bt.columns if c.startswith("loss_")}
    base_u, base_v = losses.get("unconditional"), losses.get("vol_scaled")
    out = pd.DataFrame({"mean_pinball_loss": pd.Series(losses)})
    if base_u:
        out["skill_vs_unconditional"] = 1.0 - out["mean_pinball_loss"] / base_u
    if base_v:
        out["skill_vs_vol_scaled"] = 1.0 - out["mean_pinball_loss"] / base_v
    return out.sort_values("mean_pinball_loss")


# --------------------------------------------------------------------------------------
# Point-in-time severity for the PM card
# --------------------------------------------------------------------------------------


def current_severity(
    momentum: pd.Series,
    market: pd.Series,
    *,
    as_of: pd.Timestamp | str | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    q: float = 0.05,
    extra_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Conditional VaR and ES at each horizon, fitted only on data available at ``as_of``.

    ES is the mean realised return below the fitted quantile *in the training sample*,
    which keeps it tied to observed tail behaviour rather than to a distributional
    assumption that the tail itself violates.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else momentum.index.max()
    rows = []

    for h in horizons:
        fwd = forward_return(momentum, h)
        design = build_design(momentum, market, extra_state)
        df = pd.concat([fwd.rename("y"), design], axis=1)

        train = df.loc[:as_of].dropna()
        current = design.loc[:as_of].dropna().tail(1)
        if len(train) < 250 or not len(current):
            continue

        cols = [c for c in design.columns if c in train.columns]
        Xtr = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in cols])
        beta = _fit_quantile(Xtr, train["y"].to_numpy(), q)
        x_now = np.concatenate([[1.0], current[cols].to_numpy().ravel()])

        var_cond = float(x_now @ beta)
        fitted = Xtr @ beta

        # Conditional ES, conditioned on TODAY like the VaR beside it.
        #
        # This used to be the mean realised return below the fitted quantile across the whole
        # training sample -- an average exceedance that does not depend on today's state at
        # all. Printed next to a VaR that does, it produced ES less severe than VaR whenever
        # today was more extreme than the sample average, contradicting the page's own line
        # that ES "is always the worse number". At 2020-10-31 every row was inverted: 10-day
        # VaR -0.0947 against ES -0.0586.
        #
        # The fix keeps the estimator's shape and makes it conditional: estimate the tail's
        # MULTIPLICATIVE severity in-sample -- how much worse an exceedance is than the
        # quantile it breached -- and apply that ratio to today's VaR. The ratio is >= 1 by
        # construction on the rows it averages, so ES can no longer be milder than VaR.
        y_tr = train["y"].to_numpy()
        breach = y_tr <= fitted
        if breach.sum() >= 20 and np.all(fitted[breach] < 0):
            ratio = float(np.mean(y_tr[breach] / fitted[breach]))
            es_cond = var_cond * ratio
        else:
            es_cond = np.nan

        uncond_var = float(np.quantile(train["y"], q))
        uncond_tail = train["y"][train["y"] <= uncond_var]

        rows.append(
            {
                "horizon_days": h,
                "quantile": q,
                "var_conditional": var_cond,
                "es_conditional": es_cond,
                "var_unconditional": uncond_var,
                "es_unconditional": float(uncond_tail.mean()) if len(uncond_tail) else np.nan,
                "var_pctile_of_history": float((fitted < var_cond).mean() * 100),
                "n_train": len(train),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Attribution over the model's own terms
# --------------------------------------------------------------------------------------


def shapley_over_terms(momentum: pd.Series, market: pd.Series, as_of: pd.Timestamp, *,
                       horizon: int = 10, q: float = 0.05,
                       extra_state: pd.DataFrame | None = None,
                       min_train: int = 1000) -> pd.DataFrame:
    """Each term's Shapley contribution to the fitted quantile at one date.

    EXHAUSTIVE over subsets rather than sampled. The design carries a handful of terms, so the
    2^k refits are affordable and the number is exact -- a sampled Shapley would put Monte Carlo
    error into a decomposition whose whole purpose is to add up.

    The baseline is the intercept-only fit, so the contributions plus the baseline equal the full
    model's quantile exactly. That identity is asserted by the caller rather than assumed.
    """
    from itertools import combinations
    from math import factorial

    design = build_design(momentum, market, extra_state)
    fwd = forward_return(momentum, horizon)
    df = pd.concat([fwd.rename("y"), design], axis=1).dropna()
    train = df.loc[:as_of]
    if len(train) < min_train:
        raise ValueError(f"need {min_train} training rows at {as_of.date()}, have {len(train)}")
    cur = design.loc[:as_of].dropna().tail(1)
    if cur.empty:
        raise ValueError(f"no design row at or before {as_of.date()}")
    terms = [c for c in design.columns if c in train.columns]
    ytr = train["y"].to_numpy()

    cache: dict[tuple, float] = {}

    def value(sub: tuple) -> float:
        if sub not in cache:
            Xtr = np.column_stack([np.ones(len(train))] + [train[c].to_numpy() for c in sub])
            beta = _fit_quantile(Xtr, ytr, q)
            x = np.concatenate([[1.0], cur[list(sub)].to_numpy().ravel()])
            cache[sub] = float(x @ beta)
        return cache[sub]

    n = len(terms)
    base = value(())
    full = value(tuple(terms))
    rows = []
    for i, t in enumerate(terms):
        others = [x for x in terms if x != t]
        phi = 0.0
        for k in range(len(others) + 1):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for sub in combinations(others, k):
                phi += w * (value(tuple(sorted(sub + (t,)))) - value(tuple(sorted(sub))))
        rows.append({"term": t, "shapley": phi, "value_now": float(cur[t].iloc[0])})
    out = pd.DataFrame(rows).sort_values("shapley")
    out.attrs.update({"baseline": base, "full": full, "n_subsets": len(cache),
                      "reconstruction_error": abs(base + out["shapley"].sum() - full)})
    return out


# --------------------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------------------
# Ten registered claims cite this module's reproduce command. Until now it had no entry
# point, so `python -m unstructured_momentum.model.severity --backtest` imported, printed
# nothing and exited 0 -- which reads as success to anything checking a return code. A
# reviewer found the same fault across most of the registry. See trp-76 and trp-78.

def main(argv=None) -> int:
    import argparse

    from ..config import SEALED_START
    from ..data import french

    p = argparse.ArgumentParser(prog="severity")
    p.add_argument("--backtest", action="store_true",
                   help="expanding-window out-of-sample backtest against three baselines")
    p.add_argument("--audit", action="store_true",
                   help="per-spec skill table, the comparison that keeps losing to vol")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--quantile", type=float, default=0.05)
    p.add_argument("--sealed", action="store_true",
                   help="deliberately include the sealed years; not for routine use")
    a = p.parse_args(argv)
    if not (a.backtest or a.audit):
        p.error("nothing asked for: pass --backtest or --audit")

    mom = french.momentum()
    # french.market() already returns decimals; dividing again scales the series by a further
    # hundred and flips the bear-state flag on 3.1 percent of days. See trp-87.
    mkt = french.market()["Mkt-RF"].reindex(mom.index).ffill()

    # The first version of this entry point read the full series, which runs past the tier
    # seal into the years reserved for a one-shot out-of-sample test. That is the breach this
    # repo has already logged once. The seal is applied here rather than left to the caller,
    # because a generator cited by ten claims must not depend on every caller remembering.
    if not a.sealed:
        mom = mom[mom.index.date < SEALED_START]
        mkt = mkt.reindex(mom.index)
    else:
        print("!! READING SEALED DATA -- this consumes the one-shot out-of-sample test")

    print(f"WML {mom.index[0].date()} -> {mom.index[-1].date()}, n={len(mom):,}"
          f"   tier: {'SEALED INCLUDED' if a.sealed else 'DESIGN+VALIDATE, sealed withheld'}")
    print(f"horizon {a.horizon}d, quantile {a.quantile:.0%}")

    bt = expanding_backtest(mom, mkt, horizon=a.horizon, q=a.quantile)
    summary = summarise_backtest(bt)
    print(f"\nblocks: {len(bt)}")
    print(summary.to_string(float_format=lambda v: f"{v: .6f}"))

    if a.audit:
        best = summary["mean_pinball_loss"].idxmin()
        print(f"\nlowest pinball loss: {best}")
        print("A spec only earns its seat by beating vol_scaled, not by beating "
              "unconditional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
