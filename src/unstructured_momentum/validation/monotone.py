"""exp-061 — the monotone-quantile protocol: is text a tail signal or a level signal?

The registration (project/experiments.yaml, id exp-061) is binding and this module implements
it and nothing else. One ladder, one contrast, one feature order, no tuning knobs exposed.

What the protocol does
----------------------
Baseline is the existing calibrated severity spine (`model/severity.py`) ALREADY augmented
with the mandatory price null: the CBOE SKEW index and the implied-minus-realised volatility
wedge. The augmented model is that spine plus ONE candidate feature and no other change.
Both produce out-of-sample conditional quantiles of the h-day forward momentum return by
expanding-window quantile regression, refit every 21 days, and both are scored by pinball
loss on every test day at each rung of the declared ladder tau in (0.50, 0.25, 0.10, 0.05,
0.01). Losses are paired daily and differenced.

PRIMARY test is the SHAPE, not any single rung:

    contrast = mean(skill at 0.05, 0.01) - mean(skill at 0.50, 0.25)     predicted > 0
    spearman = rank correlation of skill with tail depth across the five rungs, predicted > 0

A feature that improves the centre as much as the tail FAILS, however significant its
individual rungs are, because that pattern is a level signal and the theory is wrong.

Why skill is a RATIO and not a loss difference
----------------------------------------------
The registration says "skill" at each rung and leaves the normalisation open. It cannot be
left open, because raw pinball losses are not comparable across tau: for a correctly
specified forecaster the expected pinball loss is sigma * phi(z_tau), so the loss at tau=0.01
is an order of magnitude smaller than at tau=0.50 purely from the scoring rule's geometry.
Differencing raw losses across rungs would report "more tail skill" for any feature at all,
which would be a protocol that flatters itself -- the exact failure the second certificate
exists to catch. So skill at a rung is the RELATIVE reduction

    skill_tau = 1 - mean(L_augmented) / mean(L_baseline)

which is unitless and, under the same sigma * phi(z_tau) geometry, is EXACTLY FLAT across tau
for a pure location signal and rises into the tail for a scale signal. That is the property
the two certificates verify empirically rather than assume.

Inference
---------
Stationary block bootstrap (Politis-Romano), mean block twenty-one days, drawn ONCE per
replicate and applied to all five rungs together so that the contrast and the rank
correlation inherit the same resampling. Newey-West lag six on the paired daily loss
difference for the per-rung t-statistics, as declared. Benjamini-Hochberg across features
within each rung.

Tier discipline
---------------
This is the SEVERITY program, so the sealed-2023 boundary of `config.py` is in force
(docs/tier_rescope.md, point 1). Scoring uses DESIGN-tier dates only, and a decision date is
admitted only if its whole forward window is also DESIGN -- otherwise the outcome being
scored is partly a VALIDATE- or SEALED-tier return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..model import severity as sv

# --------------------------------------------------------------------------------------
# Frozen protocol constants. Registered; do not tune.
# --------------------------------------------------------------------------------------

#: The declared quantile ladder, centre to tail. Order is meaningful: it IS the tail depth.
LADDER: tuple[float, ...] = (0.50, 0.25, 0.10, 0.05, 0.01)

#: Rungs entering each side of the declared contrast.
TAIL_RUNGS: tuple[float, ...] = (0.05, 0.01)
CENTRE_RUNGS: tuple[float, ...] = (0.50, 0.25)

#: Headline severity horizon (config.HEADLINE_HORIZON / severity.HORIZONS).
HORIZON: int = 10

#: Mean block length for the stationary bootstrap, in trading days. Registered.
MEAN_BLOCK: int = 21

#: Newey-West lag. Registered as six.
NW_LAG: int = 6

#: Refit cadence of the expanding window, inherited from severity.expanding_backtest.
STEP: int = 21

#: Ladder item 1: the existing state-series text features, on their frozen definitions.
#: Mapped to the four constructs the registration names.
TEXT_FEATURES: dict[str, str] = {
    "top_share": "top-theme saturation",
    "eff": "effective rank (book)",
    "w_eff": "effective rank (winner leg)",
    "l_eff": "effective rank (loser leg)",
    "w_shared": "shared mass (winner leg)",
    "l_shared": "shared mass (loser leg)",
    "w_att_conc": "attention concentration (winner leg)",
    "l_att_conc": "attention concentration (loser leg)",
}

#: Present in the state series but NOT one of the four constructs the ladder names. Scored
#: and reported, but held OUT of the Benjamini-Hochberg family so it cannot dilute or
#: borrow from the declared family.
UNDECLARED_FEATURES: dict[str, str] = {"n_nodes": "corpus coverage (undeclared control)"}


# --------------------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------------------


def _cboe(name: str) -> pd.DataFrame:
    p = config.RAW / "cboe" / "index" / f"{name}.csv"
    df = pd.read_csv(p)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("DATE").sort_index()


def price_null(market: pd.Series) -> pd.DataFrame:
    """The mandatory price null: option-implied skew, and the implied-minus-realised wedge.

    SKEW is the market's own daily risk-neutral skewness -- free, published since 1990, and
    exactly the quantity a "text is a higher-moment signal" claim has to beat. The wedge is
    30-day implied volatility minus trailing realised volatility of the market over the same
    horizon, which is the variance-risk-premium term the same options market prices.

    Both are levels quoted at the close of the date they carry, and both are lagged by one
    day downstream in `severity.build_design` along with every other state variable.
    """
    skew = _cboe("SKEW")["SKEW"].rename("skew")
    vix = _cboe("VIX")["CLOSE"].rename("vix")
    rv = (market.rolling(21).std() * np.sqrt(252) * 100.0).rename("rv21")
    out = pd.concat([skew, vix, rv], axis=1)
    out["vrp_wedge"] = out["vix"] - out["rv21"]
    return out[["skew", "vrp_wedge"]].dropna()


def _design_tier_mask(index: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """True where the decision date AND its whole forward window are DESIGN tier.

    The second half is the part that is easy to skip and dishonest to skip: a forecast made
    on the last DESIGN day is scored against ten days of returns that the tier protocol has
    not opened.
    """
    tier = pd.Series([config.tier_of(d.date()) for d in calendar], index=calendar)
    is_design = (tier == config.Tier.DESIGN).astype(int)
    # window t+1 .. t+horizon must be all design
    fwd_ok = is_design.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    ok = (is_design == 1) & (fwd_ok == 1)
    return ok.reindex(index).fillna(False)


def assemble(*, horizon: int = HORIZON) -> pd.DataFrame:
    """The scoring frame: forward return, spine, price null, and every candidate feature.

    State variables are built on the FULL history so that the 126-day volatility window and
    the 504-day bear indicator are properly warmed at the start of the text sample; only the
    scored ROWS are restricted to DESIGN tier. Restricting the inputs instead would silently
    throw away the first two years of an already short sample.
    """
    from ..data import french

    mom = french.momentum()
    mkt = french.market()["Mkt-RF"]

    text = pd.read_parquet(config.PROCESSED / "state_series.parquet")
    extras = price_null(mkt).join(text, how="inner")

    design = sv.build_design(mom, mkt, extras)          # lags every state column by one day
    fwd = sv.forward_return(mom, horizon).rename("y")
    frame = pd.concat([fwd, design], axis=1).dropna()

    mask = _design_tier_mask(frame.index, mom.index, horizon)
    frame = frame.loc[mask.to_numpy()]
    frame.attrs["horizon"] = horizon
    return frame


SPINE_COLS: tuple[str, ...] = ("realised_vol", "bear", "mkt_var", "bear_x_var")
NULL_COLS: tuple[str, ...] = ("skew", "vrp_wedge")


# --------------------------------------------------------------------------------------
# Expanding-window scoring
# --------------------------------------------------------------------------------------


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score on TRAINING moments only.

    Not cosmetic. The design mixes n_nodes ~ 2e2, SKEW ~ 1.3e2 and attention concentration
    ~ 1e-2, and an ill-conditioned design is where `severity._fit_quantile` falls back
    SILENTLY to the unconditional quantile -- which would read as "the feature added
    nothing" and be indistinguishable from a real null. Training moments only, so nothing
    from the test block enters the fit.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def expanding_losses(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    taus: tuple[float, ...] = LADDER,
    min_train: int = 756,
    step: int = STEP,
    y: np.ndarray | None = None,
) -> tuple[pd.DatetimeIndex, dict[float, np.ndarray], int]:
    """Per-observation pinball loss of one specification, at every rung.

    Returns (test dates, {tau: loss array}, number of silent fallbacks). Reuses the spine's
    own `_fit_quantile` and `pinball_loss` so the fitted object is the same estimator the
    severity line publishes, not a second implementation of it.
    """
    yv = frame["y"].to_numpy() if y is None else np.asarray(y, dtype=float)
    X = frame[cols].to_numpy(dtype=float) if cols else np.zeros((len(frame), 0))
    n = len(frame)

    dates: list[pd.Timestamp] = []
    losses: dict[float, list[np.ndarray]] = {t: [] for t in taus}
    fallbacks = 0
    collected_dates = False

    for start in range(min_train, n - 1, step):
        tr = slice(0, start)
        te = slice(start, min(start + step, n))
        if te.stop <= te.start:
            continue
        Xtr_raw, Xte_raw = X[tr], X[te]
        if X.shape[1]:
            Xtr_s, Xte_s = _standardise(Xtr_raw, Xte_raw)
        else:
            Xtr_s, Xte_s = Xtr_raw, Xte_raw
        Xtr = np.column_stack([np.ones(Xtr_s.shape[0]), Xtr_s]) if X.shape[1] else np.ones((start, 1))
        Xte = np.column_stack([np.ones(Xte_s.shape[0]), Xte_s]) if X.shape[1] else np.ones((te.stop - te.start, 1))
        ytr, yte = yv[tr], yv[te]

        if not collected_dates:
            pass
        dates.extend(frame.index[te])

        for tau in taus:
            beta = sv._fit_quantile(Xtr, ytr, tau)
            if X.shape[1] and np.all(beta[1:] == 0.0):
                fallbacks += 1
            pred = Xte @ beta
            d = yte - pred
            losses[tau].append(np.maximum(tau * d, (tau - 1.0) * d))

    idx = pd.DatetimeIndex(dates)
    return idx, {t: np.concatenate(v) for t, v in losses.items()}, fallbacks


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def newey_west_t(x: np.ndarray, lag: int = NW_LAG) -> tuple[float, float]:
    """t-statistic and two-sided p for the mean of a serially dependent series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    e = x - m
    g0 = float(e @ e) / n
    var = g0
    for l in range(1, lag + 1):
        if l >= n:
            break
        gl = float(e[l:] @ e[:-l]) / n
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    var = max(var, 1e-300)
    se = np.sqrt(var / n)
    t = m / se if se > 0 else 0.0
    from scipy import stats

    return float(t), float(2.0 * stats.norm.sf(abs(t)))


def stationary_bootstrap_index(n: int, n_boot: int, mean_block: int, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, (n_boot, n).

    Geometric block lengths with mean `mean_block`, wrapped circularly. Chosen over a fixed
    block because the resampled series is then stationary, which matters when the statistic
    is a ratio of means rather than a mean.
    """
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_boot, n))
    newblock = rng.random((n_boot, n)) < p
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(newblock[:, t], starts[:, t], cont)
    return idx


def _skills(lb: dict[float, np.ndarray], la: dict[float, np.ndarray],
            taus: tuple[float, ...]) -> np.ndarray:
    return np.array([1.0 - la[t].mean() / lb[t].mean() for t in taus])


def _contrast(sk: np.ndarray, taus: tuple[float, ...]) -> float:
    ti = [taus.index(t) for t in TAIL_RUNGS]
    ci = [taus.index(t) for t in CENTRE_RUNGS]
    return float(sk[ti].mean() - sk[ci].mean())


def _spearman_depth(sk: np.ndarray) -> float:
    """Rank correlation of skill with tail depth. Ladder order IS increasing depth."""
    depth = np.arange(1, len(sk) + 1, dtype=float)
    r = pd.Series(sk).rank().to_numpy()
    if np.std(r) == 0:
        return 0.0
    return float(np.corrcoef(depth, r)[0, 1])


def score_pair(
    base: dict[float, np.ndarray],
    aug: dict[float, np.ndarray],
    *,
    taus: tuple[float, ...] = LADDER,
    n_boot: int = 2000,
    seed: int = 61,
) -> dict:
    """The whole verdict for one feature: five rungs, the contrast, the rank correlation."""
    rng = np.random.default_rng(seed)
    n = len(base[taus[0]])
    sk = _skills(base, aug, taus)
    contrast = _contrast(sk, taus)
    rho = _spearman_depth(sk)

    idx = stationary_bootstrap_index(n, n_boot, MEAN_BLOCK, rng)
    B = np.empty((n_boot, len(taus)))
    for j, t in enumerate(taus):
        B[:, j] = 1.0 - aug[t][idx].mean(axis=1) / base[t][idx].mean(axis=1)
    ti = [taus.index(t) for t in TAIL_RUNGS]
    ci = [taus.index(t) for t in CENTRE_RUNGS]
    Bc = B[:, ti].mean(axis=1) - B[:, ci].mean(axis=1)
    depth = np.arange(1, len(taus) + 1, dtype=float)
    Br = np.apply_along_axis(lambda r: _spearman_depth(r), 1, B)

    rungs = []
    for j, t in enumerate(taus):
        d = base[t] - aug[t]                      # positive => augmented is better
        tstat, p = newey_west_t(d, NW_LAG)
        t10, p10 = newey_west_t(d, HORIZON)       # robustness only; not the declared stat
        rungs.append({
            "tau": t,
            "skill": float(sk[j]),
            "skill_ci": [float(np.quantile(B[:, j], 0.025)), float(np.quantile(B[:, j], 0.975))],
            "mean_loss_base": float(base[t].mean()),
            "mean_loss_aug": float(aug[t].mean()),
            "mean_loss_diff": float(d.mean()),
            "nw6_t": tstat, "nw6_p": p,
            "nw10_t": t10, "nw10_p": p10,
        })

    def _p2(draws, stat):
        lo = float((draws <= 0).mean())
        return float(min(1.0, 2.0 * min(lo, 1.0 - lo)))

    return {
        "n_obs": int(n),
        "rungs": rungs,
        "contrast": float(contrast),
        "contrast_ci": [float(np.quantile(Bc, 0.025)), float(np.quantile(Bc, 0.975))],
        "contrast_p": _p2(Bc, contrast),
        "contrast_positive_frac": float((Bc > 0).mean()),
        "spearman_depth": float(rho),
        "spearman_ci": [float(np.quantile(Br, 0.025)), float(np.quantile(Br, 0.975))],
        "spearman_positive_frac": float((Br > 0).mean()),
        "monotone_pass": bool(contrast > 0 and float(np.quantile(Bc, 0.025)) > 0 and rho > 0),
    }


def benjamini_hochberg(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """BH within a rung, across the declared feature family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, passed = {}, 0
    for i, (k, p) in enumerate(items, start=1):
        thr = alpha * i / m
        if p <= thr:
            passed = i
        out[k] = {"p": float(p), "bh_threshold": float(thr), "rank": i}
    for i, (k, _) in enumerate(items, start=1):
        out[k]["survives_bh"] = i <= passed
    return out


# --------------------------------------------------------------------------------------
# Certificates: plant a shape, check the protocol reports that shape and no other
# --------------------------------------------------------------------------------------


def placebo_covariate(n: int, rng, rho: float = 0.9) -> np.ndarray:
    """A persistent standardised nuisance, independent of everything in the frame.

    AR(1) rather than white noise so the plant has the slow-moving character of a real state
    variable; an i.i.d. plant would be easier to detect than anything the ladder contains and
    would issue a certificate against the wrong shape.
    """
    e = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = e[0]
    s = np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + s * e[t]
    return (z - z.mean()) / z.std()


def plant_tail(y: np.ndarray, z: np.ndarray, k: float) -> np.ndarray:
    """A TAIL-ONLY signal: z moves the conditional SCALE, leaving the centre alone.

    y -> m + (y - m) * exp(k z - k^2/2), with m the sample median. The multiplier has unit
    mean, so the location is untouched; the 0.50 quantile sits at m where (y-m) = 0 and is
    therefore insensitive to z by construction, while the 0.01 quantile is roughly nine
    percent away from m and scales with the full multiplier. That is the shape the theory
    predicts, planted rather than hoped for.
    """
    m = float(np.median(y))
    return m + (y - m) * np.exp(k * z - 0.5 * k * k)


def plant_level(y: np.ndarray, z: np.ndarray, c: float) -> np.ndarray:
    """A LEVEL signal: z shifts every quantile by the same amount.

    y -> y + c * sd(y) * z. This is the plant that matters. A protocol which reports a
    positive monotone contrast here is worthless, because it would find tail shape in a
    signal that has none, and every text result it produced would be an artefact of the
    scoring rule rather than a fact about text.
    """
    return y + c * float(np.std(y)) * z
