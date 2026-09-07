"""Gate 1 -- the quantitative spine and its baseline family.

Step 1 is a DATA SANITY CHECK, not a result: reproduce Barroso and Santa-Clara's constant
volatility momentum in its published form. If the roughly doubled Sharpe and the largely
removed negative skew do not appear, the data is wrong and nothing after this means anything.

Step 2 builds the baseline family in real-time implementable form. Every later result in this
project is reported as incremental over ALL THREE of these, never standalone:
  * constant-volatility scaling (Barroso-Santa-Clara)
  * the dynamic mean-variance rule (Daniel-Moskowitz)
  * a skewness-adjusted rule (Bianchi)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.config import SEALED_START  # noqa: E402
from unstructured_momentum.data import french  # noqa: E402

TRADING_DAYS = 252
#: Barroso and Santa-Clara target the unscaled strategy's own full-sample volatility.
VOL_WINDOW = 126


def stats_of(r: pd.Series, name: str) -> dict:
    r = r.dropna()
    ann = r.mean() * TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    # Skew and kurtosis on MONTHLY returns, as the papers report them: daily skew is dominated
    # by single-day outliers and is not the quantity the crash literature discusses.
    m = (1 + r).resample("ME").prod() - 1
    dd = (1 + r).cumprod()
    mdd = float((dd / dd.cummax() - 1).min())
    return {"name": name, "n_days": int(len(r)), "ann_return": float(ann), "ann_vol": float(vol),
            "sharpe": float(ann / vol), "monthly_skew": float(stats.skew(m.dropna())),
            "monthly_kurtosis": float(stats.kurtosis(m.dropna(), fisher=True)),
            "worst_month": float(m.min()), "max_drawdown": mdd}


def main() -> None:
    wml_full = french.momentum()
    wml = wml_full[wml_full.index.date < SEALED_START]
    print(f"WML {wml.index[0].date()} -> {wml.index[-1].date()}, n={len(wml):,}   "
          f"(sealed years withheld)\n")

    # ---- STEP 1: the published form, as a data sanity check ---------------------------
    # Realised variance from the prior 126 daily returns, applied with a one-day lag so the
    # scalar uses only information available before the return it scales.
    rv = wml.rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    target = wml.std() * np.sqrt(TRADING_DAYS)          # the strategy's own full-sample vol
    w = (target / rv).shift(1)
    scaled = (w * wml).dropna()
    plain = wml.reindex(scaled.index)

    a, b = stats_of(plain, "WML unscaled"), stats_of(scaled, "WML constant-vol (BSC)")
    print("STEP 1 -- BARROSO & SANTA-CLARA, PUBLISHED FORM (in-sample scalar, sanity check)")
    hdr = f"  {'':26s}{'Sharpe':>8}{'ann ret':>9}{'ann vol':>9}{'skew':>8}{'kurt':>8}{'worst m':>9}{'maxDD':>9}"
    print(hdr)
    for s in (a, b):
        print(f"  {s['name']:26s}{s['sharpe']:8.2f}{s['ann_return']:9.2%}{s['ann_vol']:9.2%}"
              f"{s['monthly_skew']:8.2f}{s['monthly_kurtosis']:8.2f}{s['worst_month']:9.2%}"
              f"{s['max_drawdown']:9.2%}")
    ratio = b["sharpe"] / a["sharpe"]
    skew_fix = b["monthly_skew"] - a["monthly_skew"]
    print(f"\n  Sharpe ratio multiple {ratio:.2f}x   skew {a['monthly_skew']:.2f} -> "
          f"{b['monthly_skew']:.2f} ({skew_fix:+.2f})")
    ok = ratio > 1.5 and b["monthly_skew"] > a["monthly_skew"]
    print(f"  SANITY CHECK: {'PASS' if ok else 'FAIL'} "
          f"(need Sharpe multiple > 1.5 and skew moving toward zero)")
    if not ok:
        print("\n  STOPPING. The published result does not reproduce, so the data is wrong "
              "and nothing downstream means anything.")
        json.dump({"sanity": "FAIL", "sharpe_multiple": ratio}, 
                  open("reports/gate1_spine.json", "w"), indent=2)
        raise SystemExit(1)

    # ---- STEP 2: the baseline family, real-time implementable ------------------------
    # Everything here uses only information available before the return it weights: the
    # variance forecast is lagged, and every fitted quantity is estimated on an EXPANDING
    # window. The published in-sample numbers above are the sanity check, not the bar.
    mkt = (french.market()["Mkt-RF"] / 100.0).reindex(wml.index).ffill()

    # Real-time constant-vol: target is the trailing realised vol to date, not the
    # full-sample figure, which is unknowable at the time.
    target_rt = (wml.expanding(TRADING_DAYS * 2).std() * np.sqrt(TRADING_DAYS)).shift(1)
    w_cv = (target_rt / rv.shift(1)).clip(upper=5.0)

    # Daniel-Moskowitz dynamic mean-variance. Expected return is forecast from the bear
    # state and the variance forecast, refit on an expanding window; the weight is the
    # mean-variance optimum mu/(2*lambda*sigma^2), scaled to the same average gross as the
    # constant-vol arm so the comparison is about TIMING and not about leverage.
    bear = ((1 + mkt).rolling(TRADING_DAYS * 2).apply(np.prod, raw=True) - 1 < 0).astype(float)
    var_f = (rv ** 2).shift(1)
    X = pd.DataFrame({"const": 1.0, "bear": bear.shift(1),
                      "bear_var": (bear.shift(1) * var_f)}).dropna()
    y = wml.reindex(X.index)
    mu = pd.Series(index=X.index, dtype=float)
    step, start = 21, TRADING_DAYS * 10
    beta = None
    for i in range(start, len(X)):
        if (i - start) % step == 0:
            Xi, yi = X.iloc[:i].to_numpy(), y.iloc[:i].to_numpy()
            beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        mu.iloc[i] = float(X.iloc[i].to_numpy() @ beta)
    w_dm = (mu / var_f.reindex(mu.index)).clip(lower=0.0, upper=None)

    # Bianchi skewness-adjusted: penalise gross when the trailing distribution is
    # left-skewed, which is the state the crash literature says precedes the tail.
    sk = wml.rolling(VOL_WINDOW).skew().shift(1)
    w_sk = (target_rt / rv.shift(1)) * (1.0 / (1.0 + np.exp(-2.0 * sk.clip(-3, 3))))

    def norm_to(w: pd.Series, ref: pd.Series) -> pd.Series:
        """Match average gross so the arms differ in TIMING, not in leverage.

        EXPANDING, not full-sample. The first version scaled by the full-sample mean gross,
        which is a number nobody had at the time. Sharpe is scale-invariant so it could not
        have moved the bar, but a bar with a full-sample constant in it is contaminated the
        same way a signal would be, and the argument that it happens not to matter is exactly
        the argument this project refuses elsewhere.
        """
        w = w.reindex(ref.index)
        scale = (ref.abs().expanding(252).mean() / w.abs().expanding(252).mean()).shift(1)
        return (w * scale).dropna()

    idx = w_cv.dropna().index.intersection(w_dm.dropna().index).intersection(w_sk.dropna().index)
    arms = {
        "constant-vol (real-time)": (w_cv.reindex(idx) * wml.reindex(idx)),
        "dynamic mean-variance (DM)": (norm_to(w_dm, w_cv.reindex(idx)) * wml.reindex(idx)),
        "skewness-adjusted (Bianchi)": (norm_to(w_sk, w_cv.reindex(idx)) * wml.reindex(idx)),
        "WML unscaled": wml.reindex(idx),
    }
    print("\nSTEP 2 -- BASELINE FAMILY, REAL-TIME IMPLEMENTABLE")
    print(f"  common span {idx[0].date()} -> {idx[-1].date()}, n={len(idx):,}")
    print(hdr)
    fam = []
    for nm, r in arms.items():
        st = stats_of(r, nm)
        fam.append(st)
        print(f"  {st['name']:26s}{st['sharpe']:8.2f}{st['ann_return']:9.2%}{st['ann_vol']:9.2%}"
              f"{st['monthly_skew']:8.2f}{st['monthly_kurtosis']:8.2f}{st['worst_month']:9.2%}"
              f"{st['max_drawdown']:9.2%}")
    print("\n  Every later result in this project is reported as incremental over ALL THREE,")
    print("  never standalone, and never against the in-sample numbers in step 1.")

    json.dump({"sanity": "PASS", "sharpe_multiple": ratio,
               "published_form": {"unscaled": a, "constant_vol": b},
               "baseline_family": fam,
               "family_span": [str(idx[0].date()), str(idx[-1].date())]},
              open("reports/gate1_spine.json", "w"), indent=2)
    print("\nwrote reports/gate1_spine.json")


if __name__ == "__main__":
    main()
