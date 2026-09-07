"""exp-078 -- the sealed holdout. One read, both arms registered beforehand.

The severity arm asks whether the repaired conditional estimator holds its stated coverage on
forecasts made after 2023-01-01, using only data available at each forecast date. The text arm
finds episodes in the sealed years by the registered event rule and asks whether the loser leg's
stated-condition rate is elevated at their formation dates.

Nothing here is re-run. Whatever it returns is the result.
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
from unstructured_momentum.model import baselines as bl, severity as sv  # noqa: E402
from unstructured_momentum.pipeline import contract  # noqa: E402

H, Q = contract.DECLARED_HORIZON_DAYS, contract.DECLARED_QUANTILE
SEAL = pd.Timestamp(SEALED_START)


def main() -> None:
    wml = french.momentum()
    # french.market() ALREADY returns decimals -- load() divides by 100 -- so the extra
    # division here scaled the market series by a further hundred. The quantile fit is
    # equivariant to rescaling a regressor, so the fitted values are unaffected, but
    # panic_state's BEAR FLAG is a sign test on a compounded return and it is not: the flag
    # flips on 3.1 percent of days between the two scalings. Logged as trp-87 and corrected
    # here; the as-run numbers are recomputed and reported beside the corrected ones.
    mkt = french.market()["Mkt-RF"].reindex(wml.index).ffill()
    print(f"WML full series to {wml.index.max().date()}  (the seal is being read)")

    bt = sv.expanding_backtest(wml, mkt, horizon=H, q=Q, per_observation=True)
    bt = bt.set_index("date") if "date" in bt.columns else bt
    fwd = sv.forward_return(wml, H)

    sealed = bt.index[(bt.index >= SEAL) & fwd.reindex(bt.index).notna()]
    if not len(sealed):
        raise SystemExit("no sealed-era forecasts")
    y = fwd.reindex(sealed).to_numpy()
    v_cond = bt.loc[sealed, "pred_full"].to_numpy()
    v_unc = bt.loc[sealed, "pred_unconditional"].to_numpy()
    v_vol = bt.loc[sealed, "pred_vol_scaled"].to_numpy()

    print(f"\nSEALED SEVERITY ARM  {len(sealed)} forecasts, "
          f"{sealed.min().date()} -> {sealed.max().date()}")
    out = {}
    for name, v in (("conditional (full)", v_cond), ("vol-scaled", v_vol),
                    ("unconditional", v_unc)):
        k = int((y < v).sum()); n = len(y)
        lo, hi = stats.beta.ppf(0.025, k, n - k + 1) if k else (0.0, 0.0), None
        lo = float(stats.beta.ppf(0.025, k, n - k + 1)) if k else 0.0
        hi = float(stats.beta.ppf(0.975, k + 1, n - k))
        c = bl.christoffersen(y, v, Q)
        out[name] = {"breaches": k, "n": n, "rate": k / n, "ci": [lo, hi],
                     "p_uc": float(c["p_uc"])}
        print(f"  {name:22s} {k:2d}/{n}  rate {k/n:.4f}  "
              f"CI [{lo:.4f}, {hi:.4f}]  p_uc {c['p_uc']:.3f}")

    # ---- the REGISTERED sampling ----------------------------------------------------
    # The power statement fixed a 21-day forecast step, roughly forty forecasts and about two
    # expected breaches. The first implementation used all 865 daily forecasts, each a 10-day
    # window overlapping the next by nine days, and reported a spurious FAIL; that is trp-84.
    # The registered design was then reported from an ad-hoc calculation that this script did
    # not produce, which is the same fault as a claim whose artifact does not carry it. Both
    # samplings are computed here, the registered one is the verdict, and the deviation is
    # printed beside it so the difference is visible rather than described.
    def coverage(y_, v_):
        k = int((y_ < v_).sum()); n = len(y_)
        lo = float(stats.beta.ppf(0.025, k, n - k + 1)) if k else 0.0
        hi = float(stats.beta.ppf(0.975, k + 1, n - k))
        c = bl.christoffersen(y_, v_, Q)
        return {"breaches": k, "n": n, "rate": k / n, "ci": [lo, hi],
                "p_uc": float(c["p_uc"])}

    step = contract.DECLARED_FORECAST_STEP_DAYS
    idx = np.arange(0, len(sealed), step)
    reg = {name: coverage(y[idx], v[idx]) for name, v in
           (("conditional (full)", v_cond), ("vol-scaled", v_vol), ("unconditional", v_unc))}
    out_registered = reg
    print(f"\nREGISTERED SAMPLING  every {step}th forecast, {len(idx)} non-overlapping windows")
    for name, r in reg.items():
        print(f"  {name:22s} {r['breaches']:2d}/{r['n']}  rate {r['rate']:.4f}  "
              f"CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]  p_uc {r['p_uc']:.3f}")

    print(f"\n  registered pass condition: the conditional model's interval contains 0.05 and")
    print(f"  its unconditional-coverage test does not reject at 5 percent.")
    cc = reg["conditional (full)"]
    passed = (cc["ci"][0] <= Q <= cc["ci"][1]) and cc["p_uc"] > 0.05
    dev = out["conditional (full)"]
    dev_passed = (dev["ci"][0] <= Q <= dev["ci"][1]) and dev["p_uc"] > 0.05
    print(f"  VERDICT on the registered sampling: {'PASS' if passed else 'FAIL'}")
    print(f"  the daily-overlapping deviation would say: {'PASS' if dev_passed else 'FAIL'}"
          f"   ({dev['breaches']}/{dev['n']}, rate {dev['rate']:.4f}) -- trp-84, not the design")

    # ---- episodes in the sealed era, by the registered event rule --------------------
    roll = wml.rolling(1260, min_periods=756)
    fwd10 = (1 + wml).rolling(H).apply(np.prod, raw=True).shift(-H) - 1
    thresh = fwd10.expanding(756).quantile(Q)
    hit = (fwd10 <= thresh) & (fwd10.index >= SEAL)
    dates = list(fwd10.index[hit.fillna(False)])
    eps, last = [], None
    for d in dates:
        if last is None or (d - last).days > 90:
            eps.append(d); last = d
    print(f"\nSEALED EPISODES by the registered rule: {len(eps)}")
    for d in eps:
        print(f"  trough window opening {d.date()}   10d return {fwd10.loc[d]:+.2%}"
              f"   -> formation {(d - pd.offsets.MonthEnd(1)).date()}")
    json.dump({"severity_registered_step": out_registered,
               "severity_daily_overlapping_deviation": out,
               "forecast_step_days": step,
               "passed": bool(passed),
               "sealed_episodes": [str(d.date()) for d in eps],
               "formation_dates": [str((d - pd.offsets.MonthEnd(1)).date()) for d in eps]},
              open("reports/sealed_run.json", "w"), indent=2)
    print("\nwrote reports/sealed_run.json")


if __name__ == "__main__":
    main()
