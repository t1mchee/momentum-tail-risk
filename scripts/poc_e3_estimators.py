"""exp-086 -- a negative control on the scaling gain, and the head-to-head the design skipped.

Registered at commit 7588dfb, before this file existed.

ARM A. The tail experiment's one clean positive result -- volatility scaling cuts pinball loss by
14 percent -- had no negative control. Every refutation in this register carries a planted effect
showing the instrument could see what it failed to find; the positive carried nothing equivalent.
So the volatility series is SHUFFLED across time, which preserves its distribution exactly and
destroys its alignment with returns. A scaled forecast built on it carries no conditioning
information, and if the gain survives that, the gain is arithmetic and the result is withdrawn.

ARM B. The design specifies a tail model with no fitted parameters and dropped a fitted quantile
regression on trailing volatility and the bear-state terms. Nothing tested whether that was right,
and the example page still prints the fitted model's number while the validation rests on the
parameter-free one. This scores them on the same dates with the same machinery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unstructured_momentum.config import SEALED_START  # noqa: E402
from unstructured_momentum.data import french  # noqa: E402
from unstructured_momentum.model import severity as sv  # noqa: E402

import poc_e1_tail as E1  # noqa: E402

OUT = Path("reports/poc")
N_SHUFFLE = 200
SEED = 20260904


# --------------------------------------------------------------------------------------
# Arm A: does the gain survive a volatility series that cannot know anything?
# --------------------------------------------------------------------------------------


def arm_a(d: pd.DataFrame, f: pd.DataFrame, rng) -> dict:
    """Rebuild the scaled quantile on a shuffled volatility series, many times."""
    y = f["realised"].to_numpy()
    l_unc = E1.pinball(y, f["var_uncond"].to_numpy())
    real = float(E1.pinball(y, f["var_scaled"].to_numpy()).sum() / l_unc.sum())

    train_all = d.dropna(subset=["fwd", "vol"])
    ratios = []
    for _ in range(N_SHUFFLE):
        # One permutation of the whole volatility series, reused for training and for today.
        perm = rng.permutation(train_all["vol"].to_numpy())
        shuffled = pd.Series(perm, index=train_all.index)
        preds = []
        for t in f.index:
            pos = train_all.index.get_indexer([t], method="ffill")[0]
            tr = train_all.iloc[: max(pos - E1.H + 1, 0)]
            if len(tr) < E1.MIN_TRAIN:
                preds.append(np.nan); continue
            sv_tr = shuffled.iloc[: len(tr)]
            z = tr["fwd"].to_numpy() / sv_tr.to_numpy()
            preds.append(float(np.quantile(z, E1.Q) * shuffled.loc[t]))
        p = np.asarray(preds)
        ok = np.isfinite(p)
        ratios.append(float(E1.pinball(y[ok], p[ok]).sum() / l_unc[ok].sum()))
    r = np.asarray(ratios)
    return {
        "real_ratio": real, "n_shuffles": N_SHUFFLE,
        "shuffled_median": float(np.median(r)),
        "shuffled_band_5_95": [float(np.percentile(r, 5)), float(np.percentile(r, 95))],
        "share_of_shuffles_beating_the_real_gain": float((r <= real).mean()),
        "gain_survives_shuffling": bool(np.median(r) < 0.95),
    }


# --------------------------------------------------------------------------------------
# Arm B: the fitted regression against the parameter-free quantile
# --------------------------------------------------------------------------------------


def arm_b(f: pd.DataFrame) -> dict:
    """Score the fitted model on the same formation dates as the parameter-free one."""
    wml = french.momentum()
    mkt = french.market()["Mkt-RF"].reindex(wml.index).ffill()
    bt = sv.expanding_backtest(wml, mkt, horizon=E1.H, q=E1.Q, per_observation=True)
    bt = bt.set_index("date") if "date" in bt.columns else bt

    idx = f.index.intersection(bt.index)
    g = f.loc[idx].copy()
    g["var_fitted"] = bt.loc[idx, "pred_full"].to_numpy()
    g = g.dropna(subset=["var_fitted"])
    y = g["realised"].to_numpy()

    rng = np.random.default_rng(SEED)
    ratio = E1.block_ratio(E1.pinball(y, g["var_fitted"].to_numpy()),
                           E1.pinball(y, g["var_scaled"].to_numpy()), 12, rng)
    out = {"n_common_dates": int(len(g)), "fitted_vs_parameter_free": ratio,
           "coverage": {}, "conditional_coverage": {}}
    g2 = g.rename(columns={"var_fitted": "var_fit"})
    for name, col in (("parameter_free", "var_scaled"), ("fitted", "var_fit")):
        out["coverage"][name] = E1.kupiec(y, g2[col].to_numpy())
    tmp = g2.copy()
    tmp["tercile"] = pd.qcut(tmp["sigma"], 3, labels=["low vol", "mid vol", "high vol"])
    for name, col in (("parameter_free", "var_scaled"), ("fitted", "var_fit")):
        br = tmp["realised"] < tmp[col]
        by = {str(k): float(br[tmp["tercile"] == k].mean()) for k in tmp["tercile"].cat.categories}
        out["conditional_coverage"][name] = {
            "by_vol_tercile": by,
            "spread": float(max(by.values()) - min(by.values()))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = E1.build()
    f = E1.forecasts(d)
    f = f[f.index < pd.Timestamp(SEALED_START)]
    rng = np.random.default_rng(SEED)

    print(f"{len(f)} formation dates, {f.index.min().date()} to {f.index.max().date()}\n")
    print(f"ARM A  shuffling the volatility series {N_SHUFFLE} times ...", flush=True)
    a = arm_a(d, f, rng)
    print(f"  real scaled gain          ratio {a['real_ratio']:.4f}")
    print(f"  shuffled volatility       median {a['shuffled_median']:.4f}  "
          f"5-95 band [{a['shuffled_band_5_95'][0]:.4f}, {a['shuffled_band_5_95'][1]:.4f}]")
    print(f"  shuffles matching the real gain: "
          f"{a['share_of_shuffles_beating_the_real_gain']:.1%}")
    print(f"  -> the gain is {'AN ARTIFACT' if a['gain_survives_shuffling'] else 'INFORMATIONAL'}"
          f": shuffling {'preserves' if a['gain_survives_shuffling'] else 'destroys'} it")

    print(f"\nARM B  the fitted regression against the parameter-free quantile ...", flush=True)
    b = arm_b(f)
    r = b["fitted_vs_parameter_free"]
    print(f"  {b['n_common_dates']} common dates")
    print(f"  pinball ratio fitted/parameter-free {r['ratio']:.4f}  "
          f"90% CI [{r['ci90'][0]:.4f}, {r['ci90'][1]:.4f}]")
    print(f"  -> {'the fitted model wins' if r['ratio'] < 1 and r['ci90'][1] < 1 else ('the parameter-free wins' if r['ci90'][0] > 1 else 'cannot be told apart on loss')}")
    print(f"\n  {'estimator':<18}{'breaches':>10}{'rate':>9}{'Kupiec p':>10}"
          f"{'low vol':>10}{'high vol':>10}{'spread':>9}")
    for k in ("parameter_free", "fitted"):
        c = b["coverage"][k]; cc = b["conditional_coverage"][k]
        print(f"    {k:<16}{c['breaches']:>10}{c['rate']:>9.4f}{c['p']:>10.3f}"
              f"{cc['by_vol_tercile']['low vol']:>10.4f}{cc['by_vol_tercile']['high vol']:>10.4f}"
              f"{cc['spread']:>9.4f}")

    (OUT / "e3_estimators.json").write_text(json.dumps({"arm_a": a, "arm_b": b}, indent=2))
    print(f"\nwrote {OUT}/e3_estimators.json")


if __name__ == "__main__":
    main()
