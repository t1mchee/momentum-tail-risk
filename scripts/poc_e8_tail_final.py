"""E1's final artifact: the page's tail line, and the three conditioning variables that failed.

Tim's decision: the page prints the PARAMETER-FREE volatility-scaled quantile -- today's
volatility times the 5 percent quantile of past scaled returns -- with no conditioning state. The
bear state, the calendar state and the volatility tercile are reported as three tested-and-not-
used conditioning variables, each with its ratio and interval. The catalyst-window increase line
is dropped; the calendar state stays as page FIELDS (catalyst dates, window catalyst days) and
not as a model input.

This consolidates what four registered experiments already produced rather than recomputing any
of it, so the page cannot disagree with the register:

    exp-084  the two lines and the three-block-length ratio, on the published factor
    exp-086  the shuffled-volatility control, and the fitted-regression tie
    exp-089  coverage on the reconstructed book, the calendar state, the volatility tercile

It also prints concentration PER LEG per date, which nothing did: exp-089's weight norm is over
the whole book, and the sentence the memo will use is about the long leg alone.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("reports/poc")


def per_leg_concentration() -> pd.DataFrame:
    """Effective N and the weight norm for each leg on each formation date.

    Value-weighted, as the spec has it and as Tim's decision confirms: market value from
    quantity times price, never the published weight column, which is 0.00 for the smallest
    half of the universe.
    """
    legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    W = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2013:
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "price", "quantity"])
        d = d[d["price"].notna() & (d["price"] > 0) & d["quantity"].notna()]
        d["mv"] = d["quantity"] * d["price"]
        W[f] = d.pivot_table(index="as_of", columns="ticker", values="mv", aggfunc="last")
    Wm = pd.concat(W.values()).sort_index()
    Wm = Wm[~Wm.index.duplicated(keep="last")]

    rows = []
    for t in sorted(legs):
        prior = Wm.index[Wm.index <= t]
        if not len(prior) or (t - prior[-1]).days > 7:
            continue
        w0 = Wm.loc[prior[-1]]
        r = {"formation": t}
        for side, key in (("winners", "long"), ("losers", "short")):
            v = w0.reindex(legs[t][side]).astype(float)
            v = v[v > 0]
            if len(v) < 20:
                continue
            v = v / v.sum()
            r[f"{key}_n_names"] = int(len(v))
            r[f"{key}_effective_n"] = float(1.0 / (v ** 2).sum())
            r[f"{key}_weight_l2"] = float(np.sqrt((v ** 2).sum()))
            r[f"{key}_top1"] = float(v.max())
            r[f"{key}_top5"] = float(v.nlargest(5).sum())
        if "long_effective_n" in r:
            rows.append(r)
    return pd.DataFrame(rows).set_index("formation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e084 = json.loads((OUT / "e1_summary.json").read_text())
    e086 = json.loads((OUT / "e3_estimators.json").read_text())
    e089 = json.loads((OUT / "e5_book_tail_summary.json").read_text())

    conc = per_leg_concentration()
    conc.to_csv(OUT / "e8_concentration_by_leg.csv")

    art = {
        "page_line": {
            "estimator": "parameter-free volatility-scaled empirical quantile",
            "definition": ("today's trailing volatility times the 5 percent quantile of past "
                           "ten-day returns each divided by the volatility at their own start"),
            "conditioning_state": None,
            "decided_by": "Tim, 2026-09-04",
        },
        "the_two_lines_on_the_book": {
            "n_formations": e089["n_months"],
            "span": [e089["first"], e089["last"]],
            "book_ann_vol": e089["ann_vol_of_book"],
            "unconditional": e089["coverage"]["uncond"],
            "scaled": e089["coverage"]["scaled"],
            "reading": ("on the reconstructed book the UNCONDITIONAL line fails coverage and the "
                        "scaled line passes; on the published factor it was the other way round, "
                        "so the over-breach was a property of that series and not of the estimator"),
        },
        "pinball_ratio": {
            "on_the_book": e089["vs_uncond"]["scaled"],
            "on_the_published_factor": e084["primary_scaled_vs_uncond"]["12"],
            # The INTERVAL, not the ratio. `block_ratio` returns la.sum()/lb.sum() as its
            # point estimate, which has no block-length dependence at all, so reporting the
            # ratio at three block lengths printed the same number three times to sixteen
            # decimals and called it a sensitivity check. What block length actually governs
            # is the width of the bootstrap interval, and that is what varies here.
            "block_lengths_published_factor": {
                b: {"ci90": e084["primary_scaled_vs_uncond"][b]["ci90"],
                    "excludes_one": e084["primary_scaled_vs_uncond"][b]["excludes_one"]}
                for b in ("6", "12", "24")},
            "block_length_note": (
                "the point ratio is invariant to block length by construction; only the "
                "interval moves, so only the interval is reported"),
        },
        "shuffle_control": {
            "real_ratio": e086["arm_a"]["real_ratio"],
            "shuffled_median": e086["arm_a"]["shuffled_median"],
            "shuffled_band_5_95": e086["arm_a"]["shuffled_band_5_95"],
            "share_of_shuffles_matching": e086["arm_a"]["share_of_shuffles_beating_the_real_gain"],
            "reading": ("shuffled, the forecast is WORSE than the unconditional quantile it is "
                        "compared against; the gain is the volatility series knowing something "
                        "about the date it scales, not the transformation"),
        },
        "fitted_regression_tie": {
            "ratio": e086["arm_b"]["fitted_vs_parameter_free"]["ratio"],
            "ci90": e086["arm_b"]["fitted_vs_parameter_free"]["ci90"],
            "n_dates": e086["arm_b"]["n_common_dates"],
            "reading": ("the fitted quantile regression and the parameter-free line cannot be "
                        "told apart on loss, so the simplification is vindicated on parsimony "
                        "and not on performance"),
        },
        "conditioning_variables_tested_and_not_used": [
            {"name": "bear state (Daniel-Moskowitz)", "tested_on": "published factor",
             "ratio": e084["secondary_cond_vs_scaled"]["12"]["ratio"],
             "ci90": e084["secondary_cond_vs_scaled"]["12"]["ci90"],
             "verdict": "refuted: the interval excludes one on the wrong side, it makes the forecast worse"},
            {"name": "calendar state (FOMC and earnings-heavy days)", "tested_on": "reconstructed book",
             "ratio": e089["state_vs_scaled"]["ratio"], "ci90": e089["state_vs_scaled"]["ci90"],
             "verdict": "no improvement: the interval contains one",
             "caveat": ("effectively FOMC-only here; on at 22 of 81 formations, 18 from FOMC and "
                        "4 from earnings, median earnings weight in window 0.29 percent against a "
                        "10 percent threshold, because the 8-K corpus covers 1,579 of about 2,900 "
                        "tickers and begins in 2018")},
            {"name": "volatility tercile", "tested_on": "reconstructed book",
             "ratio": e089["terc_vs_scaled"]["ratio"], "ci90": e089["terc_vs_scaled"]["ci90"],
             "verdict": "no improvement: the interval contains one"},
        ],
        "conditional_coverage_on_the_book": e089["conditional_coverage"],
        "expected_shortfall_and_probability": {
            "mean_10d_es_scaled": e089["mean_es_scaled"],
            "mean_prob_below_registered_level": e089["mean_prob_below_registered_level"],
        },
        "concentration_per_leg": {
            "weighting": "value-weighted within each leg, market value from quantity times price",
            "median_long_effective_n": float(conc["long_effective_n"].median()),
            "median_short_effective_n": float(conc["short_effective_n"].median()),
            "median_long_weight_l2": float(conc["long_weight_l2"].median()),
            "median_short_weight_l2": float(conc["short_weight_l2"].median()),
            "median_long_top1": float(conc["long_top1"].median()),
            "median_long_names": int(conc["long_n_names"].median()),
            "latest": {k: (float(v) if isinstance(v, float) else int(v))
                       for k, v in conc.iloc[-1].items()},
            "latest_date": str(conc.index[-1].date()),
        },
    }
    (OUT / "e8_tail_final.json").write_text(json.dumps(art, indent=2, default=str))

    t = art["the_two_lines_on_the_book"]
    print(f"THE PAGE LINE: {art['page_line']['estimator']}, no conditioning state\n")
    print(f"on the reconstructed book, {t['n_formations']} formations, "
          f"{t['span'][0]} to {t['span'][1]}, book vol {t['book_ann_vol']:.1%}")
    for k in ("unconditional", "scaled"):
        c = t[k]
        print(f"  {k:<16}{c['breaches']:>3} of {c['n']}   {c['rate']:>7.2%}   "
              f"Kupiec p {c['p']:.3f}   {'PASSES' if c['p']>0.10 else 'FAILS'}")
    print(f"\npinball ratio, scaled vs unconditional: book "
          f"{art['pinball_ratio']['on_the_book']['ratio']:.4f}, published factor "
          f"{art['pinball_ratio']['on_the_published_factor']['ratio']:.4f}")
    s = art["shuffle_control"]
    print(f"shuffle control: real {s['real_ratio']:.4f} vs shuffled median {s['shuffled_median']:.4f}"
          f"; {s['share_of_shuffles_matching']:.1%} of shuffles matched")
    f = art["fitted_regression_tie"]
    print(f"fitted vs parameter-free: {f['ratio']:.4f} [{f['ci90'][0]:.4f}, {f['ci90'][1]:.4f}] "
          f"-- cannot be told apart\n")
    print("three conditioning variables, tested and not used:")
    for c in art["conditioning_variables_tested_and_not_used"]:
        print(f"  {c['name']:<42}{c['ratio']:>8.4f}  [{c['ci90'][0]:.4f}, {c['ci90'][1]:.4f}]")
    cp = art["concentration_per_leg"]
    print(f"\nconcentration, value-weighted (median across {len(conc)} formations):")
    print(f"  long leg  {cp['median_long_names']} names, effective N "
          f"{cp['median_long_effective_n']:.1f}, largest {cp['median_long_top1']:.1%}, "
          f"||w||2 {cp['median_long_weight_l2']:.4f}")
    print(f"  short leg effective N {cp['median_short_effective_n']:.1f}, "
          f"||w||2 {cp['median_short_weight_l2']:.4f}")
    print(f"\nwrote {OUT}/e8_tail_final.json and e8_concentration_by_leg.csv")


if __name__ == "__main__":
    main()
