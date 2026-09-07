"""exp-089 -- the tail model as designed, on the reconstructed book.

Registered at commit fef25f9a4, before this file existed.

Every tail result so far is on French's published factor, because no daily series for the
reconstructed book existed. It does now, and the book is a different object: roughly 32 percent
annualised against the published factor's 13, because value-weighting inside a decile is
extremely concentrated. An estimator validated on a 13 percent series has not been validated on
a 32 percent one, and the page is about the book.

The conditioning state changes too. Bear state was used before because it is computable on a
century of published returns; the design specifies a CALENDAR state -- a formation is "on" when
sessions t+1 to t+10 contain an FOMC decision day, or an earnings day on which reporting
companies hold at least ten percent of the book's gross weight. That needs book weights, so it
needed this series.

Four lines, each estimated only on starts whose ten-day target is already realised.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unstructured_momentum.config import SEALED_START  # noqa: E402
from unstructured_momentum.data import eightk  # noqa: E402

import poc_e1_tail as E1  # noqa: E402

OUT = Path("reports/poc")
H, Q = 10, 0.05
VOL_WINDOW = 126
MIN_TRAIN = 250
EARN_WEIGHT = 0.10          # the design's threshold on reporting companies' gross weight
LEVEL = -0.0432             # the registered event level, for the probability line
SEED = 20260904


def book() -> pd.DataFrame:
    b = pd.read_parquet("data/processed/book_returns.parquet")
    d = pd.DataFrame({"r": b["r"], "formation": b["formation"]})
    d["fwd"] = (1 + d["r"]).rolling(H).apply(np.prod, raw=True).shift(-H) - 1
    d["vol"] = d["r"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    return d[d["vol"] > 0]


def calendar_state(d: pd.DataFrame, formations: list) -> tuple[dict, dict]:
    """FOMC days, and earnings days heavy enough in the book, over sessions t+1..t+10."""
    fomc = set(pd.to_datetime(pd.read_parquet("data/raw/fed/fomc_statement_dates.parquet")["date"]).dt.date)
    ix = eightk.load_index()
    ix = ix[ix["items"].fillna("").str.contains("2.02")]
    acc = pd.to_datetime(ix["accepted_at"], errors="coerce", utc=True).dt.tz_localize(None)
    earn = pd.DataFrame({"ticker": ix["ticker"].values, "day": acc.dt.normalize().values}).dropna()

    legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    W = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        if int(f.split("_")[-1][:4]) < 2013:
            continue
        x = pd.read_parquet(f, columns=["as_of", "ticker", "price", "quantity"])
        x = x[x["price"].notna() & (x["price"] > 0) & x["quantity"].notna()]
        x["mv"] = x["quantity"] * x["price"]
        W[f] = x.pivot_table(index="as_of", columns="ticker", values="mv", aggfunc="last")
    Wm = pd.concat(W.values()).sort_index()
    Wm = Wm[~Wm.index.duplicated(keep="last")]

    state, norms = {}, {}
    idx = d.index
    for t in formations:
        win = idx[(idx > t)][:H]
        if not len(win):
            continue
        on_fomc = any(x.date() in fomc for x in win)

        prior = Wm.index[Wm.index <= t]
        gross_hit, wnorm, gk = 0.0, np.nan, np.nan
        if len(prior) and t in legs:
            w0 = Wm.loc[prior[-1]]
            names, sign = [], []
            for side, s in (("winners", 1.0), ("losers", -1.0)):
                for n in legs[t][side]:
                    names.append(n); sign.append(s)
            v = w0.reindex(names).astype(float).fillna(0.0)
            gross = v.sum()
            if gross > 0:
                wv = (v / gross).to_numpy()
                wnorm = float(np.sqrt((wv ** 2).sum()))     # correction 17: the weight norm
                gk = float(np.abs(wv).sum())
                rep = set(earn.loc[earn["day"].isin(win), "ticker"])
                gross_hit = float(v[[n in rep for n in names]].sum() / gross)
        state[t] = bool(on_fomc or gross_hit >= EARN_WEIGHT)
        norms[t] = {"weight_l2_norm": wnorm, "gross_weight": gk,
                    "earnings_weight_in_window": gross_hit, "fomc_in_window": bool(on_fomc)}
    return state, norms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = book()
    formations = sorted(d["formation"].unique())
    print(f"book series {d.index.min().date()} to {d.index.max().date()}, "
          f"{len(d):,} days, {len(formations)} formations")
    state, norms = calendar_state(d, formations)
    print(f"calendar state ON at {sum(state.values())} of {len(state)} formations "
          f"(earnings coverage begins 2018; FOMC covers the whole span)\n")

    rows = []
    for t in formations:
        if t not in d.index or t not in state:
            near = d.index[d.index <= t]
            if not len(near):
                continue
            t_row = near[-1]
        else:
            t_row = t
        if not np.isfinite(d["vol"].get(t_row, np.nan)) or not np.isfinite(d["fwd"].get(t_row, np.nan)):
            continue
        pos = d.index.get_loc(t_row)
        train = d.iloc[: max(pos - H + 1, 0)].dropna(subset=["fwd", "vol"])
        if len(train) < MIN_TRAIN:
            continue
        sig = float(d["vol"].loc[t_row]); st = state.get(t, False)
        z = train["fwd"] / train["vol"]
        tr_state = pd.Series([state.get(f, False) for f in train["formation"]], index=train.index)
        terc = pd.qcut(train["vol"], 3, labels=False, duplicates="drop")
        my_terc = int((train["vol"] < sig).mean() * 3)
        my_terc = min(my_terc, 2)

        same_state = z[tr_state == st]
        same_terc = z[terc == my_terc] if terc is not None else z
        rows.append({
            "formation": t, "sigma": sig, "calendar_state": st,
            "var_uncond": float(np.quantile(train["fwd"], Q)),
            "var_scaled": float(np.quantile(z, Q) * sig),
            "var_state": float(np.quantile(same_state, Q) * sig) if len(same_state) >= 100
                          else float(np.quantile(z, Q) * sig),
            "var_terc": float(np.quantile(same_terc, Q) * sig) if len(same_terc) >= 100
                         else float(np.quantile(z, Q) * sig),
            "es_scaled": float(z[z <= np.quantile(z, Q)].mean() * sig),
            # The design's probability line is at the BOOK's own unconditional 5% quantile.
            # -432bp is French's long-run reference and is printed beside it, labelled, not as
            # the threshold.
            "prob_below_level": float((z * sig <= LEVEL).mean()),
            "prob_below_own_var": float(
                (z * sig <= float(np.quantile(train["fwd"], Q))).mean()),
            "realised": float(d["fwd"].loc[t_row]),
            **{k: norms.get(t, {}).get(k) for k in
               ("weight_l2_norm", "earnings_weight_in_window", "fomc_in_window")},
        })
    f = pd.DataFrame(rows).set_index("formation")
    f = f[f.index < pd.Timestamp(SEALED_START)]
    f.to_csv(OUT / "e5_book_tail.csv")
    y = f["realised"].to_numpy()
    rng = np.random.default_rng(SEED)
    lines = {k: f[f"var_{k}"].to_numpy() for k in ("uncond", "scaled", "state", "terc")}
    loss = {k: E1.pinball(y, v) for k, v in lines.items()}

    res = {"n_months": int(len(f)), "first": str(f.index.min().date()),
           "last": str(f.index.max().date()),
           "calendar_state_on": int(f["calendar_state"].sum()),
           "ann_vol_of_book": float(d["r"].std() * np.sqrt(252)),
           "median_weight_l2_norm": float(f["weight_l2_norm"].median()),
           "coverage": {k: E1.kupiec(y, v) for k, v in lines.items()},
           "mean_pinball": {k: float(v.mean()) for k, v in loss.items()},
           "mean_es_scaled": float(f["es_scaled"].mean()),
           "mean_prob_below_registered_level": float(f["prob_below_level"].mean()),
           "mean_prob_below_own_var": float(f["prob_below_own_var"].mean()),
           "vs_uncond": {k: E1.block_ratio(loss[k], loss["uncond"], 6, rng)
                         for k in ("scaled", "state", "terc")},
           "state_vs_scaled": E1.block_ratio(loss["state"], loss["scaled"], 6, rng),
           "terc_vs_scaled": E1.block_ratio(loss["terc"], loss["scaled"], 6, rng)}
    cc = {}
    tmp = f.copy(); tmp["tv"] = pd.qcut(tmp["sigma"], 3, labels=["low", "mid", "high"])
    for k in lines:
        br = tmp["realised"] < tmp[f"var_{k}"]
        cc[k] = {"by_vol_tercile": {str(x): float(br[tmp["tv"] == x].mean())
                                    for x in tmp["tv"].cat.categories},
                 "by_calendar_state": {"on": float(br[tmp.calendar_state].mean()),
                                       "off": float(br[~tmp.calendar_state].mean())}}
    res["conditional_coverage"] = cc
    (OUT / "e5_book_tail_summary.json").write_text(json.dumps(res, indent=2, default=str))

    print(f"{len(f)} scored formations, {res['first']} to {res['last']}; "
          f"book vol {res['ann_vol_of_book']:.1%}\n")
    print(f"{'line':<10}{'breaches':>10}{'rate':>8}{'Kupiec p':>10}{'pinball':>12}{'vs uncond':>12}")
    for k in ("uncond", "scaled", "state", "terc"):
        c = res["coverage"][k]
        v = res["vs_uncond"].get(k)
        print(f"  {k:<8}{c['breaches']:>10}{c['rate']:>8.4f}{c['p']:>10.3f}"
              f"{res['mean_pinball'][k]:>12.6f}"
              f"{(f'{v[chr(114)+chr(97)+chr(116)+chr(105)+chr(111)]:.4f}' if v else '--'):>12}")
    print(f"\n  calendar state vs scaled: {res['state_vs_scaled']['ratio']:.4f} "
          f"[{res['state_vs_scaled']['ci90'][0]:.4f}, {res['state_vs_scaled']['ci90'][1]:.4f}]")
    print(f"  vol tercile   vs scaled: {res['terc_vs_scaled']['ratio']:.4f} "
          f"[{res['terc_vs_scaled']['ci90'][0]:.4f}, {res['terc_vs_scaled']['ci90'][1]:.4f}]")
    print(f"\n  mean 10d ES (scaled) {res['mean_es_scaled']:+.2%}; "
          f"mean P(10d return <= {LEVEL:+.2%}) = {res['mean_prob_below_registered_level']:.1%}")
    print(f"  median ||w||_2 {res['median_weight_l2_norm']:.4f} "
          f"(equal weight over 470 names would be {1/np.sqrt(470):.4f})")
    print(f"\nwrote {OUT}/e5_book_tail.csv and e5_book_tail_summary.json")


if __name__ == "__main__":
    main()
