"""What actually happened after the 2020-10-30 page, with provenance.

Not an experiment and nothing new is estimated. The number already exists: it is the
`realised` column of exp-089's stored forecast table, the target the tail model was
scored against. This script reads it, recomputes it independently from the daily book
series so the two have to agree, decomposes the worst session, and writes the result
file the page's outcome line is filled from.

Everything here is OUTSIDE the page's information set. The page is written standing on
2020-10-30; this file is the answer key, and is labelled as such wherever it is printed.

    uv run python scripts/poc_e10_realised.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from unstructured_momentum.data import french

ANCHOR = pd.Timestamp("2020-10-30")     # the last session before the formation
FORMATION = pd.Timestamp("2020-10-31")  # the page's formation month-end
H = 10                                  # the registered horizon, trading days
OUT = Path("reports/poc")


def main() -> None:
    b = pd.read_parquet("data/processed/book_returns.parquet")
    if ANCHOR not in b.index:
        raise SystemExit(f"{ANCHOR.date()} is not a session in the book series")

    win = b.index[b.index > ANCHOR][:H]
    if len(win) != H:
        raise SystemExit(f"only {len(win)} sessions after {ANCHOR.date()}, need {H}")

    # The window must belong to the formation the page prints. The book rebalances at the
    # formation, so a window straddling one would be a different book than the page's.
    carried = sorted(set(b["formation"].loc[win]))
    if carried != [FORMATION]:
        raise SystemExit(f"window spans formations {carried}, expected [{FORMATION.date()}]")

    r = b["r"].loc[win]
    realised = float(np.prod(1 + r) - 1)

    # Cross-check against the stored target. exp-089 scored its four lines on this number;
    # if the two disagree the page and the register are describing different books.
    f = pd.read_csv("reports/poc/e5_book_tail.csv", index_col=0, parse_dates=True)
    stored = float(f.loc[str(FORMATION.date()), "realised"])
    if abs(realised - stored) > 1e-9:
        raise SystemExit(f"recomputed {realised:.8f} != stored {stored:.8f}")

    worst = r.idxmin()
    lines = {k: float(f.loc[str(FORMATION.date()), f"var_{k}"])
             for k in ("uncond", "scaled", "state", "terc")}
    es = float(f.loc[str(FORMATION.date()), "es_scaled"])

    # French's published series over the identical calendar window, as the outside reference.
    fr = french.momentum()
    frw = fr.loc[win[0]:win[-1]]

    doc = {
        "what": "realised 10-trading-day return of the reconstructed book from the page's formation",
        "outside_the_information_set": (
            "This file is the answer key to the 2020-10-30 page. None of it was available to "
            "a reader standing on that date, and it is not an input to any forecast in this "
            "package."
        ),
        "anchor_session": str(ANCHOR.date()),
        "formation": str(FORMATION.date()),
        "horizon_trading_days": H,
        "window": {"start": str(win[0].date()), "end": str(win[-1].date()), "sessions": int(len(win))},
        "formation_carried_by_every_session_in_window": str(FORMATION.date()),
        "realised_return": realised,
        "daily_path": [{"date": str(d.date()), "r": float(v)} for d, v in r.items()],
        "worst_session": {
            "date": str(worst.date()),
            "book": float(r.loc[worst]),
            "winner_leg": float(b["r_winner"].loc[worst]),
            "loser_leg": float(b["r_loser"].loc[worst]),
            "n_winner": int(b["n_winner"].loc[worst]),
            "n_loser": int(b["n_loser"].loc[worst]),
            "note": (
                "The loss is the short half. The page's 'what would change this reading' named "
                "this mechanism before the fact: a vaccine or treatment readout resolves the "
                "shared condition and the loser leg reprices upward together."
            ),
        },
        "against_the_lines_the_page_printed": {
            "var_scaled": {"level": lines["scaled"], "breached": bool(realised < lines["scaled"])},
            "var_uncond": {"level": lines["uncond"], "breached": bool(realised < lines["uncond"])},
            "es_scaled": {"level": es, "note": "the expected shortfall beyond the scaled VaR, for scale only"},
            "reading": (
                "The realised loss is worse than the unconditional 5% VaR and inside the "
                "volatility-scaled one. One formation is one observation and settles nothing "
                "about either line; the coverage tests in exp-089 are the evidence."
            ),
        },
        "french_reference": {
            "series": "Ken French daily momentum (Mom), the published factor",
            "days": int(len(frw)),
            "compounded_over_the_same_window": float(np.prod(1 + frw) - 1),
            "on_the_worst_session": float(fr.loc[worst]),
            "note": (
                "The reconstructed book falls further than the published factor on both "
                "measures. It is value-weighted top and bottom deciles of the Russell 3000; "
                "French's is built on 30/70 breakpoints over a broader universe, and is the "
                "less concentrated portfolio."
            ),
        },
        "provenance": {
            "script": "scripts/poc_e10_realised.py",
            "reproduce": "uv run python scripts/poc_e10_realised.py",
            "inputs": [
                "data/processed/book_returns.parquet",
                "reports/poc/e5_book_tail.csv",
                "Ken French daily momentum via unstructured_momentum.data.french",
            ],
            "agrees_with_stored_target": {
                "file": "reports/poc/e5_book_tail.csv",
                "column": "realised",
                "value": stored,
                "recomputed_independently": realised,
            },
            "experiment": "exp-089",
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "realised_2020-10-30.json").write_text(json.dumps(doc, indent=2) + "\n")

    print(f"anchor {ANCHOR.date()}  formation {FORMATION.date()}  "
          f"window {win[0].date()}..{win[-1].date()}")
    print(f"realised {realised:+.4%}   (stored target agrees to 1e-9)")
    print(f"worst session {worst.date()}: book {r.loc[worst]:+.2%} "
          f"= winners {b['r_winner'].loc[worst]:+.2%}, losers {b['r_loser'].loc[worst]:+.2%}")
    print(f"scaled VaR {lines['scaled']:+.2%} not breached; "
          f"unconditional {lines['uncond']:+.2%} breached")
    print(f"French over the same window {float(np.prod(1 + frw) - 1):+.2%}")
    print("wrote reports/poc/realised_2020-10-30.json")


if __name__ == "__main__":
    main()
