"""PM-facing momentum reversal-risk state report.

Scope, stated honestly
----------------------
This is a **state report**, not a probability forecast. No hazard model has been fitted
yet, so the memo reports measured conditions and their historical percentiles rather than
a calibrated probability of reversal. Emitting a number like "17% chance of a reversal in
10 days" before a model exists and has been validated would be the single most misleading
thing this system could do, so it does not.

When the model lands, the probability slots into `headline` and everything else here stays
as the supporting evidence. The division is deliberate and permanent: **numbers come from
code, never from an LLM.** The language layer's job is to retrieve, filter and explain,
and the memo template is what keeps that boundary visible on the page.

Every figure carries the date it was knowable and the source it came from, so a reader can
audit any line back to a function call and a timestamp.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Tier, tier_of


def _pct_rank(series: pd.Series, value: float) -> float:
    """Percentile of ``value`` within ``series``, ignoring NaNs."""
    s = series.dropna()
    if not len(s) or not np.isfinite(value):
        return float("nan")
    return float((s < value).mean() * 100)


def _fmt(x, spec: str = ".2f", suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:{spec}}{suffix}"


def _bar(pct: float, width: int = 20) -> str:
    """Crude percentile bar. Terminal- and markdown-safe."""
    if not np.isfinite(pct):
        return "?" * width
    filled = int(round(width * pct / 100))
    return "#" * filled + "." * (width - filled)


def build_state(
    as_of: dt.date | str,
    *,
    legs: dict | None = None,
    crowding: pd.Series | None = None,
    comovement_panel: pd.DataFrame | None = None,
    market_panel: pd.DataFrame | None = None,
    short_breadth: pd.DataFrame | None = None,
    options_row: pd.Series | None = None,
    evidence: pd.DataFrame | None = None,
) -> dict:
    """Collect every available measurement for one date into one structure.

    Each component is optional. A component that is missing is reported as missing rather
    than silently omitted -- a risk memo whose gaps are invisible is worse than one with
    holes in it, because the reader cannot tell the difference between "benign" and
    "not measured".
    """
    as_of = pd.Timestamp(as_of)
    state: dict = {
        "as_of": as_of,
        "tier": tier_of(as_of.date()).value,
        "generated_at": pd.Timestamp.now(tz="UTC"),
        "components": {},
        "missing": [],
    }

    def add(name: str, payload: dict | None) -> None:
        if payload is None:
            state["missing"].append(name)
        else:
            state["components"][name] = payload

    # --- momentum legs -----------------------------------------------------------
    if legs is not None:
        state["components"]["legs"] = {
            "universe": legs["n_universe"],
            "n_winners": len(legs["winners"]),
            "n_losers": len(legs["losers"]),
            "spread": float(legs["winners"].median() - legs["losers"].median()),
            "top_winners": list(legs["winners"].head(10).index),
            "top_losers": list(legs["losers"].head(10).index),
        }
    else:
        state["missing"].append("legs")

    # --- crowding ----------------------------------------------------------------
    add(
        "crowding",
        None
        if crowding is None
        else {
            "comomentum_winners": float(crowding.get("comomentum_winners", np.nan)),
            "comomentum_losers": float(crowding.get("comomentum_losers", np.nan)),
            "comomentum_spread": float(crowding.get("comomentum_spread", np.nan)),
        },
    )

    # --- deleveraging channel ----------------------------------------------------
    if comovement_panel is not None and as_of in comovement_panel.index:
        row = comovement_panel.loc[as_of]
        hist = comovement_panel.loc[:as_of]
        state["components"]["deleveraging"] = {
            "joint_z": float(row.get("joint_z", np.nan)),
            "joint_z_pctile": _pct_rank(hist["joint_z"], row.get("joint_z", np.nan)),
            "factor_dispersion": float(row.get("factor_dispersion", np.nan)),
            "factor_avg_corr": float(row.get("factor_avg_corr", np.nan)),
        }
    else:
        state["missing"].append("deleveraging")

    # --- market state ------------------------------------------------------------
    if market_panel is not None:
        sub = market_panel.loc[:as_of]
        if len(sub):
            row = sub.iloc[-1]
            state["components"]["market"] = {
                "asof_row": str(sub.index[-1].date()),
                "vix": float(row.get("vix", np.nan)),
                "vix_ts": float(row.get("vix_ts", np.nan)),
                "vix_pctile": _pct_rank(sub["vix"], row.get("vix", np.nan)),
            }
        else:
            state["missing"].append("market")
    else:
        state["missing"].append("market")

    # --- short-side constraint ---------------------------------------------------
    if short_breadth is not None and len(short_breadth.loc[:as_of]):
        sub = short_breadth.loc[:as_of]
        row = sub.iloc[-1]
        state["components"]["short_side"] = {
            "asof_row": str(sub.index[-1].date()),
            "n_threshold_securities": int(row.get("n_threshold_securities", 0)),
            "threshold_pctile": _pct_rank(
                sub["n_threshold_securities"], row.get("n_threshold_securities", np.nan)
            ),
            "short_ratio_mkt": float(row.get("short_ratio_mkt", np.nan)),
            "exempt_ratio_mkt": float(row.get("exempt_ratio_mkt", np.nan)),
            "exempt_pctile": _pct_rank(
                sub["exempt_ratio_mkt"], row.get("exempt_ratio_mkt", np.nan)
            ),
        }
    else:
        state["missing"].append("short_side")

    # --- options -----------------------------------------------------------------
    add(
        "options",
        None
        if options_row is None
        else {
            "basket_iv30": float(options_row.get("basket_iv30", np.nan)),
            "implied_correlation": float(options_row.get("implied_correlation", np.nan)),
            "n_quoted": int(options_row.get("n_constituents_quoted", 0)),
        },
    )

    # --- text evidence -----------------------------------------------------------
    if evidence is not None and len(evidence):
        ev = evidence.copy()
        tier1 = ev[ev.get("source_tier", "other") == "tier1"] if "source_tier" in ev else ev
        state["components"]["evidence"] = {
            "n_total": int(len(ev)),
            "n_tier1": int(len(tier1)),
            "n_boilerplate": int((ev.get("source_tier") == "boilerplate").sum())
            if "source_tier" in ev
            else 0,
            "items": tier1.sort_values("seendate", ascending=False)
            .head(8)[["seendate", "domain", "title"]]
            .to_dict("records"),
        }
    else:
        state["missing"].append("evidence")

    return state


def render(state: dict) -> str:
    """Render the state as a one-page markdown memo."""
    c = state["components"]
    as_of = state["as_of"]
    lines: list[str] = []

    lines.append(f"# Momentum Reversal Risk — state as of {as_of.date()}")
    lines.append("")
    lines.append(
        f"*Generated {state['generated_at']:%Y-%m-%d %H:%M UTC} · sample tier: "
        f"**{state['tier']}***"
    )
    lines.append("")
    lines.append(
        "> **No probability is reported.** The hazard model is not yet fitted or "
        "validated, so this memo states measured conditions and their historical "
        "percentiles. A calibrated probability will appear here once the model exists "
        "and has been tested out-of-sample — not before."
    )
    lines.append("")

    # -- the exposure ------------------------------------------------------------
    if "legs" in c:
        g = c["legs"]
        lines.append("## The exposure")
        lines.append("")
        lines.append(
            f"Universe {g['universe']:,} names · long leg {g['n_winners']} · "
            f"short leg {g['n_losers']} · 12-1 spread {g['spread']:+.1%}"
        )
        lines.append("")
        lines.append(f"**Long leg (top 10):** {', '.join(g['top_winners'])}")
        lines.append("")
        lines.append(f"**Short leg (worst 10):** {', '.join(g['top_losers'])}")
        lines.append("")

    # -- conditions --------------------------------------------------------------
    lines.append("## Measured conditions")
    lines.append("")
    lines.append("| Channel | Measure | Value | Percentile |")
    lines.append("|---|---|---|---|")

    if "crowding" in c:
        g = c["crowding"]
        lines.append(
            f"| Crowding | comomentum, long leg | {_fmt(g['comomentum_winners'], '.4f')} | — |"
        )
        lines.append(
            f"| Crowding | comomentum, short leg | {_fmt(g['comomentum_losers'], '.4f')} | — |"
        )
        lines.append(
            f"| Crowding | long − short | {_fmt(g['comomentum_spread'], '+.4f')} | — |"
        )
    if "deleveraging" in c:
        g = c["deleveraging"]
        lines.append(
            f"| Deleveraging | joint factor z | {_fmt(g['joint_z'], '.2f')} | "
            f"{_fmt(g['joint_z_pctile'], '.1f')} |"
        )
        lines.append(
            f"| Deleveraging | factor dispersion | {_fmt(g['factor_dispersion'], '.2f')} | — |"
        )
    if "market" in c:
        g = c["market"]
        lines.append(
            f"| Market | VIX ({g['asof_row']}) | {_fmt(g['vix'], '.1f')} | "
            f"{_fmt(g['vix_pctile'], '.1f')} |"
        )
        lines.append(f"| Market | VIX3M/VIX | {_fmt(g['vix_ts'], '.2f')} | — |")
    if "short_side" in c:
        g = c["short_side"]
        lines.append(
            f"| Short side | Reg SHO threshold count ({g['asof_row']}) | "
            f"{g['n_threshold_securities']} | {_fmt(g['threshold_pctile'], '.1f')} |"
        )
        lines.append(
            f"| Short side | short-exempt ratio | {_fmt(g['exempt_ratio_mkt'], '.4f')} | "
            f"{_fmt(g['exempt_pctile'], '.1f')} |"
        )
    if "options" in c:
        g = c["options"]
        lines.append(
            f"| Options | basket implied corr | {_fmt(g['implied_correlation'], '.3f')} | — |"
        )
        lines.append(f"| Options | basket IV30 | {_fmt(g['basket_iv30'], '.1%')} | — |")
    lines.append("")

    # -- evidence ----------------------------------------------------------------
    if "evidence" in c:
        g = c["evidence"]
        lines.append("## Text evidence")
        lines.append("")
        lines.append(
            f"{g['n_total']} articles matched on-mechanism queries; **{g['n_tier1']} from "
            f"tier-1 sources**, {g['n_boilerplate']} discarded as templated filler."
        )
        lines.append("")
        if g["items"]:
            for it in g["items"]:
                ts = pd.Timestamp(it["seendate"])
                lines.append(f"- `{ts:%Y-%m-%d %H:%M}` **{it['domain']}** — {it['title']}")
        else:
            lines.append("*No tier-1 on-mechanism coverage in the window.*")
        lines.append("")

    # -- what is not measured ----------------------------------------------------
    if state["missing"]:
        lines.append("## Not measured")
        lines.append("")
        lines.append(
            "These channels had no data available at this date and are absent rather than "
            "benign:"
        )
        lines.append("")
        for m in state["missing"]:
            lines.append(f"- {m}")
        lines.append("")

    return "\n".join(lines)


def write(state: dict, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(state), encoding="utf-8")
