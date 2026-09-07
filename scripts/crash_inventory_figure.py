"""The crash inventory as one figure: the smear the memo's section 4.2 said would be a finding.

Section 4.2 specified winner-leg loss share against DAYS TO TROUGH. That axis is degenerate
under the registered event definitions: the fast rule fixes a ten-day window, so 164 of the
195 episodes sit at exactly ten sessions and the axis carries no information. The depth of
the loss is plotted instead and the substitution is stated here rather than left for a reader
to notice from the picture.

If the three routes are real and separable from returns alone, the episodes fall into three
clusters. They fall into two, and the largest group is in neither.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("docs/figures/crash_inventory.png")
COLOURS = {
    "Route 1": "#8c2f39",
    "Route 2": "#2f5d8c",
    "Route 3": "#3f7a52",
    "FITS-NONE": "#9a9a9a",
}


def main() -> None:
    inv = pd.read_csv(
        "reports/crash_inventory/inventory.csv", parse_dates=["peak_at", "trough_at"]
    )
    inv = inv[np.isfinite(inv["winner_share"])]

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    for route, g in inv.groupby("route"):
        for defn, mark in (("fast", "o"), ("slow", "D")):
            h = g[g["definition"] == defn]
            if not len(h):
                continue
            ax.scatter(
                h["winner_share"].clip(-1.5, 2.5),
                100 * h["depth"].abs(),
                s=26 if defn == "fast" else 34,
                marker=mark,
                c=COLOURS.get(route, "#000"),
                alpha=0.6,
                linewidths=0.5,
                edgecolors="white",
                label=f"{route} ({len(g)})" if defn == "fast" else None,
            )
    ax.axvline(0.60, color="#444", lw=0.8, ls="--")
    ax.axvline(0.40, color="#444", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(3.5, 70)
    ax.set_yticks([5, 10, 20, 50])
    ax.set_yticklabels(["5%", "10%", "20%", "50%"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.text(0.63, 4.0, "Route 2 needs winner share ≥ 0.60", fontsize=7, color="#444")
    ax.text(0.37, 4.0, "Route 1 needs loser share ≥ 0.60", fontsize=7, color="#444",
            ha="right")
    ax.set_xlabel("winner leg's share of the loss  (loser share is 1 − this)")
    ax.set_ylabel("depth of the loss")
    ax.set_title(
        "195 episodes, 1926–2026, coloured by the rule; diamonds are slow-definition",
        fontsize=9, loc="left",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", scatterpoints=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # The residue's one shape: no bear state, loser-driven, market up.
    res = inv[inv["route"] == "FITS-NONE"]
    oth = inv[inv["route"] != "FITS-NONE"]
    bx.scatter(oth["mkt_prior_2y"].clip(-1, 2), oth["mkt_during"].clip(-0.2, 0.5),
               s=10, c=[COLOURS[r] for r in oth["route"]], alpha=0.5,
               linewidths=0.5, edgecolors="white")
    bx.scatter(res["mkt_prior_2y"].clip(-1, 2), res["mkt_during"].clip(-0.2, 0.5),
               s=14, c=COLOURS["FITS-NONE"], alpha=0.75,
               linewidths=0.5, edgecolors="white")
    bx.axvline(0, color="#8c2f39", lw=1.0)
    bx.axhline(0, color="#444", lw=0.6)
    bx.text(0.03, 0.46, "bear-state clause →", fontsize=7.2, color="#8c2f39", ha="left")
    nov = inv[inv["trough_at"] == "2020-11-18"]
    if len(nov):
        n = nov.iloc[0]
        bx.annotate(
            "2020-11-18",
            xy=(min(n["mkt_prior_2y"], 2), min(n["mkt_during"], 0.5)),
            xytext=(0.85, 0.20), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#333"),
        )
    bx.set_xlabel("trailing 2-year market return at the peak")
    bx.set_ylabel("market return through the episode")
    bx.set_title(
        "the residue sits right of the clause and above zero", fontsize=9, loc="left"
    )
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
