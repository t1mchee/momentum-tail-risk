"""Map 13F CUSIPs to index tickers, and prove each match with a price.

The problem
-----------
Form 13F identifies securities by CUSIP. iShares index holdings identify them by ticker.
Neither publishes the other, and every free crosswalk covering 2007 is either a paid
identifier service or a scraped table of uncertain provenance. The one bridge both sides
carry is the issuer name, and names are exactly the field that does not join cleanly:
filers write "AT&T INC. EQUITY", "APPLE COMPUTER INC COM", "D AMKOR TECHNOLOGIES INC",
carrying column codes, class descriptors and their own abbreviations.

Why this is nonetheless safe
----------------------------
Because the match can be checked against something the names know nothing about. A 13F
holding gives a value and a share count, and their ratio is a price on the position date.
The index file gives a price for the same date. If a name match is correct those two prices
agree; if it has crossed two different issuers they will not. So every candidate match is
validated numerically, and the match rate reported here is the rate of matches that survive
that check rather than the rate of names that happened to look similar.

This matters more than the usual tidiness argument. A crosswalk that silently mismatches a
few percent of names would attach the wrong owners to the wrong stocks, which is precisely
the kind of fault that produces a clean-looking crowding measure of nothing at all.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: Leading column codes some filers put before the issuer name -- a lone letter, or a
#: discretion flag. Seen as "D AMKOR TECHNOLOGIES INC" in 2007 filings.
_LEAD_CODE = re.compile(r"^(?:[A-Z]|SOLE|DEFINED|OTHER)\s+(?=[A-Z0-9])")

#: Words that describe the instrument or the filer's bookkeeping rather than the issuer.
_NOISE = re.compile(
    r"\b(COMMON\s+STOCK|COMMON|COM|STOCK|EQUITY|SHS|SH|CL\s*[A-Z]|CLASS\s*[A-Z]|"
    r"ADR|ADRS|SPON(?:S|SORED)?|ORD|SBI|BEN\s*INT|DEP\s*RCPT|UNIT|WT|WTS|"
    r"NEW|DEL|DELAWARE|THE)\b",
    re.I,
)

#: Corporate-form suffixes. Dropped from both sides so "APPLE INC" meets "APPLE COMPUTER".
_SUFFIX = re.compile(
    r"\b(INC(?:ORPORATED)?|CORP(?:ORATION)?|CO|COMPANY|LTD|LIMITED|PLC|LP|LLC|"
    r"TR|TRUST|HLDG?S?|HOLDINGS?|GROUP|GRP|INTL|INTERNATIONAL|SA|NV|AG)\b",
    re.I,
)


def normalise(name: str | None) -> str:
    """A comparable key from an issuer name written by anyone."""
    if not name or not isinstance(name, str):
        return ""
    s = name.upper()
    s = _LEAD_CODE.sub("", s)
    s = re.sub(r"[^A-Z0-9&\s]", " ", s)
    s = re.sub(r"\bAND\b", "&", s)
    s = _NOISE.sub(" ", s)
    s = _SUFFIX.sub(" ", s)
    # Stripping a corporate suffix can strand the ampersand that joined it: "JPMORGAN
    # CHASE & CO" became "JPMORGAN CHASE &" and stopped matching "JPMORGAN CHASE", which
    # cost the single heaviest unmatched name in the index.
    s = re.sub(r"\s*&\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def implied_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-CUSIP implied price and the issuer names filers used for it.

    The price is a median across filers, which is what makes it robust: a single filer's
    misreport cannot move it, and the spread across filers is itself a quality signal.
    """
    eq = panel[(panel["kind"] == "SH") & (panel["shares"] > 0) & (panel["value_usd"] > 0)].copy()
    eq["px"] = eq["value_usd"] / eq["shares"]
    g = eq.groupby("cusip")
    out = pd.DataFrame(
        {
            "px_13f": g["px"].median(),
            "n_filers": g["cik"].nunique(),
            "px_dispersion": g["px"].apply(
                lambda s: float(np.nan if len(s) < 3 else (s.quantile(0.75) - s.quantile(0.25)) / max(s.median(), 1e-9))
            ),
            "value_usd": g["value_usd"].sum(),
        }
    )
    names = (
        eq.dropna(subset=["issuer"])
        .groupby("cusip")["issuer"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
    )
    out["issuer"] = names
    return out.reset_index()


def build(
    panel: pd.DataFrame,
    index_holdings: pd.DataFrame,
    *,
    price_tol: float = 0.02,
    name_col: str = "name",
    price_col: str = "price",
    ticker_col: str = "ticker",
) -> tuple[pd.DataFrame, dict]:
    """Match 13F CUSIPs to index tickers on name, keep only price-confirmed matches.

    ``index_holdings`` is one as-of snapshot with ticker, name and price columns, and its
    date must be the 13F position date -- matching a September index file to June positions
    would compare prices three months apart and reject every correct match.

    Returns the confirmed crosswalk and a report whose ``weight_matched`` is the statistic
    the experiment gates on.
    """
    left = implied_prices(panel)
    left["key"] = left["issuer"].map(normalise)

    right = index_holdings.copy()
    right["key"] = right[name_col].map(normalise)
    right = right[right["key"] != ""]

    # Ambiguity on the INDEX side is unresolvable: two distinct issuers really do share the
    # name "FIRST BANCORP", and a price check cannot say which one a filer meant.
    #
    # Ambiguity on the 13F side is a different thing entirely and must not be dropped. A
    # large issuer routinely carries several CUSIPs across filers -- the common stock plus a
    # warrant, an ADR line, a share class, or a transcription error -- and an earlier version
    # discarded every one of those names before testing them. That silently removed Exxon,
    # AT&T, Microsoft, Pfizer and Altria, which is to say it removed the index by weight
    # while reporting a 99.5 percent confirmation rate on what survived. The price check is
    # precisely the instrument that resolves this case, so it is applied first and the
    # ambiguity settled afterwards.
    dup_r = set(right.loc[right["key"].duplicated(keep=False), "key"])

    l = left[left["key"] != ""]
    r = right[~right["key"].isin(dup_r)]
    cand = l.merge(r, on="key", how="inner", suffixes=("", "_idx"))

    px_idx = pd.to_numeric(
        cand[price_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    cand["px_index"] = px_idx
    cand["px_gap"] = (cand["px_13f"] / cand["px_index"] - 1).abs()
    confirmed = cand[cand["px_gap"] <= price_tol].copy()

    # Where several CUSIPs still survive the price check for one index name, keep the line
    # institutions actually hold: most filers first, then most value. A warrant priced near
    # its underlying is the case this guards against.
    contested = int(confirmed[ticker_col].duplicated(keep=False).sum())
    confirmed = (
        confirmed.sort_values(["n_filers", "value_usd"], ascending=False)
        .drop_duplicates(subset=[ticker_col], keep="first")
    )

    wcol = next((c for c in ("weight_pct", "Weight (%)", "weight") if c in right.columns), None)
    if wcol:
        w = pd.to_numeric(right[wcol].astype(str).str.replace(",", "", regex=False), errors="coerce")
        total_w = float(w.sum())
        matched_w = float(
            pd.to_numeric(
                confirmed[wcol].astype(str).str.replace(",", "", regex=False), errors="coerce"
            ).sum()
        )
    else:
        total_w = matched_w = float("nan")

    report = {
        "cusips_in_13f": int(len(left)),
        "index_names": int(len(right)),
        "index_names_ambiguous_dropped": int(len(dup_r)),
        "contested_after_price_check": contested,
        "name_candidates": int(len(cand)),
        "price_confirmed": int(len(confirmed)),
        "price_rejected": int(len(cand) - len(confirmed)),
        "confirm_rate": float(len(confirmed) / len(cand)) if len(cand) else 0.0,
        "weight_matched": float(matched_w / total_w) if total_w and total_w == total_w else float("nan"),
        "median_px_gap": float(confirmed["px_gap"].median()) if len(confirmed) else float("nan"),
    }
    cols = ["cusip", "issuer", ticker_col, "px_13f", "px_index", "px_gap", "n_filers", "value_usd"]
    return confirmed[[c for c in cols if c in confirmed.columns]], report
