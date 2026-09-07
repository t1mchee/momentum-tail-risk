"""Official-sector commentary: the BIS Quarterly Review.

Why this is the last text hypothesis worth testing
---------------------------------------------------
Three routes for text have now been tested and closed. News coverage of factor unwinds is
retrospective. The catalyst calendar was refuted across a 3x3 grid of event definitions.
And the pre-event crowding corpus contained zero genuine equity-factor positioning
evidence at any source tier.

Official-sector commentary is different in kind, and has the strongest prior of the three:
regulators write about leverage and crowded positioning as a *financial-stability* concern,
before it breaks, because warning is their job. The BIS Quarterly Review runs quarterly
from 1996, is free, and carries a fixed release date — so it is timestamped, long, and
point-in-time clean.

The test that actually discriminates
-------------------------------------
It is trivially true that the **September 2007** issue discusses the quant quake: it was
published a month after the event. That is retrospective and proves nothing. The sharp
test is whether the **June 2007** issue — published before August — already flagged
crowded quant positioning or leverage in market-neutral strategies.

Same structure for Sept 2019: does the **June 2019** issue anticipate the momentum
unwind, or does only the December 2019 issue describe it?

Point-in-time
-------------
Issues are published on a fixed quarterly schedule. Exact release dates are not encoded in
the URL, so ``available_at`` uses a deliberately **conservative** end-of-release-month
convention: an issue is treated as unavailable until the last day of its nominal month.
This can only make the test harder — it delays availability rather than advancing it — so
a positive finding under this convention is real rather than an artefact of timestamp
optimism.
"""

from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .. import pit
from ..config import CORPUS, USER_AGENT

BIS_QR_URL = "https://www.bis.org/publ/qtrpdf/r_qt{yymm}.pdf"

#: The Quarterly Review appears in March, June, September and December.
BIS_MONTHS = (3, 6, 9, 12)

#: Mechanism vocabulary. Deliberately narrow: these are the words a regulator uses when
#: writing about the specific fragility this project monitors, not general risk language.
#: Generic terms ("volatility", "uncertainty") are excluded because they appear in every
#: issue ever written and would produce a series with no variance.
MECHANISM_TERMS: dict[str, tuple[str, ...]] = {
    "crowding": ("crowded", "crowding", "consensus position", "similar position"),
    "quant": (
        "quantitative strateg",
        "quantitative fund",
        "market-neutral",
        "market neutral",
        "statistical arbitrage",
        "quantitative equity",
    ),
    "deleveraging": ("deleverag", "de-leverag", "unwind", "forced sale", "forced selling",
                     "degross", "margin call"),
    "leverage": ("leverage", "leveraged", "gearing"),
    "hedge_funds": ("hedge fund",),
    "momentum": ("momentum",),
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=30))
def _fetch(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    return resp.content


def _cache_dir():
    d = CORPUS / "bis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def issue_id(year: int, month: int) -> str:
    return f"{year % 100:02d}{month:02d}"


def release_available_at(year: int, month: int) -> pd.Timestamp:
    """Conservative availability: the last day of the issue's nominal month.

    The true release is earlier in the month, so this is a lower bound on how much the
    system could have known — it makes any positive result harder to obtain, not easier.
    """
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last.tz_localize(pit.NY).tz_convert(pit.TZ)


def fetch_issue(year: int, month: int, *, refresh: bool = False) -> str:
    """Download one Quarterly Review and return its extracted text."""
    if month not in BIS_MONTHS:
        raise ValueError(f"BIS Quarterly Review is published in {BIS_MONTHS}, not month {month}")

    yymm = issue_id(year, month)
    txt_path = _cache_dir() / f"r_qt{yymm}.txt"
    if txt_path.exists() and not refresh:
        return txt_path.read_text(encoding="utf-8")

    pdf = _fetch(BIS_QR_URL.format(yymm=yymm))
    (_cache_dir() / f"r_qt{yymm}.pdf").write_bytes(pdf)

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    txt_path.write_text(text, encoding="utf-8")
    return text


def count_terms(text: str) -> dict[str, int]:
    """Count mechanism-vocabulary hits, normalised for case and whitespace."""
    flat = re.sub(r"\s+", " ", text.lower())
    return {
        family: sum(flat.count(term) for term in terms)
        for family, terms in MECHANISM_TERMS.items()
    }


def scan(
    start_year: int = 2005,
    end_year: int | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build a point-in-time panel of mechanism-term intensity across issues."""
    end_year = end_year or dt.date.today().year
    rows = []

    for year in range(start_year, end_year + 1):
        for month in BIS_MONTHS:
            if dt.date(year, month, 1) > dt.date.today():
                continue
            try:
                text = fetch_issue(year, month, refresh=refresh)
            except (requests.RequestException, ValueError, Exception):  # noqa: BLE001
                continue
            if len(text) < 5_000:  # extraction failed or issue is a stub
                continue

            counts = count_terms(text)
            # Per-10k-words so a longer issue does not read as a louder warning.
            words = max(len(text.split()), 1)
            row = {
                "issue": f"{year}-{month:02d}",
                "observed_at": pd.Timestamp(year=year, month=month, day=1),
                "available_at": release_available_at(year, month),
                "n_words": words,
                **counts,
                **{f"{k}_per10k": v / words * 10_000 for k, v in counts.items()},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = pit.stamp(df, observed_at="observed_at", available_at="available_at", name="bis:qr")
    return df


def before_event(panel: pd.DataFrame, event_date: str | dt.date) -> pd.DataFrame:
    """Issues genuinely knowable before an event date."""
    return pit.as_of(panel, pd.Timestamp(event_date)).sort_values("observed_at")
