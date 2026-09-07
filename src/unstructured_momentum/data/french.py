"""Ken French Data Library loaders, with vintage discipline.

Two findings drive the design of this module.

**1. The daily momentum file is far too stale to drive a live monitor.**
The entire library is rebuilt in one monthly batch keyed to the CRSP monthly release.
Observed 2026-08-21: every daily zip carries the same ``Last-Modified`` (2026-08-03),
covering data through 2026-06-30 -- a 34-day lag at publication, decaying to ~65 days
before the next refresh. Today's file is 52 days stale.

    => French is the *historical calibration and validation* series. The live edge must
       come from an in-house WML replication. See `factor.wml`.

**2. French restates history and does not version the files.**
The URL is overwritten in place. A value pulled for 2013-08-15 in 2024 may differ from
what the same URL returns today. Documented restatement episodes:

* 2005 -- CRSP merged 1925-1962 NYSE daily data; changed month-end prices and dividend
  ex-dates.
* 2014 -- CRSP backfilled pre-1947 shares outstanding, changing ME denominators and
  therefore every value-weighted factor return in affected periods.
* Jan 2025 -- CRSP retired the Legacy FIZ flat-file format for CIZ. The whole library was
  regenerated from a different source format.

Exactly one archived vintage is published: the December 2024 (pre-CIZ) snapshot under
``ftp_202412/``. So we hash and retain every download alongside its ``Last-Modified``
header and the embedded ``"created by using the NNNNNN CRSP database"`` provenance line,
and `restatement_diff()` quantifies the CIZ-transition revision on our own backtest
window. Any tail-risk calibration fitted on the current vintage and presented as an
out-of-sample 2009 result is contaminated by at least three rounds of retroactive
revision unless that revision has been measured.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, asdict

import pandas as pd
import requests

from .. import pit
from ..config import RAW, USER_AGENT

FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
#: The single published historical vintage (pre-CIZ), for restatement measurement.
FTP_202412 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp_202412"

DATASETS: dict[str, str] = {
    "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "size_mom_25_daily": "25_Portfolios_ME_Prior_12_2_Daily_CSV.zip",
    "mom_10_daily": "10_Portfolios_Prior_12_2_Daily_CSV.zip",
    "industry_49_daily": "49_Industry_Portfolios_daily_CSV.zip",
}

#: French uses these as missing-value sentinels. They are *not* NaN in the raw file, and
#: -99.99 is a plausible-looking percentage return, so failing to mask them silently
#: injects a -99.99% day into the series.
SENTINELS = (-99.99, -999.0, -99.99e0)


@dataclass(frozen=True)
class Vintage:
    """Provenance for one download. Retained so restatements are detectable."""

    dataset: str
    url: str
    sha256: str
    last_modified: str | None
    crsp_vintage: str | None   # e.g. "202606", parsed from the file preamble
    fetched_at: str
    n_bytes: int


def _vintage_dir(dataset: str):
    d = RAW / "french" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(dataset: str, *, archive: bool = False) -> tuple[str, Vintage]:
    """Fetch a dataset, record provenance, and persist the raw bytes.

    ``archive=True`` pulls the December-2024 pre-CIZ vintage instead of current.
    """
    fname = DATASETS[dataset]
    url = f"{FTP_202412 if archive else FTP}/{fname}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()

    payload = resp.content
    digest = hashlib.sha256(payload).hexdigest()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")

    m = re.search(r"created by using the (\d{6}) CRSP database", text)
    v = Vintage(
        dataset=dataset,
        url=url,
        sha256=digest,
        last_modified=resp.headers.get("Last-Modified"),
        crsp_vintage=m.group(1) if m else None,
        fetched_at=pd.Timestamp.now(tz="UTC").isoformat(),
        n_bytes=len(payload),
    )

    tag = f"{'archive202412' if archive else v.crsp_vintage or 'unknown'}_{digest[:12]}"
    d = _vintage_dir(dataset)
    (d / f"{tag}.csv").write_text(text, encoding="utf-8")
    (d / f"{tag}.json").write_text(json.dumps(asdict(v), indent=2), encoding="utf-8")
    return text, v


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


_DATE_ROW = re.compile(r"^\s*(\d{8})\s*,")


def _parse_blocks(text: str) -> list[pd.DataFrame]:
    """Split a French CSV into its stacked data blocks.

    Daily portfolio files contain two blocks (value-weighted then equal-weighted) with a
    repeated header between them; factor files contain one. Rather than hard-coding
    ``skiprows`` -- which differs per file (13 for momentum, 3 for FF5, 11 for the 25
    size/momentum portfolios) and has changed over time -- we detect header rows as
    "line before the first YYYYMMDD row in each run".
    """
    lines = text.splitlines()
    blocks: list[pd.DataFrame] = []
    header: str | None = None
    rows: list[str] = []

    def flush() -> None:
        if header and rows:
            df = pd.read_csv(io.StringIO("\n".join([header, *rows])))
            df.columns = ["date", *[c.strip() for c in df.columns[1:]]]
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
            blocks.append(df.set_index("date"))

    for i, ln in enumerate(lines):
        if _DATE_ROW.match(ln):
            if not rows:  # first data row of a run -> previous line is the header
                header = lines[i - 1]
            rows.append(ln)
        elif rows:
            flush()
            header, rows = None, []
    flush()

    if not blocks:
        raise ValueError("no data blocks found; the file format has changed")
    return blocks


def load(dataset: str = "momentum_daily", *, block: int = 0, refresh: bool = False) -> pd.DataFrame:
    """Load a French daily dataset as decimal returns with a DatetimeIndex.

    ``block=0`` is value-weighted, ``block=1`` equal-weighted where applicable.
    Values are converted from percent to decimal and sentinels are masked to NaN.
    """
    d = _vintage_dir(dataset)
    cached = sorted(p for p in d.glob("*.csv") if not p.name.startswith("archive"))
    if refresh or not cached:
        text, _ = download(dataset)
    else:
        text = cached[-1].read_text(encoding="utf-8")

    df = _parse_blocks(text)[block]
    df = df.apply(pd.to_numeric, errors="coerce")
    for s in SENTINELS:
        df = df.mask(df == s)
    return df / 100.0


def momentum(*, refresh: bool = False) -> pd.Series:
    """Daily WML (``Mom``) as decimal returns. The academic target series."""
    df = load("momentum_daily", refresh=refresh)
    return df["Mom"].rename("wml").dropna()


def market(*, refresh: bool = False) -> pd.DataFrame:
    """Daily FF3: ``Mkt-RF``, ``SMB``, ``HML``, ``RF``."""
    return load("ff3_daily", refresh=refresh)


def momentum_pit(*, refresh: bool = False) -> pd.DataFrame:
    """Daily WML with point-in-time stamps reflecting French's real publication lag.

    ``available_at`` is derived from the vintage's ``Last-Modified`` header where known,
    which is when the batch containing that observation was actually published. Anything
    published after that date is simply not knowable, and a monitor that pretends
    otherwise is claiming ~50 days of hindsight.
    """
    d = _vintage_dir("momentum_daily")
    metas = sorted(d.glob("*.json"))
    if refresh or not metas:
        download("momentum_daily")
        metas = sorted(d.glob("*.json"))

    meta = json.loads(metas[-1].read_text())
    published = pd.Timestamp(meta["last_modified"]) if meta.get("last_modified") else pd.Timestamp.now(tz="UTC")

    s = momentum()
    df = s.reset_index().rename(columns={"date": "observed_at"})
    return pit.stamp(df, observed_at="observed_at", available_at=published, name="french:momentum")


def publication_lag() -> dict:
    """Quantify how stale the momentum file is right now.

    Reported in the memo to justify why a live monitor cannot depend on French.
    """
    d = _vintage_dir("momentum_daily")
    metas = sorted(d.glob("*.json"))
    if not metas:
        download("momentum_daily")
        metas = sorted(d.glob("*.json"))
    meta = json.loads(metas[-1].read_text())

    s = momentum()
    published = pd.Timestamp(meta["last_modified"]) if meta.get("last_modified") else None
    last_obs = s.index.max()
    now = pd.Timestamp.now(tz="UTC")
    return {
        "crsp_vintage": meta.get("crsp_vintage"),
        "published": str(published),
        "last_observation": str(last_obs.date()),
        "lag_at_publication_days": (published.tz_localize(None) - last_obs).days if published else None,
        "staleness_today_days": (now.tz_localize(None) - last_obs).days,
    }


def restatement_diff(dataset: str = "momentum_daily", column: str = "Mom") -> pd.DataFrame:
    """Compare the current vintage against the Dec-2024 pre-CIZ archive.

    Returns per-observation differences on the overlapping window. This is the evidence
    for how much of a historical "result" is an artefact of retroactive revision.
    """
    d = _vintage_dir(dataset)
    arch = sorted(d.glob("archive202412_*.csv"))
    if not arch:
        text, _ = download(dataset, archive=True)
    else:
        text = arch[-1].read_text(encoding="utf-8")

    old = _parse_blocks(text)[0].apply(pd.to_numeric, errors="coerce")
    for s in SENTINELS:
        old = old.mask(old == s)
    old = old / 100.0

    new = load(dataset)
    idx = old.index.intersection(new.index)
    out = pd.DataFrame({"archive_202412": old.loc[idx, column], "current": new.loc[idx, column]})
    out["diff"] = out["current"] - out["archive_202412"]
    out["changed"] = out["diff"].abs() > 1e-9
    return out
