"""SEC fails-to-deliver: settlement stress observed rather than inferred from returns.

Comomentum, the project's main crowding measure, is built from how similarly stocks inside the
book move together. That is endogenous to the outcome it conditions -- it partly measures the
co-movement it is meant to predict. Fails-to-deliver are a different kind of evidence: an
observed count of shares that did not settle, published by the SEC, with no return input.

Two traps live in this source and both are handled here rather than in the caller. The archive
is served under three different path prefixes across eras, so file lists must be scraped from
the index and never constructed -- a constructed URL returned 53KB of HTML under a 404 status.
And securities are identified by CUSIP and by a POINT-IN-TIME symbol, so joining on a
current-vintage ticker map would silently mis-assign every renamed and delisted name.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/ftd")
OUT = Path("data/processed/ftd.parquet")

#: The SEC publishes each half-month file roughly two weeks after the period it covers. No
#: publication date appears in the file, so availability is the settlement date plus this.
PUBLICATION_LAG_DAYS = 15


def _read_one(z: Path) -> pd.DataFrame:
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    # Six half-months are not UTF-8 -- 2019-12, 2020-01 and 2021-10 among them, which is the
    # COVID run-up. The first version of this parser skipped them on UnicodeDecodeError, and a
    # silently absent half-month reads downstream as a quiet market rather than as missing
    # data. Decode is therefore attempted in order and only raises if every codec fails.
    last: Exception | None = None
    for codec in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, encoding=codec,
                             on_bad_lines="skip", engine="python")
            break
        except UnicodeDecodeError as exc:
            last = exc
    else:  # pragma: no cover - latin-1 decodes any byte, so this is unreachable in practice
        raise RuntimeError(f"no codec decoded {z.name}") from last
    df.columns = [c.strip().upper() for c in df.columns]
    ren = {"SETTLEMENT DATE": "settlement_date", "CUSIP": "cusip", "SYMBOL": "symbol",
           "QUANTITY (FAILS)": "fails", "DESCRIPTION": "description", "PRICE": "price"}
    df = df.rename(columns=ren)
    keep = [c for c in ("settlement_date", "cusip", "symbol", "fails", "price") if c in df]
    df = df[keep]
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], format="%Y%m%d",
                                           errors="coerce")
    df["fails"] = pd.to_numeric(df["fails"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["settlement_date", "symbol", "fails"])


def build(*, refresh: bool = False) -> pd.DataFrame:
    if OUT.exists() and not refresh:
        return pd.read_parquet(OUT)
    files = sorted(RAW.glob("cnsfails*.zip"))
    if not files:
        raise RuntimeError(f"no FTD archives in {RAW}")
    parts = []
    for i, f in enumerate(files, 1):
        try:
            parts.append(_read_one(f))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{f.name} failed to parse: {type(exc).__name__}: {exc}. A skipped half-month "
                f"reads downstream as a quiet market, so this raises instead of continuing."
            ) from exc
        if i % 60 == 0:
            print(f"  parsed {i}/{len(files)}")
    d = pd.concat(parts, ignore_index=True)
    d["available_at"] = d["settlement_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    d["fail_value"] = d["fails"] * d["price"]
    d = d.sort_values(["settlement_date", "symbol"]).reset_index(drop=True)
    d.to_parquet(OUT, index=False)
    return d


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="ftd")
    p.add_argument("--build", action="store_true")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)
    d = build(refresh=a.refresh)
    print(f"rows {len(d):,}   {d.settlement_date.min().date()} -> {d.settlement_date.max().date()}")
    print(f"symbols {d.symbol.nunique():,}   settlement dates {d.settlement_date.nunique():,}")
    print(f"availability lag {PUBLICATION_LAG_DAYS}d (no publication date in the file)")
    print(d.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
