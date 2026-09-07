"""The nightly run: one dated brief per calendar day, whatever coverage exists.

Monitoring is proven by calendar time and nothing else. A system demonstrated on request is a
backtest with a nice renderer; a system that has been emitting dated, versioned briefs for weeks
-- with its could-not-measure list visibly shrinking as stages come online -- is the only honest
evidence that the thing monitors.

So this runs before any experiment lands, at 29 percent coverage, gaps and all. The gaps are the
point: a reader can watch them close.

Every run writes four things. The daily X-ray, run first because the brief reads its named bet;
the brief itself; a manifest row recording what was measured and what was not, including how the
X-ray stage fared; and -- when the translation stage produced one -- a falsifier logged at
generation time so it can be scored later by lookup rather than by memory.

The X-ray stage is quarantined in a subprocess. The brief is the older, load-bearing product and
predates the X-ray by weeks of calendar record; an X-ray that fails costs the brief one
could_not_measure line, and must never cost it a night.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd

from .brief import render
from .run import OUT, run

MANIFEST = Path("data/processed/nightly_manifest.jsonl")
LOG = Path("data/processed/nightly.log")

#: Resolved from the package rather than from the working directory, because a scheduler that
#: fires from the wrong cwd should lose the X-ray's OUTPUT paths, not the script itself.
XRAY_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "daily_xray.py"
XRAY_TIMEOUT_S = 1800


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_xray(asof: pd.Timestamp) -> dict:
    """The X-ray, run FIRST so the brief can read its named bet, and fully quarantined.

    Two things make the quarantine real rather than nominal. It runs in a SUBPROCESS, so the
    X-ray's heavier dependency stack -- igraph, leidenalg, a network call to name the bet --
    cannot take the brief down by failing at import, hanging, or dying below the Python level.
    And the return value is a status record, never an exception: the brief is the older,
    load-bearing product, and a broken X-ray must cost it one could_not_measure line and
    nothing else.

    The severity block inside the X-ray is computed at THIS date, which is the same key the
    brief's severity cache will ask for a moment later, so the pair costs one fit and not two.
    """
    started = pd.Timestamp.now(tz="UTC")
    row: dict = {"as_of": str(asof.date()), "started_at": started.isoformat()}
    try:
        if not XRAY_SCRIPT.exists():
            raise FileNotFoundError(f"{XRAY_SCRIPT} not found from {Path.cwd()}")
        p = subprocess.run([sys.executable, str(XRAY_SCRIPT), str(asof.date())],
                           capture_output=True, text=True, timeout=XRAY_TIMEOUT_S)
        row["returncode"] = p.returncode
        if p.returncode == 0:
            row["status"] = "ok"
            row["stdout"] = p.stdout[-600:]
        else:
            row["status"] = "failed"
            row["error"] = (p.stderr or p.stdout)[-1200:]
    except subprocess.TimeoutExpired:
        row |= {"status": "failed",
                "error": f"timed out after {XRAY_TIMEOUT_S}s"}
    except Exception as exc:  # noqa: BLE001 -- an X-ray failure must never reach the brief
        row |= {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-800:]}
    row["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return row


def run_once(as_of: dt.date | str | None = None, *, translate: bool = True,
             xray: bool = True) -> dict:
    """One night's work. Never raises: a failed run must still leave a record of failing.

    Stage order is a dependency, not a preference: the X-ray writes `reports/xray/<date>.json`
    and the brief's `named_bet` stage reads it, so the X-ray goes first or the brief prints a
    could_not_measure for a bet that was about to exist.
    """
    asof = pd.Timestamp(as_of or dt.date.today()).normalize()
    started = pd.Timestamp.now(tz="UTC")
    row: dict = {"as_of": str(asof.date()), "started_at": started.isoformat(),
                 "commit": _git_sha()}
    if xray:
        row["xray"] = run_xray(asof)
    else:
        row["xray"] = {"status": "skipped", "error": "--no-xray"}
    try:
        res = run(asof, translate=translate)
        text = render(res)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"brief_{asof.date()}.txt"
        path.write_text(text)
        row |= {
            "status": "ok",
            "coverage": round(res.coverage, 4),
            "measured": sorted(res.values),
            "could_not_measure": {k: v[:160] for k, v in res.could_not_measure.items()},
            "brief": str(path),
        }
        tr = res.values.get("translation")
        if tr:
            row["driver"] = tr.get("driver")
            row["falsifier"] = tr.get("falsifier")
    except Exception as exc:  # noqa: BLE001
        row |= {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1200:]}
    row["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def history() -> pd.DataFrame:
    """Every night so far. Coverage over time is the monitoring evidence."""
    if not MANIFEST.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="nightly")
    p.add_argument("--as-of", default=None)
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--no-xray", action="store_true")
    p.add_argument("--history", action="store_true")
    a = p.parse_args(argv)
    if a.history:
        h = history()
        if h.empty:
            print("no nightly runs yet")
            return 0
        if "xray" in h.columns:
            h["xray_status"] = h["xray"].map(
                lambda d: d.get("status") if isinstance(d, dict) else None)
        cols = [c for c in ("as_of", "status", "coverage", "xray_status", "driver", "commit")
                if c in h.columns]
        print(h[cols].to_string(index=False))
        return 0
    row = run_once(a.as_of, translate=not a.no_translate, xray=not a.no_xray)
    print(f"{row['as_of']}: {row['status']}"
          + (f", coverage {row['coverage']:.0%}" if "coverage" in row else "")
          + f", xray {row.get('xray', {}).get('status', 'n/a')}"
          + (f", driver {row['driver']!r}" if row.get("driver") else ""))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
