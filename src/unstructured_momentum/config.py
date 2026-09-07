"""Project-wide configuration: paths, sample tiers, and target definitions.

The holdout protocol lives here as executable configuration rather than as prose in the
memo, so that `tier_of(date)` is the single source of truth and an accidental peek at
sealed data is a code change someone can see in a diff.
"""

from __future__ import annotations

import datetime as dt
import os
from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"          # immutable downloads, exactly as fetched
INTERIM = DATA / "interim"  # parsed but not yet feature-ready
PROCESSED = DATA / "processed"
CORPUS = DATA / "corpus"    # text documents + retrieval index
REPORTS = ROOT / "reports"

for _p in (RAW, INTERIM, PROCESSED, CORPUS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# Identify ourselves politely. SEC *requires* this and 403s without it; several other
# sources rate-limit anonymous traffic more aggressively.
CONTACT_EMAIL = os.environ.get("UM_CONTACT_EMAIL", "chee.timothy@gmail.com")
USER_AGENT = f"unstructured-momentum research (contact: {CONTACT_EMAIL})"


def load_env(path=None) -> bool:
    """Load `.env` into this process only.

    Deliberately process-scoped rather than exported to the shell. Claude Code resolves
    ``ANTHROPIC_API_KEY`` ahead of a subscription, so a globally-exported key silently
    switches an interactive session from the user's plan to metered API billing. Loading
    here means the key is live while a project script runs and nowhere else.

    Does not overwrite a variable that is already set, so an explicitly-exported value
    still wins if someone wants that.
    """
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    return True


#: Loaded on import so any entry point picks it up without ceremony.
load_env()


# --------------------------------------------------------------------------------------
# Sample tiers — the holdout protocol
# --------------------------------------------------------------------------------------


class Tier(str, Enum):
    """Which question a date is allowed to answer.

    DESIGN   freely inspected; features and scoring rules are built here.
    VALIDATE opened once, after the feature spec is frozen.
    SEALED   opened once, at the very end. Result reported whatever it is.
    """

    DESIGN = "design"
    VALIDATE = "validate"
    SEALED = "sealed"


#: Everything from this date forward is sealed. Chosen because it is partly past the
#: assistant's training cutoff, which makes the LLM-hindsight test real rather than
#: simulated. Do not move this date to make a result look better.
SEALED_START = dt.date(2023, 1, 1)

#: Episodes reserved for the VALIDATE tier. Dense-text era, but not used for design.
#: These are *approximate* windows used to route dates to tiers; the authoritative
#: episode dates come from the return-based detector in `events.detector`.
VALIDATE_WINDOWS: tuple[tuple[dt.date, dt.date], ...] = (
    (dt.date(2018, 1, 15), dt.date(2018, 3, 31)),   # Feb 2018 vol spike
    (dt.date(2020, 10, 15), dt.date(2020, 12, 31)),  # Nov 9 2020 vaccine-day rotation
    (dt.date(2021, 1, 1), dt.date(2021, 3, 31)),    # Jan 2021 short squeeze
    (dt.date(2022, 10, 1), dt.date(2022, 12, 31)),  # Nov 2022 CPI rotation
)

#: Episodes used for feature design. Documented here for provenance; the detector must
#: rediscover these unprompted or the detector is wrong.
DESIGN_EPISODES: dict[str, tuple[dt.date, dt.date]] = {
    "quant_quake_2007": (dt.date(2007, 8, 1), dt.date(2007, 8, 31)),
    "crisis_rebound_2009": (dt.date(2009, 3, 1), dt.date(2009, 5, 31)),
    "momentum_unwind_2019": (dt.date(2019, 8, 15), dt.date(2019, 9, 30)),
}


def tier_of(date: dt.date) -> Tier:
    """Return which sample tier a date belongs to."""
    if date >= SEALED_START:
        return Tier.SEALED
    for start, end in VALIDATE_WINDOWS:
        if start <= date <= end:
            return Tier.VALIDATE
    return Tier.DESIGN


class SealedDataAccess(RuntimeError):
    """Raised when code touches a tier it has not been unlocked for."""


#: Set via `unlock(...)` once the spec is frozen. Guards analysis, not data download --
#: downloading sealed-era data is fine and necessary; *looking* at it is what is gated.
_UNLOCKED: set[Tier] = {Tier.DESIGN}


def unlock(tier: Tier, reason: str) -> None:
    """Open a tier for analysis. Deliberately noisy: prints an audit line."""
    _UNLOCKED.add(tier)
    print(f"[TIER UNLOCK] {tier.value}: {reason}")


def require_unlocked(tier: Tier) -> None:
    if tier not in _UNLOCKED:
        raise SealedDataAccess(
            f"Tier {tier.value!r} is sealed. Freeze the feature spec first, then call "
            f"config.unlock(Tier.{tier.name}, reason=...). This guard exists to stop "
            f"accidental peeking, so think before you disable it."
        )


# --------------------------------------------------------------------------------------
# Target definition
# --------------------------------------------------------------------------------------

#: Forecast horizons in trading days. 10d is the headline; 5 and 21 expose the term
#: structure of reversal risk, which distinguishes an imminent unwind from a slow bleed.
HORIZONS: tuple[int, ...] = (5, 10, 21)
HEADLINE_HORIZON = 10

#: PM-facing ("absolute") event: cumulative WML return over the horizon below this.
#: Scaled by horizon so the three horizons are roughly comparable in severity.
ABSOLUTE_THRESHOLDS: dict[int, float] = {5: -0.05, 10: -0.07, 21: -0.10}

#: Statistical ("vol-standardized") event: horizon return divided by trailing realized
#: WML vol, below this unconditional percentile. Strips out the part of a crash that is
#: merely "volatility was high", which the Barroso-Santa-Clara baseline already forecasts.
VOL_STANDARDIZED_PCTILE = 5.0
REALIZED_VOL_WINDOW = 100  # trading days
