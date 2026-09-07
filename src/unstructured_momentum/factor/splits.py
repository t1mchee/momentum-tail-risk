"""Corporate-action detection from share counts, not from return magnitude.

The previous rule flagged any move beyond +/-45% as a corporate action. That is a proxy
for the real question and it is frequency-dependent: on daily data a 45% move usually IS
an action; on monthly data during a crisis it usually is not. Applied to a mixed-frequency
panel it erased Bank of America's genuine -53% in Jan-2009 and +73% in Mar-2009, which is
exactly the signal a momentum-reversal sensor exists to capture.

The direct test uses share counts, which the holdings files already carry. A split changes
price and share count inversely, so their product is preserved. A market move changes the
price and leaves the share count alone.

    split      price_ratio * quantity_ratio ~= 1
    real move  price_ratio * quantity_ratio ~= price_ratio

Frequency-agnostic by construction. Measured on the IWV panel: of 11,391 jumps beyond 45%,
643 are splits and 10,748 are real moves -- the old rule destroyed 94% of what it touched.
Validated against known actions (Chipotle 50:1 Jun-2024 at price x0.020 / qty x50.000;
Cano Health 1:100 Nov-2023; 2U 1:30; Hippo 1:25; Booking 25:1) and against BAC's crisis
series, where all three real moves survive.
"""
from __future__ import annotations
import numpy as np, pandas as pd

#: Product within this band of 1 counts as a preserved position. Measured: at 0.10 the
#: detector picks up 29,973 cells at MIN_JUMP=0.10 (noise); at 0.05 it picks up a clean
#: distribution of recognisable split ratios.
TOL = 0.05
#: The first version used 0.35, which cannot see a 3-for-2 (price x0.667, a 33% fall) or
#: any 4-for-3 or 5-for-4. Those are common: at 0.15 the detector finds 226 actions at a
#: 1.2 ratio and 82 at 1.5 that 0.35 missed entirely. Missing them leaves real corporate
#: actions in the return series, which is the same error as the old rule in the opposite
#: direction.
MIN_JUMP = 0.15


def detect(price: pd.DataFrame, quantity: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where a (date, ticker) cell is a corporate action."""
    q = quantity.reindex(index=price.index, columns=price.columns)
    pr = price / price.shift(1)
    qr = q / q.shift(1)
    prod = pr * qr
    jump = pr.notna() & ((pr > 1 + MIN_JUMP) | (pr < 1 / (1 + MIN_JUMP)))
    return jump & (prod >= 1 - TOL) & (prod <= 1 + TOL)


def adjust(price: pd.DataFrame, quantity: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rescale each series backwards through its detected actions so returns are clean.

    Real market moves are left untouched, which is the entire difference from the old rule.
    """
    acts = detect(price, quantity)
    if not acts.values.any():
        return price.copy(), 0
    pr = price / price.shift(1)
    factor = pd.DataFrame(1.0, index=price.index, columns=price.columns)
    factor = factor.mask(acts, pr)
    cum = factor.iloc[::-1].cumprod().iloc[::-1]      # forward-looking cumulative action factor
    cum = cum.shift(-1).fillna(1.0).replace(0, 1.0)
    # MULTIPLY, not divide. For a 50:1 split the price ratio is 0.02, and pre-split
    # prices must be scaled DOWN by that factor to sit continuously with post-split ones.
    # Dividing inflates them fiftyfold and was measurably worse than the rule it replaced
    # (corr to the published factor 0.686 vs 0.851) -- which is how the sign error surfaced.
    return price * cum, int(acts.values.sum())
