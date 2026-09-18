"""Constants of the frozen model, and where files land.

Every number here is fixed by the research and never estimated at runtime.
Units follow the research pipeline: zero yields in percentage points,
continuously compounded; coupons and par yields in percentage points,
semiannual; returns as decimals over the holding period.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("TSYFWD_CACHE", PROJECT_ROOT / ".cache"))
SITE_DIR = Path(os.environ.get("TSYFWD_SITE", PROJECT_ROOT / "site"))

# Bond universe: synthetic par coupon Treasuries at these tenors (years)
TENORS = [2, 5, 10, 20, 30]

# Holding-period horizons in months
HORIZONS = [1, 3, 12]

MAX_MATURITY_MONTHS = 360

# CMT points out to 1Y are bills: quoted bond-equivalent, read as zeros
BILL_MAX_M = 12

# The one frozen parameter: the error-correction speed, per year
PHI_FROZEN = 0.15

# Trend-inflation anchor: EWMA decay on monthly core CPI yoy (half-life ~53m)
TAU_NU = 0.987
TAU_MIN = 120  # months of yoy history before tau is defined

# RM2 risk model (research s06): rolling window of monthly par-yield changes
RM2_WINDOW = 60
RM2_MIN = 36

# Years of daily history the dashboard plots, and the extra years loaded
# before it so RM2's rolling window is full on the first displayed day
HISTORY_YEARS = 3
RISK_LOOKBACK_YEARS = 6

TZ = "America/New_York"


def phi_h(phi: float, h: int | float):
    """Fraction of a gap closed over h months by error correction at speed
    phi per year: 1 - (1 - phi)^(h/12). Equals phi at h = 12, is within
    12% of the linear (h/12) phi below a year, and stays below one at long
    horizons where the linear form does not. Works elementwise on arrays."""
    return 1.0 - (1.0 - phi) ** (h / 12.0)
