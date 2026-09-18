"""The inflation anchors: pi (current core CPI yoy) and tau (its EWMA trend).

Timing follows the research convention: month-m CPI is stamped at m+1, the
month it is known by. Daily, month-m CPI counts as known from the 15th of
m+1 (BLS releases land on the 10th-15th), or from the latest curve date if
that comes first, since a print already downloaded is public.

October 2025 CPI was never collected (appropriations lapse). Days before the
November print therefore use September; tau skips the missing yoy row exactly
as the research EWMA does, carrying its last value; and pi, which goes
missing for the Oct-2026 print, is filled from a geometrically interpolated
Oct-2025 index and flagged on the page.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .sources import day_key


def pi_yoy(cpi: pl.DataFrame) -> pl.DataFrame:
    """(month_id, pi_yoy) log yoy inflation, stamped at the month it is known.
    `cpi` is (month_id, cpi) on a contiguous monthly grid, nulls allowed."""
    return (
        cpi.sort("month_id")
        .with_columns(((pl.col("cpi") / pl.col("cpi").shift(12)).log()).alias("pi_yoy"))
        .drop_nulls(subset=["pi_yoy"])
        .select((pl.col("month_id") + 1).alias("month_id"), "pi_yoy")
    )


def tau_ewma(pi: pl.DataFrame, nu: float = config.TAU_NU) -> dict[int, float]:
    """Normalized discounted mean of (already publication-lagged) yoy inflation:
    a bias-corrected EWMA that steps over rows, so a missing print costs no
    extra decay. Undefined until TAU_MIN observations."""
    arr = pi["pi_yoy"].to_numpy()
    mids = pi["month_id"].to_numpy()
    out: dict[int, float] = {}
    ewma, wsum = 0.0, 0.0
    for i in range(len(arr)):
        ewma = nu * ewma + (1 - nu) * arr[i]
        wsum = nu * wsum + (1 - nu)
        if i + 1 >= config.TAU_MIN:
            out[int(mids[i])] = ewma / wsum
    return out


def monthly_anchors(cpi: pl.DataFrame) -> pl.DataFrame:
    """(month_id, tau, pi_yoy) on the research stamping."""
    pi = pi_yoy(cpi)
    tau = tau_ewma(pi)
    return pl.DataFrame(
        {"month_id": list(tau), "tau": list(tau.values())},
        schema={"month_id": pl.Int64, "tau": pl.Float64},
    ).join(pi.select(pl.col("month_id").cast(pl.Int64), "pi_yoy"), on="month_id", how="left")


def research_anchors(cpi_csv: Path) -> pl.DataFrame:
    """monthly_anchors from a FRED-format CPILFENS CSV (observation_date,
    CPILFENS). Used by the private repo's s30 check, which compares this
    model against the published research forecasts on their own inputs."""
    raw = pl.read_csv(cpi_csv)
    series = [c for c in raw.columns if c != "observation_date"][0]
    cpi = raw.with_columns(pl.col("observation_date").str.to_date().alias("d")).select(
        (pl.col("d").dt.year() * 12 + pl.col("d").dt.month()).alias("month_id"),
        pl.col(series).cast(pl.Float64).alias("cpi"),
    )
    return monthly_anchors(cpi)


def _filled_pi(cpi: pl.DataFrame) -> dict[int, float]:
    """yoy with missing index levels geometrically interpolated from neighbours."""
    v = cpi.sort("month_id")["cpi"].to_numpy().astype(float)
    mids = cpi.sort("month_id")["month_id"].to_numpy()
    for i in np.where(~np.isfinite(v))[0]:
        if 0 < i < len(v) - 1 and np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]):
            v[i] = np.sqrt(v[i - 1] * v[i + 1])
    return {int(mids[i]) + 1: float(np.log(v[i] / v[i - 12]))
            for i in range(12, len(v))
            if np.isfinite(v[i]) and np.isfinite(v[i - 12])}


def daily_anchors(keys: list[int], cpi: pl.DataFrame, latest: dt.date) -> pl.DataFrame:
    """(month_id=day key, tau, pi_yoy, cpi_month, pi_filled) as known each day."""
    anch = monthly_anchors(cpi)
    tau_map = dict(zip(anch["month_id"].to_list(), anch["tau"].to_list()))
    pi_map = {m: p for m, p in zip(anch["month_id"].to_list(), anch["pi_yoy"].to_list())
              if p is not None}
    pi_fill = _filled_pi(cpi)

    cpi = cpi.sort("month_id")
    # month_id = year*12 + month, so (year, month) is divmod on month_id - 1
    obs = [(lambda y, i: (y, i + 1))(*divmod(int(m) - 1, 12))
           for m, v in zip(cpi["month_id"], cpi["cpi"]) if v is not None]
    last_obs = obs[-1]

    def avail(ym: tuple[int, int]) -> dt.date:
        y, m = (ym[0] + 1, 1) if ym[1] == 12 else (ym[0], ym[1] + 1)
        d = dt.date(y, m, 15)
        return min(d, latest) if ym == last_obs else d

    avail_keys = np.array([day_key(avail(ym)) for ym in obs])
    tau_stamps = np.array(sorted(tau_map))
    rows = []
    for k in keys:
        i = int(np.searchsorted(avail_keys, k, side="right")) - 1
        if i < 0:
            continue
        y, m = obs[i]
        stamp = y * 12 + m + 1
        j = int(np.searchsorted(tau_stamps, stamp, side="right")) - 1
        if j < 0:
            continue
        tau = tau_map[int(tau_stamps[j])]
        pi = pi_map.get(stamp)
        filled = pi is None
        if filled:
            pi = pi_fill.get(stamp)
        if pi is None:
            continue
        rows.append({"month_id": k, "tau": tau, "pi_yoy": pi,
                     "cpi_month": f"{y:04d}-{m:02d}", "pi_filled": filled})
    return pl.DataFrame(rows, schema={"month_id": pl.Int64, "tau": pl.Float64,
                                      "pi_yoy": pl.Float64, "cpi_month": pl.Utf8,
                                      "pi_filled": pl.Boolean})
