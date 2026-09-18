"""Carry + roll-down: the h-aged par bond priced on today's curve.

For a par bond issued at t with tenor N years and coupon c (pct), held h
months (h in {1,3,12}):
- coupons received in the window: c/2 per payment date 6j <= h
- remaining flows at the pricing date sit at months 6j - h (coupons with
  6j > h) plus principal at 12N - h; all integer monthly maturities
- P_aged(t)   prices those flows on the curve at t   -> carry+rolldown
- R_CR = (P_aged(t) + coupons received)/100 - 1
- rf(h) = 1/D_t(h) - 1
"""

from __future__ import annotations

import polars as pl

from . import config


def _aged_schedule(tenors: list[int] | None = None,
                   horizons: list[int] | None = None) -> pl.DataFrame:
    """Remaining cash flows of an h-aged N-year par bond, by pricing-time maturity."""
    rows = []
    for n in (tenors if tenors is not None else config.TENORS):
        for h in (horizons if horizons is not None else config.HORIZONS):
            j_min = h // 6 + 1
            for j in range(j_min, 2 * n + 1):
                rows.append(
                    {"tenor_years": n, "horizon_m": h, "flow_month": 6 * j - h, "kind": "coupon"}
                )
            rows.append(
                {"tenor_years": n, "horizon_m": h, "flow_month": 12 * n - h, "kind": "principal"}
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("tenor_years").cast(pl.Int32),
        pl.col("horizon_m").cast(pl.Int32),
        pl.col("flow_month").cast(pl.Int32),
    )


def build_aged_pv_components(dfs: pl.DataFrame,
                             tenors: list[int] | None = None,
                             horizons: list[int] | None = None) -> pl.DataFrame:
    """A = sum of D over remaining coupon months, B = D at principal month,
    per (pricing month_id, tenor, horizon)."""
    sched = _aged_schedule(tenors, horizons)
    joined = sched.join(dfs.rename({"maturity_months": "flow_month"}), on="flow_month", how="inner")
    return (
        joined.group_by("month_id", "tenor_years", "horizon_m")
        .agg(
            pl.col("df").filter(pl.col("kind") == "coupon").sum().alias("coup_sum"),
            pl.col("df").filter(pl.col("kind") == "principal").sum().alias("prin_df"),
            pl.len().alias("n_flows"),
        )
        # complete schedules only: early-sample curves lack long maturities
        .filter(pl.col("n_flows") == 2 * pl.col("tenor_years") - pl.col("horizon_m") // 6 + 1)
        .drop("n_flows")
    )
