"""The frozen forecast, and the RM2 risk model around it.

    yhat = (R_CR - rf) + phi_h * D * 1/2 [(z_10Y - tau) + (z_N - pi)]

Nothing is estimated: phi is frozen at 0.15/yr, the two anchors get equal
weight, and tau is a fixed EWMA of core CPI. R_CR prices the h-aged par bond
on today's curve (carry plus roll-down), rf is the h-month zero, and D is the
new par bond's modified duration. All returns are decimals over h months.

The private research repo's s30 check runs this same code on the Liu-Wu curve
and confirms it reproduces the published s19 forecasts exactly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from . import anchors as anchors_mod
from . import config, curve, returns, sources


def frozen_from_zeros(zeros: pl.DataFrame, anchors: pl.DataFrame,
                      tenors: list[int] | None = None,
                      horizons: list[int] | None = None) -> pl.DataFrame:
    """The frozen forecast on any zero curve. `zeros` is long (month_id,
    maturity_months, yield_pct); `anchors` is (month_id, tau, pi_yoy, ...),
    where month_id is whatever integer date key the curve uses."""
    tenors = tenors or config.TENORS
    horizons = horizons or config.HORIZONS
    zeros = zeros.with_columns(pl.col("month_id").cast(pl.Int64),
                               pl.col("maturity_months").cast(pl.Int32))
    anchors = anchors.with_columns(pl.col("month_id").cast(pl.Int64))
    dfs = curve.build_discount_factors(zeros)
    par = curve.build_par_bonds(dfs).filter(pl.col("tenor_years").is_in(tenors)).select(
        "month_id", "tenor_years", "par_coupon_pct", "modified_dur")
    comp = returns.build_aged_pv_components(dfs, tenors=tenors, horizons=horizons)
    rf = dfs.filter(pl.col("maturity_months").is_in(horizons)).select(
        "month_id", pl.col("maturity_months").alias("horizon_m"),
        (1.0 / pl.col("df") - 1.0).alias("rf_simple"))
    z_n = zeros.filter((pl.col("maturity_months") % 12 == 0)
                       & (pl.col("maturity_months") // 12).is_in(tenors)).select(
        "month_id", (pl.col("maturity_months") // 12).cast(pl.Int32).alias("tenor_years"),
        (pl.col("yield_pct") / 100.0).alias("z_n"))
    z10 = zeros.filter(pl.col("maturity_months") == 120).select(
        "month_id", (pl.col("yield_pct") / 100.0).alias("z_10"))
    phi = pl.DataFrame({"horizon_m": horizons,
                        "phi_h": [config.phi_h(config.PHI_FROZEN, h) for h in horizons]},
                       schema={"horizon_m": pl.Int32, "phi_h": pl.Float64})

    out = (
        par.join(comp, on=["month_id", "tenor_years"], how="inner")
        .join(rf, on=["month_id", "horizon_m"], how="inner")
        .join(z_n, on=["month_id", "tenor_years"], how="inner")
        .join(z10, on="month_id", how="inner")
        .join(anchors, on="month_id", how="inner")
        .join(phi, on="horizon_m", how="inner")
        .with_columns(
            ((pl.col("par_coupon_pct") / 2.0 * pl.col("coup_sum") + 100.0 * pl.col("prin_df")
              + pl.col("par_coupon_pct") / 2.0 * (pl.col("horizon_m") // 6)) / 100.0 - 1.0
             ).alias("r_cr"),
            (pl.col("z_10") - pl.col("tau")).alias("gap_cycle"),
            (pl.col("z_n") - pl.col("pi_yoy")).alias("gap_value"),
        )
        .with_columns((pl.col("r_cr") - pl.col("rf_simple")).alias("m0"))
    )
    # same operation order as the research code: average the two single-anchor
    # forecasts, so the ensemble matches it bit for bit
    fc = pl.col("m0") + pl.col("phi_h") * pl.col("modified_dur") * pl.col("gap_cycle")
    fv = pl.col("m0") + pl.col("phi_h") * pl.col("modified_dur") * pl.col("gap_value")
    out = out.with_columns(((fc + fv) / 2.0).alias("yhat")).with_columns(
        (pl.col("yhat") - pl.col("m0")).alias("tilt"))
    return out.drop("coup_sum", "prin_df").sort("month_id", "horizon_m", "tenor_years")


# --------------------------------------------------------------------------
# risk (research RM2 + the conditional Sharpe ratio)
# --------------------------------------------------------------------------

def month_end_keys(panel: pl.DataFrame) -> pl.DataFrame:
    """(month_id, key) of the last Treasury business day of each month."""
    return (
        panel.with_columns((pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).alias("mid"))
        .group_by("mid").agg(pl.col("key").max())
        .sort("mid").rename({"mid": "month_id"})
    )


def rm2_sig_dy(zeros: pl.DataFrame, me: pl.DataFrame) -> pl.DataFrame:
    """RM2's yield-vol leg: the rolling std of monthly par-coupon changes, per
    tenor, stamped at the month it uses through."""
    me_zeros = (
        zeros.join(me.select("month_id", pl.col("key").alias("month_id_day")),
                   left_on="month_id", right_on="month_id_day", how="inner")
        .select(pl.col("month_id_right").alias("month_id"), "maturity_months", "yield_pct")
    )
    par = curve.build_par_bonds(curve.build_discount_factors(me_zeros)).filter(
        pl.col("tenor_years").is_in(config.TENORS))
    return (
        par.sort("tenor_years", "month_id")
        .with_columns((pl.col("par_coupon_pct").diff().over("tenor_years") / 100.0).alias("dy"))
        .with_columns(pl.col("dy").rolling_std(window_size=config.RM2_WINDOW,
                                               min_samples=config.RM2_MIN)
                      .over("tenor_years").alias("sig_dy"))
        .drop_nulls("sig_dy")
        .join(me, on="month_id", how="inner")
        .select("month_id", "key", "tenor_years", "sig_dy")
    )


def attach_rm2(fc: pl.DataFrame, sig: pl.DataFrame) -> pl.DataFrame:
    """sigma = D x sig_dy x sqrt(h) from the latest month-end on or before each
    day, plus the conditional Sharpe ratio E[rx] / sigma."""
    out = fc.sort("month_id").join_asof(
        sig.sort("key").select("key", "tenor_years", "sig_dy"), left_on="month_id",
        right_on="key", by="tenor_years", strategy="backward", check_sortedness=False)
    return out.with_columns(
        (pl.col("modified_dur") * pl.col("sig_dy") * pl.col("horizon_m").sqrt()).alias("sigma")
    ).with_columns((pl.col("yhat") / pl.col("sigma")).alias("sharpe"))


# --------------------------------------------------------------------------
# one call for the dashboard
# --------------------------------------------------------------------------

def build(history_years: int = config.HISTORY_YEARS,
          today: dt.date | None = None) -> dict:
    """Daily forecasts from Jan 1 of `history_years` ago to the latest close.
    The panel reaches further back so RM2's rolling window is full on the
    first displayed day."""
    today = today or dt.date.today()
    warnings: list[str] = []
    panel = sources.load_par_panel(
        today.year - history_years - config.RISK_LOOKBACK_YEARS, today, warnings)
    zeros = curve.cmt_to_zeros(panel)
    sig = rm2_sig_dy(zeros, month_end_keys(panel))

    first = sources.day_key(dt.date(today.year - history_years, 1, 1))
    panel = panel.filter(pl.col("key") >= first)
    zeros = zeros.filter(pl.col("month_id") >= first)
    cpi = sources.load_core_cpi(today, warnings)
    keys = panel["key"].to_list()
    anchors = anchors_mod.daily_anchors(keys, cpi, sources.key_date(max(keys)))
    fc = attach_rm2(frozen_from_zeros(zeros, anchors), sig).with_columns(
        pl.col("month_id").map_elements(sources.key_date, return_dtype=pl.Date).alias("date"))
    return {
        "today": today,
        "forecasts": fc,
        "zeros": zeros,
        "par": panel,
        "asof": fc["date"].max(),
        "cmt_last": panel["date"].max(),
        "warnings": warnings,
    }
