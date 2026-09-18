"""Zero curve from CMT par yields, then synthetic par bonds.

Copied verbatim from the research pipeline (bondlab/curve.py) apart from the
bootstrap, which is the live-data entry point; s30 in the private repo checks
that forecasts built this way reproduce the published research ones exactly.

Conventions:
- D_t(m) = exp(-(m/12) * y_t(m)/100), m in months, y in pct points (cc, ann.)
- par bonds pay semiannual coupons, face 100; the annual coupon rate (pct)
  solves 100 = (c/2) * sum_j D(6j) + 100 * D(12N)
- Macaulay duration is PV-weighted time using zero-curve discounting
- modified duration = Macaulay / (1 + c/200) (semiannual par-yield convention)
- convexity is effective: parallel +/-10bp bump of the cc zero curve,
  (P+ + P- - 2P0) / (P0 * dy^2) with dy = 0.001
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import config

BUMP = 0.001  # 10bp parallel bump, decimal


def _coupon_schedule(max_tenor: int = 30) -> pl.DataFrame:
    """Rows (tenor_years, j, flow_month=6j) for j = 1..2N."""
    rows = [
        {"tenor_years": n, "j": j, "flow_month": 6 * j}
        for n in range(1, max_tenor + 1)
        for j in range(1, 2 * n + 1)
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("tenor_years").cast(pl.Int32),
        pl.col("j").cast(pl.Int32),
        pl.col("flow_month").cast(pl.Int32),
    )


def build_discount_factors(yields_monthly: pl.DataFrame) -> pl.DataFrame:
    return yields_monthly.select(
        "month_id",
        "maturity_months",
        ((-(pl.col("maturity_months") / 12.0) * (pl.col("yield_pct") / 100.0)).exp()).alias("df"),
    )


def build_par_bonds(dfs: pl.DataFrame) -> pl.DataFrame:
    """Par coupon, price check, durations, convexity for annual tenors 1..30."""
    sched = _coupon_schedule()
    joined = sched.join(
        dfs.rename({"maturity_months": "flow_month"}), on="flow_month", how="inner"
    ).with_columns(
        (pl.col("df") * ((-pl.col("flow_month") / 12.0) * BUMP).exp()).alias("df_up"),
        (pl.col("df") * ((pl.col("flow_month") / 12.0) * BUMP).exp()).alias("df_dn"),
        (pl.col("j") / 2.0 * pl.col("df")).alias("t_df"),
    )
    sums = joined.group_by("month_id", "tenor_years").agg(
        pl.col("df").sum().alias("ann_sum"),
        pl.col("df_up").sum().alias("ann_sum_up"),
        pl.col("df_dn").sum().alias("ann_sum_dn"),
        pl.col("t_df").sum().alias("tw_sum"),
        pl.len().alias("n_flows"),
    ).filter(pl.col("n_flows") == 2 * pl.col("tenor_years")).drop("n_flows")
    fin = dfs.with_columns((pl.col("maturity_months") // 12).alias("tenor_years")).filter(
        pl.col("maturity_months") % 12 == 0
    ).select(
        "month_id",
        pl.col("tenor_years").cast(pl.Int32),
        pl.col("df").alias("d_fin"),
        (pl.col("df") * ((-pl.col("maturity_months") / 12.0) * BUMP).exp()).alias("d_fin_up"),
        (pl.col("df") * ((pl.col("maturity_months") / 12.0) * BUMP).exp()).alias("d_fin_dn"),
    )
    out = sums.join(fin, on=["month_id", "tenor_years"], how="inner").with_columns(
        (2.0 * 100.0 * (1.0 - pl.col("d_fin")) / pl.col("ann_sum")).alias("par_coupon_pct")
    )
    out = out.with_columns(
        (
            pl.col("par_coupon_pct") / 2.0 * pl.col("ann_sum")
            + 100.0 * pl.col("d_fin")
        ).alias("price_check"),
        (
            (
                pl.col("par_coupon_pct") / 2.0 * pl.col("tw_sum")
                + 100.0 * pl.col("tenor_years") * pl.col("d_fin")
            )
            / 100.0
        ).alias("macaulay_dur"),
    )
    out = out.with_columns(
        (pl.col("macaulay_dur") / (1.0 + pl.col("par_coupon_pct") / 200.0)).alias("modified_dur"),
        (
            (
                pl.col("par_coupon_pct") / 2.0 * (pl.col("ann_sum_up") + pl.col("ann_sum_dn"))
                + 100.0 * (pl.col("d_fin_up") + pl.col("d_fin_dn"))
                - 2.0 * pl.col("price_check")
            )
            / (pl.col("price_check") * BUMP * BUMP)
        ).alias("convexity"),
    )
    return out.select(
        "month_id",
        "tenor_years",
        "par_coupon_pct",
        "price_check",
        "macaulay_dur",
        "modified_dur",
        "convexity",
    ).sort("month_id", "tenor_years")


def _lin(x_nodes: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of y (rows x nodes) at x, flat outside."""
    xc = np.clip(x, x_nodes[0], x_nodes[-1])
    i = np.clip(np.searchsorted(x_nodes, xc, side="right"), 1, len(x_nodes) - 1)
    w = (xc - x_nodes[i - 1]) / (x_nodes[i] - x_nodes[i - 1])
    return y[:, i - 1] * (1.0 - w) + y[:, i] * w


def cmt_to_zeros(panel: pl.DataFrame, method: str = "pchip") -> pl.DataFrame:
    """Long (month_id=key, maturity_months, yield_pct) continuously compounded
    zero yields in percent on the 1..max monthly grid, one curve per row.

    Par yields are interpolated (PCHIP by default: s30 shows it tracks the
    Liu-Wu forecasts best since 2016) to a semiannual grid; points out to 1Y
    are bills, read as bond-equivalent zeros, and longer points bootstrap
    semiannual-coupon discount factors. Zeros between nodes are linear.
    Rows are processed in groups sharing the same available maturities."""
    mcols = [c for c in panel.columns if c.startswith("cmt_")]
    mats = np.array([float(c[4:]) for c in mcols])
    vals = panel.select(mcols).to_numpy().astype(float)
    keys_all = panel["key"].to_numpy().astype(np.int64)
    avail = np.isfinite(vals)
    patterns, inverse = np.unique(avail, axis=0, return_inverse=True)
    keys, mm, yy = [], [], []
    for p_idx, pattern in enumerate(patterns):
        m = mats[pattern]
        if m.size < 5 or m.max() < 120 or m.min() > config.BILL_MAX_M:
            continue
        rows = np.where(inverse.ravel() == p_idx)[0]
        y = vals[rows][:, pattern]
        max_m = int(m.max())
        semi = np.arange(6, max_m + 1, 6, dtype=float)
        if method == "pchip":
            from scipy.interpolate import PchipInterpolator
            ps = PchipInterpolator(m, y, axis=1)(np.clip(semi, m[0], m[-1]))
        else:
            ps = _lin(m, y, semi)
        dfs = np.empty_like(ps)
        cum = np.zeros(len(rows))
        for i, t in enumerate(semi):
            if t <= config.BILL_MAX_M:
                dfs[:, i] = (1.0 + ps[:, i] / 200.0) ** (-2.0 * t / 12.0)
            else:
                c = ps[:, i] / 2.0
                dfs[:, i] = (100.0 - c * cum) / (100.0 + c)
            cum += dfs[:, i]
        # bill quotes inside six months enter directly as zeros
        short = m < 6
        nodes = np.concatenate([m[short], semi])
        z_nodes = np.hstack([2.0 * np.log1p(y[:, short] / 200.0), -np.log(dfs) / (semi / 12.0)])
        grid = np.arange(1, max_m + 1, dtype=float)
        z = 100.0 * _lin(nodes, z_nodes, grid)
        keys.append(np.repeat(keys_all[rows], max_m))
        mm.append(np.tile(grid.astype(np.int32), len(rows)))
        yy.append(z.ravel())
    return pl.DataFrame({
        "month_id": np.concatenate(keys),
        "maturity_months": np.concatenate(mm),
        "yield_pct": np.concatenate(yy),
    }).sort("month_id", "maturity_months")
