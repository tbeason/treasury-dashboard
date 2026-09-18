"""Offline regression test: fixed inputs must produce fixed outputs.

The math here was copied from a research pipeline that validates it against
published results (its s30 check reproduces the study's forecasts exactly).
This test is the guard that the copy keeps behaving: it pins one as-of date's
forecasts, risk and anchors to the last bit. It needs no network and no key.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from tsyfwd import anchors, config, curve, model, sources

FIXTURES = Path(__file__).parent / "fixtures"
ASOF = dt.date(2026, 9, 17)
TOL = 1e-12
VALUE_COLS = ["yhat", "m0", "tilt", "sigma", "sharpe", "modified_dur",
              "par_coupon_pct", "rf_simple", "z_n", "z_10", "tau", "pi_yoy"]


@pytest.fixture(scope="module")
def computed() -> pl.DataFrame:
    panel = pl.read_csv(FIXTURES / "panel.csv", try_parse_dates=True).with_columns(
        pl.col("date").map_elements(sources.day_key, return_dtype=pl.Int64).alias("key"))
    cpi = pl.read_csv(FIXTURES / "cpi.csv").select(
        (pl.col("year") * 12 + pl.col("month")).alias("month_id"), "cpi")
    zeros = curve.cmt_to_zeros(panel)
    sig = model.rm2_sig_dy(zeros, model.month_end_keys(panel))
    anch = anchors.daily_anchors(panel["key"].to_list(), cpi, ASOF)
    fc = model.attach_rm2(model.frozen_from_zeros(zeros, anch), sig)
    return fc.filter(pl.col("month_id") == sources.day_key(ASOF)).sort("horizon_m", "tenor_years")


@pytest.fixture(scope="module")
def expected() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "expected.csv").sort("horizon_m", "tenor_years")


def test_shape(computed, expected):
    assert computed.height == expected.height == len(config.TENORS) * len(config.HORIZONS)


@pytest.mark.parametrize("col", VALUE_COLS)
def test_values_match(computed, expected, col):
    got = computed.sort("horizon_m", "tenor_years")[col].to_numpy()
    want = expected[col].to_numpy()
    worst = max(abs(g - w) for g, w in zip(got, want))
    assert worst < TOL, f"{col} drifted by {worst:.3e}"


def test_identity_yhat_is_m0_plus_tilt(computed):
    d = computed.select((pl.col("yhat") - pl.col("m0") - pl.col("tilt")).abs().max()).item()
    assert d < 1e-15


def test_tilt_is_phi_times_duration_times_mean_gap(computed):
    """The frozen tilt, spelled out independently of how model.py builds it."""
    for r in computed.iter_rows(named=True):
        gap = 0.5 * ((r["z_10"] - r["tau"]) + (r["z_n"] - r["pi_yoy"]))
        want = config.phi_h(config.PHI_FROZEN, r["horizon_m"]) * r["modified_dur"] * gap
        assert abs(r["tilt"] - want) < 1e-15


def test_sharpe_is_yhat_over_sigma(computed):
    d = computed.select((pl.col("sharpe") - pl.col("yhat") / pl.col("sigma")).abs().max()).item()
    assert d < 1e-15


def test_phi_h_is_annual_phi_at_one_year():
    assert config.phi_h(config.PHI_FROZEN, 12) == pytest.approx(config.PHI_FROZEN, abs=1e-15)
