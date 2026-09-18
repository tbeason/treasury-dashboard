"""Public data: the Treasury par yield curve and core CPI.

Each source has a fallback and a cache, because the build runs unattended:

  curve    Treasury daily par yield CSVs  ->  FRED DGS*  ->  cache
  core CPI FRED CPILFENS                  ->  BLS v1     ->  cache

Treasury leads for the curve because its file carries the 1.5/2/4-month
points FRED's DGS series lack. FRED leads for CPI because the BLS public
endpoint is capped at 25 calls a day per IP, which shared CI runners can
exhaust; a FRED_API_KEY lifts that. Downloads are cached under .cache/, and
a failed download falls back to the cache and records a warning shown on the
page. Only if a source yields nothing AND has no cache does the build fail,
with a message naming what it tried.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

from . import config

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={y}&page&_format=csv"
)
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_CORE_NSA = "CUUR0000SA0L1E"  # = FRED CPILFENS
FRED_CORE_NSA = "CPILFENS"  # core CPI, not seasonally adjusted (never revised)

# Treasury CSV column -> maturity in months
CMT_COLUMNS = {
    "1 Mo": 1.0, "1.5 Month": 1.5, "2 Mo": 2.0, "3 Mo": 3.0, "4 Mo": 4.0,
    "6 Mo": 6.0, "1 Yr": 12.0, "2 Yr": 24.0, "3 Yr": 36.0, "5 Yr": 60.0,
    "7 Yr": 84.0, "10 Yr": 120.0, "20 Yr": 240.0, "30 Yr": 360.0,
}
# FRED fallback: the same CMT quotes, minus the 1.5/2/4-month points
FRED_CMT_SERIES = {
    "DGS1MO": 1.0, "DGS3MO": 3.0, "DGS6MO": 6.0, "DGS1": 12.0, "DGS2": 24.0,
    "DGS3": 36.0, "DGS5": 60.0, "DGS7": 84.0, "DGS10": 120.0, "DGS20": 240.0,
    "DGS30": 360.0,
}
EPOCH = dt.date(1970, 1, 1)
# core CPI history shipped with the package, so tau's long EWMA is right even
# on a cold run with no key; the live sources only extend it
SEED_CPI = Path(__file__).resolve().parent / "data" / "cpi_core_nsa_seed.csv"


def day_key(d: dt.date) -> int:
    return (d - EPOCH).days


def key_date(k: int) -> dt.date:
    return EPOCH + dt.timedelta(days=int(k))


def fred_key() -> str | None:
    return os.environ.get("FRED_API_KEY") or None


def _get(url: str, data: bytes | None = None, timeout: int = 45) -> bytes:
    req = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": "tsyfwd-dashboard/1.0",
                 **({"Content-Type": "application/json"} if data else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fred_observations(series_id: str, timeout: int = 45) -> list[tuple[dt.date, float]]:
    """(date, value) pairs for a FRED series; '.' marks a missing day."""
    key = fred_key()
    if not key:
        raise RuntimeError("no FRED_API_KEY")
    url = FRED_URL + "?" + urllib.parse.urlencode(
        {"series_id": series_id, "api_key": key, "file_type": "json"})
    js = json.loads(_get(url, timeout=timeout))
    out = []
    for o in js["observations"]:
        try:
            out.append((dt.date.fromisoformat(o["date"]), float(o["value"])))
        except ValueError:  # "." = no observation that day
            continue
    return out


# --------------------------------------------------------------------------
# the par yield curve
# --------------------------------------------------------------------------

def _treasury_year(year: int, today: dt.date, warnings: list[str]) -> Path | None:
    path = config.CACHE_DIR / "treasury" / f"par_{year}.csv"
    # past years are final; the current year (and last year during January,
    # when the Dec-31 print can still arrive) is refreshed on every run
    fresh = year >= today.year - (1 if today.month == 1 else 0)
    if path.exists() and not fresh:
        return path
    try:
        body = _get(TREASURY_URL.format(y=year))
        if not body.startswith(b"Date"):
            raise ValueError("unexpected response")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    except Exception as e:  # noqa: BLE001 - any network failure -> cache
        if not path.exists():
            warnings.append(f"Treasury {year}: download failed ({e}) and no cache")
            return None
        warnings.append(f"Treasury {year}: download failed ({e}); using cached copy")
    return path


def _par_panel_treasury(start_year: int, today: dt.date,
                        warnings: list[str]) -> pl.DataFrame | None:
    frames = []
    for year in range(start_year, today.year + 1):
        path = _treasury_year(year, today, warnings)
        if path is None:
            continue
        raw = pl.read_csv(path, infer_schema_length=0)
        cols = [c for c in raw.columns if c in CMT_COLUMNS]
        frames.append(raw.select(
            pl.col("Date").str.to_date("%m/%d/%Y").alias("date"),
            *[pl.col(c).cast(pl.Float64, strict=False).alias(f"cmt_{CMT_COLUMNS[c]:g}")
              for c in cols],
        ))
    return pl.concat(frames, how="diagonal_relaxed") if frames else None


def _par_panel_fred(start_year: int, warnings: list[str]) -> pl.DataFrame | None:
    frames = []
    for series, months in FRED_CMT_SERIES.items():
        try:
            obs = [(d, v) for d, v in _fred_observations(series) if d.year >= start_year]
        except Exception as e:  # noqa: BLE001
            warnings.append(f"FRED {series}: {e}")
            return None
        frames.append(pl.DataFrame(
            {"date": [d for d, _ in obs], f"cmt_{months:g}": [v for _, v in obs]},
            schema={"date": pl.Date, f"cmt_{months:g}": pl.Float64}))
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.join(f, on="date", how="full", coalesce=True)
    return panel


def load_par_panel(start_year: int, today: dt.date | None = None,
                   warnings: list[str] | None = None) -> pl.DataFrame:
    """Daily CMT par yields (percent) with columns date, key, cmt_<months>."""
    today = today or dt.date.today()
    warnings = warnings if warnings is not None else []
    panel = _par_panel_treasury(start_year, today, warnings)
    if panel is None:
        warnings.append("Treasury unavailable; falling back to FRED DGS series")
        panel = _par_panel_fred(start_year, warnings)
    if panel is None:
        raise RuntimeError(
            "no par yield curve: Treasury download failed with no cache in "
            f"{config.CACHE_DIR / 'treasury'}, and the FRED fallback did not "
            "return data (set FRED_API_KEY to enable it). "
            + "; ".join(warnings))
    mcols = sorted([c for c in panel.columns if c.startswith("cmt_")],
                   key=lambda c: float(c[4:]))
    return (
        panel.select("date", *mcols)
        .unique("date", keep="last")
        .sort("date")
        .with_columns(pl.col("date").map_elements(day_key, return_dtype=pl.Int64).alias("key"))
    )


# --------------------------------------------------------------------------
# core CPI
# --------------------------------------------------------------------------

def _cpi_fred(warnings: list[str]) -> dict[tuple[int, int], float] | None:
    if not fred_key():
        return None  # not configured: fall through to BLS without a page warning
    try:
        obs = _fred_observations(FRED_CORE_NSA)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"FRED {FRED_CORE_NSA}: {e}")
        return None
    return {(d.year, d.month): v for d, v in obs}


def _read_levels_csv(path: Path) -> dict[tuple[int, int], float]:
    if not path.exists():
        return {}
    rows = pl.read_csv(path).iter_rows(named=True)
    return {(int(r["year"]), int(r["month"])): r["cpi"] for r in rows if r["cpi"] is not None}


def _cpi_bls(warnings: list[str], today: dt.date) -> dict[tuple[int, int], float] | None:
    """BLS public v1: no key needed, but capped at 25 calls/day per IP and ten
    years per call, so it tops up recent months rather than building history."""
    levels: dict[tuple[int, int], float] = {}
    try:
        body = _get(BLS_URL, json.dumps({
            "seriesid": [BLS_CORE_NSA],
            "startyear": str(today.year - 9), "endyear": str(today.year),
        }).encode())
        js = json.loads(body)
        if js.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(js.get("message"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"BLS CPI: {e}")
        return None
    for o in js["Results"]["series"][0]["data"]:
        if not o["period"].startswith("M") or o["period"] == "M13":
            continue
        try:
            levels[(int(o["year"]), int(o["period"][1:]))] = float(o["value"])
        except ValueError:  # "-" = not collected (e.g. Oct 2025)
            continue
    return levels or None


def load_core_cpi(today: dt.date | None = None,
                  warnings: list[str] | None = None) -> pl.DataFrame:
    """(month_id, cpi) for core CPI NSA, contiguous monthly, nulls where a
    print does not exist.

    tau is an EWMA with a ~53-month half-life and needs the full history to
    match the research value, so the package ships the series as a seed and
    the live sources only extend it: seed < cache < BLS < FRED."""
    today = today or dt.date.today()
    warnings = warnings if warnings is not None else []
    cache = config.CACHE_DIR / "cpi_core_nsa.csv"

    levels = _read_levels_csv(SEED_CPI)
    levels.update(_read_levels_csv(cache))
    fresh = _cpi_fred(warnings)
    if fresh is None:
        fresh = _cpi_bls(warnings, today)
    if fresh:
        levels.update(fresh)
    elif not levels:
        raise RuntimeError(
            "no core CPI: FRED failed (set FRED_API_KEY), BLS failed, and there "
            f"is no cache at {cache} or seed at {SEED_CPI}. " + "; ".join(warnings))
    else:
        warnings.append("core CPI is from cache; it may be missing the latest print")

    latest = max(levels)
    stale_months = (today.year * 12 + today.month) - (latest[0] * 12 + latest[1])
    if stale_months > 2:
        warnings.append(f"Latest core CPI print is {latest[0]}-{latest[1]:02d}, "
                        f"{stale_months} months behind.")

    first, last = min(levels), max(levels)
    rows = []
    y, m = first
    while (y, m) <= last:
        rows.append({"year": y, "month": m, "cpi": levels.get((y, m))})
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    out = pl.DataFrame(rows, schema={"year": pl.Int32, "month": pl.Int32, "cpi": pl.Float64})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(cache)
    return out.select((pl.col("year") * 12 + pl.col("month")).alias("month_id"), "cpi")
