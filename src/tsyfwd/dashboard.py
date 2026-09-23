"""The dashboard page: one self-contained HTML file.

Python assembles a JSON payload (current table, daily history, anchors, model
card); the page draws it with inline SVG and vanilla JS, so it needs no
plotting dependency. Returns are h-month simple returns.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from . import config, model, refs

HISTORY_NAME = "forecast_history.csv"
STALE_BUSINESS_DAYS = 3
# project Pages sites inherit the account custom domain (tbeason.com);
# the tbeason.github.io address 301-redirects here
SITE_URL = "https://tbeason.com/treasury-dashboard/"


def _r(x, nd=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _value_on_or_before(hist: pl.DataFrame, d: dt.date) -> float | None:
    s = hist.filter(pl.col("date") <= d)
    return None if s.is_empty() else float(s["yhat"][-1])


def _model_card() -> dict:
    """Fixed numbers from the research study (refs.py), not recomputed here."""
    return {
        "oos": {str(h): {str(t): {"frozen": v[0], "carry": v[1]}
                         for t, v in by_tenor.items()}
                for h, by_tenor in refs.OOS_R2.items()},
        "oos_sample": refs.OOS_SAMPLE,
        "regime": {"half_B": refs.REGIME_HALF_B, "month": refs.REGIME_ASOF,
                   "band": list(refs.REGIME_BAND)},
        "basis": {str(t): list(v) for t, v in refs.CURVE_BASIS_BP.items()},
        "basis_sample": refs.CURVE_BASIS_SAMPLE,
    }


def load_history(published_url: str | None = None) -> pl.DataFrame | None:
    """The audit trail of what the page showed each day. CI runs start with an
    empty workspace, so the published copy is pulled back in first."""
    local = config.CACHE_DIR / HISTORY_NAME
    if not local.exists() and published_url:
        try:
            with urllib.request.urlopen(published_url, timeout=30) as r:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(r.read())
        except Exception:  # noqa: BLE001 - first ever run, or the page is down
            return None
    if not local.exists():
        return None
    return pl.read_csv(local, try_parse_dates=True, schema_overrides={"cpi_month": pl.Utf8})


def payload(res: dict, generated: dt.datetime) -> dict:
    fc: pl.DataFrame = res["forecasts"]
    asof: dt.date = res["asof"]
    dates = sorted(fc["date"].unique().to_list())
    date_idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    series: dict = {}
    current: dict = {}
    for h in config.HORIZONS:
        series[str(h)] = {}
        current[str(h)] = {}
        for t in config.TENORS:
            s = fc.filter((pl.col("horizon_m") == h) & (pl.col("tenor_years") == t)).sort("date")
            arr = {k: [None] * n for k in ("yhat", "m0", "tilt", "sigma", "sharpe")}
            for r in s.select("date", "yhat", "m0", "tilt", "sigma", "sharpe").iter_rows(named=True):
                i = date_idx[r["date"]]
                for k in arr:
                    arr[k][i] = _r(r[k])
            series[str(h)][str(t)] = arr
            last = s.filter(pl.col("date") == asof)
            if last.is_empty():
                continue
            r = last.row(0, named=True)
            prev = s.filter(pl.col("date") < asof)
            chg = {
                "d1": None if prev.is_empty() else r["yhat"] - float(prev["yhat"][-1]),
                "d1w": None, "d1m": None,
            }
            for key, days in (("d1w", 7), ("d1m", 30)):
                v = _value_on_or_before(s, asof - dt.timedelta(days=days))
                chg[key] = None if v is None else r["yhat"] - v
            current[str(h)][str(t)] = {
                "yhat": _r(r["yhat"]), "ann": _r(r["yhat"] * 12 / h), "m0": _r(r["m0"]),
                "tilt": _r(r["tilt"]), "rf": _r(r["rf_simple"]),
                "total": _r(r["yhat"] + r["rf_simple"]), "dur": _r(r["modified_dur"], 3),
                "z_n": _r(r["z_n"]), "coupon": _r(r["par_coupon_pct"] / 100.0),
                "breakeven_bp": _r(r["yhat"] / r["modified_dur"] * 1e4, 1),
                "sigma": _r(r["sigma"]), "sharpe": _r(r["sharpe"], 3),
                "gap_value": _r(r["gap_value"]),
                **{k: _r(v) for k, v in chg.items()},
            }

    anch = (fc.filter((pl.col("horizon_m") == config.HORIZONS[0])
                      & (pl.col("tenor_years") == config.TENORS[0]))
            .select("date", "z_10", "tau", "pi_yoy", "cpi_month", "pi_filled").sort("date"))
    anchors = {k: [None] * n for k in ("z_10", "tau", "pi")}
    for r in anch.iter_rows(named=True):
        i = date_idx[r["date"]]
        anchors["z_10"][i] = _r(r["z_10"])
        anchors["tau"][i] = _r(r["tau"])
        anchors["pi"][i] = _r(r["pi_yoy"])
    last_anchor = anch.filter(pl.col("date") == asof).row(0, named=True)

    warnings = list(res["warnings"])
    lag = int(np.busday_count(asof, res["today"]))
    if lag >= STALE_BUSINESS_DAYS:
        warnings.append(f"Latest Treasury curve is {asof:%a %d %b %Y}, {lag} business days old.")
    if last_anchor["pi_filled"]:
        warnings.append("Core CPI yoy for the latest print uses an interpolated Oct-2025 index "
                        "(BLS did not collect October 2025).")

    return {
        "asof": asof.isoformat(),
        "generated": generated.strftime("%Y-%m-%d %H:%M"),
        "cpi_month": last_anchor["cpi_month"],
        "anchors_now": {"z_10": _r(last_anchor["z_10"]), "tau": _r(last_anchor["tau"]),
                        "pi": _r(last_anchor["pi_yoy"])},
        "warnings": warnings,
        "tenors": config.TENORS,
        "horizons": config.HORIZONS,
        "phi": config.PHI_FROZEN,
        "dates": [d.isoformat() for d in dates],
        "series": series,
        "anchors": anchors,
        "current": current,
        "model": _model_card(),
        "site_url": SITE_URL,
    }


def append_history(res: dict, generated: dt.datetime, published_url: str | None = None) -> pl.DataFrame:
    """One row per (as-of, h, tenor); the latest run for an as-of date replaces
    earlier ones, so reruns correct rather than duplicate."""
    fc = res["forecasts"].filter(pl.col("date") == res["asof"]).select(
        pl.col("date").alias("asof"), pl.lit(generated.strftime("%Y-%m-%dT%H:%M")).alias("run_at"),
        "horizon_m", "tenor_years", "yhat", "m0", "tilt", "sigma", "sharpe",
        "rf_simple", "modified_dur", "z_n", "z_10", "tau", "pi_yoy", "cpi_month",
    )
    old = load_history(published_url)
    if old is not None:
        old = old.filter(pl.col("asof") != res["asof"])
        # diagonal: rows written before a column existed keep nulls for it
        fc = pl.concat([old, fc], how="diagonal_relaxed").select(fc.columns)
    fc = fc.sort("asof", "horizon_m", "tenor_years")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fc.write_csv(config.CACHE_DIR / HISTORY_NAME)
    return fc


def render(data: dict) -> str:
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__PAYLOAD__", blob)


def run(history_years: int = config.HISTORY_YEARS, published_url: str | None = None) -> dict:
    """Build the page and the history CSV into config.SITE_DIR."""
    generated = dt.datetime.now(ZoneInfo(config.TZ))
    res = model.build(history_years=history_years,
                      today=dt.datetime.now(ZoneInfo(config.TZ)).date())
    data = payload(res, generated)
    config.SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.SITE_DIR / "index.html"
    out.write_text(render(data), encoding="utf-8")
    hist = append_history(res, generated, published_url)
    hist.write_csv(config.SITE_DIR / HISTORY_NAME)

    lines = [f"Frozen-model expected excess returns, as of {data['asof']} close "
             f"(core CPI through {data['cpi_month']})",
             "  h   " + "".join(f"{t:>8}Y" for t in config.TENORS)]
    for h in config.HORIZONS:
        cur = data["current"][str(h)]
        lines.append(f"{h:>3}m  " + "".join(
            f"{100 * cur[str(t)]['yhat']:>8.2f}%" if str(t) in cur else f"{'-':>9}"
            for t in config.TENORS))
    for w in data["warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append(f"wrote {out}")
    print("\n".join(lines))
    return data


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Treasury Expected Returns</title>
<meta name="description" content="Daily expected excess returns on 2, 5, 10, 20 and 30-year Treasuries from a frozen forecasting model: carry and roll-down plus a pull of yields toward inflation anchors.">
<meta property="og:title" content="Treasury expected returns">
<meta property="og:description" content="Daily expected excess returns across the Treasury curve, from a model with one fixed parameter.">
<meta property="og:type" content="website">
<!-- unlisted for now: drop this line (and robots.txt) when the page is linked publicly -->
<meta name="robots" content="noindex, nofollow">
<!-- web fonts are optional: offline, the stacks below fall back to system faces -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --up: #12805a; --down: #c2410c;
  --warn-bg: #fff6e0; --warn-ink: #6b4a00; --warn-mark: #fab219;
  --sans: "Archivo", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --up: #3cc48f; --down: #ef7652;
    --warn-bg: #2a2310; --warn-ink: #f3d58a; --warn-mark: #fab219;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --up: #3cc48f; --down: #ef7652;
  --warn-bg: #2a2310; --warn-ink: #f3d58a; --warn-mark: #fab219;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--page); color: var(--ink); font: 13px/1.45 var(--sans);
  padding-bottom: env(safe-area-inset-bottom);
}
a { color: var(--ink); }
.wrap { max-width: 1440px; margin: 0 auto; }
.mono { font-family: var(--mono); }
.label { font: 700 12px/1.3 var(--sans); letter-spacing: .06em; text-transform: uppercase; }
.small { font: 11px var(--mono); color: var(--muted); }
.key { display: inline-block; width: 12px; height: 2px; border-radius: 1px; flex: none; }
.up { color: var(--up); } .down { color: var(--down); }

header.top {
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 16px 32px;
  padding: max(26px, env(safe-area-inset-top)) 32px 18px; border-bottom: 2px solid var(--ink);
}
.kicker { font: 600 11px var(--mono); letter-spacing: .1em; color: var(--ink-2); }
h1 { font-weight: 800; font-size: 34px; letter-spacing: -0.025em; line-height: 1.05; margin: 6px 0 0; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 28px; font: 11px var(--mono); color: var(--muted); }
.meta b { display: block; font-weight: 400; font-size: 13px; color: var(--ink); }
.warn { margin: 14px 32px 0; padding: 10px 14px; border-radius: 3px; background: var(--warn-bg); color: var(--warn-ink); }
.warn div::before { content: "\26A0\FE0E  "; }

.hero { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 32px; padding: 24px 32px; }
.hh { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }
.mx { display: grid; grid-template-columns: 56px repeat(5, minmax(0, 1fr)); gap: 4px; }
.mx .ch { display: flex; align-items: center; gap: 6px; padding: 0 0 4px 10px; font: 600 12px var(--mono); }
.mx .rh { display: flex; align-items: center; font: 600 13px var(--mono); color: var(--ink-2); }
.cell {
  height: 78px; padding: 10px; border: 0; border-radius: 3px; cursor: pointer; font: inherit; text-align: left;
  display: flex; flex-direction: column; justify-content: space-between;
  outline: 2px solid transparent; outline-offset: 1px;
}
.cell[aria-pressed="true"] { outline-color: var(--ink); }
.cell:focus-visible { outline-color: var(--s1); }
.cell .v { font: 500 21px var(--mono); letter-spacing: -0.02em; }
.cell .f { display: flex; justify-content: space-between; gap: 6px; font: 11px var(--mono); opacity: .8; }

.focus { border-left: 1px solid var(--grid); padding-left: 32px; display: flex; flex-direction: column; gap: 14px; }
.fh { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.fh .label { display: flex; align-items: center; gap: 8px; }
.fh .label .key { width: 14px; height: 3px; }
.fbig { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; }
.fbig b { font: 500 60px/1 var(--mono); letter-spacing: -0.04em; }
.fbig span { font: 13px var(--mono); color: var(--ink-2); }
.stack { display: flex; height: 12px; border-radius: 2px; overflow: hidden; background: var(--grid); }
.stack i { display: block; height: 100%; }
.stackleg { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 12px; margin-top: 6px; font: 11.5px var(--mono); }
.stackleg span { display: inline-flex; align-items: center; gap: 6px; }
.sw { display: inline-block; width: 8px; height: 8px; }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--ink); }
.stats div { padding: 9px 0; border-bottom: 1px solid var(--grid); }
.stats span { display: block; font: 10.5px var(--mono); color: var(--muted); letter-spacing: .04em; }
.stats b { font: 500 15px var(--mono); }
.close, .grab, .fchart-wrap, .backdrop { display: none; }

.bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px 28px; padding: 12px 32px;
  border-top: 1px solid var(--ink); border-bottom: 1px solid var(--grid);
}
.bar .small { margin-left: auto; }
.seg { display: inline-flex; border: 1px solid var(--ink); border-radius: 2px; overflow: hidden; }
.seg button {
  font: 500 12px var(--mono); border: 0; padding: 6px 12px; cursor: pointer;
  background: transparent; color: var(--ink-2);
}
.seg button:hover { background: var(--wash); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.seg button:focus-visible { outline: 2px solid var(--s1); outline-offset: -2px; }

.panel { background: var(--surface); border: 1px solid var(--ring); border-radius: 3px; }
.mult { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; padding: 16px 32px 0; }
.mp { padding: 10px 12px 6px; cursor: pointer; text-align: left; font: inherit; color: inherit; }
.mp.on { border-color: var(--ink); }
.mp:focus-visible { outline: 2px solid var(--s1); outline-offset: 1px; }
.ph { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.tn { display: inline-flex; align-items: center; gap: 6px; font: 600 12px var(--mono); }
.ph b { font: 500 15px var(--mono); }
.mp .dw, .mp .spark { display: none; }
.chart { position: relative; width: 100%; height: 140px; touch-action: pan-y; }
.chart.tall { height: 220px; }
.chart svg { display: block; width: 100%; height: 100%; overflow: visible; }

.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; padding: 16px 32px 0; }
.grid2 .panel { padding: 14px 16px 8px; }
.pt { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 4px 16px; margin-bottom: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font: 11px var(--mono); color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.tip {
  position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 3px; padding: 6px 9px; font: 12px var(--mono); box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  white-space: nowrap; z-index: 2;
}
.tip .d { color: var(--muted); margin-bottom: 3px; }
.tip .row { display: flex; align-items: center; gap: 8px; font-variant-numeric: tabular-nums; }
.tip .row b { font-weight: 600; min-width: 54px; text-align: right; }
.tip .row span:last-child { color: var(--ink-2); }

.sect { padding: 20px 32px 0; }
.sh { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 4px 16px; padding-bottom: 8px; border-bottom: 2px solid var(--ink); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font: 13px var(--mono); font-variant-numeric: tabular-nums; }
th, td { padding: 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 500; font-size: 11px; border-bottom-color: var(--axis); vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
th:last-child, td:last-child { padding-right: 0; }
#tbl tbody tr { cursor: pointer; }
#tbl tbody tr:hover { background: var(--wash); }
#tbl tbody tr.on { background: var(--wash); }
td.em { font-weight: 600; }
td .tn { font-weight: 400; font-size: 13px; gap: 8px; }
.anchors { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0 24px; }
.anchors div { padding: 10px 0; border-bottom: 1px solid var(--grid); }
.anchors b { display: block; font: 500 18px var(--mono); }
.anchors span { font-size: 12px; color: var(--ink-2); }
.note { color: var(--muted); font-size: 12px; margin: 8px 0 0; max-width: 1000px; text-wrap: pretty; }

.foot {
  display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr); gap: 24px 32px;
  padding: 24px 32px; margin-top: 24px; border-top: 2px solid var(--ink);
  font-size: 12.5px; color: var(--ink-2); line-height: 1.55;
}
.foot .label { color: var(--ink); margin-bottom: 8px; }
.foot p { margin: 0 0 6px; text-wrap: pretty; }
.formula { font: 12px var(--mono); color: var(--ink); margin-bottom: 8px; overflow-x: auto; white-space: nowrap; }
details.more { margin: 0 32px 32px; border-top: 1px solid var(--grid); }
details.more > summary {
  list-style: none; cursor: pointer; display: flex; justify-content: space-between; align-items: center;
  min-height: 44px; font: 700 12px var(--sans); letter-spacing: .06em; text-transform: uppercase;
}
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary::after { content: "+"; font: 400 16px var(--mono); color: var(--muted); }
details.more[open] > summary::after { content: "−"; }
details.more p { color: var(--ink-2); margin: 6px 0; max-width: 1000px; text-wrap: pretty; }
details.more table { margin: 6px 0 10px; }

@media (max-width: 1100px) {
  .hero { grid-template-columns: 1fr; }
  .focus { border-left: 0; padding-left: 0; border-top: 1px solid var(--grid); padding-top: 20px; }
  .mult { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .anchors { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .foot { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
  header.top { padding: max(20px, env(safe-area-inset-top)) 16px 12px; }
  h1 { font-size: 24px; margin-top: 4px; }
  .meta { gap: 4px 18px; }
  .meta b { font-size: 12px; }
  .warn { margin: 12px 16px 0; }
  .hero { padding: 14px 16px 0; gap: 0; }
  .mx { grid-template-columns: 32px repeat(5, minmax(0, 1fr)); gap: 3px; }
  .mx .ch { flex-direction: column; padding: 0 0 4px; font-size: 11px; gap: 4px; }
  .mx .rh { font-size: 11px; }
  .cell { height: 56px; padding: 4px 2px; align-items: center; justify-content: center; gap: 2px; }
  .cell .v { font-size: 13.5px; }
  .cell .f { font-size: 9.5px; }
  .cell .f span:last-child { display: none; }
  /* the focus panel becomes a bottom sheet, opened by tapping a cell or tenor row */
  .focus {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 10; gap: 10px;
    background: var(--surface); border: 0; border-radius: 16px 16px 0 0; box-shadow: 0 -8px 30px rgba(0,0,0,0.18);
    padding: 10px 16px max(22px, env(safe-area-inset-bottom)); max-height: 88vh; overflow-y: auto;
    transform: translateY(105%); transition: transform .25s ease; visibility: hidden;
  }
  body.sheet .focus { transform: none; visibility: visible; }
  body.sheet .backdrop { display: block; position: fixed; inset: 0; background: rgba(11,11,11,0.32); z-index: 9; }
  .grab { display: block; width: 40px; height: 4px; border-radius: 2px; background: var(--axis); margin: 0 auto 2px; }
  .close { display: block; width: 44px; height: 44px; margin: -10px -10px -10px 0; border: 0; background: transparent; font: 18px var(--mono); color: var(--ink-2); cursor: pointer; }
  .fbig b { font-size: 44px; }
  .fbig span { font-size: 12px; }
  .stats b { font-size: 13px; }
  .fchart-wrap { display: block; }
  .fchart-wrap .chart { height: 170px; }
  .bar { padding: 12px 16px; margin-top: 18px; gap: 10px 12px; }
  .bar .small { display: none; }
  .seg button { padding: 8px 10px; }
  .mult { grid-template-columns: 1fr; gap: 0; padding: 0 16px; }
  .mp {
    display: grid; grid-template-columns: 40px minmax(0, 1fr) 66px 52px; align-items: center; gap: 10px;
    min-height: 52px; padding: 0; background: none; border: 0; border-bottom: 1px solid var(--grid); border-radius: 0;
  }
  .mp.on { border-color: var(--grid); }
  .mp .ph { display: contents; }
  .mp .tn { order: 1; }
  .mp .spark { display: block; order: 2; height: 26px; }
  .mp .ph b { order: 3; text-align: right; }
  .mp .dw { display: block; order: 4; text-align: right; font: 11.5px var(--mono); }
  .mp .chart { display: none; }
  .grid2 { grid-template-columns: 1fr; padding: 16px 16px 0; }
  .grid2 .chart.tall { height: 190px; }
  .sect { padding: 18px 16px 0; }
  .anchors { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 16px; }
  .anchors b { font-size: 16px; }
  th, td { padding: 7px 6px; }
  table { font-size: 12px; }
  .foot { grid-template-columns: 1fr; padding: 20px 16px; gap: 18px; }
  details.more { margin: 0 16px 24px; }
}
@media (prefers-reduced-motion: reduce) { .focus { transition: none; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <div class="kicker" id="kicker"></div>
      <h1>Treasury expected returns</h1>
    </div>
    <div class="meta" id="meta"></div>
  </header>
  <div class="warn" id="warn" hidden></div>

  <div class="hero">
    <div>
      <div class="hh"><span class="label">Expected excess return · horizon × tenor</span><span class="small">shade = excess ÷ σ</span></div>
      <div class="mx" id="matrix" role="group" aria-label="Horizon and tenor"></div>
      <div class="small" style="margin-top:10px">Select a cell to focus it · cell footer: excess ÷ σ and 1-week change</div>
    </div>
    <div class="backdrop" id="backdrop"></div>
    <section class="focus" id="focus" aria-label="Selected bond" aria-live="polite">
      <span class="grab"></span>
      <div class="fh"><span class="label" id="f-title"></span><span class="small" id="f-sub"></span><button type="button" class="close" id="f-close" aria-label="Close">×</button></div>
      <div class="fbig"><b id="f-val"></b><span id="f-ann"></span></div>
      <div>
        <div class="stack"><i id="f-m0w" style="background:var(--s2)"></i><i id="f-tiltw" style="background:var(--s3)"></i></div>
        <div class="stackleg"><span><i class="sw" style="background:var(--s2)"></i><span id="f-m0"></span></span><span><i class="sw" style="background:var(--s3)"></i><span id="f-tilt"></span></span></div>
      </div>
      <div class="stats" id="f-stats"></div>
      <div class="fchart-wrap"><div class="hh" style="margin:4px 0 6px"><span class="label">History</span></div><div class="chart" id="cf"></div></div>
    </section>
  </div>

  <div class="bar" role="group" aria-label="Chart filters">
    <span class="label">History</span>
    <div class="seg" id="ctl-r"></div>
    <div class="seg" id="ctl-m"></div>
    <span class="small" id="bar-note"></span>
  </div>
  <div class="mult" id="mult"></div>

  <div class="grid2">
    <div class="panel">
      <div class="pt"><span class="label" id="c2-title"></span><div class="legend" id="c2-legend"></div></div>
      <div class="chart tall" id="c2"></div>
    </div>
    <div class="panel">
      <div class="pt"><span class="label">Rate anchors</span><div class="legend" id="c3-legend"></div></div>
      <div class="chart tall" id="c3"></div>
    </div>
  </div>

  <div class="sect">
    <div class="sh"><span class="label" id="tbl-title"></span><span class="small" id="tbl-sub"></span></div>
    <div class="tablewrap"><table id="tbl"></table></div>
    <p class="note">σ is modified duration × the 60-month rolling volatility of monthly par-yield changes, scaled by √h; excess ÷ σ is the conditional Sharpe ratio. Cushion is expected excess ÷ modified duration: the parallel yield rise, in bp, that would erase it. Δ columns are changes in expected excess. Coupons received are held as cash, not reinvested, which understates 12-month figures by roughly 5 bp at current short rates.</p>
  </div>

  <div class="sect">
    <div class="sh"><span class="label">Anchors today</span><span class="small">the gaps behind the tilt</span></div>
    <div class="anchors" id="anchors"></div>
    <p class="note">The rate-gap term averages, with equal weights, the 10-year zero yield minus trend inflation τ (EWMA of core CPI yoy, ν = 0.987) and each tenor's own zero yield minus current core CPI yoy π.</p>
  </div>

  <footer class="foot">
    <div>
      <div class="label">Model</div>
      <div class="formula">E[rx] = (carry + roll − rf) + [1 − (1 − φ)^(h/12)] · D · ½[(y₁₀ − τ) + (y_N − π)]</div>
      <p>Nothing is estimated: φ is frozen at 0.15/yr, the anchor weights are equal, and τ is a fixed EWMA of core CPI. The zero curve is bootstrapped daily from the Treasury par curve (PCHIP interpolation).</p>
    </div>
    <div>
      <div class="label">Track record</div>
      <p id="track"></p>
    </div>
    <div>
      <div class="label">Not investment advice</div>
      <p>Research output: a fixed model applied to public data, published so its forecasts can be watched in real time. The synthetic par bonds it prices ignore financing, bid-offer and taxes.</p>
      <p>Sources: <a href="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve">Treasury par yield curve</a> · <a href="https://fred.stlouisfed.org/series/CPILFENS">FRED CPILFENS</a>. <a href="forecast_history.csv">forecast_history.csv</a> · <a href="https://github.com/tbeason/treasury-dashboard">GitHub</a> · <a href="https://tbeason.com">Tyler Beason</a>.</p>
    </div>
  </footer>
  <details class="more" id="model"><summary>Full model card</summary><div id="model-body"></div></details>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
const TENOR_COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5"];
const state = { h: "12", range: "1Y", tenor: "10", metric: "yhat" };
const RANGES = { "3M": 92, "6M": 183, "1Y": 366, "3Y": 1100 };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const isPhone = () => window.matchMedia("(max-width: 640px)").matches;
// no leading "+": sign only when negative
const sgn = x => x < 0 ? "−" : "";
const pct = (x, nd = 2) => x == null ? "–" : sgn(x) + Math.abs(100 * x).toFixed(nd) + "%";
const pctPlain = (x, nd = 2) => x == null ? "–" : (100 * x).toFixed(nd) + "%";
const axisPct = (v, nd) => (v < -1e-12 ? "−" : "") + Math.abs(100 * v).toFixed(nd) + "%";
const bp = x => x == null ? "–" : sgn(x) + Math.abs(1e4 * x).toFixed(0) + " bp";
const dcls = x => x == null ? "" : x >= 0 ? "up" : "down";
const hLabel = h => ({ "1": "1-month", "3": "3-month", "12": "12-month" })[h];
const fmtDate = s => new Date(s + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
const el = (tag, attrs = {}, text) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
};
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const keyEl = color => { const k = el("span", { class: "key" }); k.style.background = color; return k; };
const tenorColor = t => css(TENOR_COLORS[D.tenors.indexOf(Number(t))]);

// header + warnings
document.getElementById("kicker").textContent = `DAILY · FROZEN MODEL · φ = ${D.phi}/YR`;
if (D.warnings.length) {
  const w = document.getElementById("warn");
  D.warnings.forEach(t => w.appendChild(el("div", {}, t)));
  w.hidden = false;
}
function renderMeta() {
  const box = document.getElementById("meta"); box.replaceChildren();
  const any = D.current[state.h][String(D.tenors[0])];
  [["Close", fmtDate(D.asof)], ["Core CPI", D.cpi_month], ["Updated", D.generated + " ET"],
   [`Risk-free ${state.h}M`, any ? pctPlain(any.rf) : "–"]].forEach(([l, v]) => {
    const d = el("div", {}, l.toUpperCase()); d.append(el("b", {}, v)); box.append(d);
  });
}

// controls
function seg(id, options, key) {
  const box = document.getElementById(id);
  options.forEach(([val, text]) => {
    const b = el("button", { type: "button", "aria-pressed": String(state[key] === val) }, text);
    b.addEventListener("click", () => {
      state[key] = val;
      box.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      renderAll();
    });
    box.appendChild(b);
  });
}
seg("ctl-r", Object.keys(RANGES).map(r => [r, r]), "range");
seg("ctl-m", [["yhat", "Excess"], ["sharpe", "Excess ÷ σ"]], "metric");

// bottom sheet (phone only; on wider screens the focus panel is inline)
const openSheet = () => { if (isPhone()) { document.body.classList.add("sheet"); renderFocusChart(); } };
const closeSheet = () => document.body.classList.remove("sheet");
document.getElementById("f-close").addEventListener("click", closeSheet);
document.getElementById("backdrop").addEventListener("click", closeSheet);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeSheet(); });
function select(h, t) { state.h = String(h); state.tenor = String(t); renderAll(); openSheet(); }

function renderMatrix() {
  const box = document.getElementById("matrix"); box.replaceChildren();
  let maxSr = 0;
  D.horizons.forEach(h => D.tenors.forEach(t => { const c = D.current[String(h)][String(t)]; if (c && c.sharpe != null) maxSr = Math.max(maxSr, c.sharpe); }));
  box.append(el("span"));
  D.tenors.forEach(t => { const s = el("span", { class: "ch" }); s.append(keyEl(tenorColor(t)), document.createTextNode(t + "Y")); box.append(s); });
  D.horizons.forEach(h => {
    box.append(el("span", { class: "rh" }, h + "M"));
    D.tenors.forEach(t => {
      const c = D.current[String(h)][String(t)];
      const on = String(h) === state.h && String(t) === state.tenor;
      const b = el("button", { type: "button", class: "cell", "aria-pressed": String(on), "aria-label": `${t}-year, ${h}-month` });
      const a = c && c.sharpe != null && maxSr > 0 ? Math.min(0.92, Math.max(0.08, 0.08 + 0.84 * c.sharpe / maxSr)) : 0.04;
      b.style.background = `oklch(0.52 0.13 252 / ${a.toFixed(2)})`;
      b.style.color = a > 0.5 ? "#ffffff" : css("--ink");
      b.append(el("span", { class: "v" }, c ? pct(c.yhat) : "–"));
      const f = el("span", { class: "f" });
      f.append(el("span", {}, c && c.sharpe != null ? "SR " + c.sharpe.toFixed(2) : ""), el("span", {}, c ? bp(c.d1w) : ""));
      b.append(f);
      b.addEventListener("click", () => select(h, t));
      box.append(b);
    });
  });
}

function renderFocus() {
  const h = state.h, t = state.tenor, c = D.current[h][t];
  const title = document.getElementById("f-title"); title.replaceChildren(keyEl(tenorColor(t)), document.createTextNode(`${t}-year · ${hLabel(h)}`));
  if (!c) { document.getElementById("f-val").textContent = "–"; return; }
  document.getElementById("f-sub").textContent = `zero ${pctPlain(c.z_n)} · dur ${c.dur.toFixed(2)}`;
  document.getElementById("f-val").textContent = pct(c.yhat);
  document.getElementById("f-ann").textContent = `${pct(c.ann)}/yr · total ${pct(c.total)}`;
  const tot = Math.abs(c.m0) + Math.abs(c.tilt) || 1;
  document.getElementById("f-m0w").style.width = (100 * Math.abs(c.m0) / tot) + "%";
  document.getElementById("f-tiltw").style.width = (100 * Math.abs(c.tilt) / tot) + "%";
  document.getElementById("f-m0").textContent = `carry + roll − rf ${pct(c.m0)}`;
  document.getElementById("f-tilt").textContent = `rate-gap tilt ${pct(c.tilt)}`;
  const box = document.getElementById("f-stats"); box.replaceChildren();
  [["σ", c.sigma == null ? "–" : pctPlain(c.sigma), ""],
   ["EXCESS ÷ σ", c.sharpe == null ? "–" : c.sharpe.toFixed(2), ""],
   ["CUSHION", c.breakeven_bp == null ? "–" : sgn(c.breakeven_bp) + Math.abs(c.breakeven_bp).toFixed(0) + " bp", ""],
   ["Δ1D", bp(c.d1), dcls(c.d1)], ["Δ1W", bp(c.d1w), dcls(c.d1w)], ["Δ1M", bp(c.d1m), dcls(c.d1m)]].forEach(([l, v, cls]) => {
    const d = el("div"); d.append(el("span", {}, l), el("b", cls ? { class: cls } : {}, v)); box.append(d);
  });
}
function renderFocusChart() {
  if (!document.body.classList.contains("sheet")) return;
  const h = state.h, t = state.tenor;
  lineChart("cf", [{ name: t + "Y", color: tenorColor(t), values: D.series[h][t][state.metric] }],
    state.metric === "yhat" ? axisPct : numAxis, state.metric === "yhat" ? v => pct(v) : v => v.toFixed(2), { thin: true, ticks: 4 });
}

function sparkline(box, values, color) {
  box.replaceChildren();
  const W = box.clientWidth, H = box.clientHeight; if (!W) return;
  const idx = rangeIdx();
  const pts = idx.map(([, i]) => values[i]).filter(v => v != null); if (pts.length < 2) return;
  let lo = Math.min(...pts), hi = Math.max(...pts); const pd = (hi - lo) * 0.1 || 1e-4; lo -= pd; hi += pd;
  const X = j => 2 + (W - 6) * j / (pts.length - 1), Y = v => H - 2 - (H - 4) * (v - lo) / (hi - lo);
  let d = ""; pts.forEach((v, j) => { d += (j ? "L" : "M") + X(j).toFixed(1) + "," + Y(v).toFixed(1); });
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, "aria-hidden": "true" });
  svg.style.overflow = "visible";
  svg.append(svgEl("path", { d: d + `L${X(pts.length - 1)},${H}L2,${H}Z`, fill: color, opacity: 0.1 }),
    svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round" }),
    svgEl("circle", { cx: X(pts.length - 1), cy: Y(pts[pts.length - 1]), r: 2.5, fill: color }));
  box.append(svg);
}

const numAxis = (v, nd) => (v < -1e-12 ? "−" : "") + Math.abs(v).toFixed(Math.max(nd, 1));

function renderMult() {
  const box = document.getElementById("mult"); box.replaceChildren();
  const h = state.h, m = state.metric;
  document.getElementById("bar-note").textContent = `${hLabel(h)} horizon · each panel on its own scale`;
  const charts = [];
  D.tenors.forEach(t => {
    const c = D.current[h][String(t)];
    const p = el("button", { type: "button", class: "panel mp" + (String(t) === state.tenor ? " on" : ""), "aria-label": `${t}-year detail` });
    const ph = el("div", { class: "ph" });
    const tn = el("span", { class: "tn" }); tn.append(keyEl(tenorColor(t)), document.createTextNode(t + "Y"));
    ph.append(tn, el("b", {}, !c ? "–" : m === "yhat" ? pct(c.yhat) : (c.sharpe == null ? "–" : c.sharpe.toFixed(2))));
    const dw = el("span", { class: "dw " + dcls(c && c.d1w) }, c ? bp(c.d1w) : "–");
    const ch = el("div", { class: "chart", id: "m-" + t });
    const sp = el("div", { class: "spark" });
    p.append(ph, dw, ch, sp);
    p.addEventListener("click", () => select(h, t));
    box.append(p);
    charts.push([t, ch, sp]);
  });
  charts.forEach(([t, ch, sp]) => {
    const values = D.series[h][String(t)][m];
    if (ch.clientWidth) lineChart("m-" + t, [{ name: t + "Y", color: tenorColor(t), values }],
      m === "yhat" ? axisPct : numAxis, m === "yhat" ? v => pct(v) : v => v.toFixed(2), { thin: true, ticks: 4, left: 40 });
    sparkline(sp, values, tenorColor(t));
  });
}

function renderTable() {
  const h = state.h;
  document.getElementById("tbl-title").textContent = `Expected ${hLabel(h)} returns`;
  const any = D.current[h][String(D.tenors[0])];
  document.getElementById("tbl-sub").textContent =
    `excess over the ${h}M bill (${any ? pctPlain(any.rf) : "–"}) · holding-period, not annualized unless labelled`;
  const tbl = document.getElementById("tbl"); tbl.replaceChildren();
  const phone = isPhone();
  const cush = c => c.breakeven_bp == null ? "–" : sgn(c.breakeven_bp) + Math.abs(c.breakeven_bp).toFixed(0) + " bp";
  const COLS = [
    ["Tenor", null, "", true],
    ["Excess", c => pct(c.yhat), "em", true],
    ["Ann.", c => pct(c.ann), "", !phone],
    ["Total", c => pct(c.total), "", !phone],
    ["Carry+roll−rf", c => pct(c.m0), "", !phone],
    ["Tilt", c => pct(c.tilt), "", !phone],
    ["σ", c => c.sigma == null ? "–" : pctPlain(c.sigma), "", true],
    ["Excess÷σ", c => c.sharpe == null ? "–" : c.sharpe.toFixed(2), "em", true],
    ["Cushion", cush, "", !phone],
    ["Δ1d", c => bp(c.d1), "d1", !phone],
    ["Δ1w", c => bp(c.d1w), "d1w", true],
    ["Δ1m", c => bp(c.d1m), "d1m", !phone],
    ["Zero", c => pctPlain(c.z_n), "", !phone],
    ["Dur", c => c.dur.toFixed(2), "", !phone],
  ].filter(x => x[3]);
  const tr = el("tr"); COLS.forEach(([name]) => tr.append(el("th", { scope: "col" }, name)));
  const thead = el("thead"); thead.append(tr); tbl.append(thead);
  const tb = el("tbody");
  D.tenors.forEach(t => {
    const c = D.current[h][String(t)];
    const row = el("tr", String(t) === state.tenor ? { class: "on" } : {});
    const first = el("td"); const tn = el("span", { class: "tn" }); tn.append(keyEl(tenorColor(t)), document.createTextNode(t + "Y")); first.append(tn); row.append(first);
    COLS.slice(1).forEach(([, val, cls]) => {
      if (!c) { row.append(el("td", {}, "–")); return; }
      const dk = cls && cls.startsWith("d1") ? dcls(c[cls]) : cls;
      row.append(el("td", dk ? { class: dk } : {}, val(c)));
    });
    row.addEventListener("click", () => select(h, t));
    tb.append(row);
  });
  tbl.append(tb);
}

function legend(id, items) {
  const box = document.getElementById(id); box.replaceChildren();
  items.forEach(it => { const s = el("span"); s.append(keyEl(it.color), document.createTextNode(it.name)); box.append(s); });
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const out = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(12));
  return { ticks: out, step };
}

function rangeIdx() {
  const end = new Date(D.asof + "T00:00:00");
  const start = new Date(end.getTime() - RANGES[state.range] * 864e5);
  return D.dates.map((d, i) => [new Date(d + "T00:00:00"), i]).filter(([d]) => d >= start);
}

function lineChart(id, series, yFmt, tipFmt, opt = {}) {
  const box = document.getElementById(id); box.replaceChildren();
  const days = RANGES[state.range];
  const idx = rangeIdx();
  if (!idx.length) return;
  const W = box.clientWidth, H = box.clientHeight;
  if (!W || !H) return;
  const m = { l: opt.left || 46, r: 10, t: 8, b: 22 };
  const fs = 10.5, ff = css("--mono");
  let lo = Infinity, hi = -Infinity;
  series.forEach(s => idx.forEach(([, i]) => { const v = s.values[i]; if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (!isFinite(lo)) return;
  if (lo > 0 && lo < 0.25 * hi) lo = 0;
  const pad = (hi - lo) * 0.06 || 0.001; lo -= pad; hi += pad;
  const { ticks, step } = niceTicks(lo, hi, opt.ticks || 5);
  let nd = 0;
  const scale = yFmt === numAxis ? 1 : 100;
  while (nd < 3 && Math.abs(Math.round(step * scale * 10 ** nd) - step * scale * 10 ** nd) > 1e-6) nd++;
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const t0 = idx[0][0].getTime(), t1 = idx[idx.length - 1][0].getTime() || t0 + 1;
  const X = d => m.l + (W - m.l - m.r) * (d.getTime() - t0) / Math.max(1, t1 - t0);
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  ticks.forEach(v => {
    const y = Y(v);
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: Math.abs(v) < 1e-12 ? css("--axis") : css("--grid"), "stroke-width": 1 }));
    const tx = svgEl("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", fill: css("--muted"), "font-size": fs, "font-family": ff });
    tx.textContent = yFmt(v, nd); svg.append(tx);
  });
  const months = [];
  idx.forEach(([d], k) => { if (k === 0) return; const p = idx[k - 1][0]; if (d.getMonth() !== p.getMonth()) months.push(d); });
  const every = Math.max(1, Math.ceil(months.length / Math.max(2, Math.floor((W - m.l) / 70))));
  months.forEach((d, k) => {
    if (k % every) return;
    const tx = svgEl("text", { x: X(d), y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": fs, "font-family": ff });
    tx.textContent = d.toLocaleDateString(undefined, d.getMonth() === 0 || days > 400 ? { month: "short", year: "2-digit" } : { month: "short" });
    svg.append(tx);
  });
  svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }));
  const sw = opt.thin ? 1.5 : 2, dr = opt.thin ? 3 : 4;
  series.forEach(s => {
    let dstr = "", pen = false;
    idx.forEach(([d, i]) => {
      const v = s.values[i];
      if (v == null) { pen = false; return; }
      dstr += (pen ? "L" : "M") + X(d).toFixed(1) + "," + Y(v).toFixed(1); pen = true;
    });
    svg.append(svgEl("path", { d: dstr, fill: "none", stroke: s.color, "stroke-width": sw, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  });
  const [dLast, iLast] = idx[idx.length - 1];
  series.forEach(s => {
    const v = s.values[iLast]; if (v == null) return;
    svg.append(svgEl("circle", { cx: X(dLast), cy: Y(v), r: dr, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }));
  });
  const cross = svgEl("line", { y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" });
  const dots = series.map(s => svgEl("circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }));
  svg.append(cross, ...dots);
  const hit = svgEl("rect", { x: m.l, y: 0, width: Math.max(0, W - m.l - m.r), height: H, fill: "transparent", tabindex: 0, "aria-label": "Chart: use arrow keys to inspect values" });
  svg.append(hit);
  box.append(svg);
  const tip = el("div", { class: "tip", hidden: "" }); box.append(tip);
  let cur = idx.length - 1;
  function show(k) {
    cur = Math.max(0, Math.min(idx.length - 1, k));
    const [d, i] = idx[cur]; const x = X(d);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", { class: "d" }, fmtDate(D.dates[i])));
    series.forEach((s, j) => {
      const v = s.values[i];
      if (v == null) { dots[j].setAttribute("visibility", "hidden"); return; }
      dots[j].setAttribute("cx", x); dots[j].setAttribute("cy", Y(v)); dots[j].setAttribute("visibility", "visible");
      const row = el("div", { class: "row" });
      row.append(keyEl(s.color), el("b", {}, tipFmt(v)));
      if (series.length > 1) row.append(el("span", {}, s.name));
      tip.append(row);
    });
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = (x + 14 + tw > W ? Math.max(0, x - 14 - tw) : x + 14) + "px";
    tip.style.top = m.t + "px";
  }
  function hide() { cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tip.hidden = true; }
  function nearest(px) {
    const t = t0 + (px - m.l) / Math.max(1, W - m.l - m.r) * (t1 - t0);
    let a = 0, b = idx.length - 1;
    while (b - a > 1) { const c = (a + b) >> 1; if (idx[c][0].getTime() < t) a = c; else b = c; }
    return Math.abs(idx[a][0].getTime() - t) < Math.abs(idx[b][0].getTime() - t) ? a : b;
  }
  hit.addEventListener("pointermove", e => { const r = svg.getBoundingClientRect(); show(nearest((e.clientX - r.left) * W / r.width)); });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(cur));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", e => {
    if (e.key === "ArrowLeft") { show(cur - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(cur + 1); e.preventDefault(); }
  });
}

function renderCharts() {
  const h = state.h, t = state.tenor;
  document.getElementById("c2-title").textContent = `${t}Y · ${h}M decomposition`;
  const ser = D.series[h][t];
  const s2 = [
    { name: "Excess", color: css("--s1"), values: ser.yhat },
    { name: "Carry+roll−rf", color: css("--s2"), values: ser.m0 },
    { name: "Tilt", color: css("--s3"), values: ser.tilt },
  ];
  legend("c2-legend", s2);
  lineChart("c2", s2, axisPct, v => pct(v), { thin: true });
  const s3 = [
    { name: "10Y zero", color: css("--s1"), values: D.anchors.z_10 },
    { name: "Trend τ", color: css("--s2"), values: D.anchors.tau },
    { name: "Core CPI π", color: css("--s3"), values: D.anchors.pi },
  ];
  legend("c3-legend", s3);
  lineChart("c3", s3, axisPct, v => pctPlain(v), { thin: true });
}

function renderAnchors() {
  const a = D.anchors_now, t = state.tenor, c = D.current["12"][t];
  const items = [
    [pctPlain(a.z_10), "10Y zero yield"],
    [pctPlain(a.tau), "Trend inflation τ"],
    [pctPlain(a.pi), `Core CPI yoy π (${D.cpi_month})`],
    [pct(a.z_10 - a.tau), "Cycle gap: 10Y − τ"],
    [c ? pct(c.gap_value) : "–", `Value gap: ${t}Y − π`],
  ];
  const box = document.getElementById("anchors"); box.replaceChildren();
  items.forEach(([v, l]) => { const d = el("div"); d.append(el("b", {}, v), el("span", {}, l)); box.append(d); });
}

function renderModel() {
  const M = D.model;
  const track = document.getElementById("track"); track.replaceChildren();
  if (M.oos) {
    const r12 = D.tenors.map(t => M.oos["12"][String(t)].frozen.toFixed(3)).join(" · ");
    const c12 = D.tenors.map(t => M.oos["12"][String(t)].carry.toFixed(3)).join(" · ");
    track.append(document.createTextNode(`Out-of-sample R², 12-month, ${M.oos_sample}, for ${D.tenors[0]}–${D.tenors[D.tenors.length - 1]}Y: `),
      el("span", { class: "mono", style: "color:var(--ink)" }, r12),
      document.createTextNode(`, vs carry-only ${c12}. `));
  }
  if (M.regime) {
    const [lo, hi] = M.regime.band, inBand = M.regime.half_B >= lo && M.regime.half_B <= hi;
    track.append(document.createTextNode(`Regime B̂/2 = ${M.regime.half_B.toFixed(3)} (${M.regime.month}), ${inBand ? "inside" : "OUTSIDE"} ${lo.toFixed(2)}–${hi.toFixed(2)}.`));
  }
  const box = document.getElementById("model-body"); box.replaceChildren();
  box.append(el("div", { class: "formula" }, "E[rx] = (carry + roll − rf) + [1 − (1 − φ)^(h/12)] · D · ½[(y_10Y − τ) + (y_N − π)],   φ = " + D.phi));
  box.append(el("p", {}, "Nothing is estimated: φ is frozen at 0.15/yr, the anchor weights are equal, and τ is a fixed EWMA of core CPI. The zero curve is bootstrapped each day from the Treasury par yield curve (PCHIP interpolation); core CPI (NSA) comes from FRED, with BLS as a fallback."));
  if (M.oos) {
    box.append(el("p", {}, `Out-of-sample R² vs the real-time mean, ${M.oos_sample} sample, on the research (Liu-Wu) curve, with the carry-only model in parentheses. These are fixed properties of the study, not recomputed daily:`));
    const wrap = el("div", { class: "tablewrap" }); const tbl = el("table");
    const tr = el("tr"); tr.append(el("th", { scope: "col" }, "Horizon")); D.tenors.forEach(t => tr.append(el("th", { scope: "col" }, t + "Y")));
    const thead = el("thead"); thead.append(tr); tbl.append(thead);
    const tb = el("tbody");
    D.horizons.forEach(h => {
      const row = el("tr"); row.append(el("td", {}, h + "M"));
      D.tenors.forEach(t => { const o = M.oos[String(h)][String(t)]; row.append(el("td", {}, `${o.frozen.toFixed(3)} (${o.carry.toFixed(3)})`)); });
      tb.append(row);
    });
    tbl.append(tb); wrap.append(tbl); box.append(wrap);
  }
  if (M.regime) {
    const [lo, hi] = M.regime.band;
    const inBand = M.regime.half_B >= lo && M.regime.half_B <= hi;
    box.append(el("p", {}, `Regime monitor: real-time B̂/2 = ${M.regime.half_B.toFixed(3)} as of ${M.regime.month}, ${inBand ? "inside" : "OUTSIDE"} the ${lo.toFixed(2)}–${hi.toFixed(2)} band that makes φ = 0.15 defensible. It moves only when the research study is rerun.`));
  }
  if (M.basis) {
    const parts = D.tenors.map(t => { const b = M.basis[String(t)]; return b ? `${t}Y ${b[0] < 0 ? "−" : ""}${Math.abs(b[0])}/${b[1]}` : null; }).filter(Boolean);
    box.append(el("p", {}, `Curve basis: this page bootstraps the Treasury par curve, the study used Liu-Wu. Month-end 12-month forecasts, ${M.basis_sample}, bias / median |difference| in bp: ${parts.join(" · ")}.`));
  }
}

function renderAll() {
  renderMeta(); renderMatrix(); renderFocus(); renderMult(); renderTable();
  renderCharts(); renderAnchors(); renderFocusChart();
}
renderModel();
renderAll();
let rt, wasPhone = isPhone();
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (isPhone() !== wasPhone) { wasPhone = isPhone(); closeSheet(); }
    renderAll();
  }, 120);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
if (document.fonts) document.fonts.ready.then(renderAll);
})();
</script>
</body>
</html>
"""
