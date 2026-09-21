"""The dashboard page: one self-contained HTML file.

Python assembles a JSON payload (current table, daily history, anchors, model
card); the page draws it with inline SVG and vanilla JS, so it works offline
and needs no plotting dependency. Returns are h-month simple returns.
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
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --wash: rgba(11,11,11,0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --warn-bg: #fff6e0; --warn-ink: #6b4a00; --warn-mark: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --warn-bg: #2a2310; --warn-ink: #f3d58a; --warn-mark: #fab219;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --wash: rgba(255,255,255,0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --warn-bg: #2a2310; --warn-ink: #f3d58a; --warn-mark: #fab219;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: max(20px, env(safe-area-inset-top)) 20px max(32px, env(safe-area-inset-bottom));
}
.wrap { max-width: 1180px; margin: 0 auto; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 24px; }
h1 { font-size: 22px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--ink-2); }
.muted { color: var(--muted); }
.warn { margin: 14px 0 0; padding: 10px 14px; border-radius: 8px; background: var(--warn-bg); color: var(--warn-ink); }
.warn div::before { content: "\26A0\FE0E  "; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; margin: 18px 0 14px; }
.ctl { display: flex; align-items: center; gap: 8px; }
.ctl > span { color: var(--ink-2); font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--ring); border-radius: 8px; padding: 2px; background: var(--surface); }
.seg button {
  font: inherit; font-size: 13px; color: var(--ink-2); background: transparent; border: 0;
  padding: 4px 10px; border-radius: 6px; cursor: pointer; min-width: 38px;
}
.seg button:hover { background: var(--wash); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--s1); outline-offset: 1px; }
.tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.tile .lab { color: var(--ink-2); font-size: 13px; display: flex; align-items: center; gap: 6px; }
.key { display: inline-block; width: 14px; height: 2px; border-radius: 1px; }
.tile .val { font-size: 26px; font-weight: 600; margin: 2px 0 0; letter-spacing: -0.01em; }
.tile .meta { color: var(--muted); font-size: 12.5px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px 12px; margin-top: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 6px 0 4px; font-size: 12.5px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.chart { position: relative; width: 100%; height: 260px; touch-action: pan-y; }
.chart svg { display: block; width: 100%; height: 100%; overflow: visible; }
.tip {
  position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 8px; padding: 7px 10px; font-size: 12.5px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  white-space: nowrap; z-index: 2;
}
.tip .d { color: var(--muted); margin-bottom: 3px; }
.tip .row { display: flex; align-items: center; gap: 8px; font-variant-numeric: tabular-nums; }
.tip .row b { font-weight: 600; min-width: 58px; text-align: right; }
.tip .row span:last-child { color: var(--ink-2); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13.5px; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 500; font-size: 12.5px; vertical-align: bottom; }
th:first-child, td:first-child { text-align: left; }
tr:last-child td { border-bottom: 0; }
td.em { font-weight: 600; }
.note { color: var(--muted); font-size: 12.5px; margin: 8px 0 0; }
.anchors { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 8px 16px; margin: 8px 0 4px; }
.anchors div b { display: block; font-size: 18px; font-weight: 600; }
.anchors div span { color: var(--ink-2); font-size: 12.5px; }
.model p { margin: 6px 0; color: var(--ink-2); }
footer { max-width: 1180px; margin: 18px auto 0; color: var(--muted); font-size: 12.5px; }
footer p { margin: 6px 0; }
footer a { color: var(--ink-2); }
.formula { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 13px; color: var(--ink); overflow-x: auto; white-space: nowrap; padding: 6px 0; }
@media (max-width: 900px) {
  .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .grid2 { grid-template-columns: 1fr; }
  .anchors { grid-template-columns: repeat(3, minmax(0,1fr)); }
}
details.sect > summary {
  font-size: 15px; font-weight: 600; cursor: pointer; list-style: none;
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
details.sect > summary::-webkit-details-marker { display: none; }
details.sect > summary::after {
  content: "+"; color: var(--muted); font-weight: 400; font-size: 17px; line-height: 1;
}
details.sect[open] > summary::after { content: "−"; }
details.sect:not([open]) { padding-bottom: 13px; }
.note-sect { border: 0; background: none; padding: 0; margin-top: 6px; }
.note-sect > summary { font-size: 12.5px; font-weight: 500; color: var(--muted); }
.note-sect > p { margin: 4px 0 0; }
@media (min-width: 641px) {
  /* match the [open] rule's specificity, or the marker survives on desktop */
  details.sect > summary { cursor: default; }
  details.sect > summary::after,
  details.sect[open] > summary::after { content: none; }
}
@media (max-width: 640px) {
  body { padding-left: 16px; padding-right: 16px; }
  h1 { font-size: 20px; }
  .controls { gap: 8px 14px; margin: 14px 0 12px; }
  .ctl > span { font-size: 12px; }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .tile { padding: 10px 12px; }
  .tile .val { font-size: 22px; }
  .tile .lab, .tile .meta { font-size: 12px; }
  .anchors { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .chart { height: 210px; }
  .card { padding: 12px 13px 10px; }
  th, td { padding: 6px 8px; }
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Treasury expected returns</h1>
    <div class="sub" id="asof"></div>
  </header>
  <div class="warn" id="warn" hidden></div>

  <div class="controls" role="group" aria-label="Filters">
    <div class="ctl"><span>Horizon</span><div class="seg" id="ctl-h"></div></div>
    <div class="ctl"><span>History</span><div class="seg" id="ctl-r"></div></div>
    <div class="ctl"><span>Tenor detail</span><div class="seg" id="ctl-t"></div></div>
    <div class="ctl"><span>Columns</span><div class="seg" id="ctl-c"></div></div>
  </div>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2 id="tbl-title"></h2>
    <div class="sub" style="font-size:12.5px" id="tbl-sub"></div>
    <div class="tablewrap"><table id="tbl"></table></div>
    <details class="sect note-sect"><summary>What these columns mean</summary>
    <p class="note">Volatility σ is the research RM2 model on the live curve: modified duration × the 60-month rolling volatility of monthly par-yield changes, scaled by √h. Excess ÷ σ is the conditional Sharpe ratio for the holding period. Yield cushion is the same expected excess divided by modified duration — the parallel yield rise that would erase it, in bp; equivalently, 1 bp of cushion is 0.01% of expected return per year of duration. Changes are in the expected excess return. Coupons received during the holding period are held as cash, not reinvested, which understates the 12-month figures by roughly 5 bp at current short rates (no coupon is received at all within 1 or 3 months).</p>
    </details>
  </div>

  <div class="card">
    <h2 id="c1-title"></h2>
    <div class="legend" id="c1-legend"></div>
    <div class="chart" id="c1"></div>
  </div>

  <details class="card sect">
    <summary id="c4-title"></summary>
    <div class="sub" style="font-size:12.5px">Expected excess return per unit of RM2 volatility. Rises when yields are high relative to the inflation anchors, or when rate volatility falls.</div>
    <div class="legend" id="c4-legend"></div>
    <div class="chart" id="c4"></div>
  </details>

  <div class="grid2">
    <details class="card sect">
      <summary id="c2-title"></summary>
      <div class="legend" id="c2-legend"></div>
      <div class="chart" id="c2"></div>
    </details>
    <details class="card sect">
      <summary>Rate anchors</summary>
      <div class="legend" id="c3-legend"></div>
      <div class="chart" id="c3"></div>
    </details>
  </div>

  <details class="card sect">
    <summary>Anchors today</summary>
    <div class="anchors" id="anchors"></div>
    <p class="note">The rate-gap term uses the 10-year zero yield minus trend inflation τ (EWMA of core CPI yoy, ν = 0.987) and each tenor's own zero yield minus current core CPI yoy π, averaged with equal weights.</p>
  </details>

  <details class="card sect model" id="model"></details>

  <footer>
    <p><strong>Not investment advice.</strong> This page is research output: a fixed model applied to public data, published so its forecasts can be watched in real time. It makes no recommendation, and the synthetic par bonds it prices ignore financing, bid-offer and taxes.</p>
    <p>Sources: <a href="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve">Treasury daily par yield curve</a> · <a href="https://fred.stlouisfed.org/series/CPILFENS">FRED CPILFENS</a> (core CPI, NSA). Daily history: <a href="forecast_history.csv">forecast_history.csv</a>. Code: <a href="https://github.com/tbeason/treasury-dashboard">github.com/tbeason/treasury-dashboard</a>.</p>
    <p>By <a href="https://tbeason.com">Tyler Beason</a>.</p>
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("payload").textContent);
const TENOR_COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5"];
const state = { h: "12", range: "1Y", tenor: "10", cols: "key" };
const RANGES = { "3M": 92, "6M": 183, "1Y": 366, "3Y": 1100 };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const isPhone = () => window.matchMedia("(max-width: 640px)").matches;
const pct = (x, nd = 2) => x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(nd) + "%";
const pctPlain = (x, nd = 2) => x == null ? "–" : (100 * x).toFixed(nd) + "%";
const axisPct = (v, nd) => (v < -1e-12 ? "−" : "") + Math.abs(100 * v).toFixed(nd) + "%";
const bp = x => x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(1e4 * x).toFixed(0) + " bp";
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

// header + warnings
document.getElementById("asof").textContent =
  `Close of ${fmtDate(D.asof)} · core CPI through ${D.cpi_month} · updated ${D.generated}`;
if (D.warnings.length) {
  const w = document.getElementById("warn");
  D.warnings.forEach(t => w.appendChild(el("div", {}, t)));
  w.hidden = false;
}

// controls
function seg(id, options, key, label) {
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
seg("ctl-h", D.horizons.map(h => [String(h), h + "M"]), "h");
seg("ctl-r", Object.keys(RANGES).map(r => [r, r]), "range");
seg("ctl-t", D.tenors.map(t => [String(t), t + "Y"]), "tenor");
seg("ctl-c", [["key", "Key"], ["all", "All"]], "cols");

function tenorColor(t) { return css(TENOR_COLORS[D.tenors.indexOf(Number(t))]); }

function renderTiles() {
  const box = document.getElementById("tiles"); box.replaceChildren();
  const cur = D.current[state.h];
  D.tenors.forEach(t => {
    const c = cur[String(t)];
    const tile = el("div", { class: "tile" });
    const lab = el("div", { class: "lab" });
    const k = el("span", { class: "key" }); k.style.background = tenorColor(t);
    lab.append(k, document.createTextNode(`${t}Y · ${state.h}M excess return`));
    tile.append(lab);
    tile.append(el("div", { class: "val" }, c ? pct(c.yhat) : "–"));
    if (c) {
      tile.append(el("div", { class: "meta" }, `${pct(c.ann, 1)}/yr · ${bp(c.d1w)} vs 1w`));
    }
    box.append(tile);
  });
}

function renderTable() {
  const h = state.h;
  document.getElementById("tbl-title").textContent = `Expected ${hLabel(h)} returns`;
  const anyTenor = D.current[h][String(D.tenors[0])];
  const rf = anyTenor ? anyTenor.rf : null;
  document.getElementById("tbl-sub").textContent =
    `Excess returns are over the ${hLabel(h)} risk-free rate, ${pctPlain(rf)} today (the ${h}-month Treasury bill). `
    + "Carry and roll-down plus a φ = 0.15 pull of yields toward the inflation anchors, for the holding period, not annualized unless labeled.";
  const tbl = document.getElementById("tbl"); tbl.replaceChildren();
  // key columns always show; the rest (decomposition, other change windows,
  // par coupon) only under Columns: All, so the default table fits the width
  const phone = isPhone();
  const COLS = [
    ["Tenor", c => c.tenor, "", true],
    ["Expected excess", c => pct(c.yhat), "em", true],
    ["Annualized", c => pct(c.ann), "", !phone],
    ["Expected total", c => pct(c.total), "", !phone],
    ["Carry + roll − rf", c => pct(c.m0), "", false],
    ["Rate-gap tilt", c => pct(c.tilt), "", false],
    ["Volatility σ", c => c.sigma == null ? "–" : pctPlain(c.sigma), "", true],
    ["Excess ÷ σ", c => c.sharpe == null ? "–" : c.sharpe.toFixed(2), "em", true],
    ["Yield cushion", c => (c.breakeven_bp >= 0 ? "" : "−") + Math.abs(c.breakeven_bp).toFixed(0) + " bp", "", !phone],
    ["Δ 1d", c => bp(c.d1), "", false],
    ["Δ 1w", c => bp(c.d1w), "", !phone],
    ["Δ 1m", c => bp(c.d1m), "", false],
    ["Zero yield", c => pctPlain(c.z_n), "", !phone],
    ["Par coupon", c => pctPlain(c.coupon), "", false],
    ["Mod. duration", c => c.dur.toFixed(2), "", !phone],
  ].filter(([, , , key]) => key || state.cols === "all");
  const tr = el("tr"); COLS.forEach(([name]) => tr.append(el("th", { scope: "col" }, name)));
  const thead = el("thead"); thead.append(tr); tbl.append(thead);
  const tb = el("tbody");
  D.tenors.forEach(t => {
    const c = D.current[h][String(t)];
    const row = el("tr");
    if (!c) { row.append(el("td", {}, t + "Y")); for (let i = 1; i < COLS.length; i++) row.append(el("td", {}, "–")); tb.append(row); return; }
    COLS.forEach(([, val, cls]) => row.append(el("td", cls ? { class: cls } : {}, val({ ...c, tenor: t + "Y" }))));
    tb.append(row);
  });
  tbl.append(tb);
}

function legend(id, items) {
  const box = document.getElementById(id); box.replaceChildren();
  items.forEach(it => {
    const s = el("span"); const k = el("span", { class: "key" }); k.style.background = it.color;
    s.append(k, document.createTextNode(it.name)); box.append(s);
  });
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const out = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(12));
  return { ticks: out, step };
}

function lineChart(id, series, yFmt, tipFmt) {
  const box = document.getElementById(id); box.replaceChildren();
  const days = RANGES[state.range];
  const end = new Date(D.asof + "T00:00:00");
  const start = new Date(end.getTime() - days * 864e5);
  const idx = D.dates.map((d, i) => [new Date(d + "T00:00:00"), i]).filter(([d]) => d >= start);
  if (!idx.length) return;
  const W = box.clientWidth, H = box.clientHeight;
  const m = { l: 48, r: 12, t: 8, b: 24 };
  let lo = Infinity, hi = -Infinity;
  series.forEach(s => idx.forEach(([, i]) => { const v = s.values[i]; if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (lo > 0 && lo < 0.25 * hi) lo = 0;
  const pad = (hi - lo) * 0.06 || 0.001; lo -= pad; hi += pad;
  const { ticks, step } = niceTicks(lo, hi, 5);
  let nd = 0;
  while (nd < 3 && Math.abs(Math.round(step * 100 * 10 ** nd) - step * 100 * 10 ** nd) > 1e-6) nd++;
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const t0 = idx[0][0].getTime(), t1 = idx[idx.length - 1][0].getTime() || t0 + 1;
  const X = d => m.l + (W - m.l - m.r) * (d.getTime() - t0) / Math.max(1, t1 - t0);
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  ticks.forEach(v => {
    const y = Y(v);
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: Math.abs(v) < 1e-12 ? css("--axis") : css("--grid"), "stroke-width": 1 }));
    const tx = svgEl("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11.5, style: "font-variant-numeric: tabular-nums" });
    tx.textContent = yFmt(v, nd); svg.append(tx);
  });
  // x ticks: month starts, thinned to fit
  const months = [];
  idx.forEach(([d], k) => { if (k === 0) return; const p = idx[k - 1][0]; if (d.getMonth() !== p.getMonth()) months.push(d); });
  const every = Math.max(1, Math.ceil(months.length / Math.max(2, Math.floor((W - m.l) / 70))));
  months.forEach((d, k) => {
    if (k % every) return;
    const x = X(d);
    const tx = svgEl("text", { x, y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": 11.5 });
    tx.textContent = d.toLocaleDateString(undefined, d.getMonth() === 0 || days > 400 ? { month: "short", year: "2-digit" } : { month: "short" });
    svg.append(tx);
  });
  svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }));
  series.forEach(s => {
    let dstr = "", pen = false;
    idx.forEach(([d, i]) => {
      const v = s.values[i];
      if (v == null) { pen = false; return; }
      dstr += (pen ? "L" : "M") + X(d).toFixed(1) + "," + Y(v).toFixed(1); pen = true;
    });
    svg.append(svgEl("path", { d: dstr, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  });
  // end dots
  const [dLast, iLast] = idx[idx.length - 1];
  series.forEach(s => {
    const v = s.values[iLast]; if (v == null) return;
    svg.append(svgEl("circle", { cx: X(dLast), cy: Y(v), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }));
  });
  // crosshair layer
  const cross = svgEl("line", { y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" });
  const dots = series.map(s => { const c = svgEl("circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }); return c; });
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
      const k2 = el("span", { class: "key" }); k2.style.background = s.color;
      row.append(k2, el("b", {}, tipFmt(v)), el("span", {}, s.name));
      tip.append(row);
    });
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = (x + 14 + tw > W ? x - 14 - tw : x + 14) + "px";
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
  document.getElementById("c1-title").textContent = `Expected ${hLabel(h)} excess return by tenor`;
  const s1 = D.tenors.map(x => ({ name: x + "Y", color: tenorColor(x), values: D.series[h][String(x)].yhat }));
  legend("c1-legend", s1);
  lineChart("c1", s1, axisPct, v => pct(v));

  document.getElementById("c4-title").textContent = `Expected ${hLabel(h)} excess return per unit of risk`;
  const s4 = D.tenors.map(x => ({ name: x + "Y", color: tenorColor(x), values: D.series[h][String(x)].sharpe }));
  legend("c4-legend", s4);
  lineChart("c4", s4, (v, nd) => v.toFixed(Math.max(nd, 1)), v => v.toFixed(2));

  document.getElementById("c2-title").textContent = `${t}-year: where the ${hLabel(h)} expectation comes from`;
  const ser = D.series[h][t];
  const s2 = [
    { name: "Expected excess", color: css("--s1"), values: ser.yhat },
    { name: "Carry + roll − rf", color: css("--s2"), values: ser.m0 },
    { name: "Rate-gap tilt", color: css("--s3"), values: ser.tilt },
  ];
  legend("c2-legend", s2);
  lineChart("c2", s2, axisPct, v => pct(v));

  const s3 = [
    { name: "10Y zero yield", color: css("--s1"), values: D.anchors.z_10 },
    { name: "Trend inflation τ", color: css("--s2"), values: D.anchors.tau },
    { name: "Core CPI yoy π", color: css("--s3"), values: D.anchors.pi },
  ];
  legend("c3-legend", s3);
  lineChart("c3", s3, axisPct, v => pctPlain(v));
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
  const box = document.getElementById("model"); box.replaceChildren();
  box.append(el("summary", {}, "Model card"));
  box.append(el("div", { class: "formula" }, "E[rx] = (carry + roll − rf) + [1 − (1 − φ)^(h/12)] · D · ½[(y_10Y − τ) + (y_N − π)],   φ = " + D.phi));
  box.append(el("p", {}, "Nothing is estimated: φ is frozen at 0.15/yr, the anchor weights are equal, and τ is a fixed EWMA of core CPI. The zero curve is bootstrapped each day from the Treasury par yield curve (PCHIP interpolation); core CPI (NSA) comes from FRED, with BLS as a fallback."));
  const M = D.model;
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
    const parts = D.tenors.map(t => { const b = M.basis[String(t)]; return b ? `${t}Y ${b[0] >= 0 ? "+" : ""}${b[0]}/${b[1]}` : null; }).filter(Boolean);
    box.append(el("p", {}, `Curve basis: this page bootstraps the Treasury par curve, the study used Liu-Wu. Month-end 12-month forecasts, ${M.basis_sample}, bias / median |difference| in bp: ${parts.join(" · ")}.`));
  }
}

function syncSections() {
  const phone = isPhone();
  document.querySelectorAll("details.sect").forEach(d => { d.open = !phone; });
}
document.querySelectorAll("details.sect").forEach(d =>
  d.addEventListener("toggle", () => { if (d.open) renderCharts(); }));

function renderAll() { renderTiles(); renderTable(); renderCharts(); renderAnchors(); renderModel(); }
syncSections();
renderAll();
let rt, wasPhone = isPhone();
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (isPhone() !== wasPhone) { wasPhone = isPhone(); syncSections(); renderAll(); }
    else renderCharts();
  }, 120);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
})();
</script>
</body>
</html>
"""
