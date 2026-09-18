# Treasury expected returns

A daily dashboard of expected excess returns on 2, 5, 10, 20 and 30-year
Treasuries, at 1-, 3- and 12-month horizons, from a forecasting model whose
only parameter is fixed in advance.

**[View the dashboard →](https://tbeason.com/treasury-dashboard/)**

## The model

For an N-year par bond held h months:

```
E[rx] = (R_CR − rf) + φ_h · D · ½[(y₁₀ − τ) + (y_N − π)],   φ_h = 1 − (1 − φ)^(h/12)
```

- **R_CR − rf** — carry and roll-down over the risk-free rate. The bond is aged
  h months and repriced on today's zero curve, so roll-down is whatever the
  curve's own shape delivers rather than a separate assumption. Coupons received
  are added at face value; the curve is assumed unchanged, not that forwards are
  realized.
- **The tilt** — yields pull toward inflation anchors at speed **φ = 0.15/year**,
  scaled by modified duration D. The cycle gap uses the 10-year zero against
  trend inflation τ (an EWMA of core CPI year-over-year, ν = 0.987); the value
  gap uses each tenor's own zero against current core CPI π. Equal weights.

Nothing is estimated at runtime: φ, the anchor weights and the EWMA decay are
all fixed, so the page is pure arithmetic on public data. The model card on the
page carries its out-of-sample record, which is a property of the underlying
study and does not move with the daily data.

The page also reports risk: **σ** is modified duration times the 60-month
rolling volatility of monthly par-yield changes, scaled by √h, and **excess ÷ σ**
is the conditional Sharpe ratio.

## Data

| What | Source | Fallback |
|---|---|---|
| Par yield curve | [Treasury daily par yields](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) | FRED `DGS*`, then cache |
| Core CPI (NSA) | FRED [`CPILFENS`](https://fred.stlouisfed.org/series/CPILFENS) | BLS public API, then the seed history shipped in this repo |

Treasury leads for the curve because its file carries the 1.5-, 2- and 4-month
points FRED's series lack. FRED leads for CPI because the keyless BLS endpoint
is capped at 25 calls a day per IP, which shared CI runners can exhaust.

The zero curve is bootstrapped from the par quotes each run: PCHIP interpolation
onto a semiannual grid, bills read as bond-equivalent zeros, then a standard
coupon bootstrap out to 30 years.

## Running it

```
uv sync
uv run scripts/build.py --open     # writes site/index.html
uv run pytest -q                   # offline regression test
```

`FRED_API_KEY` is optional locally (the build falls back to BLS plus the seed
history). `TSYFWD_SITE` and `TSYFWD_CACHE` override the output and cache
directories.

GitHub Actions rebuilds and publishes the page each weekday evening; see
`.github/workflows/dashboard.yml`.

## Layout

```
src/tsyfwd/
  config.py     the frozen constants
  sources.py    Treasury / FRED / BLS fetch, caching, fallbacks
  curve.py      par yields -> zero curve -> par bonds and durations
  returns.py    the h-aged bond: carry + roll-down
  anchors.py    pi and the tau EWMA, with publication-lag timing
  model.py      the forecast, RM2 risk, and the daily build
  refs.py       fixed model-card numbers from the study
  dashboard.py  payload + the self-contained HTML page
```

## Caveats

These are synthetic par bonds, not actual issues: no on-the-run premium, no
financing, bid-offer or taxes. The tilt is duration-only, with no convexity
term. **This is research output, not investment advice.**

By [Tyler Beason](https://tbeason.com).
