"""Fixed reference numbers for the model card.

These come from the research pipeline's last full run (Liu-Wu curve through
2025-12) and do not change when the dashboard updates: the model is frozen,
so its out-of-sample record is a property of the study, not of today's data.
Regenerate only when that study is rerun.
"""

from __future__ import annotations

# Out-of-sample R2 vs the real-time mean, matched sample 1985-11 onward, on
# the Liu-Wu curve (research table16_frozen.csv): {horizon: {tenor: (frozen,
# carry-only)}}. The frozen model is the ensemble at s = 1.
OOS_R2 = {
    1: {2: (0.030, 0.015), 5: (0.036, 0.020), 10: (0.032, 0.016),
        20: (0.026, 0.009), 30: (0.026, 0.011)},
    3: {2: (0.077, 0.039), 5: (0.092, 0.051), 10: (0.089, 0.046),
        20: (0.075, 0.029), 30: (0.072, 0.028)},
    12: {2: (0.084, -0.015), 5: (0.242, 0.126), 10: (0.276, 0.148),
         20: (0.240, 0.086), 30: (0.255, 0.076)},
}
OOS_SAMPLE = "1985-2025, matched"

# Regime monitor: the real-time B_hat/2 estimate (research table26_B_path.csv).
# phi = 0.15 is defensible while this sits in [0.10, 0.20].
REGIME_HALF_B = 0.133
REGIME_ASOF = "2025-12"
REGIME_BAND = (0.10, 0.20)

# Curve basis: this dashboard bootstraps the Treasury par curve, the research
# used Liu-Wu. Month-end 12-month forecasts 2016-2025, (bias, median |diff|)
# in bp of 12-month return (research table30_live_vs_lw.csv, cmt_pchip).
CURVE_BASIS_BP = {2: (0, 4), 5: (-0, 5), 10: (3, 12), 20: (-25, 23), 30: (5, 17)}
CURVE_BASIS_SAMPLE = "2016-2025"
