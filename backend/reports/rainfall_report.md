# Model 1 - Rainfall / heavy-rain risk: evaluation report

Dataset: `corridor_rainfall_open_meteo.csv`: real daily rainfall at each of the 35 corridor segments, downloaded from the Open-Meteo Historical Weather API (ERA5 reanalysis). Weather data by Open-Meteo.com (CC BY 4.0). Plus `corridor_rain_forecast_open_meteo.csv`: the rain forecast for each day made the day before (Open-Meteo Previous Runs API).

Split: train 2020-01-08..2024-12-31, validation 2025-01-01..2025-12-31, test 2026-01-01..2026-11-12

Forecast input available for 12,145 training rows (2024 only); all models are scored on the same 12,775 validation and 11,060 test segment-days that have a forecast made the day before.

## Algorithm comparison (validation year)

| Model | Log loss | Macro-F1 | Heavy-rain PR-AUC |
|---|---|---|---|
| LogisticRegression · past rain | 0.512 | 0.424 | 0.142 |
| RandomForest · past rain | 0.486 | 0.436 | 0.130 |
| HistGradientBoosting · past rain | 0.511 | 0.437 | 0.038 |
| LogisticRegression · past rain + forecast | 0.464 | 0.534 | 0.141 |
| RandomForest · past rain + forecast | 0.424 | 0.481 | 0.198 |
| HistGradientBoosting · past rain + forecast | 0.447 | 0.512 | 0.062 |

Selected: **RandomForest · past rain + forecast** (lowest validation log loss).

## Test period - model vs baselines (same test days)

| Model | Log loss | Macro-F1 | Accuracy | Heavy-rain PR-AUC |
|---|---|---|---|---|
| RandomForest · past rain + forecast | 0.391 | 0.529 | 0.822 | 0.526 |
| climatology (location x month frequencies) | 0.505 | 0.395 | 0.780 | 0.150 |
| persistence (tomorrow = today) | - | 0.560 | 0.804 | - |
| forecast alone (category of the forecast made the day before) | - | 0.651 | 0.833 | - |
| RandomForest · past rain (fallback, same days) | 0.420 | 0.443 | 0.819 | 0.402 |

## Heavy-rain alert rule (chosen on the validation year by F1)

| Rule | Threshold | Precision | Recall | F1 | Alerts raised | Actual heavy days |
|---|---|---|---|---|---|---|
| model | 0.0910 | 0.187 | 0.333 | 0.239 | 75 | 42 |
| persistence | - | 0.262 | 0.262 | 0.262 | 42 | 42 |
| model_or_persistence (selected) | 0.1715 | 0.277 | 0.310 | 0.292 | 47 | 42 |
| forecast | - | 0.194 | 0.167 | 0.179 | 36 | 42 |

Selected alert rule: **model_or_persistence** = heavy-rain probability at or above the alert threshold, or heavy rain on the previous day.

## Heavy-rain alert on the test period (rules fixed on validation, test seen once)

| Rule | Threshold | Precision | Recall | F1 | Alerts raised | Actual heavy days |
|---|---|---|---|---|---|---|
| model | 0.0910 | 0.246 | 0.652 | 0.357 | 61 | 23 |
| persistence | - | 0.391 | 0.391 | 0.391 | 23 | 23 |
| model_or_persistence (selected) | 0.1715 | 0.324 | 0.522 | 0.400 | 37 | 23 |
| forecast | - | 0.591 | 0.565 | 0.578 | 22 | 23 |

## Fallback when no forecast exists for a date

Model **RandomForest · past rain**, alert rule **persistence** (chosen on validation). On all 11,060 test segment-days:

| Model | Log loss | Macro-F1 | Accuracy | Heavy-rain PR-AUC |
|---|---|---|---|---|
| RandomForest · past rain | 0.420 | 0.443 | 0.819 | 0.402 |

| Rule | Threshold | Precision | Recall | F1 | Alerts raised | Actual heavy days |
|---|---|---|---|---|---|---|
| model | 0.0589 | 0.203 | 0.609 | 0.304 | 69 | 23 |
| persistence (selected) | - | 0.391 | 0.391 | 0.391 | 23 | 23 |

## Limitations
- No flood labels exist; heavy rain (>= 64.5 mm/day, IMD) is used as a flood trigger, not a flood observation.
- Rainfall is ERA5 / ERA5-Land reanalysis from Open-Meteo (a weather model fitted to observations, about 10-25 km grid), not rain-gauge readings. It smooths very local cloudbursts, so it records fewer heavy-rain days than a gauge would.
- Daily totals: the model forecasts the whole day, not individual hours.
- Heavy-rain days are rare, so alerts have low precision. The alert rule is chosen on the validation year; see the alert tables above.
- Weather forecasts are archived only from January 2024, so the 'past rain + forecast' models learn from one year (2024) and are tested on the current year only; their comparison with past-rain-only models uses exactly the same test days.
- When no forecast made the day before exists for a date (e.g. dates far in the past or future), the app uses the past-rain-only fallback model, and says so.