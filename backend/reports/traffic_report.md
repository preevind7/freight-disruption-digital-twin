# Model 3 - Festival-time traffic congestion: evaluation report

Datasets: `indian_roads_dataset.csv` (target `traffic_density`) + `significant_dates.csv` (event dates). Split: train 2022-01-01..2023-12-31, validation 2024-01-01..2024-06-30, test 2024-07-01..2025-04-15

## Algorithm comparison (validation)

| Model | Macro-F1 | Accuracy | Log loss |
|---|---|---|---|
| LogisticRegression | 0.418 | 0.475 | 1.010 |
| RandomForest | 0.445 | 0.486 | 0.989 |
| HistGradientBoosting | 0.465 | 0.490 | 0.985 |

Selected: **HistGradientBoosting** (highest validation macro-F1).

## Test period - model vs baselines

| Model | Macro-F1 | Accuracy | Balanced accuracy |
|---|---|---|---|
| HistGradientBoosting | 0.474 | 0.496 | 0.482 |
| majority class ('low') | 0.174 | 0.353 | 0.333 |
| peak-hour rule (peak=high, else low) | 0.371 | 0.486 | 0.458 |

## Share of 'high' congestion by festival / event (all rows)

| Festival / event | Share high | Rows |
|---|---|---|
| Diwali | 0.4 | 40 |
| Eid | 0.5 | 48 |
| Holi | 0.259 | 54 |
| New Year | 0.341 | 41 |
| None | 0.351 | 19531 |
| Other event | 0.353 | 286 |

## Feature importance (permutation, test set, drop in macro-F1)

| Feature | Importance |
|---|---|
| is_peak_hour | 0.0767 |
| hour_cos | 0.0717 |
| hour_sin | 0.0306 |
| day_of_week | 0.0099 |
| city | 0.0085 |
| month_sin | 0.0061 |
| month_cos | 0.0049 |
| festival_event | 0.0046 |
| road_type | 0.0026 |
| is_weekend | 0.0023 |
| traffic_signal | 0.0019 |
| weather | 0.0015 |
| is_festival_or_event | 0.0011 |
| lanes | -0.0006 |

## Limitations
- No uploaded dataset measures traffic congestion; traffic_density from accident records is used as a proxy.
- traffic_density is observed only where and when accidents happened, so it is a biased sample of traffic.
- Only 469 of 20000 rows fall on a festival or significant date, so festival effects are estimated from little data.
- pandals.csv and reviews.csv contain no traffic, crowd-count or date fields and are not used.