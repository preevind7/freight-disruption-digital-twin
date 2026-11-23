# Model 2 - Accident risk: evaluation report

Dataset: `indian_roads_dataset.csv` · Split: train 2022-01-01..2023-12-31, validation 2024-01-01..2024-06-30, test 2024-07-01..2025-04-15

Target: `pre_accident_risk = risk_score - 0.20 x (accident_severity == fatal)`

## Algorithm comparison (validation)

| Model | MAE | RMSE | R2 | Risk-band accuracy |
|---|---|---|---|---|
| LinearRegression (Ridge) | 0.0000 | 0.0018 | 0.9999 | 1.000 |
| RandomForest | 0.0005 | 0.0054 | 0.9993 | 0.999 |
| HistGradientBoosting | 0.0005 | 0.0019 | 0.9999 | 0.992 |

Selected: **LinearRegression (Ridge)** (lowest validation MAE).

## Test period - model vs baseline

| Model | MAE | RMSE | R2 | Risk-band accuracy |
|---|---|---|---|---|
| LinearRegression (Ridge) | 0.0000 | 0.0000 | 1.0000 | 1.000 |
| Baseline (training mean) | 0.1700 | 0.2083 | -0.0007 | 0.433 |

90% of validation predictions were within **±0.000** of the true score.

## Feature importance (permutation, test set, increase in MAE)

| Feature | Importance |
|---|---|
| visibility | 0.1114 |
| traffic_density | 0.0926 |
| is_peak_hour | 0.0576 |
| weather | 0.0552 |
| festival | 0.0008 |
| road_type | 0.0000 |
| hour_cos | 0.0000 |
| temperature | 0.0000 |
| city | 0.0000 |
| is_weekend | 0.0000 |
| lanes | -0.0000 |
| traffic_signal | -0.0000 |
| day_of_week | -0.0000 |
| hour_sin | -0.0000 |

## Limitations
- Every record is an accident, so the model scores risky conditions; it does not estimate the probability that an accident happens.
- The dataset's risk_score is rule-based; the model learns that rule (very high R2 is expected, not a sign of real-world accuracy).
- accident_severity could not be predicted from the inputs (no signal), so severity is not offered.