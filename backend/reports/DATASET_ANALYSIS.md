# Dataset analysis (Phase 4)

All figures below were computed from the uploaded files. Nothing is assumed about columns that are not in the data.

## Summary - which file feeds which model

| File | Rows | Used by | Role |
|---|---|---|---|
| `india_rainfall_2016_2025.csv` | 73,060 | Model 1 | Training data (rainfall history + target) |
| `indian_roads_dataset.csv` | 20,000 | Models 2 and 3 | Training data (accident risk; traffic density) |
| `significant_dates.csv` | 137 | Model 3 | Festival / event calendar feature |
| `daily-rainfall-at-state-level.csv` | 204,876 | not used | See reasons below |
| `pandals.csv` | 5,000 | not used | No traffic fields |
| `reviews.csv` | 20,000 | not used | No traffic fields |

---

## 1. `india_rainfall_2016_2025.csv` -> Model 1

- 20 cities x 3,653 consecutive days (2016-01-01 to 2025-12-31). No missing values, no duplicate rows, no gaps in dates (checked in code before building lag features).
- Columns: Date, Year, Month, Month_Name, Day, Day_of_Week, Season, Location, Country, Latitude, Longitude, Rainfall_mm, Precipitation_mm, Rain_Hours.
- **Categorical:** Location, Season, Month_Name, Day_of_Week, Country (single value). **Numeric:** Rainfall_mm, Precipitation_mm, Rain_Hours, Latitude, Longitude.
- **No flood column exists.** Flood events cannot be learned. The model predicts the IMD daily rainfall category of the next day; the probability of heavy rain (>= 64.5 mm/day) is reported as the *flood-trigger risk*.
- Class balance of daily rainfall: < 2.5 mm 52,878 days; light 14,847; moderate 4,980; heavy or above 355 (0.49%). Heavy days occur in every year (21-45 per year), so a chronological split keeps positives in every period.
- Rainfall today correlates with yesterday (lag-1 r = 0.50), so past days carry real signal.
- **Leakage removed:** `Precipitation_mm` equals `Rainfall_mm` on 99.3% of rows (r = 0.997) and `Rain_Hours` is measured on the same day. Both describe the day being predicted, so neither is an input.
- Inputs used: city, season (day-of-year sin/cos), rainfall on each of the previous 7 days, 3- and 7-day totals, 7-day maximum, number of rainy days (>= 2.5 mm) in the last 7.

## 2. `indian_roads_dataset.csv` -> Models 2 and 3

- 20,000 accident records, 8 cities, 2022-01-01 to 2025-04-15, 24 columns. No duplicates. Only `festival` has missing values (99.4% empty - only 115 rows fall on Diwali, Holi, Eid or New Year).
- **Categorical:** city, state, day_of_week, road_type, weather, visibility, traffic_density, cause, accident_severity, festival. **Numeric:** latitude, longitude, hour, is_weekend, lanes, traffic_signal, temperature, vehicles_involved, casualties, is_peak_hour, risk_score.
- Consistency checks passed: `time` = `hour`; `is_weekend` = Saturday/Sunday; `is_peak_hour` = hour in {8, 9, 10, 17, 18, 19}.
- Visibility always follows weather: clear -> high only; fog -> low only; rain -> low or medium. The API rejects pairs never seen in the data.

### Target choice for Model 2 (accident)
- *"Will an accident happen?"* - impossible: every row is an accident (no negative examples).
- *accident_severity* - not learnable: fatal / major / minor are about 15 / 30 / 55% regardless of weather, cause or road type. A gradient-boosting model reached macro-F1 0.237 without leaking columns (no better than always predicting "minor"). It only improves when given `casualties` or `risk_score`, which are known after the crash.
- *risk_score* - a deterministic rule (linear fit R2 = 0.99999). It adds fixed amounts for low/medium visibility, high traffic density, peak hour, festival **and +0.20 when the accident was fatal**. The fatal part is post-crash information, so it is removed:
  `pre_accident_risk = risk_score - 0.20 x (accident_severity == "fatal")` (range 0.10-0.90). Every one of the 42 condition combinations maps to exactly one value.
- **Excluded as leakage:** cause, accident_severity, vehicles_involved, casualties, raw risk_score.
- Expect near-perfect test scores: the model learns the dataset's rule. This shows correct implementation, not real-world accuracy.

### Target choice for Model 3 (traffic)
- No uploaded file measures congestion (speed, volume, travel time). The closest signal is `traffic_density` (low 35% / medium 30% / high 35%), recorded at each accident.
- Peak hour is the strongest driver (60% high at peak vs 27% off-peak). City and weekday barely matter.
- **Excluded as leakage:** risk_score (computed from traffic_density), visibility (determined by weather), and all post-accident columns.
- **Bias:** density is observed only where accidents happened, so it is not a random sample of traffic.

## 3. `significant_dates.csv` -> Model 3 feature
- 137 dated events, 2024-10-31 to 2026-12-31 (festivals, holidays, matches, concerts, metro changes), Bengaluru-oriented.
- Mapped to the roads dataset's festival names (Diwali/Deepavali -> Diwali, Holi, Eid/Bakrid -> Eid, New Year), all other events -> "Other event".
- Overlap with the roads data is small: 469 of 20,000 rows fall on a festival or listed date. Festival effects are therefore weak and uncertain - stated in the model report.
- At prediction time the API looks the requested date up in this file, so future festivals (e.g. 2026-11-08 Deepavali) are detected automatically.

## 4. Files not used, and why
- `daily-rainfall-at-state-level.csv` - state-level rather than city-level, ends 2024-07-31, 8.4% of `actual` missing and 15.1% of `deviation` missing. `deviation` is calculated from `actual` and `normal` (leakage), and `rfs` is undocumented. The city file is complete and longer, so it was preferred.
- `pandals.csv` / `reviews.csv` - Kolkata Durga Puja pandal attributes and user ratings, with no traffic, crowd-count or date fields, so they cannot train a congestion model. Internal checks also show the two files are unrelated: review text is spread evenly over 1-5 stars, and pandal `avg_rating` does not match the review average (r = -0.007).

## 5. What would improve the models
- Flood event records (date, location) to train a real flood classifier.
- A traffic dataset with measured speed / volume / congestion and dates (ideally Bengaluru, matching `significant_dates.csv`).
- Accident data that also includes non-accident road-hours, so accident probability can be estimated.
