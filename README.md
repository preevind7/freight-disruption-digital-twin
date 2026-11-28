# GenAI Digital Twin for Freight Disruption Prediction & Mitigation

Forecasting highway disruption in India for every vehicle on the road: trucks, buses, cars/taxis and vans.
Two pilot corridors (Mumbai → Bengaluru, Chennai → Bengaluru), 35 road segments.

## Live demo (GitHub Pages)

**https://preevind7.github.io/freight-disruption-digital-twin/**

On GitHub Pages the frontend runs in **demo mode**: no backend is needed, and it uses a saved sample day
(19 Aug 2025) from the 3 ML models (`frontend/js/demo-data.js`, answered by `frontend/js/demo-backend.js`).
You can sign in with any username and password. In demo mode the assistant gives built-in answers;
live predictions and the real AI assistant need the full app.

## What is in this repository

| Part | What it does |
|---|---|
| `backend/` | Python + FastAPI server: JWT login, REST API (Swagger docs at `/docs`), 3 ML models, live weather and traffic, GenAI assistant |
| `backend/training/` | Trains the 3 models: rainfall / heavy-rain risk, accident risk, festival traffic congestion |
| `backend/reports/` | Test results of each model on a held-out later period, against simple baselines |
| `frontend/` | Plain HTML, CSS and JavaScript web app, one screen per feature; served by the backend or by GitHub Pages |

## Data

- **Rainfall:** real daily rainfall for each of the 35 segments from Open-Meteo (ERA5 reanalysis), 2020 onwards, plus archived day-ahead forecasts from 2024.
- **Accident and traffic:** `indian_roads_dataset.csv` (2022 – Apr 2025) and `significant_dates.csv` for festivals. These look synthetic; the accident risk score is rule-based.
- **Live (full app only):** Open-Meteo forecasts and TomTom Traffic Flow (free key).
- **Vehicles, trips and delay minutes** are demo data. Delay accuracy is not measured yet, because no real trip records are available.

## Model results (test data the models never saw)

| Model | Test period | Measure | Simple baseline | Forecast alone | Our model |
|---|---|---|---|---|---|
| Rainfall | 1 Jan – 12 Nov 2026 | Heavy-rain PR-AUC | 0.150 | – | 0.526 |
| Rainfall | 1 Jan – 12 Nov 2026 | Log loss | 0.505 | – | 0.391 |
| Rainfall | 1 Jan – 12 Nov 2026 | Accuracy | 78.0% | 83.3% | 82.2% |
| Heavy-rain alerts | 1 Jan – 12 Nov 2026 | Precision · recall | 39% · 39% | 59% · 57% | 32% · 52% |
| Traffic | Jul 2024 – Apr 2025 | Accuracy | 48.6% | – | 49.7% |
| Accident | Jul 2024 – Apr 2025 | R² | 0.00 | – | 1.00 (learns the dataset's formula) |

Baselines: rain frequency by segment and month; heavy rain yesterday (alerts); a peak-hour rule (traffic).
The test period had only 23 heavy-rain segment-days. Full details are in `backend/reports/`.

## Scenario simulator

Monte Carlo comparison in the browser (100 – 5,000 runs; 1,000 runs take well under a second), with optional
alternate-route comparison and current-forecast, heavy-rain, closure and congestion scenarios. Results show
P50 and P90 delay and the chance of arriving on time.

## GenAI assistant (full app)

Answers questions in plain language from the ML forecast. Provider: Gemini, with Groq as an automatic backup
(set in `backend/.env`). Every number in an answer is compared with the model data, and unsupported numbers
are flagged.

## Run the full app

```bash
cd backend
pip install -r requirements.txt
python -m training.train_all       # trains the 3 models
python -m pytest -q                # 118 tests
python -m app.manage_users secret  # put the result in .env as JWT_SECRET
uvicorn app.main:app --port 8000   # open http://127.0.0.1:8000
```

Copy `backend/.env.example` to `backend/.env` and add your keys (Gemini / Groq, TomTom) to turn on the real AI and live traffic.
GitHub Actions retrains the models and runs all tests on every push.
