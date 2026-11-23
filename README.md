# Freight Disruption Digital Twin

A digital twin for freight corridors in India (Mumbai → Bengaluru and Chennai → Bengaluru) that forecasts
**disruption risk per road segment and per hour** from three machine-learning models, and explains it
through a web app and an AI assistant.

| Folder | What it is |
|---|---|
| [`backend/`](backend/) | FastAPI + scikit-learn: data download, model training, evaluation reports, prediction API, login, AI assistant |
| [`frontend/`](frontend/) | The web app (vanilla JavaScript): dashboard, corridor risk map, segment details, scenario simulator, GenAI assistant |

## What the models do
- **Rainfall / heavy-rain (flood trigger) risk** – next-day rainfall category for each of the 35 corridor segments,
  trained on **real daily rainfall at each segment** (Open-Meteo / ERA5, 2020 onward) plus the **rain forecast made the
  day before** (Open-Meteo Previous Runs API). The heavy-rain alert rule is chosen on a validation year, never on the test year.
- **Accident risk** – road risk score from road, weather, visibility, traffic and time of day.
- **Festival / event traffic** – congestion level by hour, using a festival and event calendar; for **today** the current
  hour uses **live traffic speeds from TomTom**.

The combined segment risk is `1 − (1 − 0.7·rain)(1 − 0.5·accident)(1 − 0.3·congestion)`; these weights are documented
assumptions because no disruption records exist to learn them from.

**Honest evaluation:** every model is trained on older data, tuned on a validation period and scored once on a later
test period it never saw, and compared with simple baselines. Results and limitations are in
[`backend/reports/`](backend/reports/) and on the app's *Model & Validation* page.

## Run it locally
Requirements: Python 3.11+ and (optional) free API keys.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # then add your own keys (AI assistant, TomTom) - optional
python -m training.download_rainfall   # real rainfall per segment (free, no key; about 1 hour, resumable)
python -m training.download_forecasts  # rain forecasts made the day before (free, no key)
python -m training.train_all           # trains the 3 models and writes backend/reports/
uvicorn app.main:app                   # then open http://127.0.0.1:8000 and click "Create account"
```
The backend serves the `frontend/` folder at `/`, so one address runs the whole app.
Details: [`backend/README.md`](backend/README.md). Tests: `python -m pytest -q`.

## Data sources
- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0): Historical Weather API (ERA5) and Previous Runs API.
- Live traffic: [TomTom Traffic Flow API](https://developer.tomtom.com/) (own key required).
- Road accidents, traffic and festival/event datasets: uploaded project datasets in `backend/datasets/` (see the reports for their limitations).

Keys are never stored in this repository (`.env` is excluded); each user adds their own.
