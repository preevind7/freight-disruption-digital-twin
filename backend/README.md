# Freight Disruption Digital Twin - ML Backend

FastAPI backend with **three independent scikit-learn models**, wired to the existing frontend:

| # | Model | Endpoint | Trained on | Predicts |
|---|---|---|---|---|
| 1 | Rainfall & flood-trigger risk | `POST /api/predict/rainfall` | `india_rainfall_2016_2025.csv` | Next-day IMD rainfall category + heavy-rain (>= 64.5 mm) probability |
| 2 | Accident risk | `POST /api/predict/accident` | `indian_roads_dataset.csv` | Pre-accident road risk score 0-1 (Low / Moderate / High) |
| 3 | Festival traffic congestion | `POST /api/predict/traffic` | `indian_roads_dataset.csv` + `significant_dates.csv` | Congestion level low / medium / high |

Also available: `GET /api/corridor/forecast` (all three models for every highway segment and hour — drives the Dashboard, Risk Map, Simulator, Assistant and Alerts), `GET /api/health`, `GET /api/models` (metrics + allowed inputs), `GET /api/rainfall/history`, `POST /api/models/reload`, and accounts: `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`.

**Login is required.** Every endpoint except `/api/health` and the login endpoints needs a token: `Authorization: Bearer <token>`. See [Login & accounts](#login--accounts).

Why these targets, and what each dataset can and cannot support: **[reports/DATASET_ANALYSIS.md](reports/DATASET_ANALYSIS.md)**.

---

## Folder structure

```
project/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, CORS, error handlers, routers
│   │   ├── config.py               paths, CORS origins (.env overrides)
│   │   ├── schemas/                Pydantic request/response models (one file per model)
│   │   ├── routes/                 API endpoints (one file per model + system routes)
│   │   ├── services/               prediction logic + auth_service (login, token check) + user_store
│   │   ├── manage_users.py         command-line tool for accounts and the token secret
│   │   ├── ml_models/              feature builders shared by training AND serving + model registry
│   │   └── utils/                  preprocessing, validation, festival calendar lookup
│   ├── training/                   train_rainfall.py, train_accident.py, train_traffic.py, train_all.py
│   ├── datasets/                   the CSVs used for training (unused ones in datasets/unused/)
│   ├── trained_models/             *.joblib bundles (pipeline + metadata + metrics)
│   ├── reports/                    evaluation reports (.md + .json) and DATASET_ANALYSIS.md
│   ├── tests/                      73 automated tests (API, login, sign-up, corridor forecast, AI assistant)
│   ├── users.json                  login accounts (password hashes only)
│   ├── requirements.txt
│   ├── .env / .env.example
│   └── README.md
└── frontend/                       your existing site + ML pages (see "Frontend changes")
```

The same feature-building code (`app/ml_models/*_model.py`) is used for training and for API requests. This prevents the classic bug where the API builds slightly different features from those the model learned on.

---

## Setup in VS Code (Phase 3)

Open the `project` folder in VS Code, then open a terminal with **Terminal → New Terminal**. Run every command below in that terminal.

### 1. Go to the backend folder
```bash
cd backend
```

### 2. Create a virtual environment
Windows:
```powershell
python -m venv .venv
```
macOS / Linux:
```bash
python3 -m venv .venv
```
Expected: no output, and a new `.venv` folder appears. Use **Python 3.10 – 3.14** (check with `python --version`).

### 3. Activate it
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
If PowerShell says *running scripts is disabled*, run this once, then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Windows CMD: `.venv\Scripts\activate.bat`  ·  macOS / Linux: `source .venv/bin/activate`

Expected: the prompt starts with `(.venv)`. In VS Code press **Ctrl+Shift+P → Python: Select Interpreter** and pick the `.venv` interpreter.

### 4. Install the dependencies
```bash
pip install -r requirements.txt
```
Expected: it ends with `Successfully installed fastapi-… scikit-learn-1.6.1 …`.

### 5. Train the three models (Phase 5)
```bash
python -m training.train_all
```
This takes about 30 seconds. Expected log lines:
```
INFO  ================ RAINFALL ================
INFO  Selected HistGradientBoosting; heavy-rain alert threshold tuned on validation = 0.0531
INFO  Saved model bundle -> ...trained_models/rainfall_model.joblib
INFO  ================ ACCIDENT ================
INFO  Selected LinearRegression (Ridge) (lowest validation MAE)
INFO  ================ TRAFFIC ================
INFO  Selected HistGradientBoosting (highest validation macro-F1)
INFO  All models trained. Reports are in backend/reports/.
```
The trained files are already included, but re-training on your machine guarantees they match your installed library versions. You can also train one model at a time: `python -m training.train_rainfall` (or `train_accident`, `train_traffic`).

### 6. Set up login (secret + accounts)
The project already includes a secret in `.env` and four demo accounts in `users.json`. To make your own (recommended):
```bash
python -m app.manage_users secret        # new random JWT_SECRET written to .env
python -m app.manage_users list          # show accounts
```
Demo accounts (one per target user from the proposal):

| Username | Password | Role |
|---|---|---|
| `dispatcher` | `Dispatch@123` | Freight & Logistics Dispatcher |
| `operator` | `Fleet@123` | Fleet & Transport Operator |
| `planner` | `Planner@123` | Highway & Policy Planner |
| `researcher` | `Research@123` | Researcher / Student |

Change them before showing the app to others: `python -m app.manage_users passwd dispatcher`.

### 7. Start the API server
```bash
uvicorn app.main:app --reload
```
Expected:
```
INFO  app.ml_models.registry  Loaded rainfall model (HistGradientBoosting)
INFO  app.ml_models.registry  Loaded accident model (LinearRegression (Ridge))
INFO  app.ml_models.registry  Loaded traffic model (HistGradientBoosting)
INFO:     Uvicorn running on http://127.0.0.1:8000
```
Keep this terminal open. Open **http://127.0.0.1:8000/docs** for Swagger UI.

### 8. Start the frontend (second terminal)
Click **+** in the terminal panel, then run:
```bash
cd frontend
python -m http.server 5500
```
(on macOS / Linux use `python3`). Open **http://127.0.0.1:5500**, click a role card (it fills in the username), enter the password, and click **Sign in**. Then use the **ML PREDICTIONS** section of the sidebar.
The VS Code **Live Server** extension also works, because it uses port 5500 by default.

> Don't open `index.html` by double-clicking it (`file://`): browsers block API calls from `file://` pages. Use the local server above.

---

## Corridor forecast: the three models on every screen

`GET /api/corridor/forecast?corridor_id=mumbai-bengaluru&date=YYYY-MM-DD` (date optional, default today IST) returns, for every segment of the corridor:

| Step | Model | What it gets |
|---|---|---|
| 1 | Rainfall | the segment's own location (real Open-Meteo data, see below) — or, before that is installed, the nearest of the 20 uploaded rainfall cities — + the last 7 days of rain → rain category, heavy-rain probability, expected mm/day |
| 2 | Traffic | nearest road-dataset city, each hour 0–23, festival from `significant_dates.csv`, weather from step 1 → congestion probabilities |
| 3 | Accident | same city, each hour, weather/visibility from step 1, traffic density from step 2 → risk score |
| 4 | Rule | `risk = 1 − (1 − 0.7·rain)(1 − 0.5·accident)(1 − 0.3·congestion)` per hour, plus alerts |

**Where the last 7 days of rain come from** (shown on every screen as `rain: …`):
1. `dataset` — observed values from the rainfall file the model was trained on (`corridor_rainfall_open_meteo.csv`, or `india_rainfall_2016_2025.csv`)
2. `live` — Open-Meteo daily precipitation at the segment's own location (free, no key; needs internet on the backend computer)
3. `replay` — if there is no internet: the same calendar date from 2025 in the dataset, clearly labelled

**Honest limits:** there are no disruption or delay records in the data, so the step-4 weights are stated assumptions (returned in the response under `method`), and delay minutes in the Simulator are rule-of-thumb estimates from the risk. Segments that share the nearest data city share traffic/accident values; distances are shown on each segment page.

The **Forecast date** box in the top bar switches every screen to another date — e.g. `2025-06-19` (107.8 mm observed in Mumbai) shows the models reacting to a heavy-rain day.

Settings in `.env`: `LIVE_WEATHER=true|false`, `LIVE_WEATHER_TIMEOUT=4`, `FORECAST_CACHE_MINUTES=30`.

## Real rainfall for every segment (Open-Meteo)

The uploaded rainfall file has only 20 cities, so a highway segment used the rain of the nearest city.
`training/download_rainfall.py` replaces it with **real daily rainfall at the exact location of each of the 35 segments**:

```powershell
python -m training.download_rainfall      # 2020-01-01 to 6 days ago; ~1 hour (paced), resumes if stopped
python -m training.train_rainfall          # retrain on it; reports/rainfall_report.md is rewritten
```
Then restart the API (Ctrl+C, `uvicorn app.main:app --reload`).

- **Source:** Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis), free for non-commercial use, no key. *Weather data by Open-Meteo.com, CC BY 4.0.*
- **Free limits:** 5,000 calls/hour, 10,000/day; a request longer than 2 weeks counts as ~1 call per 14 days. The script stays under 4,000/hour, saves every segment in `datasets/open_meteo_cache/`, and continues where it stopped. Running it again later only downloads the new days.
- **Split:** chosen from the data: validation = 2024, test = 2025 up to the download date (never seen in training).
- **Safe switch:** the app always reads the rainfall file its loaded model was trained on, so downloading alone changes nothing until you retrain. To go back, delete `datasets/corridor_rainfall_open_meteo.csv` and retrain.
- **Honest limit:** ERA5 is a weather model fitted to observations (about 10–25 km grid), not a rain gauge; it smooths very local cloudbursts, so it records fewer heavy-rain days than a gauge would. There are still no flood records, so heavy rain (≥ 64.5 mm/day) remains the flood trigger.

## Rain forecast made the day before (optional model input)

Forecasting heavy rain from past rain alone is hard. `training/download_forecasts.py` downloads, for every segment, the rain
**forecast for each day that was made 24 hours earlier** (Open-Meteo Previous Runs API, `precipitation_previous_day1`, archived from January 2024):

```powershell
python -m training.download_forecasts     # 2024-01-01 to 2 days ago; ~2,700 Open-Meteo calls, paced, resumes
python -m training.train_rainfall          # compares "past rain" vs "past rain + forecast" on the same days
```
Then restart the API.

- **No look-ahead:** each hourly value was predicted 24 h before that hour, so the model only gets what was known before the day began.
- **Honest comparison:** with forecasts, the split becomes validation = last year, test = the current year. Every candidate (3 algorithms × 2 input sets) is scored on exactly the same validation and test segment-days; "forecast alone" is a baseline in the report. The forecast model is used only if it wins on validation.
- **Fallback:** dates without a forecast (before 2024, far future, replay) use the best past-rain-only model and its own alert rule; every screen says which model was used.
- **Live:** for today, the backend asks the Previous Runs API for today's forecast made yesterday (same definition as in training). A failure there does not affect the live 7-day rain history (separate circuit breakers).
- **Alert rules** now also include `forecast` (forecast ≥ 64.5 mm), chosen on validation like the others.

## Live traffic (TomTom)

The traffic model was trained on a dataset without real traffic counts, so for **today** the backend measures congestion directly:

1. Create a free key at developer.tomtom.com (no card) and add `TOMTOM_API_KEY=...` to `backend/.env`; restart the API.
2. For each segment, TomTom's Flow Segment Data gives the current speed and the free-flow speed of the nearest road.
   `congestion = min(1, (1 − current/free-flow) / 0.5)`; level low (≥ 80% of free flow) / medium / high (< 50%, or road closed).
3. That measured value replaces the traffic model's estimate **for the current hour** (other hours still use the model). Road closures raise a High alert, slow traffic a Moderate one.
4. Readings are cached 30 minutes (35 segments × 48 = 1,680 calls/day, under the free 2,500) and the backend stops at `TRAFFIC_DAILY_BUDGET`.
5. Every reading is appended to `datasets/traffic_observations.csv` (speed, free-flow speed, hour, weekday, festival). After some weeks this is real data to train a traffic model on.

`GET /api/traffic/live/status` shows whether it is on, requests used today and readings saved (never the key). Tests never call TomTom.

## Share a public link (free, from your own computer)

The backend also serves the frontend at `/` when the frontend folder exists (default `Desktop\DEMO FOLDER`, or set `FRONTEND_DIR`).
So one address works for both, and a Cloudflare Quick Tunnel (no account) can make it public:

```powershell
winget install --id Cloudflare.cloudflared         # once
uvicorn app.main:app --reload                       # terminal 1: the app on http://127.0.0.1:8000
cloudflared tunnel --url http://localhost:8000      # terminal 2: prints https://<random>.trycloudflare.com
```
- Works while this computer is on and both terminals run; the address changes every time the tunnel starts.
- Hidden files (`.git`, `.env`, ...) are never served. Keys stay in `backend/.env` on this computer.
- Visitors share your free AI / TomTom quotas (per-user assistant rate limit and the TomTom daily budget still apply).
- Cloudflare offers Quick Tunnels for testing and demos; for a permanent site use a server with about 2 GB RAM.

## One-click public link (Windows)

Double-click `Start Freight Twin.bat` (copy it from `scripts\` to the Desktop): it starts or reuses the backend and the Cloudflare tunnel in two windows, then shows the public link, copies it to the clipboard and opens it. `Stop Freight Twin.bat` stops both.

## Online 24/7 (free Oracle Cloud server)

See `deploy/README_DEPLOY.md`: `deploy/package_for_server.ps1` (PC) makes the package, `deploy/setup_server.sh` (server) installs it as an auto-starting service behind HTTPS.

## AI assistant (real LLM)

The **GenAI Assistant** page sends each question to a large language model (LLM) together with the live facts of the app.

1. **Facts sent to the AI:** the ML corridor forecast for the selected corridor, date and hour (all 3 models), the alerts, your trips and the models' test results (`app/services/assistant_service.py → build_context`).
2. **Instructions:** the AI is told to use only those facts and never to invent numbers (`SYSTEM_PROMPT`).
3. **Number check:** every number in the answer is compared with the facts of the segment named in the same sentence, at the precision written. The page shows ✓ when all numbers match, or ⚠ with the numbers it could not verify.
4. **Backup:** if no provider is configured, or the provider fails, the page uses built-in answers built from the same forecast, and says so.

**Setup:** add these lines to `backend/.env`, then restart uvicorn.

| Provider | Cost | `.env` lines | Where to get the key |
|---|---|---|---|
| Google Gemini | free tier | `LLM_PROVIDER=gemini` · `LLM_API_KEY=your-key` | Google AI Studio → *Get API key* |
| Groq | free tier | `LLM_PROVIDER=groq` · `LLM_API_KEY=gsk_...` | console.groq.com/keys |
| Ollama (on your PC) | free, offline | `LLM_PROVIDER=ollama` | install from ollama.com, then run `ollama pull llama3.2` |
| Anthropic Claude | paid | `LLM_PROVIDER=anthropic` · `LLM_API_KEY=sk-ant-...` | console.anthropic.com |
| OpenAI | paid | `LLM_PROVIDER=openai` · `LLM_API_KEY=sk-...` · `LLM_MODEL=<model>` | platform.openai.com |

- Optional: `LLM_MODEL=` (other model name), `LLM_BASE_URL=` (other address), `ASSISTANT_REQUESTS_PER_MINUTE=10`.
- **Backup provider (recommended):** free AI tiers are sometimes overloaded (HTTP 503). Add `LLM_FALLBACK_PROVIDER=groq` and `LLM_FALLBACK_API_KEY=gsk_...`:
  when the main provider is busy, too slow (25 s) or over quota, the same question goes to the backup and the answer says so. A rejected key is reported, not hidden.
  Groq's default model is `openai/gpt-oss-120b` (Groq deprecated `llama-3.3-70b-versatile`).
- Check it with `GET /api/assistant/status` (in Swagger), or open the assistant page: it shows **AI connected** with the model name.
- **Keep the key secret:** it stays in `backend/.env` (which `.gitignore` excludes) and is never sent to the browser.
- **Privacy:** questions, trips and forecast facts are sent to the provider. On free tiers, providers may use this data to improve their products; read the provider's terms before entering real company data.
- **Errors:** messages such as "API key rejected", "rate limit or free quota reached" or "model not found (404)" are shown in the chat. A 404 usually means the provider renamed the model; set `LLM_MODEL` to a current name from the provider's model list.

## Login & accounts

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/auth/register` | public | JSON `{"name","username","password","role"}` → creates the account (then sign in). Turn off with `ALLOW_REGISTRATION=false` |
| `POST /api/auth/login` | public | JSON `{"username","password"}` → `{access_token, expires_at, user}` |
| `POST /api/auth/token` | public | Same, as a form (used by the Swagger **Authorize** button) |
| `GET /api/auth/me` | token | Current user; the frontend uses it to check a saved token is still valid |
| `GET /api/health` | public | Server and model status |
| all `/api/predict/*`, `/api/models*`, `/api/rainfall/history` | token | Return **401** without a valid token |

How it works:
- **Passwords** are stored only as salted PBKDF2-SHA256 hashes in `users.json`.
- **Tokens** are JWTs signed with `JWT_SECRET` and valid for 8 hours (`TOKEN_EXPIRE_MINUTES`).
- **Lockout:** after 5 wrong passwords an account is locked for 5 minutes (HTTP 429).
- **Frontend:** the token is saved in the browser and sent with every API call. When it expires, or the server rejects it, the user is signed out and returned to the login page. **Log out** clears it.

Manage accounts (run in `backend/`):
```bash
python -m app.manage_users add priya --role dispatcher --name "Priya Deshmukh"   # asks for the password
python -m app.manage_users passwd priya
python -m app.manage_users remove priya
python -m app.manage_users seed-demo     # re-create the 4 demo accounts
```
Roles: `dispatcher`, `operator`, `planner`, `researcher`. All four roles can use all three models; the role decides which screens the frontend shows first.

**Testing in Swagger:** open `/docs` → click **Authorize** (top right) → enter a username and password → **Authorize**. Every request after that sends the token.

## API reference (Phase 6)

All prediction endpoints accept and return JSON. Errors always look like this:
```json
{ "detail": "human-readable message", "errors": [{ "field": "visibility", "message": "..." }] }
```

| Status | When |
|---|---|
| 200 | Prediction returned |
| 401 | Not logged in, wrong password, or token expired / invalid |
| 429 | Too many wrong passwords (temporary lockout) |
| 404 | `/api/rainfall/history` date outside the dataset |
| 422 | Missing field, wrong type, out-of-range number, value not seen in training (e.g. city "Paris"), or impossible combination (clear weather + low visibility) |
| 503 | Model file missing or unreadable. The message gives the training command; the other models keep working |
| 500 | Unexpected error (details are logged in the backend terminal, never sent to the browser) |

### Example requests (also pre-filled in Swagger)
```jsonc
// POST /api/predict/rainfall   (index 0 = the day before forecast_date)
{ "city": "Mumbai", "forecast_date": "2025-07-15", "rainfall_last_7_days_mm": [42.0, 18.5, 60.2, 12.0, 5.0, 0.0, 22.4] }

// POST /api/predict/accident
{ "city": "Mumbai", "road_type": "highway", "lanes": 4, "traffic_signal": false, "weather": "rain",
  "visibility": "low", "temperature": 27, "traffic_density": "high", "hour": 18, "day_of_week": "Friday", "festival": "None" }

// POST /api/predict/traffic   (festival omitted -> detected from significant_dates.csv)
{ "city": "Bangalore", "date": "2026-11-08", "hour": 18, "road_type": "urban", "lanes": 3, "traffic_signal": true, "weather": "clear" }
```
Allowed values for every field come from the training data. See `GET /api/models` → `input_options`.

---

## Model results (test period = most recent dates, never seen in training)

Full reports: `reports/rainfall_report.md`, `reports/accident_report.md`, `reports/traffic_report.md`, also shown on the **Model & Validation** page.

**Model 1 – Rainfall** (HistGradientBoosting; train 2016–2022, tune 2023, test 2024–2025)

| | Log loss ↓ | Macro-F1 | Accuracy | Heavy-rain PR-AUC |
|---|---|---|---|---|
| Model | **0.486** | 0.449 | **0.799** | **0.058** |
| Climatology baseline | 0.604 | 0.357 | 0.761 | 0.028 |
| Persistence baseline (tomorrow = today) | – | **0.491** | 0.781 | – |

The model gives clearly better probabilities than climatology. However, the simple persistence rule still has a higher macro-F1, and heavy rain remains hard to forecast from rainfall history alone: the alert caught 14 of 82 heavy days (recall 0.17) with 184 alerts (precision 0.08).

**Model 2 – Accident risk** (Ridge; test Jul 2024 – Apr 2025): MAE 0.000, R² 1.000, risk-band accuracy 1.000, against MAE 0.170 for the mean baseline. The perfect score is because the dataset's `risk_score` is a formula and the model recovers it exactly. It confirms the pipeline is correct; it is not evidence of real-world accuracy.

**Model 3 – Traffic congestion** (HistGradientBoosting; test Jul 2024 – Apr 2025)

| | Macro-F1 | Accuracy |
|---|---|---|
| Model | **0.474** | **0.496** |
| Peak-hour rule baseline | 0.371 | 0.486 |
| Majority-class baseline | 0.174 | 0.353 |

Modest but real gain. Hour of day dominates; festival effects are small and based on only 469 rows.

---

## Testing (Phase 8)

```bash
python -m pytest -v
```
Expected: `73 passed`. The tests cover:
- health and CORS preflight
- valid predictions for all three models
- the same input always giving the same output, and different inputs giving different outputs (no hardcoded results)
- invalid and missing inputs (422)
- dates outside the dataset (404)
- a deleted model file returning 503 while the other models keep working
- login: correct and wrong passwords, every protected route returning 401 without a token, tampered and expired tokens, lockout, and no password hashes ever returned

The tests use their own temporary users file, so your `users.json` is never changed.

Manual checks in Swagger (`/docs`): **Authorize** first, then click an endpoint → **Try it out** → **Execute**.

---

## Frontend changes (Phase 7)

All existing screens still work as before. Added or changed:

| File | Change |
|---|---|
| `js/api.js` (new) | One place for the backend URL, `fetch` with timeout, error parsing, shared UI helpers |
| `js/pages/predict-rainfall.js` (new) | Rainfall form, **Fill from dataset** button, result with probabilities and the observed outcome |
| `js/pages/predict-accident.js` (new) | Accident form (visibility limited to valid options for the chosen weather), score, risk drivers |
| `js/pages/predict-traffic.js` (new) | Traffic form with festival auto-detection, level and probabilities |
| `js/pages/model.js` | The empty placeholder now shows the real evaluation reports from the backend |
| `css/ml.css` (new) | Layout for the new pages, using your existing design tokens |
| `js/pages/login.js` | Same design; "Your name" replaced by **Username + Password**, checked by `POST /api/auth/login` |
| `js/store.js`, `js/router.js`, `js/app.js` | Token saved with the session; every screen except login needs a valid token; Log out clears it |
| `index.html`, `js/router.js`, `js/app.js` | Script and style tags, 3 routes, "ML PREDICTIONS" sidebar group, banner text |

Every form shows a loading spinner while waiting and a readable error box if the request fails (backend down, invalid input, model missing).
To point the frontend at another backend address without editing code, run this in the browser console: `localStorage.setItem("fdt-api-base", "http://host:port")`.

---

## Final validation checklist (Phase 9)

- [x] Three separate datasets → three separate pipelines → three separate `.joblib` files and endpoints
- [x] Each form calls only its own endpoint (`/api/predict/rainfall` | `accident` | `traffic`)
- [x] Predictions come only from the loaded models. There are no random or hardcoded results (checked by tests)
- [x] Preprocessing is saved inside each pipeline; the same feature code is used for training and serving
- [x] Chronological train / validation / test split, compared against baselines, with metrics documented in `reports/`
- [x] Invalid input → 422 with a field message; missing model → 503; unexpected → 500 without internal details
- [x] CORS limited to the configured frontend origins (`.env` → `CORS_ORIGINS`)
- [x] Login required: the whole frontend and every data/prediction API need a valid token; passwords hashed; tokens expire
- [ ] Other existing screens (Dashboard, Risk Map, Simulator, …) still use the prototype's synthetic data. Connecting them would need segment-level, hourly data that the uploaded datasets don't contain.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Login says **Backend unreachable** | The backend must be running to sign in: start uvicorn |
| Sent back to the login page after a while | The token expired (8 h) or `JWT_SECRET` changed. Sign in again |
| **429 Too many failed attempts** | Wait 5 minutes, or restart uvicorn (lockouts are kept in memory) |
| Forgot a password | `python -m app.manage_users passwd <username>` |
| Frontend shows **Backend unreachable** | Start `uvicorn app.main:app --reload` in `backend/` and check http://127.0.0.1:8000/api/health |
| Browser console shows a **CORS** error | Open the frontend at an address listed in `CORS_ORIGINS` (e.g. http://127.0.0.1:5500), or add yours to `.env` and restart uvicorn |
| **503 Model file not found** | `python -m training.train_all`, then `POST /api/models/reload` or restart the server |
| Warning *trained with scikit-learn X but Y installed* | Re-run `python -m training.train_all` |
| Numbers differ slightly from this README | Newer scikit-learn versions train slightly different models; your own `reports/` are the reference |
| `[Errno 10048]` / *address already in use* | Another server is on port 8000: close it, or run `uvicorn app.main:app --reload --port 8001` and set `fdt-api-base` accordingly |
| The live GitHub Pages site can't reach your local API | Browsers may block an HTTPS site from calling `http://127.0.0.1`. Use the local frontend for development; deploying the API publicly (HTTPS) is the next step |
