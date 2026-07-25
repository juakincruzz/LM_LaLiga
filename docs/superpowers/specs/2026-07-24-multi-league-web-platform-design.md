# Design Spec: LM_LaLiga Multi-League Web Platform

**Date:** 2026-07-24
**Approach:** FastAPI + Jinja2 + HTMX + SQLite (Opción B)
**Scope:** Expand from La Liga terminal-only to 5 European leagues via web dashboard with job execution and off-season transfer analysis.

---

## 1. Goals

- Replace terminal-only usage with a local web dashboard.
- Support **5 leagues:** La Liga (SP1), Premier League (E0), Bundesliga (D1), Serie A (I1), Ligue 1 (F1).
- Allow triggering data updates, model training, season simulation, and Monte Carlo from the browser.
- Provide **off-season mode:** when a league is not active, show previous season probabilities + next-season projection based on transfer analysis.
- Integrate **Transfermarkt** data (via local `transfermarkt-api`) to model how incoming/outgoing players affect team strength.
- Keep architecture deployable for future sharing with a close circle.

---

## 2. Architecture

### 2.1 Stack
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Web Framework | **FastAPI** | API endpoints, background jobs, server-side rendering |
| Templates | **Jinja2** | HTML rendered server-side |
| Interactivity | **HTMX** | AJAX without custom JS: partial page updates, job polling |
| CSS | **Tailwind CSS (CDN)** | Utility-first styling, zero build step |
| Database | **SQLite** | Persistence of matches, teams, predictions, jobs, transfers |
| Data Sources | `football-data.co.uk`, Transfermarkt API (self-hosted), ESPN/BBC scrapers |
| ML | Existing stack (scikit-learn, XGBoost, LightGBM) | No change to modeling layer |
| Job Queue | **FastAPI BackgroundTasks** | For personal use this is sufficient |
| Icons | **Lucide (CDN)** | Clean, modern icons |

### 2.2 High-Level Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Browser   │────▶│  FastAPI     │────▶│  Background │
│  (HTMX)     │◀────│   Server     │◀────│   Tasks     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                          │                     │
                    ┌─────┴─────┐          ┌────┴────┐
                    │  SQLite   │          │   ML    │
                    │  (data)   │          │ Pipeline│
                    └───────────┘          └─────────┘
                          ▲
                          │
              ┌───────────┴───────────┐
              │ football-data.co.uk   │
              │ Transfermarkt API     │
              │ ESPN/BBC Scrapers     │
              └───────────────────────┘
```

---

## 3. Database Schema (SQLite)

```sql
-- leagues
id INTEGER PRIMARY KEY
code TEXT UNIQUE NOT NULL  -- 'SP1', 'E0', 'D1', 'I1', 'F1'
name TEXT NOT NULL         -- 'La Liga', 'Premier League', ...
country TEXT
season TEXT                -- '2526', '2425', etc.
last_updated DATETIME
is_active BOOLEAN          -- true if season in progress

-- teams
id INTEGER PRIMARY KEY
league_id INTEGER FK
name TEXT NOT NULL
code TEXT                  -- short code if available
alias TEXT                 -- name variations for fuzzy matching

-- matches
id INTEGER PRIMARY KEY
league_id INTEGER FK
date DATE NOT NULL
home_team_id INTEGER FK
away_team_id INTEGER FK
home_goals INTEGER
away_goals INTEGER
result TEXT                -- 'H', 'D', 'A'
season TEXT
matchday INTEGER
status TEXT                -- 'played', 'scheduled'

-- model_runs
id INTEGER PRIMARY KEY
league_id INTEGER FK
created_at DATETIME
status TEXT                -- 'pending', 'running', 'completed', 'failed'
model_type TEXT            -- 'xgboost', 'random_forest', etc.
accuracy REAL
log TEXT                   -- training log / error traceback

-- predictions
id INTEGER PRIMARY KEY
model_run_id INTEGER FK
match_id INTEGER FK
predicted_result TEXT      -- 'H', 'D', 'A'
prob_home REAL
prob_draw REAL
prob_away REAL

-- simulations (Monte Carlo)
id INTEGER PRIMARY KEY
model_run_id INTEGER FK
sim_count INTEGER
standings_json TEXT        -- JSON array of final standings

-- transfers
id INTEGER PRIMARY KEY
league_id INTEGER FK
team_id INTEGER FK
player_name TEXT
position TEXT              -- 'GK', 'DEF', 'MID', 'FWD'
transfer_type TEXT         -- 'in', 'out'
market_value_eur REAL      -- from Transfermarkt
from_club TEXT
to_club TEXT
date DATE

-- job_status (for HTMX polling)
id INTEGER PRIMARY KEY
job_type TEXT              -- 'train', 'simulate', 'update_data', 'analyze_transfers'
league_id INTEGER FK
status TEXT                -- 'queued', 'running', 'done', 'failed'
progress INTEGER           -- 0-100
result_json TEXT
created_at DATETIME
updated_at DATETIME
```

---

## 4. Data Pipeline Abstraction

The current pipeline is **hardcoded** to La Liga (`SP1`, season `2526`). We abstract it:

### 4.1 Config-Driven League Support
A new `config/leagues.yaml` (or Python dict) maps league codes to football-data.co.uk codes and metadata:

```yaml
SP1:
  name: "La Liga"
  data_code: "SP1"
  country: "Spain"
  teams: 20
E0:
  name: "Premier League"
  data_code: "E0"
  country: "England"
  teams: 20
D1:
  name: "Bundesliga"
  data_code: "D1"
  country: "Germany"
  teams: 18
I1:
  name: "Serie A"
  data_code: "I1"
  country: "Italy"
  teams: 20
F1:
  name: "Ligue 1"
  data_code: "F1"
  country: "France"
  teams: 18
```

### 4.2 Generic Data Manager
`scripts/data_manager.py` currently does:
```python
# Hardcoded
SEASON = "2526"
URL = f"https://www.football-data.co.uk/mmz4281/{SEASON}/SP1.csv"
```

Refactored to:
```python
def download_league_data(league_code: str, season: str) -> Path:
    config = LEAGUE_CONFIG[league_code]
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{config['data_code']}.csv"
    ...
```

### 4.3 Feature Engineering Reuse
`scripts/preprocess.py` already works generically on columns (`FTHG`, `FTAG`, etc.) present in **all** football-data CSVs. It only needs:
- League-specific team name normalization (dictionary per league for aliases).
- Season parameter instead of hardcoded string.

### 4.4 Model Training Per League
Each league gets its own trained model stored in `models/{league_code}_predictor.pkl`. The `LaLigaPredictor` class becomes `LeaguePredictor` parameterized by `league_id`.

---

## 5. Web Application Structure

### 5.1 Project Layout
```
LM_LaLiga/
├── app/                          # FastAPI web layer (NEW)
│   ├── main.py                   # FastAPI app factory, middleware
│   ├── config.py                 # Settings, league config
│   ├── database.py               # SQLite connection & table creation
│   ├── models.py                 # Pydantic models for API
│   ├── dependencies.py           # DB session, auth (future)
│   ├── routers/
│   │   ├── dashboard.py          # Main dashboard, league selector
│   │   ├── leagues.py            # League CRUD & status
│   │   ├── matches.py            # Match listings, results
│   │   ├── predictions.py        # Predictions & probabilities
│   │   ├── simulations.py        # Monte Carlo results
│   │   ├── jobs.py               # Background job trigger + status polling
│   │   └── transfers.py          # Transfer analysis & off-season mode
│   ├── services/
│   │   ├── data_service.py       # Wrapper around existing data_manager
│   │   ├── ml_service.py         # Model training, evaluation
│   │   ├── simulation_service.py # Monte Carlo & season sim
│   │   └── transfer_service.py   # Transfermarkt API integration
│   ├── static/
│   │   └── css/                  # Tailwind CDN reference (no build)
│   └── templates/
│       ├── base.html             # Layout with HTMX, Tailwind, Lucide
│       ├── dashboard.html        # League cards, status overview
│       ├── league_detail.html    # Standings, upcoming matches, predictions
│       ├── predictions.html      # Match-by-match probabilities
│       ├── simulations.html      # Monte Carlo charts (Chart.js CDN)
│       ├── transfers.html        # Transfer list & impact scores
│       ├── jobs/                 # Job status partials for HTMX
│       │   ├── job_status.html
│       │   ├── job_result.html
│       │   └── progress.html
│       └── partials/             # HTMX swap fragments
│           ├── match_row.html
│           ├── standings_table.html
│           └── probability_bar.html
├── config/
│   └── leagues.yaml              # League metadata
├── scripts/                      # REFACTORED existing code
│   ├── data_manager.py           # Now league-generic
│   ├── preprocess.py             # Now league-generic
│   ├── model_evaluation.py
│   ├── season_simulator.py
│   └── scraper_laliga.py         # -> scraper_league.py generic
├── models/                       # REFACTORED
│   ├── predictor.py              # Generic LeaguePredictor
│   └── {league_code}_predictor.pkl  # One per league
├── data/
│   ├── raw/{league_code}/
│   └── processed/{league_code}/
├── transfermarkt-api/            # Git submodule or clone
├── requirements.txt              # + fastapi, uvicorn, jinja2, aiosqlite
└── run_web.py                    # Entry point: `python run_web.py`
```

### 5.2 Key Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Dashboard: cards for each league, last update, status |
| `/league/{code}` | GET | League detail: standings, upcoming matches, buttons |
| `/league/{code}/data` | POST | Trigger data update for league (BackgroundTask) |
| `/league/{code}/train` | POST | Trigger model training (BackgroundTask) |
| `/league/{code}/simulate` | POST | Trigger Monte Carlo (BackgroundTask) |
| `/league/{code}/predictions` | GET | Match predictions page |
| `/league/{code}/simulations` | GET | Monte Carlo results page with charts |
| `/league/{code}/transfers` | GET | Transfer analysis page |
| `/api/jobs/{job_id}` | GET | Job status JSON for HTMX polling |
| `/api/jobs/{job_id}/poll` | GET | HTML partial with progress for HTMX swap |

### 5.3 HTMX Interaction Patterns

**Triggering a job:**
```html
<button hx-post="/league/SP1/train"
        hx-target="#job-status"
        hx-swap="innerHTML">
  Entrenar Modelo
</button>
<div id="job-status">
  <!-- Replaced with job card that auto-polls -->
</div>
```

**Polling job progress:**
The returned job card contains:
```html
<div hx-get="/api/jobs/123/poll"
     hx-trigger="every 2s"
     hx-swap="outerHTML">
  <div class="progress-bar" style="width: {{ progress }}%"></div>
  <p>{{ status }}</p>
</div>
```
When status reaches `done`, the poll endpoint returns the final results partial.

---

## 6. Off-Season & Transfer Analysis

### 6.1 Detecting Off-Season
A league is **inactive** if:
- The latest match in the DB is > 30 days old, OR
- A manual toggle in the dashboard is set, OR
- The configured season has no matches in the current year.

When inactive, the dashboard shows:
1. **Previous Season Summary:** Final standings, champion probability retrospect.
2. **Next Season Preview:** Projected standings based on transfer analysis.

### 6.2 Transfermarkt Integration
1. Clone `felipeall/transfermarkt-api` as a **Git submodule** in `transfermarkt-api/`.
2. Run it as a **separate process** (Docker or `python app/main.py` on another port, e.g. `localhost:8001`).
3. Our FastAPI app calls it via HTTP:
   ```
   GET http://localhost:8001/clubs/{club_id}/transfers
   ```
4. Map Transfermarkt club IDs to our team names via a lookup table in the DB.

### 6.3 Transfer Impact Scoring
For each team, calculate a **net squad change score**:

```
for each incoming player:
    impact += market_value_eur × position_weight(position)
for each outgoing player:
    impact -= market_value_eur × position_weight(position)

position_weight = {
    'FWD': 1.2,   # attackers have high goal impact
    'MID': 1.0,
    'DEF': 0.9,
    'GK': 0.8
}
```

Then adjust the team's **projected goals for/against** from the previous season baseline:
- Positive net impact → slightly higher projected GF, lower GA.
- Negative net impact → lower GF, higher GA.

This adjusted baseline feeds into the **pre-season Monte Carlo simulation** using the previous season's schedule template (same teams, reshuffled fixtures or using announced schedule if available).

### 6.4 Transfer UI
- List of all transfers per team with player photo (if available), value, and position badge.
- Team-by-team **squad change radar** showing: attack impact, defense impact, net spend, age profile change.
- Projected table for next season with confidence intervals.

---

## 7. Job Execution Design

FastAPI `BackgroundTasks` is sufficient for personal use. For each job:

1. **Create DB row** in `job_status` with `status='queued'`.
2. **Add to BackgroundTasks** a function that:
   - Updates row to `status='running'`.
   - Updates `progress` periodically (e.g., 10%, 50%, 90%).
   - Runs the actual script logic.
   - On success: stores results JSON, sets `status='done'`.
   - On failure: stores traceback, sets `status='failed'`.
3. **Return job_id** immediately to the frontend.
4. **Frontend polls** every 2 seconds until `done` or `failed`.

**Job types:**
- `update_data`: Downloads CSV, loads to DB.
- `train`: Runs temporal split, trains models, compares, saves best.
- `simulate`: Runs Monte Carlo with current model.
- `analyze_transfers`: Fetches transfermarkt data, computes impact scores.

---

## 8. Error Handling & Resilience

- **Data source fallback:** If `football-data.co.uk` fails, try ESPN/BBC scraper. If both fail, show "Datos no disponibles" but don't crash.
- **Transfermarkt API down:** Show manual fallback message. Allow user to input transfers manually in a form as temporary workaround.
- **Model training failure:** Show log in the job card. Allow re-try.
- **Database locked:** SQLite with `aiosqlite` handles async access.

---

## 9. Testing Strategy

This project currently has **zero tests**. We will add:

- **Unit tests** for config loading, league code mapping, data normalization.
- **Integration tests** for FastAPI endpoints using `TestClient`.
- **End-to-end** manual test: click through dashboard, trigger a job, verify results appear.

No CI/CD initially (personal use), but tests will allow safe refactoring when expanding.

---

## 10. Deployment for Friends

To share with the close circle:

1. **Option: Local network**
   ```bash
   python run_web.py --host 0.0.0.0 --port 8000
   ```
   Friends access via `http://TU_IP:8000` on the same WiFi.

2. **Option: Cloud (future)**
   - Railway / Fly.io: Free tier supports FastAPI + SQLite (with persistent volume).
   - Dockerize the app.
   - Add a simple `SECRET_KEY` + basic HTTP auth for friends.

---

## 11. Migration Plan from Current Repo

1. **Phase 1: Abstraction**
   - Create `config/leagues.yaml`.
   - Refactor `scripts/data_manager.py` to accept `league_code`.
   - Refactor `scripts/preprocess.py` to accept `league_code`.
   - Rename `models/predictor.py` to generic `LeaguePredictor`.
   - Verify La Liga still works end-to-end.

2. **Phase 2: Database**
   - Create `app/database.py` with schema.
   - Write migration script that imports existing `data/processed/*.csv` into SQLite.

3. **Phase 3: Web Layer**
   - Build FastAPI app, Jinja2 templates, HTMX interactions.
   - Implement job system with BackgroundTasks.
   - Connect to abstracted pipeline.

4. **Phase 4: Multi-League**
   - Download data for E0, D1, I1, F1.
   - Verify preprocessing works for each.
   - Train models for each league.
   - Add league selector UI.

5. **Phase 5: Transfer Analysis**
   - Add transfermarkt-api as submodule.
   - Build transfer mapping and impact scoring.
   - Implement off-season detection and next-season preview.

6. **Phase 6: Polish**
   - Add tests.
   - Add visualizations (Chart.js for Monte Carlo, standings evolution).
   - Docker support.

---

## 12. Spec Self-Review Checklist

- [x] **Placeholder scan:** No TBD or TODO sections.
- [x] **Internal consistency:** BackgroundTasks used consistently; database schema supports all described features.
- [x] **Scope check:** This is focused on a single deployable unit (FastAPI app). Transfermarkt API is an external dependency, not part of the core build.
- [x] **Ambiguity check:** Job polling defined explicitly (2s). Off-season detection criteria explicit. League codes explicit.

---

**Status:** Ready for implementation planning.
