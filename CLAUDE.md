# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The virtualenv lives in `.venv` (Python 3.14).

```bash
source .venv/bin/activate
pip install -e ".[dev]"           # deps declared in pyproject.toml (there is no requirements.txt)

python run_web.py                 # FastAPI dashboard on http://127.0.0.1:8000 (uvicorn --reload)
python -m app.seed                # init DB schema + insert the 5 leagues from config/leagues.yaml
```

`manage.py` is a thin CLI over the same `app/services/*` the web app calls — no logic of its own:

```bash
python manage.py update   --league SP1
python manage.py train    --league SP1
python manage.py simulate --league SP1 --sims 500 --seed 42
python manage.py project  --league SP1
python manage.py evaluate --league SP1     # prints the last saved evaluation, no retraining
```

### Tests

```bash
pytest                                          # the whole suite (slow tests deselected)
pytest tests/test_api.py::test_league_detail    # single test
pytest -m slow                                  # the networked end-to-end test only
```

Config lives in `pyproject.toml` under `[tool.pytest.ini_options]`: `testpaths = ["tests"]` keeps
pytest away from the `transfermarkt-api/` submodule, and `asyncio_mode = "auto"` means async tests
need no `@pytest.mark.asyncio` marker.

Two **autouse** fixtures in [tests/conftest.py](tests/conftest.py) make it impossible for a test to
touch real state — keep it that way when adding tests:

- `temp_db` redirects `DB_PATH` to a tmp file and runs `init_db` + `seed_leagues`. It patches the
  binding in `app.database`, `app.routers.jobs` and `app.seed`, because those do
  `from app.database import DB_PATH` (import-by-value; patching only the origin is not enough).
- `data_root` redirects `scripts.data_manager_generic.DATA_DIR` to a tmp dir seeded with
  `tests/fixtures/SP1_2526.csv` (a frozen full season). Every module resolves CSV paths through
  `league_csv_path()`, so patching that one constant covers all of them.
- `mock_download` fakes `requests.get` and routes by the season in the URL to the matching
  `tests/fixtures/SP1_{season}.csv` (5 seasons committed, `2122`..`2526`) — training now spans multiple
  seasons, so a fixture that ignored the season and always returned `2526` made every "historical"
  season an identical copy and the model scored a meaningless 100 %.

`tests/test_integration.py` is marked `slow`: it hits the network and trains a real model (~1–3 min).

### Transfermarkt API (optional, for transfer analysis)

`transfermarkt-api/` is a git submodule (`git submodule update --init`). `app/services/transfer_service.py` calls it at `http://localhost:8001`, but the submodule's own README defaults to port 8000, which collides with `run_web.py` — start it explicitly on 8001:

```bash
cd transfermarkt-api && PYTHONPATH=$PWD uvicorn app.main:app --port 8001
```

The `analyze_transfers` job fails cleanly if it isn't running; every other feature works without it.

## Architecture

Three layers:

- **`core/`** — shared library, imported by everything. `dataset.py` is the *only* place that reads and
  normalizes football-data.co.uk CSVs. `state.py` holds `LeagueState`, the *only* implementation of team
  state, the 35 feature formulas and the standings. `features.py` owns `FEATURE_COLS` and turns state
  into a training matrix. `evaluation.py` has the season-based train/test split and the three baselines.
  `goals.py` samples Poisson scorelines conditioned on the model's outcome. `errors.py` holds the domain
  exceptions whose Spanish messages reach the UI directly (`str(e)` lands in `job_status.log`).
- **`app/`** — FastAPI dashboard over 5 leagues: routers → services → `core`.
- **`models/` + `scripts/`** — the ML core (`LeaguePredictor` in `models/predictor.py`;
  `scripts/season_simulator.py` orchestrates `SeasonSimulator`, `scripts/preprocess_generic.py` and
  `scripts/data_manager_generic.py` are thin wrappers around `core/`).
- **`manage.py`** — CLI over `app/services/*`, nothing else. The constraint that matters: it must never
  grow its own standings/feature logic, or the duplication phase 3 removed comes back.

The legacy La Liga-only layer (root `run_*.py`, `scripts/data_manager.py`, `load_data.py`,
`scraper_laliga.py`, `fetch_updated_data.py`, `scripts/preprocess.py`, `scripts/model_evaluation.py`,
`LaLigaPredictor`) is deleted — phase 6 of the plan below. It's recoverable from the `pre-cleanup` git
tag if something turns out to still be needed.

An audit and a 6-phase remediation plan live at
`~/.claude/plans/buenas-quiero-que-hagas-wobbly-flame.md`. All 6 phases are done.
Design docs for the web layer: [docs/superpowers/specs/](docs/superpowers/specs/) and [docs/superpowers/plans/](docs/superpowers/plans/).

### League identity

`config/leagues.yaml` is the single source of truth. League codes are football-data.co.uk codes: `SP1`, `E0`, `D1`, `I1`, `F1`. They flow through everything as a **string** `league_code` — it is the FK target (`leagues.code`) in every table, the raw-data directory, and the model filename. Never introduce an integer league id.

Path conventions derived from the code:
- Raw data: `data/raw/{league_code}/{season}.csv`, season as `"2526"` for 2025-26.
- Models: `models/{league_code}_predictor.pkl`.
- DB: `data/db.sqlite` (relative path — the server and scripts must be run from the repo root).

`config/leagues.yaml`'s top-level `defaults:` block (`current_season`, `history_depth`) is the single
source of truth for "what season is it" — read through `app/config.py::get_current_season()` /
`get_training_seasons()`. Season codes are mechanically derivable (`"2526" → "2425"` via
`previous_season()`), so there's no per-league list to maintain. `EXCLUDED_SEASONS = {"2021"}` in
`app/config.py` drops 2020-21 from training regardless of `history_depth`: empty stadiums that season
distorted home advantage, one of the 35 features.

`next_season.promoted_teams` in the YAML ships with placeholder names (`"Ascendido 1"`, …) that must be replaced with real promoted clubs; their count also determines how many teams get relegated in the projection.

### Web request flow

Browser (HTMX) → FastAPI router → BackgroundTask → service → existing ML code → SQLite → HTMX polls a partial.

- `POST /api/jobs/?job_type=X&league_code=Y` validates against `JOB_HANDLERS` (400 on unknown type,
  404 on unknown league) **before** inserting, then returns the **HTML partial**
  `partials/job_status.html` (not JSON), which HTMX appends to `#job-container` and polls at
  `/api/jobs/{id}/poll`.
- Adding a job type = writing one `async def handler(ctx: JobContext, league_code) -> dict` and
  registering it in `JOB_HANDLERS` in [app/routers/jobs.py](app/routers/jobs.py). The returned dict is
  stored as `result_json`; its `mensaje` key is what the user reads.
- **Each background task opens its own `aiosqlite.connect(DB_PATH)`.** The request-scoped
  `get_db_dependency` connection is closed when the response is sent, so it cannot be passed into a task.
- **CPU work goes in `asyncio.to_thread` inside the service**, not in the router. These tasks run on
  the event loop; a synchronous XGBoost fit or Monte Carlo there freezes the whole server, including
  the HTMX poll that is supposed to show progress.
- **All jobs are serialized through `_job_lock` (`asyncio.Lock`).** Nothing stops the browser from
  firing all five buttons at once, and each `train` claims every core via XGBoost's `n_jobs=-1` — five
  at a time meant five processes fighting for CPU plus five concurrent SQLite writers. The lock makes
  `queued` mean it, in FIFO order (`asyncio.Lock` serves waiters in acquire order). In tests, the lock
  gets reset per test by an autouse fixture in [tests/conftest.py](tests/conftest.py) — `asyncio.Lock`
  binds to whichever event loop first calls `acquire()`, and pytest-asyncio hands each test function its
  own loop, so a bare module-level lock would raise `bound to a different event loop` on the second test.
  Not an issue in production: uvicorn owns one loop for the process's whole life.
- Long-running handlers get incremental progress via `ctx.progress_threadsafe(done, total)`, called from
  inside the `asyncio.to_thread` worker (see `SeasonSimulator.monte_carlo_simulation`'s `on_progress`
  callback) — it schedules the DB write back onto the event loop with `run_coroutine_threadsafe`.
- `_update_job` only writes the columns you pass. Progress messages go in `message`, errors in `log`.
- Failures land in `job_status.status='failed'` with `TipoDeError: mensaje` in `log`; check that column
  when a job "does nothing".

### Persistence

Schema is a single `SCHEMA` string in [app/database.py](app/database.py), executed via `executescript`
on every startup (`lifespan` → `init_db`). There is no migration tool: add new columns to `SCHEMA`
**and** to the `MIGRATIONS` list (each statement is tried and its failure ignored, meaning "already
exists"). `INDEXES` is applied the same way; the `UNIQUE` ones are what make ingestion idempotent.

`app/services/data_service.py::update_league_data` is the single ingestion point — it is what
populates `teams` and `matches`, always re-downloading so the button actually refreshes data.
Everything downstream (season state, match counts, prediction linkage) depends on it having run.

The league page does **not** compute standings; it renders the most recent `simulations` row **for that
league** (`WHERE league_code = ?` — the column is required, otherwise projections leak across leagues).
`label` distinguishes a current-season Monte Carlo from a next-season projection. So "the table didn't
change" usually means no new `simulations` row was written.

Season state (`app/services/season_state.py`) is derived from `matches`, never configured:
`SIN_DATOS | PRETEMPORADA | EN_CURSO | FINALIZADA`. Simulating is only meaningful when matches remain.

### ML core

`LeaguePredictor` ([models/predictor.py](models/predictor.py)) is the real class; `LaLigaPredictor` is an alias at the bottom of the file kept for the legacy scripts.

Match result target is `{-1: away, 0: draw, 1: home}` everywhere in feature/label code.
`LeaguePredictor._train()` encodes it to `{0,1,2}` for **every** model (not just XGBoost/LightGBM —
Random Forest and Gradient Boosting used to train on the raw `{-1,0,1}` labels, which is what made
`evaluate()` mislabel their classification reports). `predict_match()` and `_probs_by_outcome()` always
index by `model.classes_` and translate back through `_inverse_target_map`; there is no
`model_name in (...)` branching left anywhere in the predictor.

**Training spans multiple seasons and reported accuracy must always be out-of-sample.**
`app/services/ml_service.py` trains on `app.config.get_training_seasons()` (5 prior seasons + the
current one by default) and holds out the **current season entirely** as test
(`core/evaluation.py::season_train_test_split`) — there's deliberately no separate validation season,
since the pipeline trains a single fixed-hyperparameter XGBoost with no model selection to consume one.
Alongside `accuracy`, `model_runs` stores three baselines (`baseline_home` = always predict home,
`baseline_majority` = the training set's majority class, `baseline_bookmaker` = following Bet365's
favourite via the `Odds_Home/Draw/Away` columns, `None` if a CSV lacks them) plus `logloss`/`brier` —
accuracy alone hides a model that never predicts draws. Runs with `baseline_home IS NULL` predate this
and are hidden from the UI, because their accuracy was measured on the training set.
`tests/test_ml_service.py` and `tests/test_evaluation.py` cover this; the majority-class baseline is
computed from **train**, not test — `scripts/model_evaluation.py`'s old walk-forward baseline used the
test set's own mode, an oracle baseline that always looks better than it is.

**`core.state.LeagueState` is the single implementation** of team state, the 35 features and the
standings. Both consumers drive the same object: `build_training_matrix` (batch, for training) and
`SeasonSimulator` (incremental, for simulation). `scripts/preprocess.py` is now a ~25-line adapter kept
only for the legacy scripts. Never add a feature formula anywhere else — that duplication is exactly
what let the model be trained on one vector and predicted with another.

Invariants when touching it:
- `features()` may only read past matches. Anything derived from the current match is leakage.
- Sort with `kind="mergesort"`. The default quicksort is unstable, so same-day matches got reordered
  between runs and ~2 % of feature rows were non-deterministic.
- `tests/test_feature_parity.py` guards all of this against a frozen matrix in `tests/fixtures/`.
  If it fails, you changed the features — regenerate the snapshot only if that was intentional.
- Feed the model `feature_vector()` (ndarray), not `feature_row()` (DataFrame): the pandas→XGBoost
  conversion of a single row costs ~14x the prediction itself, and the Monte Carlo does one per match
  per simulation.
- `build_training_matrix(df, reset_per_season=True)` calls `state.start_new_season()` whenever the
  `Season` column changes: accumulators and `last_match_date` reset (a team doesn't carry last season's
  points, and days-of-rest shouldn't span the summer break), but `h2h_records` is **kept** — it's the
  only thing that makes the three H2H features informative instead of a constant, since within a single
  season each ordered `(home, away)` pair only meets once.

Two independent simulators exist:
- `simulate_season()` in [scripts/season_simulator.py](scripts/season_simulator.py) — in-season: loads
  the predictor (raising `ModelNotTrainedError` if absent) and simulates the remaining fixtures. Raises
  `SeasonFinishedError` when nothing is pending; without that guard it replays the real season and
  reports a 100 % champion, which looks like a model output but carries no information. The model
  decides the outcome (H/D/A); goals are then sampled from `core/goals.py::sample_scoreline` — a
  rejection-sampled Poisson using `LeagueState.expected_goals()` (blends each team's own
  attack/defense average, falling back to a neutral `GOLES_LIGA_NEUTRO` below 3 matches of history) —
  not three fixed scorelines, which used to make goal difference always ±1.
- `project_next_season()` in [app/services/next_season_service.py](app/services/next_season_service.py)
  — off-season: does not use the ML model at all. Attack/defense ratings from the finished season,
  relegated teams swapped for the configured promoted ones, capped transfer adjustment, its own Poisson
  round-robin Monte Carlo. Deliberately **not** unified with `core/goals.py`: there's no ML outcome to
  condition on here, so goals are the primary simulated variable instead of something sampled after the
  fact — different problem, not duplicated code.

Both take a `seed` and thread an explicit `np.random.Generator`; results must be reproducible.

### Frontend

Jinja2 + HTMX + Tailwind/Lucide via CDN — no build step, no bundler, no npm. Anything swapped by HTMX
goes in `app/templates/partials/`.

Import the shared `templates` from [app/templating.py](app/templating.py) — one Jinja env, path resolved
from `__file__`, with `ligas` exposed as a global so the nav comes from `config/leagues.yaml` instead of
a hardcoded list. Do not instantiate `Jinja2Templates` in a router.

## Conventions

- User-facing strings, docstrings and comments are in **Spanish**; identifiers and DB columns in English. Match the surrounding file.
- Team names come from football-data.co.uk (e.g. `"Ath Madrid"`); mapping them to Transfermarkt is handled by `_normalize()` + the `TEAM_ALIASES` table in [app/services/transfer_service.py](app/services/transfer_service.py) — extend that table rather than renaming data.
- Nothing under `data/` is tracked any more: the SQLite DB, `data/raw/**/*.csv` and `data/processed/`
  are all gitignored and regenerated locally. The single exception is `tests/fixtures/*.csv`, which is
  negated in `.gitignore` on purpose.
