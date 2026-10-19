# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The system Python is 3.14 (there is no 3.12). As of 2026-10-19 both `.venv/` and `venv/` in the repo
are **broken** — `venv/` was moved from `~/Escritorio/PROYECTOS/` and its shebangs point there, its
packages are for 3.12, and `.venv/` has no `bin/`. Recreate one before running anything:

```bash
rm -rf .venv venv && python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"           # deps declared in pyproject.toml (there is no requirements.txt)

python run_web.py                 # FastAPI app on http://127.0.0.1:8000 (uvicorn --reload)
python -m app.seed                # init DB schema + insert La Liga from config/leagues.yaml
```

`manage.py` is a thin CLI over the same `app/services/*` the web app calls — no logic of its own:

```bash
python manage.py update
python manage.py train
python manage.py simulate --sims 500 --seed 42
python manage.py project
python manage.py evaluate     # prints the last saved evaluation, no retraining
```

### Tests

```bash
pytest                                          # the whole suite (slow tests deselected)
pytest tests/test_api.py::test_portada_es_la_liga    # single test
pytest -m slow                                  # the networked end-to-end test only
```

Config lives in `pyproject.toml` under `[tool.pytest.ini_options]`: `testpaths = ["tests"]` keeps
pytest out of the virtualenvs, and `asyncio_mode = "auto"` means async tests
need no `@pytest.mark.asyncio` marker.

**Autouse** fixtures in [tests/conftest.py](tests/conftest.py) make it impossible for a test to
touch real state — keep it that way when adding tests:

- `temporada_de_fixtures` freezes `current_season` to `"2526"` (the season of the committed CSVs),
  so `config/leagues.yaml` can follow the real season (`2627` today) without breaking every test.

- `temp_db` redirects `DB_PATH` to a tmp file and runs `init_db` + `seed_leagues`. It patches the
  binding in `app.database`, `app.routers.jobs` and `app.seed`, because those do
  `from app.database import DB_PATH` (import-by-value; patching only the origin is not enough).
- `data_root` redirects `scripts.data_manager_generic.DATA_DIR` to a tmp dir seeded with
  `tests/fixtures/SP1_2526.csv` (a frozen full season). Every module resolves CSV paths through
  `league_csv_path()`, so patching that one constant covers all of them.
- `mock_download` fakes `requests.get` and routes by the season in the URL to the matching
  `tests/fixtures/SP1_{season}.csv` (5 seasons committed, `2122`..`2526`) — training now spans multiple
  seasons, so a fixture that ignored the season and always returned `2526` made every "historical"
  season an identical copy and the model scored a meaningless 100 %. Calendar URLs (openfootball)
  get a 404, so `update_league_data` carries on with results only. Tests that need pending
  fixtures use the `media_temporada` fixture in [tests/test_prediccion.py](tests/test_prediccion.py)
  (25/26 cut to 190 played + a 380-match calendar).

`tests/test_integration.py` is marked `slow`: it hits the network and trains a real model (~1–3 min).

## Architecture

Three layers:

- **`core/`** — shared library, imported by everything. `dataset.py` is the *only* place that reads and
  normalizes football-data.co.uk CSVs. `state.py` holds `LeagueState`, the *only* implementation of team
  state, the 35 feature formulas and the standings. `features.py` owns `FEATURE_COLS` and turns state
  into a training matrix. `evaluation.py` has the season-based train/test split and the three baselines.
  `goals.py` samples Poisson scorelines conditioned on the model's outcome. `errors.py` holds the domain
  exceptions whose Spanish messages reach the UI directly (`str(e)` lands in `job_status.log`).
- **`app/`** — FastAPI app for La Liga only: routers → services → `core`.
- **`models/` + `scripts/`** — the ML core (`LeaguePredictor` in `models/predictor.py`;
  `scripts/season_simulator.py` orchestrates `SeasonSimulator`, `scripts/preprocess_generic.py` and
  `scripts/data_manager_generic.py` are thin wrappers around `core/`).
- **`manage.py`** — CLI over `app/services/*`, nothing else. The constraint that matters: it must never
  grow its own standings/feature logic, or the duplication phase 3 removed comes back.

The legacy layer (root `run_*.py`, `simulate_season.py`, `simulator/`, `scripts/data_manager.py`,
`scripts/preprocess.py`, `scripts/data_sources.py`, `load_data.py`, `scraper_laliga.py`, …) is deleted.
It's recoverable from the `pre-cleanup` git tag if something turns out to still be needed.

The prioritized improvement backlog is [docs/ROADMAP.md](docs/ROADMAP.md). The model-reliability study behind
it is reproducible with `python experimentos/fiabilidad_modelo.py` (walk-forward over 4 test seasons;
downloads seasons back to 14/15 into `data/raw/SP1/`).

### League identity

**The project covers La Liga only** (the other four leagues were removed on 2026-10-19 to focus on
developing one league in depth). `config/leagues.yaml` holds just `SP1`, and `app.config.LEAGUE_CODE`
is the constant to use instead of writing `"SP1"`. The **string** `league_code` plumbing is kept on
purpose: it's the FK target (`leagues.code`) in every table, the raw-data directory and the model
filename, and services still take it as a parameter. Don't rip it out, and never introduce an integer
league id.

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

- `POST /api/jobs/?job_type=X` (`league_code` optional, defaults to `LEAGUE_CODE`) validates against
  `JOB_HANDLERS` (400 on unknown type, 404 on unknown league) **before** inserting, then returns the **HTML partial**
  `partials/job_status.html` (not JSON), which HTMX appends to `#job-container` and polls at
  `/api/jobs/{id}/poll`.
- Adding a job type = writing one `async def handler(ctx: JobContext, league_code) -> dict` and
  registering it in `JOB_HANDLERS` in [app/routers/jobs.py](app/routers/jobs.py), plus its Spanish name
  in `JOB_LABELS` and a button in the `acciones` list at the top of `liga.html` (which also drives the
  1–4 keyboard shortcuts and the bottom key bar; extend the regex in `retro.js` if you add a fifth). The returned dict is stored as `result_json`; its
  `mensaje` key is what the user reads.
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

**Two data sources.** football-data.co.uk's CSV only contains *played* matches, so on its own an
in-progress season looks finished (pending = 0). The fixtures come from openfootball's calendar
(`scripts/data_manager_generic.py::download_calendar`, saved as `data/raw/SP1/{season}_calendario.csv`
with team names already translated by [core/equipos.py](core/equipos.py) — add a newly promoted club
there if the download fails with `EquipoDesconocidoError`). `core/dataset.py::load_season` merges it
by **(home, away) pair, not date** (postponements change the date; each ordered pair plays once per
season) and fills `Matchday`. Ingestion deletes and re-inserts `scheduled` rows on every update so a
rescheduled match doesn't leave a ghost row. If the calendar can't be downloaded, the update still
succeeds and says so in its message.

The home page (`/`, [app/routers/liga.py](app/routers/liga.py); `/league/SP1` 301-redirects there) does
**not** compute standings; it renders the most recent `simulations` row **of the current season**
(`WHERE league_code = ? AND season = ?` — `simulations.season` is the season the run started from;
rows without it are hidden). `label` distinguishes a current-season Monte Carlo from a next-season
projection. So "the table didn't change" usually means no new `simulations` row was written.

Season state (`app/services/season_state.py`) is derived from `matches` **of the current season**
(`matches` keeps every season ever ingested), never configured:
`SIN_DATOS | PRETEMPORADA | EN_CURSO | FINALIZADA`. Simulating is only meaningful when matches remain.

**Match predictions** ([app/services/prediction_service.py](app/services/prediction_service.py), served
as HTMX partials by [app/routers/prediccion.py](app/routers/prediccion.py): `/prediccion/jornada?n=` and
`/prediccion/partido?local=&visitante=&fecha=`). The model gives 1X2; scorelines come from
`core/goals.py::scoreline_distribution`, the analytic twin of `sample_scoreline` (Poisson rescaled so
each outcome zone sums to the model's probability — the most likely score never contradicts the
pick). The (state, predictor) context is cached keyed on the mtimes of the `.pkl`, the season CSV and
the calendar. "Next matchday" = first matchday with pending matches whose **median** date hasn't
passed — using the max date let one postponed match turn an old matchday into "next". Expected
errors come back as the partial with a message and status 200, because HTMX doesn't swap 4xx/5xx.

### ML core

`LeaguePredictor` ([models/predictor.py](models/predictor.py)) is the only predictor class (there is no
`LaLigaPredictor` alias any more).

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
`SeasonSimulator` (incremental, for simulation). Never add a feature formula anywhere else — that duplication is exactly
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

**The state used to predict must be built like the one used to train.**
`scripts/season_simulator.py::build_current_state` replays the previous training seasons with
`core.features.build_state` (same season-change semantics as `build_training_matrix`, which share the
`_recorrer` loop) and then the current season's played matches, then `keep_only()`s this season's
teams. Building it from the current season alone silently zeroed the H2H features (0.0 previous
meetings vs 2.86 in training). Both the simulator and the prediction service use it.

`LeaguePredictor.load_model()` sets `n_jobs=1`: predicting one row at a time with `n_jobs=-1` is
3.6× slower (thread overhead) and pins every core.

Two independent simulators exist:
- `simulate_season()` in [scripts/season_simulator.py](scripts/season_simulator.py) — in-season: loads
  the predictor (raising `ModelNotTrainedError` if absent) and simulates the remaining fixtures.
  Points, GF and GA in the returned table are **Monte Carlo means**, from the same run as the
  probabilities (they used to come from one separate random simulation). Raises
  `SeasonFinishedError` when nothing is pending; without that guard it replays the real season and
  reports a 100 % champion, which looks like a model output but carries no information. The model
  decides the outcome (H/D/A); goals are then sampled from `core/goals.py::sample_scoreline` — a
  rejection-sampled Poisson using `LeagueState.expected_goals()` (blends each team's own
  attack/defense average, falling back to a neutral `GOLES_LIGA_NEUTRO` below 3 matches of history) —
  not three fixed scorelines, which used to make goal difference always ±1.
- `project_next_season()` in [app/services/next_season_service.py](app/services/next_season_service.py)
  — off-season: does not use the ML model at all. Attack/defense ratings from the finished season,
  relegated teams swapped for the configured promoted ones, its own Poisson
  round-robin Monte Carlo. Deliberately **not** unified with `core/goals.py`: there's no ML outcome to
  condition on here, so goals are the primary simulated variable instead of something sampled after the
  fact — different problem, not duplicated code.

Both take a `seed` and thread an explicit `np.random.Generator`; results must be reproducible.

### Frontend

Jinja2 + HTMX (pinned with SRI) + one hand-written stylesheet — no Tailwind, no icon library, no build
step, no npm. Anything swapped by HTMX goes in `app/templates/partials/`.

The look is **retro MS-DOS/VGA**: [app/static/css/retro.css](app/static/css/retro.css) is all the CSS
(the 16 EGA colours as custom properties, double-bordered "windows" with hard shadows, CRT scanlines
toggleable from the menu bar), and [app/static/js/retro.js](app/static/js/retro.js) handles the 1–4
shortcuts, the clock and the CRT toggle. Fonts are Google Fonts' "Press Start 2P" and "VT323".
**Press Start 2P squashes accented capitals (Ó, Í…) into its 8 px cell so they read as lowercase** —
only use it for text that can't carry an uppercase accent (the big title, button names, numbers);
window titles and table headers use VT323 for that reason.

Import the shared `templates` from [app/templating.py](app/templating.py) — one Jinja env, path resolved
from `__file__`, with the league config (`liga`), `league_code` and `temporada_actual` exposed as
globals, `temporadas` and `equipo` (display name, [core/equipos.py](core/equipos.py)) filters, and
`version_estaticos` for cache-busting `?v=` on the CSS/JS URLs. Do not instantiate `Jinja2Templates` in a router.

## Conventions

- User-facing strings, docstrings and comments are in **Spanish**; identifiers and DB columns in English. Match the surrounding file.
- Team names come from football-data.co.uk (e.g. `"Ath Madrid"`) and are used as-is.
- The transfer analysis (Transfermarkt API submodule, `transfers` table, `analyze_transfers` job) was
  removed on 2026-10-19 at the user's request. Don't reintroduce it without asking.
- Nothing under `data/` is tracked any more: the SQLite DB, `data/raw/**/*.csv` and `data/processed/`
  are all gitignored and regenerated locally. The single exception is `tests/fixtures/*.csv`, which is
  negated in `.gitignore` on purpose.
