# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The system Python is 3.14 (there is no 3.12). Use `.venv/` (recreated on 2026-10-20; the old `venv/`
was moved from another folder and is gone). If it breaks again, recreate it:

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

- `temporada_de_fixtures` freezes `current_season` to `"2526"` (the season of the committed CSVs)
  and `first_season` to `"2122"` (the oldest one), so `config/leagues.yaml` can follow the real
  season (`2627` today, training from `1415`) without breaking every test.

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

- `understat_offline` (autouse) makes `scripts.understat._get` raise `ConnectionError`, so
  `update_league_data` never reaches Understat in a test; `mock_understat` serves
  `tests/fixtures/understat/` instead (25/26 cut to Betis's last 6 matches, real lineups). It patches
  `_get`, not `requests.get`: `requests` is one shared module, and patching it clobbered `mock_download`.

`tests/test_integration.py` is marked `slow`: it hits the network and trains a real model (~1–3 min).

## Architecture

Three layers:

- **`core/`** — shared library, imported by everything. `dataset.py` is the *only* place that reads and
  normalizes football-data.co.uk CSVs. `state.py` holds `LeagueState`, the *only* implementation of team
  state, the 13 feature formulas (Elo + continuous form) and the standings. `features.py` owns `FEATURE_COLS` and turns state
  into a training matrix. `evaluation.py` has the walk-forward splits, the report and the baselines.
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
downloads seasons back to 14/15 into `data/raw/SP1/`). `python experimentos/calibracion.py` is the
calibration study (2026-10-20): the model is already well calibrated and neither temperature nor
Dirichlet calibration lowers the walk-forward log-loss, so **no calibration layer** — don't add one
without re-running it. `python experimentos/elo_ascendidos.py` (2026-10-20) tested seeding promoted
teams' Elo from Segunda (`SP2`): points per game and a shared two-division Elo both did no better
than the fixed 1430, so `ELO_ASCENDIDO` stays as is. `python experimentos/xg_understat.py` (2026-10-20): Understat xG
exists for every match since 14/15 and joins cleanly, but an EWMA-xG feature gave −0.002 log-loss
over 8 seasons (CI includes 0), nothing in validation and worse in 26/27, and Understat's xG scale
drifts (0.98 → 1.12 xG per goal), so **xG is not a model feature**. If revisited, use Understat
only — football-data's 26/27 `HxG`/`AxG` is a different xG model.

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

`config/leagues.yaml`'s top-level `defaults:` block (`current_season`, `first_season`) is the single
source of truth for "what season is it" — read through `app/config.py::get_current_season()` /
`get_training_seasons()` (every season from `first_season`, `"1415"`, to the current one: more
history measurably helps). Season codes are mechanically derivable (`"2526" → "2425"` via
`previous_season()`), so there's no per-league list to maintain. `EXCLUDED_SEASONS = {"2021"}` in
`app/config.py`: 2020-21 (empty stadiums distorted home advantage) is never a training or test
example, but its matches **do** feed the state, so Elo and form don't jump over a season.
`SP1.relegated` (3) sets the relegation places for the in-season Monte Carlo and the table colours.

`next_season.promoted_teams` in the YAML ships with placeholder names (`"Ascendido 1"`, …) that must be replaced with real promoted clubs; their count also determines how many teams get relegated in the projection. The projection **refuses** to run while they are placeholders (`PromotedTeamsNotConfiguredError`) — it used to show "Ascendido 2" rows in the table.

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
- On startup, `lifespan` calls `app.database.mark_interrupted_jobs()`: anything still `queued`/`running`
  was lost with the old process (BackgroundTasks don't survive a restart) and is marked `failed`.

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

**Third source: Understat** (player data and lineups, for the team page).
[scripts/understat.py](scripts/understat.py) is the only module that knows it (unofficial JSON
endpoints, no API key): `download_understat` runs inside `update_league_data` (in a thread, with
progress) and caches `data/raw/SP1/understat/liga_{year}.json` plus one `partidos/{id}.json` per
played match, downloaded once; `load_lineups` reads that cache into one row per player and match.
Team names go through `UNDERSTAT` in [core/equipos.py](core/equipos.py) — add a promoted club there
too. If Understat fails, the update still succeeds with an `aviso_jugadores`. Swapping the source
(e.g. a paid API before publishing) means reimplementing those two functions only.

The home page (`/`, [app/routers/liga.py](app/routers/liga.py); `/league/SP1` 301-redirects there) shows
two tables. **"Clasificación"** is the real one, computed on each request by
[app/services/standings_service.py](app/services/standings_service.py) from the season CSV through
`LeagueState.standings()` (tiebreak: GD then GF — La Liga uses head-to-head first, the page says so).
**"Proyección final"** renders the most recent `simulations` row **of the current season** (while
matches are pending, only `label = 'Temporada actual'` rows: a next-season projection saved mid-season
by an older version is ignored)
(`WHERE league_code = ? AND season = ?` — `simulations.season` is the season the run started from;
rows without it are hidden). `label` distinguishes a current-season Monte Carlo from a next-season
projection. So "the table didn't change" usually means no new `simulations` row was written.

**Accuracy history** ("Historial de aciertos", `/prediccion/historial`, lazy-loaded,
[app/services/history_service.py](app/services/history_service.py)): every played match of the
current season, predicted by `models/{league}_predictor_historial.pkl` — a model trained **only on
earlier seasons** (`ml_service` saves the current season's walk-forward fold model there) — from the
state just before the match. It's strictly out-of-sample and picks up new matchdays after "Actualizar
datos" without retraining. Per matchday: model vs market hits and log-loss; the market is only the
reference (see Conventions). Cached on the mtimes of that `.pkl`, the season CSV and the calendar.

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

**Team page** (`/equipo/{nombre}`, football-data name URL-encoded, [app/routers/equipo.py](app/routers/equipo.py)
→ [app/services/team_service.py](app/services/team_service.py)): header (position from `LeagueState`),
last-5 form (only the V/E/D chips — no points-per-game line, the user read it as betting odds),
strengths/weaknesses (15 per-match measures vs the season's teams, z-scored; only |z| ≥ `Z_MINIMO`
(0.75), at most one per `familia` (total/home/away goals say the same thing), nothing before
`PARTIDOS_MINIMOS` (5) matches; xG ones only when the CSV has `HxG`/`AxG`, i.e. from 26/27), likely XI and top players (from Understat lineups, summed per
team so mid-season transfers only count for their current club), and honours from
[config/palmares.yaml](config/palmares.yaml) — hand-edited, RFEF/UEFA-recognised titles only, rule at
the top of the file. It **never calls Understat**: it reads the cache. No odds anywhere on it.

Team **crests** are our own design, not the clubs' trademarked ones (the project may be published):
`ESCUDOS` in [core/equipos.py](core/equipos.py) (pattern + EGA colours + initials, add a promoted club
there) drawn by [app/escudos.py](app/escudos.py) as inline 16×16 SVG — inline so the initials can use
the page's pixel font. Unknown teams get a grey crest with initials. Jinja globals: `escudo(nombre,
tam)`, `tiene_ficha(nombre)` (false only for the "Ascendido N" placeholders).

### ML core

`LeaguePredictor` ([models/predictor.py](models/predictor.py)) is the only predictor class (there is no
`LaLigaPredictor` alias any more).

Match result target is `{-1: away, 0: draw, 1: home}` everywhere in feature/label code.
`LeaguePredictor._train()` encodes it to `{0,1,2}` for **every** model (not just XGBoost/LightGBM —
Random Forest and Gradient Boosting used to train on the raw `{-1,0,1}` labels, which is what made
`evaluate()` mislabel their classification reports). `predict_match()` and `_probs_by_outcome()` always
index by `model.classes_` and translate back through `_inverse_target_map`; there is no
`model_name in (...)` branching left anywhere in the predictor.

**Reported metrics are walk-forward and always out-of-sample.** `app/services/ml_service.py` builds
the matrix over `get_training_seasons()` and, for each of the last `TEMPORADAS_DE_TEST` (4) seasons
with matches, trains on the seasons before it only and predicts it
(`core/evaluation.py::walk_forward_splits` → `evaluate_walk_forward`). The metrics aggregate all those
predictions (~1,200–1,500 matches); testing on the current season alone gave a number from ~80
matches. The **saved** model is then trained on everything played, current season included. There's
no separate validation season: one fixed-hyperparameter XGBoost, no model selection to consume it.
`model_runs` stores `accuracy`, `logloss`, `brier` and the references, all computed per fold with
what was known then: `baseline_home`, `baseline_majority` (the fold's **train** majority — using the
test set's own mode is an oracle), `baseline_bookmaker` (Bet365's favourite), `logloss_bookmaker`
(Bet365's margin-free probabilities) and `logloss_frequencies` (training outcome frequencies = knowing
nothing). Log-loss is the headline: the Monte Carlo and the matchday use the probabilities, not just
the pick. Runs with `baseline_home IS NULL` predate honest metrics and are hidden from the UI.

The XGBoost is **regularized** (depth 2, 400 trees at 0.02, subsampling, L2): the old depth-4 one
overfit. The `.pkl` stores `feature_names`; `load_model()` raises `ModelNotTrainedError` ("retrain")
if they differ from `FEATURE_COLS`, instead of an XGBoost shape error mid-prediction.

**`core.state.LeagueState` is the single implementation** of team state, the features and the
standings. The 13 features are an **Elo** (K 20, home edge 60, goal-difference multiplier, regressed a
third towards the mean at each season change, promoted teams start at 1430) and a **continuous form**
(EWMA α 0.12 of goals and shots on target for/against, never reset), plus diffs. They replaced 35
season accumulators that restarted at zero every summer, which left the model knowing nothing until
matchday 6 (docs/ROADMAP.md §2 has the study). Both consumers drive the same object: `build_training_matrix` (batch, for training) and
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
  `Season` column changes: the table accumulators reset (they feed standings, `expected_goals` and the
  team page, not the model) and Elo regresses; the EWMA form and `h2h_records` (shown as "precedentes",
  not a feature) are kept.
- `apply_match` takes optional shots on target. A simulated match only has goals, so the shots form
  doesn't move during a Monte Carlo — by design, not a bug.

**The state used to predict must be built like the one used to train.**
`scripts/season_simulator.py::build_current_state` replays the previous training seasons with
`core.features.build_state` (same season-change semantics as `build_training_matrix`, which share the
`_recorrer` loop) and then the current season's played matches, then `keep_only()`s this season's
teams. Building it from the current season alone would start every Elo at 1500 and every form at the
default. Both the simulator and the prediction service use it.

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
  rejection-sampled Poisson using `LeagueState.expected_goals()` (average of one team's EWMA
  attack and the other's EWMA defence — the same never-reset form the model uses — times
  `FACTOR_GOLES_LOCAL`/`_VISITANTE` for home advantage; it beat the season averages it replaced on
  real 22/23–26/27 goals, most clearly in matchdays 1–6) —
  not three fixed scorelines, which used to make goal difference always ±1.
- `project_next_season()` in [app/services/next_season_service.py](app/services/next_season_service.py)
  — off-season: does not use the ML model at all. Reads the finished season through
  `core.dataset.load_season`, like everything else, and raises `SeasonInProgressError` if it has
  pending matches (button 4 is disabled via `ResumenTemporada.puede_proyectar`). Run mid-season, it
  used to treat the played matches as the whole season and relegate whoever was bottom that day. Attack/defense ratings from the finished season,
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
- **Betting odds are never model features** (user decision, 2026-10-20): the model must stay its own,
  not a copy of the market. Odds are only an aggregate reference in the metrics (`baseline_bookmaker`,
  `logloss_bookmaker`, the model card and the accuracy history). Don't add them as inputs, and don't
  show per-match odds next to predictions (also rejected, same date), without asking.
- The transfer analysis (Transfermarkt API submodule, `transfers` table, `analyze_transfers` job) was
  removed on 2026-10-19 at the user's request. Don't reintroduce it without asking.
- Nothing under `data/` is tracked any more: the SQLite DB, `data/raw/**/*.csv` and `data/processed/`
  are all gitignored and regenerated locally. The single exception is `tests/fixtures/*.csv`, which is
  negated in `.gitignore` on purpose.
