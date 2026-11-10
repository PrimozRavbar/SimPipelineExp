# SimPipeline: notes for Claude and other agents

SimPipeline simulates a production recommendation system (World → Kafka → Flink → Redis →
two-tower retrieval → ModelServer → World) plus an offline retraining loop
(DataLake → Spark → feature store → training → HNSW). The goal is a game where players
build, break, diagnose and fix ML systems.

## Setup and commands

```bash
pip install -e ".[dev]"          # torch, pandas, numpy, pytest, jupyter
python -m pytest -q              # all tests (~2.5 min; several need torch)
python -m pytest -q tests/test_determinism.py   # fast checks, no torch needed
```

Run notebooks from the repo root. `SimPipExp_quickstart.ipynb` is the reference demo.
`SimPipExp_GH_1p0.ipynb` is the owner's working notebook: don't edit it.

## Rules

- **Reproducibility.** Never use the global `random` module or unseeded torch. Take an
  `rng` (a `random.Random`) or `seed` argument and get it from
  `build_recommendation_system(seed=...)`. Same seed must give the same events and model.
- **No machine-specific paths.** No `/content/...` or absolute paths. Data defaults to
  `world.external_data_loader.DEFAULT_DATA_PATH`.
- **Keep the smoke test green.** `tests/test_smoke.py` must pass before any merge. Add a
  test for every bug you fix.
- **Stay in your files.** When several agents work in parallel, each one edits only the
  files it owns (see the work split below). Ask the orchestrator before touching others.
- **Keep interfaces swappable.** New components implement the existing ABCs
  (`ANNIndex`, `RetrievalService`, `UserTower`, `ItemTower`) rather than replacing callers.
- **Small, reviewable branches.** One topic per branch; describe what changed and how you
  checked it.

## Work split for parallel agents

| Agent       | Owns                                                        |
|-------------|-------------------------------------------------------------|
| World       | `world/*`                                                   |
| Metrics     | `ml_system/evaluation.py`, `ml_system/metrics.py`, `dashboard/` |
| Features    | `ml_system/flink.py`, `ml_system/offline_*`, `ml_system/features.py`, `TwoTowerRetrieval.retrieve` |
| Performance | `ml_system/ann.py`, `ml_system/training.py`                 |
| Console     | `console/` (SimSession, API server, live UI), `SimPipExp_console.ipynb` |
| Orchestrator| `pyproject.toml`, `tests/`, notebooks, this file; merges branches |

Screenshots of the live console are in `docs/console/`.

## Relevance (ground truth)

"Relevant" is defined once, in `world/relevance.py`, and every component (simulated users,
metrics, game scoring) uses that module. Don't redefine it elsewhere.

- **Genre share.** For user *u*, `share_u(g) = genre_preferences[g] / Σ genre_preferences`.
  `genre_preferences` is the rating-weighted genre count from
  `world/external_data_loader.py:initialize_user_states`.
- **Relevance score** `rel(u, m) ∈ [0, 1]` is the cosine similarity between *u*'s
  genre-share vector and movie *m*'s 0/1 genre vector (over `GENRE_NAMES`). It is 0 if
  either vector is empty.
- **Relevant (binary).** `rel(u, m) ≥ RELEVANCE_THRESHOLD`, which is currently 0.65. At that
  threshold the median user finds about 10% of the catalog relevant. The spread is wide and
  skewed: p25 is about 6%, p75 about 34%, and the mean is about 17%. Users whose taste
  matches a large genre such as Drama have many relevant movies. A lower threshold makes
  almost everything relevant for them. Use this label for
  Recall/Precision/NDCG-style metrics against ground truth.
- **World-side only.** It is computed from the World's `user_states`, which the ML system
  must never read directly. It is deterministic and uses no RNG.
- **Slate.** `ModelServer` retrieves `num_candidates = 100` items and shows the user only
  the top `slate_size = 10`. Both are set via `build_recommendation_system(...)`. Only the
  shown slate is logged in `RecommendationShownEvent`, so training negatives
  (`TrainingDatasetGenerator`) are always items the user saw and skipped.
- **Click model (cascade with position bias).** The user scans the slate in rank order and
  examines rank *i* (0-based) with probability `1 / log2(i + 2)`, looking at the first
  `max_examined = 10` items at most. An examined item is clicked with probability
  `p_noise + (p_max − p_noise) · lift`, where
  `lift = max(0, (rel − RELEVANCE_THRESHOLD) / (1 − RELEVANCE_THRESHOLD))`, `p_noise = 0.02`
  and `p_max = 0.5`. A non-relevant movie is clicked only at the noise rate. The first click
  ends the scan. No click means no watch and no rating. These are `UserSimulator`
  constructor arguments.
- **Expected CTR** over all 943 users: an oracle slate (sorted by `rel`) gets about 69%, a
  random slate about 17% (`ml_system.metrics.reference_ctrs(world)` computes both exactly), and the least-relevant slate under 10%. A better model therefore
  earns more clicks, and putting relevant items near the top matters.
- **Outcomes after a click** (`outcome_model="relevance"`, the default): watch probability
  `0.4 + 0.55·rel`, completion `0.15 + 0.8·rel + N(0, 0.15)` clipped to [0.05, 1],
  `watch_time = 120 · completion`, rating `round(1 + 4·rel + N(0, 0.75))` clipped to 1..5,
  rating probability 0.1. `outcome_model="random"` keeps the old uniform outcomes with
  the same RNG calls.
- `world.relevance.relevance_matrix(user_states, movies)` is the same score vectorised
  (numpy), for evaluation code that needs every (user, movie) pair.

## Configuration switches (`build_recommendation_system`)

| Argument | Default | Meaning |
|---|---|---|
| `seed` | 42 | Every RNG and torch init derives from it |
| `num_candidates`, `slate_size` | 100, 10 | Retrieved vs shown items |
| `users_per_tick` | 1 | Distinct users served per tick (seeded sample) |
| `outcome_model` | `"relevance"` | `"relevance"` or `"random"` watch/completion/rating |
| `user_tower_kwargs` | `{"use_id_embedding": False, "use_activity_features": False}` | Trained user tower inputs |
| `item_tower_kwargs` | `{"use_id_embedding": False, "use_rating_feature": False}` | Trained item tower inputs |
| `log_metrics`, `runs_dir`, `run_id` | `True`, `"runs"`, auto | Metrics log at `runs_dir/run_id/metrics.jsonl` |

The tower classes themselves default to every input on (so old checkpoints load); the
defaults above are `two_tower.DEFAULT_USER_TOWER_KWARGS` / `DEFAULT_ITEM_TOWER_KWARGS`,
also used by `RetrievalEvaluator`. The untrained bootstrap towers always keep ID
embeddings: they are the exploration policy (see "Why CTR was flat").
`recommendation_id` is a per-World counter, unique per shown slate.

## Online metrics

`system["ctr_tracker"]` (`ml_system/metrics.py:CTRTracker`) consumes shown and clicked events
from Kafka and buckets them by event tick. Use `per_tick()`, `rolling_ctr(window=100)`,
`ctr_between(start, end)` (for example, before and after a retrain) and `overall_ctr()`.
With the default `users_per_tick=1`, raw per-tick CTR is 0 or 1, so plot the rolling series;
with more users per tick (the quickstart uses 10) it is a real rate. `per_tick()` rows also
carry `active_users`.

## Metrics log and dashboard

Every built system (unless `log_metrics=False`) has `system["metrics_logger"]`
(`ml_system/metrics.py:MetricsLogger`) and `system["run_dir"]`. It writes one JSON object per
line to `<run_dir>/metrics.jsonl`; every record has `type` and `tick`:

| type | fields |
|---|---|
| `run_start` | `run_id`, `seed`, `created_at` (wall clock), `config` (users_per_tick, outcome_model, num_candidates, slate_size, num_users, num_movies), `reference` (`random_ctr`, `oracle_ctr`), `model_version` (0) |
| `tick` | `shown`, `clicks`, `ctr` (null if nothing shown), `active_users`, `model_version` |
| `retrain` | `training_examples`, `model_version` (the new one) |
| `deploy` | `model_version` (serves from the next tick) |
| `evaluation` | free-form; the quickstart logs `k`, `recall_at_k`, `random_recall_at_k`, `popularity_recall_at_k` |

`retrain(system)` logs retrain + deploy. Log anything else with
`system["metrics_logger"].log_event(type, tick, **fields)`. `read_metrics(run_dir)` loads it.
An explicit `run_id` restarts that directory's log; default ids never collide. `runs/` is
git-ignored, and tests run from a temp working directory (`tests/conftest.py`).

`dashboard.render_dashboard(run_dir) -> Path` reads only the log and writes one
self-contained `<run_dir>/dashboard.html` (HTML + CSS + inline SVG, no JavaScript, no
external files): header (run id, seed, simulated ticks, model version), four tiles (CTR
since the last deploy vs before, Recall@20 vs random, active users per tick, retrains),
rolling-CTR chart with retrain markers and random/oracle reference lines, and an event log.
`dashboard.display_dashboard(run_dir)` also shows it inline in Jupyter/Colab. The rolling
window defaults to about 250 slates (`window=` ticks to override).

## Live console (`console/`)

A player console connected to one running simulation. `mockups/console/index.html` is the
original fake-data mockup (keep it unchanged); the live UI is `console/static/index.html`.

```bash
pip install -e ".[console]"      # fastapi, uvicorn
python -m console                # http://localhost:8000  (--port, --seed, --users-per-tick)
```

On Colab, `SimPipExp_console.ipynb` starts the server in the background and opens it with
`google.colab.output.serve_kernel_port_as_window`.

- `console/session.py:SimSession` has no web code. It wraps `build_recommendation_system`
  (default 10 users per tick), runs the clock on a background thread (speed 1×/10×/100× =
  2/20/200 ticks per second, as fast as possible when ticks are slower) and exposes
  JSON-ready views. `step(n)` advances synchronously (tests, scripts).
- **Determinism.** `submit(name, **params)` queues an action (`retrain`, `rollback`, `spark`,
  `settings`); queued actions apply only at the next tick boundary (after tick T, before
  T+1), in submission order. Retraining builds its features and training set on the clock
  thread at the boundary, then trains, embeds and builds the HNSW index on a background
  thread while the current model keeps serving. The new model deploys exactly
  `deploy_delay` ticks (default 50) after the retrain started; if training is not done by
  then, the clock waits. Same seed + same actions at the same ticks = same run.
- **Models.** Every version (v0 = untrained bootstrap) keeps its own towers and index in
  memory, so roll back redeploys the previously served version instantly. The trainer
  keeps training its own towers across retrains (warm start); deployed versions are copies.
  Held-out Recall@20 is not computed live (it needs a second training run); the UI shows it
  as pending and shows online CTR per version instead.
- **Spark.** The offline feature job runs on its own schedule (`spark_every`, default 100
  ticks) and as the first stage of every retrain (logged as `spark` with `by: "retrain vN"`;
  covered by the retrain's cost). So the Pipeline's Spark node and the feature-freshness
  tile reflect both.
- **Settings** (Pipeline screen, `POST /api/settings`): `spark_every`, `retrain_every`
  (0 = off), `deploy_delay`, `epochs`, `negatives`, `slate_size`, `num_candidates`,
  `rebuild_index` (off: a deploy swaps the towers but keeps the old index, the classic
  stale-index bug), and the alert rule's `alert_drop_pct`, `alert_ticks`, `alert_window`.
  Settings with no simulated counterpart (Kafka partitions, TTLs, canary...) are shown
  disabled.
- **Alert.** After each deploy the session compares the rolling CTR over `alert_window`
  ticks with the CTR over the `2·alert_window` ticks before the deploy; a drop of more than
  `alert_drop_pct` % within `alert_ticks` ticks opens an incident (with runbook). A later
  deploy (e.g. a rollback) marks it Mitigated.
- **Budget** is a game resource: retrain 40 credits, Spark run 5, of 1000.
- **Latency** is the wall-clock time of `ModelServer.recommend` in the server process
  (real but machine-dependent, never fed back into the simulation).
- Session events also go to the metrics log: `retrain_start`, `retrain`, `deploy` (with
  `index_version`, `by`), `spark`, `settings`, `incident`.

API (`console/server.py`; the UI uses nothing else):

| Endpoint | |
|---|---|
| `GET /api/state` | tick, playing, speed, model/index version, training in flight, budget, reference CTRs |
| `GET /api/metrics?since=T` | per-tick rows for ticks > T: shown, clicks, ctr, active_users, model_version, latency; plus deploys |
| `GET /api/events?since=SEQ` | event log entries |
| `GET /api/models` | versions with training stats and online CTR, rollback target |
| `GET /api/pipeline` | component status and current settings |
| `GET /api/incidents` | incidents opened by the alert rule |
| `POST /api/play`, `/api/pause`, `/api/speed {speed}`, `/api/step {ticks}` | clock (step only while paused) |
| `POST /api/retrain`, `/api/rollback`, `/api/spark`, `/api/settings {...}` | queued for the next tick boundary |

## Why CTR was flat (fixed)

Main used to go 18.3% → 18.7% after a retrain. Expected-CTR analysis per user showed the
retrained top-100 contained good items (oracle rerank of it: 47%) but the learned scores did
not rank them. Causes, all fixed:
1. HNSW linked each node to its M nearest only, so clusters were islands (recall@10 ≈ 0.5).
   It now uses the neighbour-selection heuristic and 2·M links on layer 0 (≈ 0.96).
2. Train/serve skew: serving invented a uniform cold-start profile (wrong genre names) and a
   3.5 rating, and never updated `interactions`. Serving now uses the training defaults.
3. Monotonic activity counts (always out of the training range at serving time), per-ID
   embeddings (about 1 click per user) and the noisy item rating swamped the genre signal.
   They are switched off in the trained towers.
4. The untrained bootstrap towers need ID embeddings to explore. Without them every user
   sees the same 10 movies and the first retrain has nothing to learn from.

Quickstart now (seed 42, 10 users/tick, 1000 ticks, retrain 5 epochs, 300 ticks): CTR
**17.2% → 29.7%** (seed 1: 13.2% → 30.1%). With only ~5000 sessions before the retrain the
result is unstable: the model collapses to a few dozen near-global slates.

## Known issues (next phases)

- `SimPip_DEMO_1p2.ipynb` (legacy Colab notebook) still scores Recall@K on training data
  and uses `/content/...` paths. The quickstart uses the held-out
  `RetrievalEvaluator(movie_objects, num_users=..., seed=...)`.
- **Offline Recall@K mostly measures exposure.** Held-out clicks come from the bootstrap
  policy, which shows a narrow set of items, so a global popularity list scores ~83%
  Recall@20 in the quickstart (no leak: it pads from the catalog) while the two-tower model
  scores ~7% yet lifts online CTR from 17% to 30%. Trust online CTR (or expected CTR from
  `UserSimulator.expected_click_rate`) over offline recall here.
- CTR after a retrain (~30%) is still far from the oracle (~69%): most users have 0–3
  watched movies when the model is trained, and users without history all get one global
  slate. A shared genre projection across both towers gave about 2 more points in an
  experiment (not merged: it couples the towers).
- Online and offline features still disagree on popularity (Flink: watches per item in
  Redis `item_popularity`, unused by serving; offline: watches in the feature store).
- `retrain()` recomputes offline features that the Spark job already computed.
- Console: a stale index (`rebuild_index` off) does not visibly hurt CTR yet, because the
  trainer warm-starts from the previous towers, so v(n) user vectors still match v(n-1)
  item vectors. The CTR-drop alert is unit-tested, but no built-in scenario reliably fires it.
  Missions and data drift in the console are still mock.
- With `users_per_tick > 1`, every active user in a tick is served from the same model
  state; nothing models concurrency or latency yet.
