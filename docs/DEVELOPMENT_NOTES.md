# SimPipeline: development notes (full history and measurements; CLAUDE.md is the short version)

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
| Features    | `ml_system/flink.py`, `ml_system/offline_*`, `ml_system/features.py`, `ml_system/feast.py`, `TwoTowerRetrieval.retrieve` |
| Performance | `ml_system/ann.py`, `ml_system/ann_indexes.py`, `ml_system/training.py`, `ml_system/ranking.py` |
| Ops         | `ml_system/faults.py`, `ml_system/fallback.py`, `ml_system/experiments.py`, `ml_system/model_registry.py`, `ml_system/kafka.py`, `ml_system/redis.py`, `ml_system/kserve.py` |
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
- **Rich world** (`build_recommendation_system(taste="rich")`, console `--taste rich`; the
  default `"genre"` world above is unchanged). `world/taste.py` fits a biased matrix
  factorisation of the real MovieLens ratings once per process (ALS, 16 factors, fixed
  settings and seed, so the true tastes are the same in every game): `rating ~ mu + b_movie +
  p_user . q_movie`. Relevance becomes `0.4 * genre cosine + 0.6 * taste score`, with taste
  score `clip((mu + b_movie + p . q - 1) / 4, 0, 1)` (the user bias is left out: generous raters
  are not more interested in everything). `b_movie` is movie quality; `p . q` are hidden
  tastes beyond genres. The threshold is calibrated (`calibrate_threshold`) so the median user
  finds 10% of the catalog relevant (0.660; p25 5.8%, p75 16.9%). All relevance functions take
  `taste=` (the World's `world.taste`, None in the genre world); `relevance_threshold(taste)`
  gives the threshold. Reference CTRs: random 12.2%, oracle 68.5%. The best slate chosen by
  genre alone reaches only 40.1%, hidden tastes alone 52.3%, one global best-quality list 34.9%:
  there is a lot to learn beyond genres. Taste drift in the dynamic world still moves the genre
  part only.
- **Personal world** (`taste="personal"`, console `--taste personal`): the rich world with hidden
  tastes counting `PERSONAL_HIDDEN_SCALE` = 3 times (`TasteModel.scaled()`, threshold recalibrated
  to 0.691). Per user, relevance varies about as much with hidden taste (0.080) as with the genre
  match (0.085) instead of 0.029, and a slate chosen by genre and quality alone holds 82% relevant
  movies (rich: 98%). Oracle ~78% (rich ~69%). The rich world is unchanged.

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
| `world_dynamics` | `None` (static world) | Dict: taste drift, users joining, fatigue, rewatch (see "Realism features"); `world.dynamics.DYNAMIC_WORLD` turns all on |

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
  (default 20 users per tick; the CLI and notebook too), runs the clock on a background thread (speed 1×/10×/100× =
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
| `GET /api/snapshot?since=T&events_since=S` | all the reads below in one response; the UI polls only this, once a second, with a 5 s timeout per request |
| `POST /api/insight` | stage report: where clicks are lost (uses ground truth; computed on request) |
| `GET /api/state` | tick, playing, speed, model/index version, training in flight, budget, reference CTRs |
| `GET /api/metrics?since=T` | per-tick rows for ticks > T: shown, clicks, ctr, active_users, model_version, latency; plus deploys |
| `GET /api/events?since=SEQ` | event log entries |
| `GET /api/models` | versions with training stats and online CTR, rollback target |
| `GET /api/pipeline` | component status and current settings |
| `GET /api/incidents` | incidents opened by the alert rule |
| `POST /api/play`, `/api/pause`, `/api/speed {speed}`, `/api/step {ticks}` | clock (step only while paused) |
| `POST /api/retrain`, `/api/rollback`, `/api/spark`, `/api/settings {...}` | queued for the next tick boundary |
| `GET /api/faults` | faults so far (hidden chaos faults only counted while active), fault kinds, repairs |
| `POST /api/fault {kind, duration}`, `/api/repair/{restart_flink\|rebuild_index\|restart_model_server}` | drills and repairs, next tick boundary |
| `POST /api/canary/promote`, `/api/canary/abort`, `/api/models/{v}/export` | canary control, model export |
| `POST /api/swap {component, to}` | swap a component for another (`ann_index`: `hnsw`, `ivf`, `exact`); costs money; next tick boundary |
| `POST /api/repair/restart_kafka_broker` | ends a `kafka_broker_down` fault |

`GET /api/state` also carries `health` (lag, fallback share, index self-test, skew, coverage,
drift PSI, feature age), `canary` (per-arm counts and the test) and `spark_job`.

## Realism features (all opt-in)

Everything below is off by default: a default build produces byte-identical events to
the version before it (`tests/test_defaults_unchanged.py` holds a fingerprint taken on
main). New randomness uses its own string-seeded RNGs (`random.Random("world-dynamics-<seed>")`,
`"chaos-<seed>"`), never the shared seed stream, so switching a feature on never shifts
another component's random draws.

- **Changing world** (`world/dynamics.py`, `UserSimulator(fatigue=, rewatch=)`): taste drift
  (every N ticks a share of users moves part of its taste to one genre; `taste_drift_mode`:
  `"abrupt"`, the original, moves it at once to a random genre; `"gradual"`, the dynamic world's
  default, starts a drift journey towards a genre related to the user's taste (catalog genre
  co-occurrence x share, away from genres they already mostly like) and gets there in
  `taste_drift_steps` (4) equal steps over the next N ticks, same total shift), users joining over
  time (`World.active_user_ids`, cold start), fatigue (each repeat impression of a movie
  multiplies its click probability by `1 - fatigue`), no rewatching (`rewatch` multiplier for
  movies watched in the simulation), fatigue wearing off (`fatigue_recovery=H`: a (user,
  movie) impression count halves every H ticks since its last impression; the dynamic world
  uses 1000, console `fatigue_recovery_ticks`; 0 = never, the original). `expected_click_rate`
  applies the same factors.
  `reference_ctrs` is computed at build time and goes stale once tastes drift.
- **Browsing** (`World(browsing={"rate": p})`, own RNG `"browsing-<seed>"`; console `browse_pct`,
  30 by default in both worlds; the build default is off, so the golden fingerprint holds): on a
  visit, with probability p, the user first browses a genre drawn from their *current* true
  genre share (`BrowseEvent(genre)`, before the slate). Kafka topic and data lake like every
  event. Rich profiles count it (`features.update_genre_counts` kind `"browse"`,
  `BROWSE_WEIGHT` 0.5 with the rich decay; one definition for Flink `track_browsing` and Spark;
  the original `"watches"` mode ignores it). The preference model keeps a per-user browse vector
  (decay 0.8 per browse) and learns a weight `kb` on "browse match" (movie genres . recent
  browsing share), taken when the slate is shown (it learns ~0.8).
- **Richer user signals** (world, console Users node; the build default is off, so the golden
  fingerprint holds):
  - *Interest signals* (`interest_save_pct` 100, `interest_dismiss_pct` 50 by default in both
    worlds; own RNG `"interest-<seed>"`, so clicks are unchanged): every movie the user examined
    in the cascade but did not click (`UserSimulator.examined`, recorded without extra draws) is
    saved with probability save x relevance lift (relevant movies only) or marked "not
    interested" with probability dismiss (irrelevant movies only): `InterestEvent(movie_id,
    kind)`. Data lake topic; rich profiles count saves (`features.update_genre_counts` kind
    `"save"`, `SAVE_WEIGHT` 0.4; Flink `track_interest` and Spark); the preference model takes a
    save as a grade row with target 0.85 and a dismissal with target 0 (`interest_weight` 1).
  - *Ratings* (`rating_pct`, `UserSimulator.rating_probability`, 10 = the original).
  - *Sessions* (`session_slates`, 1 = the original): slates per visit; the model server skips
    movies already shown to the user earlier in the same tick (`ModelServer._session`).
  Measured (rich, 20 users a tick, default setup, mean CTR ticks 800-2000; dynamic s42 baseline
  26.3): ratings 50% 25.3; interest 50/20 28.5 (s1 28.2; static s42 50.7 vs 50.0); interest
  100/50 **29.7** (s1 30.6; static s42 51.5); interest 50/20 + ratings 50% 28.4; two slates per visit 22.4 per slate (the
  second row holds second-best picks and users tire twice as fast; about 0.45 clicks per visit
  vs 0.26); everything 24.4. Signals on examined-but-unclicked movies are the evidence the
  click alone never gave. Defaults: interest 100/50, ratings 10%, one slate.
- **Targeted exploration** (`ml_system/exploration.py:SlotExplorer`, the last re-ranker; needs the
  preference model): at `explore_rate_pct` of slates (`explore_boost_pct` for users
  `PreferenceModel.drifting()` flags: recent click rate below half their long-run rate, or recent
  browsing pointing away from their learned top-3 genres), up to `explore_slots` slots from
  position 4 down get a well-rated movie (one of the 5 best by learned quality, at random) the
  user has rarely been shown, from a genre chosen by Thompson sampling over the user's learned
  taste and an uncertainty that shrinks with genre impressions (`explore_policy="thompson"`) or
  towards under-shown genres (`"uniform"`); candidates come from the whole catalog, unlike
  `ExplorationReranker`. Every shown movie's propensity goes to `SlotExplorer.propensities`
  ((tick, user, movie)); with `explore_ips` the preference model weights rows by min(10, 1/p).
  Own RNG `"explorer-<seed>"`; the stage report skips it. Pipeline shows the explored share,
  their click rate and the users flagged as drifting.
  Measured (rich, 20 users a tick, gradual drift, mean CTR over ticks 800-2000; one 400-tick
  window is noisy, +-2-3 points): dynamic s42 gradual only 25.4, + browsing 30% 26.0, +
  exploration only 24.6, + browsing 30% + exploration **26.3** (flattest: 26.9 / 26.2 / 25.9 /
  26.1), + browsing 50% + exploration 26.7, stronger exploration (2 slots, 15%, 80% boosted)
  25.6; s1 gradual only 26.2, + browsing + exploration 26.3. Abrupt drift (old) s42 25.0 mean,
  19.5 at tick 2000 vs gradual 21.5. Static s42: no browsing 50.5, browsing 30% 50.0, + exploration
  49.6 (nothing to find when tastes never move). So the best dynamic setup adds exploration
  (`BEST_DYNAMIC_EXTRAS`, applied with `--world dynamic --preset best`) and the static one does
  not. At tick 2000 (dynamic, browsing + exploration) drifted users' learned taste correlates 0.23
  with the truth (0.15 before), the browse weight is 0.82, and the drift detector flags about
  half the users (loose). Swapping in the true genre taste would still lift the catalog top 10
  from 17.3% to 31.9%: with fatigue and no rewatching the system cannot re-serve known
  favourites and needs much finer taste estimates to pick fresh movies.
- **Pipeline infra:** `OfflineFeatureStore` binary-search point-in-time lookups and
  `clear_history()` (the offline pipeline clears before recomputing, so repeated Spark runs
  no longer stack duplicate snapshots); `Kafka.poll(max_records=)`, `Kafka.lag(consumer)`;
  `Flink.paused`, `Flink.max_events_per_tick` (lag builds up), `Flink.track_watched`;
  `Redis.read_transforms` (fault hook); `Spark.failing`.
- **Faults** (`ml_system/faults.py`): `FaultInjector` (a Service, registered last; a fault
  starting at tick S is switched on at the end of tick S-1, or immediately via
  `apply_due(now)`). Kinds: `flink_outage`, `feature_corruption` (serving reads rotated genre
  preferences), `index_id_mismatch` (`RemappedIndex`: right neighbours, wrong ids),
  `model_server_outage` (ModelServer serves `PopularityRetrieval`), `traffic_spike`,
  `spark_failure`. Repairs: `restart_flink`, `rebuild_index`, `restart_model_server`.
- **ModelServer** gained optional `fallback` + `retriever_down`, a `router(user_id) ->
  (retriever, ranker)` for canaries, and applies `reranker` after `ranker` (all None by default).
- **Batched training** (`TwoTowerTrainer(batched=True)`, the default): each tower runs once
  per batch (sampled negatives padded and masked) instead of once per example. Same loss;
  weights match the original loop (`batched=False`) to ~1e-5 after 2 epochs
  (`tests/test_batched_training.py`). 400 ticks of data, 5 epochs: 10.1 s -> 1.5 s on CPU.
  The quickstart result is unchanged (seed 42: 17.2% -> 29.7%; seed 1: 13.2% -> 30.2%).
  Console: "Batched training" switch (Pipeline > Training).
- **Ranking** (`ml_system/ranking.py`): `Ranker`/`Reranker` ABCs. `LogisticRanker` uses
  ML-side features only (genre match from learned preferences, has-profile, log popularity,
  rating) and is trained on logged slates under the cascade (positive = click, negatives =
  movies shown above it; unclicked slates give negatives at every position) with slate
  position as a feature fixed at serving (position debiasing). `WatchedFilter` drops
  already-watched movies (needs `Flink.track_watched`). Seed 42, 10 users/tick, retrain at
  400: CTR over the next 300 ticks 17.9% without the ranker, 23.6% with it.
- **MLP ranker** (`MLPRanker`): 9 features (`FEATURES_V2`: the 4 above, genre match x
  popularity, no-profile x popularity, genre match with the last 5 watches, watched
  recently, times shown in the last 3 slates), a 16-unit MLP plus a training-only
  position "shallow tower" dropped at serving, seeded with `torch.random.fork_rng`.
  `RecentHistory` defines the recent-history features once: Flink (`track_recent`) keeps
  them in Redis for serving (`recent_watches`, `recent_slates`; it subscribes to shown
  slates with `Kafka.subscribe(from_end=True)`), and training replays the event log so
  each slate sees its history as of just before it. Same protocol as above:
  static 18.0% none / 23.7% logistic / 23.4% MLP; dynamic 9.5% / 21.1% / 22.7%.
- **Re-rankers:** `MMRReranker` (maximal marginal relevance on genre Jaccard over the top
  30 candidates; relevance from the incoming order), `ExplorationReranker` (each visible
  slot is, with probability `rate`, a random lower candidate; own seeded RNG
  `"explore-<seed>"`), `RerankerChain`. Console order: watched filter -> diversity ->
  exploration. With the MLP ranker: dynamic world baseline 22.7%, diversity 30% 24.9%,
  diversity 60% 13.3%, exploration 10% 18.6% (catalog coverage 13.5% -> 23%); static world
  baseline 23.4%, diversity 30% 21.2%. Variety pays when users tire of repeats, costs when
  they don't, and too much always hurts.
- **Personalisation** (all off by default):
  - *Profile candidates* (`ml_system/candidate_sources.py`): `ProfileRetrieval` retrieves by
    the user's learned genre profile (`genre_match`, ties by popularity) and
    `ModelServer.extra_sources` merges it round-robin with the two-tower candidates.
  - *Onboarding* (`World(onboarding={"genres": 3, "noise": 0.3})`): the first time a user is
    active they declare favourite genres (`OnboardingEvent`, before their first slate; their
    true top genres, each replaced by a random genre with probability `noise`; own RNG
    `"onboarding-<seed>"`). Flink (`track_onboarding`) and the offline pipeline add
    `ONBOARDING_WEIGHT` (2) per declared genre to the profile; the data lake stores them.
  - *Rich profiles* (`ml_system/features.update_genre_counts`, one definition for Flink and
    the offline pipeline): `profile_mode="rich"` counts clicks (0.5), watch completion and
    ratings ((r-3)/2, floored at 0) with decay 0.97 per new signal. `"watches"` is the
    original code path, untouched.
  - *Per-genre ranker features* (`MLPRanker(feature_set="v3")`): v2 + profile cosine + one
    taste-match feature per genre.

  Seed 42, MLP ranker on, retrains at 400 and 800, CTR over ticks 852-1200 (after v2):
  static world baseline 23.5% -> all four **48.5%** (seed 1: 19.0% -> 45.5%); dynamic world
  22.2% -> 27.2% (29.3% with diversity 30%). Leaving one out of "all four" (static):
  without profile candidates 24.6%, without onboarding 30.7%, without rich profiles 43.7%,
  without per-genre features 45.6%. Onboarding is what gives every user a profile, and the
  profile candidates are what put matching movies in front of the ranker; alone, each moves
  CTR only a little.
- **Retrieval v2** (all off by default; batched training only):
  - `TwoTowerTrainer(logq_correction=True)`: subtract log(item frequency among positives) from
    the positive and in-batch logits (sampling-bias correction, Yi et al. 2019).
  - `random_negatives=k` with `trainer.item_pool`: k random catalog movies per example as
    easy negatives (seeded per training call).
  - `graded=True` with `TrainingDatasetGenerator(graded=True)`: each click weighted by what
    followed (`training_dataset.grade`: 0.3 if not watched, else 0.5 + completion + 0.25 per
    star above 3).
  - `SimpleUserTower(use_history=True, num_movies=...)`: mean of a learned embedding of the
    last 5 watched movies (`user_features["recent_watches"]`: Redis `recent_watches` online,
    replayed per click by `TrainingDatasetGenerator(history=True)` offline).
  - `MLPRanker(multitask=True)`: a second head predicts watch completion given a click;
    serving orders by p(click) x expected completion.
  - Console `train_from_scratch` (and switching `tower_history`) builds fresh, seeded towers.

  Measured on top of the personalisation features (MLP ranker v3, profile candidates 30, rich
  profiles, onboarding), CTR over ticks 852-1200:

  | | static s42 | static s1 | dynamic s42 | dynamic s1 |
  |---|---|---|---|---|
  | base | 48.5% | 45.5% | 27.2% | 27.8% |
  | + logQ | 49.1% | 47.6% | 29.6% | 30.9% |
  | all five | 51.0% | 48.5% | 28.5% | 30.4% |

  logQ is the one that clearly works: the best 10 of the two-tower candidates go from 42.7 to
  59.5 (static s42; relevant share 32% -> 46%). Random negatives, graded positives, history
  and the multi-task ranker are each within +-2 points (noise for one seed). Why the others
  barely help here: relevance in this World is a function of genres only, so item identity
  and watch history add little beyond the genre profile, and noise clicks are only 2%.
- **Stage report** (`console/insight.py`, `POST /api/insight`, Models > "Where clicks are
  lost"): exact expected CTR (ground truth, on request, no RNG) of the served slate, the best
  10 of the two-tower candidates, of the extra sources, of all candidates, of the candidates
  ordered by the system's own learned profile, and of the catalog (oracle). With everything
  on (static s42) served 48.8 ~ profile-ordered 48.3 < candidates 58.2 < oracle 66.9: the
  ranker uses what the system knows as well as plain profile ordering; the rest of the
  ranking loss is imperfect knowledge of users, not ranker modelling.
- **Models in the rich world** (seed 42 unless noted, retrains at 400 and 800, CTR over ticks
  852-1200, then the stage report at 1200): retrain only 14.7% (static) / 11.4% (dynamic); the
  best genre-era stack (MLP ranker v3, profile candidates 30, rich profiles, onboarding, logQ)
  37.2% static (s1 37.6%) / 19.7% dynamic. Static stage report: oracle 68.8, best of all
  candidates 58.0, served 39.4: retrieval is now good, the ranker loses ~19 points because all
  its features are genre-based.
- **Movie signals and co-watching** (`ml_system/item_signals.py`, rich-world phase B):
  `ItemSignals` is one definition updated event by event, online by Flink
  (`track_item_signals`, Redis `("item_signals", "all")`) and by a point-in-time replay in
  training. Per movie: smoothed click rate, completion, rating, impressions and clicks over
  expected clicks (COEC: clicks vs the average click rate of the slots it was shown in, a
  position-debiased quality signal); co-watch similarity (count / sqrt(watches a x watches b)
  over each user's last 20 watches). `MLPRanker(feature_set="v4")` = v3 + those + co-watch score
  with the recent watches; `CoWatchRetrieval` ("because you watched X") is an extra candidate
  source. Console: `ranker_signals`, `cowatch_candidates`. Rich world, on top of the best
  genre-era stack (37.2% static / 19.7% dynamic): both 43.8% static (s1 41.1%, base 37.6%) and
  26.0% dynamic; each alone +0.5 / -2.5 static. Flink applies each tick's events to the signals
  in the data lake's topic order (shown, clicked, watch, rating): COEC needs a click's slate
  first, and with Flink's own order (clicks before slates) online and training COEC differed,
  a train/serve skew that dropped CTR to 29.7% until fixed (`tests/test_item_signals.py`
  checks Flink's signals equal a replay of the log). `Kafka.subscribe(from_tick=T)`: Flink now
  subscribes to slates from the start of the current tick (it used to miss that tick's slates).
- **Two-tower inputs for the rich world** (all off by default; a change of tower inputs gives
  the next retrain fresh, seeded towers): `SimpleUserTower(use_engagement=True)` (per-genre
  click-through "genre_engagement" = skips, and "profile_strength"; `features.update_engagement`
  is one definition for Flink `track_engagement` and the offline pipeline `engagement`),
  `demographics=` (fixed table from `ml_system/demographics.py`: age bracket, gender,
  occupation), `history_table=` (the item tower's movie ID table shared with the watch history;
  `SimpleItemTower(id_init_std=0.01)` starts movie IDs small). Console: `tower_engagement`,
  `tower_demographics`, `tower_shared_ids`. Rich world, on top of signals + co-watch (43.8%
  static / 26.0% dynamic): engagement 39.3 / 22.8, demographics 39.5 / 26.8, shared IDs
  42.5 / 22.3, all three 39.4 / 24.2. They raise the quality of the two-tower candidates (best
  10 up to 60.1 vs ~55) but not served CTR: in the rich world the ranker is the bottleneck.
- **Retrieval score as a ranker feature** (`MLPRanker(feature_set="v5")`, console
  `ranker_tower_score`): v4 + user vector . movie vector. A first version scored the training
  rows with the freshly trained towers: label leakage (those towers were trained on the same
  clicks), CTR 29.0 / 15.2 with the richer towers. Fixed with **feature logging**:
  `ModelServer.score_logger(tick, user, slate, retriever)` is called for every served slate,
  and the console logs each shown movie's score from the retriever that served it
  (`SimSession.score_log`, (tick, user, movie) -> score); training rows join the logged scores
  (`MLPRanker.fill_tower_scores_from_log`; Models shows `logged_share`). Rich world, s42:
  v5 44.2 / 23.3 (vs 43.8 / 26.0 without), with the richer towers 43.4 / 27.9 (was 29.0 / 15.2).
  A ranker that cannot be fitted yet (no clicks) no longer fails the retrain: the towers ship
  without it.
- **Preference model** (`ml_system/online_model.py:PreferenceModel`, console `preference_model`):
  one model that learns from every logged signal, online every tick and refitted at every
  retrain. Score `s(u, m) = b_m + decade_m + g_m . (w0 + a * onboarding_u + w_u) + q_m . p_u +
  r(u, m) + h . history(u, m)`; click logit = `c + pos[i] + s` (learned position term), grade =
  `sigmoid(c2 + kappa * s)`. Rows: clicked movie (positive), movies above it (negatives), an
  unclicked slate (negatives, weight 0.5); grades: watch completion, rating, click without a
  watch (0.3). `r(u, m)`: a per-pair offset moved online by surprising clicks and skips (unseen
  pairs stay 0). History (`preference_history`): times shown (a count; it learns -0.148 per
  impression in the dynamic world vs the true log(0.85) = -0.16) and already watched (it learns
  -2.06 vs log(0.1) = -2.3; +0.36 in the static world, where rewatching is allowed).
  Flink (`Flink.preferences`) applies each tick's events in data-lake order and learns from
  that tick's slates before the next tick; a retrain snapshots it at the boundary, refits it
  in the background (full-batch Adam, 300 epochs, seeded) and `install()`s it at the deploy
  tick, replaying the rows logged since the snapshot. Serving: `PreferenceRanker` orders the
  candidates (the version's own ranker until the first fit) and `PreferenceRetrieval` adds its
  catalog-wide top `preference_candidates` (50). Options: `preference_factors` (0),
  `preference_half_life` (0: rows weigh the same whatever their age), `preference_optimism_pct`
  (UCB-style bonus for rarely shown movies, fading with impressions), `preference_genre_optimism_pct`
  (per-user bonus for genres the user has rarely been shown), `preference_taste_half_life` (a
  second fit stage re-learns each user's taste deviation on age-weighted rows, anchored to the
  full-history taste; shared parameters keep all data); all off by default (see Known issues
  for their measured effect).
  Why: the stage report showed the system's knowledge of users and movies, not model capacity,
  was the limit. With perfect knowledge of genre taste and movie quality (no hidden factors) a
  rich-world slate reaches 61.6% (oracle 68.5%); the logged data is small (by tick 1200: ~12k
  slates, ~13 per user, ~5 clicks per user), and a batch fit of this model on a recorded game
  scores 37% / 48% / 52% at ticks 400 / 800 / 1200 (it keeps rising with data).
  Leaks found on the way (tests in `tests/test_preference_model.py`): rating rows carrying
  "already watched" (taught "watched means liked"), and slate rows written after their own
  watch (taught "watched means clicked", weight +5 even where rewatches are rare; rows now take
  their history when the slate is shown). A "times shown" feature looked harmful (-16 points)
  only because of the second leak.
  Protocol (CTR ticks 852-1200 after retrains at 400 and 800), old default -> with it:
  rich static s42 44.8 -> 49.7, s1 40.9 -> 44.7; rich dynamic s42 27.6 -> 27.0, s1 24.5 -> 28.7;
  genre static s42 49.6 -> 56.2, dynamic 31.9 -> 34.8. Stage report (rich static s42): two-tower
  + other candidates now reach 64.6 of the 68.8 oracle; the ranking step loses the rest.
  Long games (rich s42, retrain every 400 ticks, CTR per 400 ticks): static, it gets there much
  faster (40.4 at 800 vs 23.0) but plateaus at ~52-53 while the old stack reaches 55.4 by 4000
  (factors k=16 do not help: 52.4); dynamic, everything decays (old 25.1 at 1200 -> 13.2 at
  4000, preference model 29.0 -> 14.5) because fatigue never recovers: a learner that knows
  genre taste and quality exhausts its best picks per user, while the oracle (still ~55) knows
  the remaining hidden tastes. A half-life (400 / 1000 ticks) gives only +1.5 / +2.6 at 2800.
- **Ranker v6** (`ranker_preference_score`): MLP v5 + the preference model's score, logged at
  serving time from the live fitted model (`SimSession.preference_log`) and joined to training
  rows like the tower score. A v6 version serves its own MLP once at least half its rows had a
  logged score; otherwise (first retrain) the preference model ranks. It combines the fast start
  of the preference model with the collaborative signals of the old stack: long static game
  (s42) 40.7 at 800, 55.9 at 4000 (vs 53.1 preference ranker, 55.4 old stack; s1 39.3 at 800,
  53.6 at 4000). Protocol: rich
  static s42 49.4, s1 46.9; rich dynamic s42 29.4, s1 27.9; genre static 53.9, dynamic 35.3
  (average 40.5 vs 40.2 preference ranker vs 36.6 old default).
- **Default setup** (console `preset="best"`, `--preset best`, the default; `"plain"` = every
  model feature off, the original console): preference model with repeat effects, MLP ranker
  v6 (genre features, movie signals, logged retrieval and preference scores), preference
  candidates 50, profile candidates 30, co-watch candidates 30, rich profiles,
  onboarding, logQ, diversity 15%; in the dynamic world also targeted exploration
  (`BEST_DYNAMIC_EXTRAS`). The world defaults: 20 users a tick, browsing 30%, and in the dynamic
  world gradual drift and fatigue recovery. Chosen over the alternatives by CTR after the second
  retrain (ticks 852-1200), averaged over static/dynamic and two seeds in the rich world:
  this setup 34.5% vs 33.6% (no retrieval score / diversity) and lower with the tower inputs
  (engagement + demographics + shared IDs: 36.0% static s1). Genre world s42: 49.6% static,
  31.9% dynamic. Players can change every setting; tests pin `preset="plain"`.
- **Infrastructure components** (all pass-throughs with their defaults: the golden fingerprint
  is unchanged; `tests/test_infra.py`). Each has `describe()` (shown in `GET /api/pipeline`
  under `infra`) and console settings on its Pipeline node:
  - *Kafka* (`ml_system/kafka.py`): partitions keyed by user (`kafka_partitions`, 3), brokers
    (3) and replication factor (1): partition p is led by broker p % brokers, replicated on the
    next ones; a partition with no live replica is unavailable and a consumer's poll stops at
    its first message (the topic's order is kept, so it waits and lag builds up). Retention
    (`kafka_retention_ticks`, 0 = forever): older messages are deleted each tick; a consumer
    that is further behind loses them (`lost`). Offsets are logical. Drill
    `kafka_broker_down` (broker 0; not in the chaos pool), repair `restart_kafka_broker`.
    The data lake and the CTR tracker are Kafka consumers too.
  - *Flink*: `parallelism` x `slot_capacity` events per tick (0 = unlimited; `flink_capacity`
    stays as a hard cap); changing the parallelism is a savepoint + restart (2 ticks without
    processing, $20); backpressure = lag that grew for 10 ticks.
  - *Redis* (`ml_system/redis.py`): `redis_max_keys` (0 = unlimited) with `allkeys-lru`,
    `volatile-ttl` or `noeviction` (writes refused); shared objects (`item_signals`,
    `preference_model`) are pinned; `redis_ttl_ticks` expires user features after their last
    write (serving then uses the defaults); hit/miss/eviction/expiration stats.
  - *Feast* (`ml_system/feast.py`, new): serving reads features through it (retriever,
    rankers, profile candidates, canary arm); training keeps reading the offline store point in
    time. A registry of stream views (Flink -> Redis) and batch views (Spark -> offline store).
    `feast_mode="direct"` (default) reads batch features from the offline store per request, as
    before; `"materialized"` serves them from an online copy refreshed every
    `feast_materialize_every` ticks ($2 per run), stale in between.
  - *KServe* (`ml_system/kserve.py`, new): replicas in front of the ModelServer
    (`ModelServer.admission`). `kserve_replica_capacity` requests per replica per tick (0 =
    unlimited); requests beyond the ready replicas go to the popularity fallback (overflow);
    a Knative-style autoscaler sizes for last tick's traffic at `kserve_target_pct` between min
    and max replicas, with `kserve_cold_start_ticks`; estimated latency 5 ms / (1 - utilisation).
  - New incident rules: `budget` (cash < 0), `kafka_partitions` (offline partitions),
    `serving_capacity` (> 10% overflow), each with a runbook.
- **Swappable component: the ANN index** (`ml_system/ann_indexes.py`). `ExactIndex` (brute
  force, perfect recall) and `IVFIndex` (FAISS-style inverted file: seeded spherical k-means into
  32 cells, searches the `ann_nprobe` nearest) implement `ANNIndex` next to `HNSWIndex`, and
  expose `nodes` like it (feature logging, the self-test). `POST /api/swap
  {"component": "ann_index", "to": ...}` (action `swap`): at the next boundary the serving
  model's index is rebuilt with the new type from its own movie vectors, golden queries are
  re-recorded, a broken index is replaced, the $500 migration is charged, and later retrains
  build the new type (older versions keep theirs, so a rollback restores its index). Simulated
  search cost (vectors touched per query; 1 µs each): HNSW `ef_search + M log2 n` (~370 here),
  IVF centroids + the nprobe largest cells (~380 at nprobe 4), exact n (1682). Running price
  per tick: HNSW $0.60, IVF $0.30, exact $1.50. `SimSession.SWAPPABLE` lists swappable
  components; the next one plugs in there.
- **Money** (`console/economy.py`, separate from the compute credits that pay for retrains and
  Spark): start with $10,000; every click earns $1; every tick the configuration costs money
  (Kafka $0.50/broker + $0.05/partition, Flink $0.80/slot, Redis $0.20 + $0.10/1000 keys,
  KServe $1/replica, the ANN index type, Feast $0.30 + $2/materialization); one-off charges:
  index swap $500, Flink rescale $20, Kafka repartition $50. The default setup costs ~$4.5 a
  tick, so at 20 users a tick an untrained model loses money and a good one earns it. Cash in
  the top bar; Pipeline > Costs shows the bill per component and the one-off ledger.
- **Experiments** (`ml_system/experiments.py`): stable hash bucketing (a new salt reshuffles
  users independently), two-proportion z-test, per-arm counting from the event stream.
- **Registry** (`ml_system/model_registry.py`): `export_bundle` / `load_bundle` (towers with
  constructor config, index embeddings, ranker parameters, metadata).

In the console (all live settings, applied at tick boundaries; see `console/session.py`
`DEFAULT_SETTINGS`): `--world dynamic`, chaos (`chaos_every`: hidden seeded faults, revealed
when they end) and drills, repair actions, `flink_capacity`, `ranker`, `filter_watched`,
`--preset best|plain`, `ranker_mlp`, `ranker_genre_features`, `ranker_multitask`, `ranker_signals`, `ranker_tower_score`,
`tower_engagement`, `tower_demographics`, `tower_shared_ids`, `profile_candidates`,
`cowatch_candidates`, `rich_profiles`,
`onboarding`, `logq_correction`, `random_negatives`, `graded_positives`, `tower_history`,
`train_from_scratch`,
`diversity_pct`, `exploration_pct`, `batched_training`,
`canary_share`/`canary_ticks` (promote unless the canary loses with 95% confidence; 50% is an
A/B test), `eval_gate` (newest `holdout_pct`% of events held out of training; both models
scored by Recall@20 on its clicks; a model more than `gate_tolerance_pct`% worse is
Rejected), `spark_throughput` (Spark jobs take events/throughput ticks; a retrain waits for
its Spark stage), `freshness_alert`, export to `<run_dir>/models/vN`.

Health checks and incident rules (`console/health.py`, `SimSession._monitor`): CTR drop at
any time (rolling window vs the 4 windows before) and after a deploy, Flink lag
(`lag_alert`), fallback share > 50%, ANN self-test (golden queries recorded from each
index version when first checked must still return the same ids: healthy = 100%, alert
< 80%), online/offline feature
skew (`skew_alert_pct`), feature freshness, click-mix drift (PSI of the genre mix of clicks in
the last 200 ticks vs the first 200 after the current deploy, so the model's own effect is
not counted as drift; alert > 0.2). Each rule opens one incident with a runbook and
resolves it when the condition clears; deploys mitigate CTR incidents only.

Measured fault effects (seed 42, after one retrain, CTR ~20%): `index_id_mismatch` drops CTR
to ~14% and fails the self-test; `feature_corruption` barely moves CTR but skew jumps to
~96%; `flink_outage` builds ~1.9 lagged events per tick; `model_server_outage` *raised* CTR
to ~26% (the popularity fallback beats an early two-tower model: see Known issues).

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
  item vectors. Faults (`index_id_mismatch`) now give the CTR alerts a real cause.
  Missions are still mock.
- Trained item embeddings are nearly collapsed (every pair has cosine > 0.92, ~250 distinct
  vectors for 1682 movies: the trained item tower sees only genres). HNSW finds the exact
  nearest neighbour for only ~75-95% of item queries on such an index (duplicates are pruned
  by the neighbour heuristic). The console's index self-test first compared against exact
  search and raised false "self-test failing" incidents after normal deploys; it now uses
  golden queries instead (`tests/test_faults.py` has the regression test).
- A popularity list beats the first retrained two-tower model online (~26% vs ~20%), so the
  `model_server_outage` fault can raise CTR early in a game.
- Switching `rich_profiles` mid-game: Flink keeps the counts it has and adds new signal in
  the new mode, while the next Spark run recomputes every profile in the new mode, so online
  and offline profiles differ until users have new activity (the skew tile shows it).
- Long dynamic games still decay, with or without fatigue recovery (rich s42, default setup,
  20 users a tick, CTR per 400 ticks): recovery 1000 ticks 27.4 at 800 -> 25.5 at 1200 -> 13.9
  at 4000 (oracle 51.2); no recovery 26.9 -> 25.4 -> 12.3 (oracle 44.3). The static world at
  20 users a tick climbs 44.2 -> 50.3 -> 56.3. In the dynamic world a user is shown more than
  half the catalog by tick 4000 and every candidate source keeps proposing the same tired
  movies (best 10 of all candidates 21.6 vs oracle 51.2): retrieval needs to be exposure-aware.
- Why long dynamic games decay (diagnosed; rich s42, 20 users a tick, default setup). A system
  that knew genre taste and movie quality perfectly and avoided tired/watched movies would
  still reach 55.4% at tick 2000 (61.2% oracle), so the world is not exhausted. Swapping the
  preference model's learned parts for the truth at tick 2000: true quality 14.3 -> 19.3, true
  genre taste -> 35.1, both 45.9 (static: no gap). The missing knowledge is the genre taste of
  users whose taste drifted: learned-vs-true correlation 0.76 for users who never drifted but
  0.15 for drifted ones (39% of users by tick 2000), equally for those who drifted before tick
  1000. Drift moves 60% of a user's genre taste to a random genre at once, usually one they
  are rarely shown, so no evidence ever arrives. One mechanic off at a time (CTR at 2000,
  baseline 19.5): no fatigue 28.0, no rewatch penalty 21.0, no drift 23.1; all three off 53.4.
  Opt-in levers measured (CTR at tick 2000): movie optimism (`preference_optimism_pct` 50 /
  100 / 200) 19.4 / 21.1 / 21.2 (static 100: 55.1 vs 54.5); taste recency
  (`preference_taste_half_life` 300 / 150) 20.1 / 21.4; genre exploration
  (`preference_genre_optimism_pct` 100) 20.9 (17.1 when the preference model ranks directly);
  broader candidates (preference 100, co-watch 60, 200 retrieved) 20.8; factors 16 19.9;
  exploration 10% 19.8. Down-weighting repeated skips (tried, removed) collapsed CTR to ~5%:
  it hid the evidence the model needs to learn fatigue. The remaining lever is the World:
  gradual drift towards related genres, or user signals beyond slates (search, browsing).
- **Collaborative filtering** (`ml_system/cf_model.py`, console `cf_model`, `cf_candidates`,
  `cf_factors`; off by default): `ImplicitMF`, weighted ALS on implicit feedback (clicks with
  confidence from completion and rating, skips above a click as weak negatives, unseen pairs as
  low-confidence zeros; ALS-WR regularisation, so users without clicks get no vector), refitted at
  every retrain in the background and versioned with deploys and rollbacks. The preference model's
  rows carry the deployed CF model's score as of when the slate was shown (fitted at the previous
  retrain: no leak), with a learned weight; `CFRetrieval` adds the best-matching unwatched movies
  from the whole catalog. Measured in the personal world (s42, 20 users a tick, mean CTR ticks
  800-2000): static 53.8 -> 54.3 (60.4 vs 59.2 at tick 2000), dynamic 26.8 -> 26.7. Even there the
  CF factors learn little of the true hidden taste (learning curve on a recorded personal static
  game, correlation 0.039 / 0.065 / 0.078 at ticks 800 / 1600 / 2400), so it stays off.
- **Precision levers measured** (rich dynamic s42, mean CTR ticks 800-2000, default 26.3):
  `preference_grade_weight_pct` 300 (watch completion and ratings x3) 26.5,
  `preference_browse_memory_pct` 95 (keep ~20 browses instead of ~5) 26.4, both 26.8: within
  noise. Defaults stay 100 / 80.
- **Hidden-taste factors do not pay in this World (measured).** Learning curve on a recorded
  6000-tick static game (rich s42, default setup, 20 users a tick), offline, vs ground truth,
  at ticks 1000 / 2000 / 4000 / 6000 (7 / 18 / 42 / 66 clicks per user): correlation of learned
  personal factors with the true hidden taste p.q, preference-model factors (k=16) 0.040 /
  0.053 / 0.059 / 0.065, implicit-feedback ALS (`ml_system/cf_model.py:ImplicitMF`, weighted
  ALS with confidence from clicks/completion/ratings/skips, ALS-WR regularisation; not wired
  in) 0.033 / 0.048 / 0.056 / 0.060; catalog top 10 by the preference model 51.8 / 54.3 / 56.5 /
  57.3 with or without factors, lower with CF mixed in at any weight. Why: per user, relevance
  varies 0.085 with genre fit, 0.057 with movie quality and only 0.029 with the hidden part
  (the taste MF is fitted with strong regularisation), and dropping the hidden part changes the
  relevant/not label of only 4% of (user, movie) pairs; learning it from cascade clicks needs
  far more data. The levers are genre-taste and quality precision (dynamic-world swap test),
  or a World with stronger hidden tastes. Recorded games also show static CTR levelling off at
  ~57% from tick ~3000 and dynamic at ~22-24% from tick ~2400 (no collapse).
- Demographics (age, gender, occupation) are still unused by the models.
- Exploration's long-run benefit (better training data for later retrains) is not measured
  yet, and explored slots are not logged with their propensity, so training treats them
  like any other impression.
- Canary statistics treat each slate as independent (users repeat), so the 95% intervals are
  somewhat too narrow. The gate's Recall@20 ignores the ranker and, like all offline recall
  here, mostly measures exposure.
- Spark jobs and model bundles cannot be resumed after a server restart; only the metrics
  log persists.
- With `users_per_tick > 1`, every active user in a tick is served from the same model
  state; nothing models concurrency or latency yet.
