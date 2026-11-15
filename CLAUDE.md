# SimPipeline: notes for Claude and other agents

A game about running a production recommender: World → Kafka → Flink → Redis/Feast → two-tower
retrieval + rankers → KServe/ModelServer → World, plus offline retraining (DataLake → Spark →
feature store → training → ANN index). Players build, break, diagnose and fix it.

**Full history, every measurement and every bug found: `docs/DEVELOPMENT_NOTES.md`.** Read the
relevant section there before changing a component. Player docs: `README.md`, `docs/USER_GUIDE.md`.

## Commands

```bash
pip install -e ".[dev,console]"
python -m pytest -q                              # all tests (~17 min; several need torch)
python -m pytest -q tests/test_defaults_unchanged.py tests/test_smoke.py   # quick gate
python -m console --taste rich [--world dynamic] # http://localhost:8000
```

`SimPipExp_GH_1p0.ipynb` is the owner's notebook: don't edit it. Use `OMP_NUM_THREADS=1` when
running several simulations in parallel.

## Rules

- **Reproducibility.** No global `random`, no unseeded torch. New randomness gets its own
  string-seeded RNG (`random.Random("<feature>-<seed>")`) so switching a feature on never shifts
  other draws. Same seed + same actions at the same ticks = same game.
- **Defaults stay byte-identical.** `tests/test_defaults_unchanged.py` fingerprints a default
  `build_recommendation_system`; new features are off in the build and switched on in the
  console's settings/presets.
- **No leaks.** Training rows only use what was known when the slate was shown (point-in-time
  features, scores logged at serving). Several leaks were found this way: see the notes.
- **Ground truth is World-side only** (`world/relevance.py`, `world/taste.py`): metrics, the
  stage report and tests may read it; models never.
- **No machine-specific paths.** Add a test for every bug fixed. Keep interfaces swappable
  (`ANNIndex`, `RetrievalService`, `Ranker`, `Reranker`, towers).
- **Branches.** One topic per branch; never push to `main`; describe what changed and how it
  was checked (measure CTR over ticks 800-2000 on two seeds, static and dynamic).

## Map

| Area | Files |
|---|---|
| World (users, true tastes, dynamics, browsing, interest signals) | `world/` |
| Streaming and storage | `ml_system/kafka.py`, `flink.py`, `redis.py`, `feast.py`, `datalake.py` |
| Offline | `ml_system/spark.py` + `spark_sim/` (cluster cost model, simulated scheduler), `offline_*`, `features.py`, `training*.py`, `training_job.py` (GPU job model), `validation.py` (data and model checks) |
| Models | `two_tower.py`, `ranking.py` (MLP v2-v6), `online_model.py` (preference model), `cf_model.py`, `item_signals.py`, `candidate_sources.py`, `exploration.py` |
| Serving | `model_server.py`, `kserve.py`, `ann.py`, `ann_indexes.py`, `fallback.py` |
| Ops | `faults.py`, `experiments.py`, `model_registry.py`, `metrics.py` |
| Game | `console/session.py` (SimSession, settings, presets), `server.py` (API), `static/index.html` (UI), `economy.py`, `insight.py`, `health.py` |

## Key facts

- **Relevance:** genre world = genre cosine; rich = 0.4 genre + 0.6 taste score (quality +
  hidden taste); personal = hidden taste x3. Cascade click model, `p_max` 0.5. Oracle CTR: rich
  ~69%, personal ~78%; random ~12%.
- **Console defaults** (`DEFAULT_SETTINGS`, preset `best`): preference model + MLP ranker v6,
  extra candidate sources, rich profiles, onboarding, logQ, diversity 15%; 20 users a tick;
  browsing 30%; saves/not-interested 100/50; dynamic world adds gradual drift, fatigue recovery
  1000 ticks and targeted exploration. Tests pin `preset="plain"`.
- **Where CTR stands** (rich, mean ticks 800-2000): static ~51%, dynamic ~30% (seed 42 29.7,
  seed 1 30.6). The dynamic gap is evidence per user, not model capacity; hidden-taste learning
  (factors, CF) does not pay at game data volumes. Details and every experiment in the notes.
- **Spark** runs on a simulated cluster (1 tick ~ 15 min): incremental, partitioned, atomic
  commit; features are byte-identical for any cluster or schedule, and a job that finishes
  inside its tick is ready at once (so the default game is unchanged). Failure modes (spot loss,
  OOM/skew, slow node, busy cluster, stuck lake sink) are faults; viral movies are World-side.
- **Retrains are workflows** (features → validate data → train → validate model → evaluate →
  deploy). Training time and failures (GPU OOM, preemption) are simulated on a GPU cluster; the
  model deploys `deploy_delay` ticks after the start or when training is done, whichever is
  later (a 5-tick delay measured slightly worse CTR, so 50 stays). Batch size and learning rate
  are real. Faults: gpu_preemption, feature_nulls (needs a backfill), scheduler_down.
- **Actions apply at tick boundaries**; retrains train in a background thread and deploy after
  `deploy_delay` ticks; every version (towers, index, preference model, CF) rolls back together.
- **API:** the UI polls `GET /api/snapshot` only; actions are POSTs queued for the next tick
  (`console/server.py` lists them).
