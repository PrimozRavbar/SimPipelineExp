# SimPipeline: notes for Claude and other agents

SimPipeline simulates a production recommendation system (World → Kafka → Flink → Redis →
two-tower retrieval → ModelServer → World) plus an offline retraining loop
(DataLake → Spark → feature store → training → HNSW). The goal is a game where players
build, break, diagnose and fix ML systems.

## Setup and commands

```bash
pip install -e ".[dev]"          # torch, pandas, numpy, pytest, jupyter
pytest -q                        # all tests (~1 min; smoke test needs torch)
pytest -q tests/test_determinism.py   # fast checks, no torch needed
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
| Metrics     | `ml_system/evaluation.py`, new `ml_system/metrics.py`       |
| Features    | `ml_system/flink.py`, `ml_system/offline_*`, `ml_system/features.py`, `TwoTowerRetrieval.retrieve` |
| Performance | `ml_system/ann.py`, `ml_system/training.py`                 |
| Orchestrator| `pyproject.toml`, `tests/`, notebooks, this file; merges branches |

## Known issues (next phases)

- Recall@K in the notebooks is scored on training data; `RetrievalEvaluator` has a
  held-out split but calls `SimpleUserTower()` without `num_users`.
- Simulated users click at ~60% regardless of model quality (100 items shown; genre-match
  rule almost always true). Ratings and watch completion are random.
- Online and offline features disagree (cold-start genre list, `interactions`,
  default average rating, popularity definition).
- `retrain()` recomputes offline features that the Spark job already computed.
