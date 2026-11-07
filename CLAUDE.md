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
  threshold about 17% of the catalog is relevant for a typical user. A lower threshold lets
  Drama-heavy taste make almost everything relevant. Use this label for
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
- **Expected CTR** over all 943 users: an oracle slate (sorted by `rel`) gets about 68%, a
  random slate about 18%, and the least-relevant slate under 10%. A better model therefore
  earns more clicks, and putting relevant items near the top matters.
- Watch completion and ratings are still random (see known issues).

## Known issues (next phases)

- Recall@K in the notebooks is scored on training data. Use
  `RetrievalEvaluator(movie_objects, num_users=..., seed=...)` for a held-out score.
  `SimPip_DEMO_1p2.ipynb` still calls the old signature.
- Ratings and watch completion are random. They should depend on `rel(u, m)`.
- `evaluate_popularity` pads its top-K list with movies drawn from the validation
  population. That leaks the answers, so its recall is inflated on small runs.
- Online and offline features disagree (cold-start genre list, `interactions`,
  default average rating, popularity definition).
- `retrain()` recomputes offline features that the Spark job already computed.
