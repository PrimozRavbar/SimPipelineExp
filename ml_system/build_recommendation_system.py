import random

import torch

from simulation.simulator import Simulator
from world.world import World

from ml_system.kafka import Kafka
from ml_system.redis import Redis
from ml_system.flink import Flink
from ml_system.datalake import DataLake
from ml_system.spark import Spark
from ml_system.model_server import ModelServer
import time
from pathlib import Path

from ml_system.metrics import CTRTracker, MetricsLogger, reference_ctrs
from ml_system.two_tower import (
    TwoTowerRetrieval,
    SimpleUserTower,
    SimpleItemTower,
    DEFAULT_USER_TOWER_KWARGS,
    DEFAULT_ITEM_TOWER_KWARGS,
)
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.feast import Feast
from ml_system.kserve import KServe
from ml_system.offline_pipeline import (
    OfflinePipeline,
    OfflineFeaturePipeline,
)
from ml_system.training_dataset import TrainingDatasetGenerator
from ml_system.training import TwoTowerTrainer, EmbeddingGenerator
from ml_system.bootstrap import (
    BootstrapPipeline,
    BootstrapFeaturePipeline,
)


def build_recommendation_system(
    movie_objects,
    user_objects,
    user_states,
    movie_states,
    seed=42,
    num_candidates=100,
    slate_size=10,
    user_tower_kwargs=None,
    item_tower_kwargs=None,
    users_per_tick=1,
    outcome_model="relevance",
    log_metrics=True,
    runs_dir="runs",
    run_id=None,
    world_dynamics=None,
    onboarding=None,
    browsing=None,
    taste="genre"
):
    """Wire up World, streaming, serving and the offline retraining loop.

    `user_tower_kwargs` / `item_tower_kwargs` are passed to
    SimpleUserTower / SimpleItemTower (feature switches such as
    `use_activity_features`); the defaults are DEFAULT_*_TOWER_KWARGS.
    `users_per_tick` is how many distinct users the World serves per tick.
    `outcome_model` is "relevance" (watch/completion/rating depend on
    rel(u, m)) or "random" (the old uniform outcomes).
    With `log_metrics` the run appends to <runs_dir>/<run_id>/metrics.jsonl
    (see MetricsLogger); `run_id` defaults to "seed<seed>-<timestamp>".
    `world_dynamics` (dict, default None = a static world) switches on
    taste drift, users joining over time, fatigue and no-rewatching; see
    world.dynamics (DYNAMIC_WORLD is a preset with everything on).
    `onboarding` (dict {"genres": 3, "noise": 0.3}, default None = off):
    new users declare favourite genres before their first slate; the
    console can also switch it on later (world.onboarding).
    `taste`: "genre" (default: relevance is the genre match only) or
    "rich" (relevance mixes the genre match with hidden tastes and movie
    quality learned from the real MovieLens ratings; see world.taste).
    """
    if taste not in ("genre", "rich"):
        raise ValueError('taste must be "genre" or "rich"')
    taste_model = None
    if taste == "rich":
        from world.relevance import calibrate_threshold
        from world.taste import fit_taste_model, ratings_from_user_states

        # Fitted on the pre-game ratings, before the simulation adds any.
        taste_model = fit_taste_model(ratings_from_user_states(user_states))
        calibrate_threshold(taste_model, [user_states[u] for u in sorted(user_states)],
                            [movie_objects[m] for m in sorted(movie_objects)])
    from world.dynamics import DYNAMICS_KEYS, EXPOSURE_KEYS, WorldDynamics

    world_dynamics = dict(world_dynamics or {})
    unknown = set(world_dynamics) - set(DYNAMICS_KEYS) - set(EXPOSURE_KEYS)
    if unknown:
        raise ValueError("unknown world_dynamics keys: %s" % sorted(unknown))

    user_tower_kwargs = dict(
        DEFAULT_USER_TOWER_KWARGS if user_tower_kwargs is None else user_tower_kwargs
    )
    item_tower_kwargs = dict(
        DEFAULT_ITEM_TOWER_KWARGS if item_tower_kwargs is None else item_tower_kwargs
    )

    # One seed controls every source of randomness, so two builds with the
    # same seed produce the same simulation and the same trained model.
    torch.manual_seed(seed)
    seeds = random.Random(seed)

    def child_rng():
        return random.Random(seeds.getrandbits(64))

    # Feature store
    feature_store = OfflineFeatureStore()

    # Bootstrap
    bootstrap_pipeline = BootstrapPipeline(
        feature_pipeline=BootstrapFeaturePipeline(
            movie_objects,
            user_objects,
            feature_store
        ),
        embedding_generator=EmbeddingGenerator(
            feature_store
        )
    )

    # The bootstrap towers are untrained, so they act as an exploration
    # policy. They keep the full input set (ID embeddings included): random
    # ID embeddings give each user a different slate. Without them every
    # cold-start user gets the same embedding, everyone is shown the same
    # 10 movies, and the first retrain has nothing to learn from.
    bootstrap_item_tower = SimpleItemTower(
        num_movies=len(movie_objects)
    )

    bootstrap_artifacts = bootstrap_pipeline.run(
        bootstrap_item_tower
    )

    # Online components
    redis = Redis()

    retriever = TwoTowerRetrieval(
        user_tower=SimpleUserTower(
            num_users=len(user_objects)
        ),
        item_tower=bootstrap_artifacts["item_tower"],
        ann_index=bootstrap_artifacts["hnsw_index"],
        redis=redis,
        feature_store=feature_store
    )

    model_server = ModelServer(
        retriever=retriever,
        num_candidates=num_candidates,
        slate_size=slate_size
    )

    # Simulator
    world = World(
        user_objects=user_objects,
        movie_objects=movie_objects,
        user_states=user_states,
        movie_states=movie_states,
        rng=child_rng(),
        users_per_tick=users_per_tick,
        outcome_model=outcome_model,
        user_simulator_kwargs={
            k: world_dynamics[k] for k in EXPOSURE_KEYS if k in world_dynamics
        },
        onboarding=onboarding,
        # Own RNG, derived from the seed directly (no draw from the shared
        # stream), so switching onboarding on never shifts other components.
        onboarding_rng=random.Random("onboarding-%s" % seed),
        browsing=browsing,
        browsing_rng=random.Random("browsing-%s" % seed),
        taste=taste_model
    )

    sim = Simulator(seed=seeds.getrandbits(64))

    sim.register(world)
    sim.register(Kafka())
    sim.register(Spark())
    sim.register(Flink())
    sim.register(redis)
    sim.register(feature_store)

    datalake = DataLake()

    sim.register(datalake)
    sim.register(model_server)

    # Feast (serving reads features through it; training reads the offline
    # store point in time) and KServe (replicas in front of the model
    # server). Both are pass-throughs with their default settings.
    feast = Feast(redis, feature_store)
    sim.register(feast)
    retriever.feature_store = feast
    kserve = KServe(model_server)
    sim.register(kserve)
    from ml_system.fallback import PopularityRetrieval
    model_server.overflow_fallback = PopularityRetrieval(redis, movie_objects)

    ctr_tracker = CTRTracker()
    sim.register(ctr_tracker)

    # Registered only when asked for, so a static world is unchanged. Its
    # RNG is derived from `seed` directly, not from the shared stream.
    dynamics = None
    dynamics_args = {k: world_dynamics[k] for k in DYNAMICS_KEYS if k in world_dynamics}
    if dynamics_args:
        dynamics = WorldDynamics(world, seed=seed, **dynamics_args)
        sim.register(dynamics)

    metrics_logger = None
    run_dir = None

    if log_metrics:
        if run_id is None:
            # Default ids never reuse a directory; an explicit run_id does
            # (the log is restarted).
            base = f"seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
            run_id, n = base, 1
            while (Path(runs_dir) / run_id).exists():
                n += 1
                run_id = f"{base}-{n}"

        run_dir = Path(runs_dir) / run_id

        # Registered after the CTRTracker, so each tick's row is complete.
        metrics_logger = MetricsLogger(
            run_dir,
            ctr_tracker,
            run_info={
                "run_id": run_id,
                "seed": seed,
                "config": {
                    "taste": taste,
                    "users_per_tick": users_per_tick,
                    "outcome_model": outcome_model,
                    "num_candidates": num_candidates,
                    "slate_size": slate_size,
                    "num_users": len(user_objects),
                    "num_movies": len(movie_objects),
                },
                "reference": reference_ctrs(world, slate_size=slate_size, seed=seed),
            },
        )
        sim.register(metrics_logger)

    # Offline pipeline
    offline_feature_pipeline = OfflineFeaturePipeline(
        movie_objects,
        feature_store
    )

    training_dataset_generator = TrainingDatasetGenerator(
        feature_store,
        rng=child_rng()
    )

    embedding_generator = EmbeddingGenerator(
        feature_store
    )

    two_tower_trainer = TwoTowerTrainer(
        user_tower=SimpleUserTower(
            num_users=len(user_objects),
            **user_tower_kwargs
        ),
        item_tower=SimpleItemTower(
            num_movies=len(movie_objects),
            **item_tower_kwargs
        )
    )

    offline_pipeline = OfflinePipeline(
        feature_pipeline=offline_feature_pipeline,
        dataset_generator=training_dataset_generator,
        trainer=two_tower_trainer,
        embedding_generator=embedding_generator
    )

    return {
        "sim": sim,
        "world": world,
        "redis": redis,
        "datalake": datalake,
        "retriever": retriever,
        "model_server": model_server,
        "offline_pipeline": offline_pipeline,
        "feature_store": feature_store,
        "feast": feast,
        "kserve": kserve,
        "spark": sim.get_service("Spark"),
        "ctr_tracker": ctr_tracker,
        "world_dynamics": dynamics,
        "metrics_logger": metrics_logger,
        "run_dir": run_dir,
        "seed": seed,
        "users_per_tick": users_per_tick,
        "outcome_model": outcome_model,
        "user_tower_kwargs": user_tower_kwargs,
        "item_tower_kwargs": item_tower_kwargs
    }
