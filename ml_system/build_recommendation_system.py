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
from ml_system.metrics import CTRTracker
from ml_system.two_tower import (
    TwoTowerRetrieval,
    SimpleUserTower,
    SimpleItemTower,
    DEFAULT_USER_TOWER_KWARGS,
    DEFAULT_ITEM_TOWER_KWARGS,
)
from ml_system.offline_feature_store import OfflineFeatureStore
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
    item_tower_kwargs=None
):
    """Wire up World, streaming, serving and the offline retraining loop.

    `user_tower_kwargs` / `item_tower_kwargs` are passed to
    SimpleUserTower / SimpleItemTower (feature switches such as
    `use_activity_features`); the defaults are DEFAULT_*_TOWER_KWARGS.
    """

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
        rng=child_rng()
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

    ctr_tracker = CTRTracker()
    sim.register(ctr_tracker)

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
        "spark": sim.get_service("Spark"),
        "ctr_tracker": ctr_tracker,
        "seed": seed,
        "user_tower_kwargs": user_tower_kwargs,
        "item_tower_kwargs": item_tower_kwargs
    }
