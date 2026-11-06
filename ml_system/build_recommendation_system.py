from simulation.simulator import Simulator
from world.world import World

from ml_system.kafka import Kafka
from ml_system.redis import Redis
from ml_system.flink import Flink
from ml_system.datalake import DataLake
from ml_system.spark import Spark
from ml_system.model_server import ModelServer
from ml_system.two_tower import (
    TwoTowerRetrieval,
    SimpleUserTower,
    SimpleItemTower,
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
    movie_states
):

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
        retriever=retriever
    )

    # Simulator
    world = World(
        user_objects=user_objects,
        movie_objects=movie_objects,
        user_states=user_states,
        movie_states=movie_states
    )

    sim = Simulator()

    sim.register(world)
    sim.register(Kafka())
    sim.register(Spark())
    sim.register(Flink())
    sim.register(redis)
    sim.register(feature_store)

    datalake = DataLake()

    sim.register(datalake)
    sim.register(model_server)

    # Offline pipeline
    offline_feature_pipeline = OfflineFeaturePipeline(
        movie_objects,
        feature_store
    )

    training_dataset_generator = TrainingDatasetGenerator(
        feature_store
    )

    embedding_generator = EmbeddingGenerator(
        feature_store
    )

    two_tower_trainer = TwoTowerTrainer(
        user_tower=SimpleUserTower(
            num_users=len(user_objects)
        ),
        item_tower=SimpleItemTower(
            num_movies=len(movie_objects)
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
        "spark": sim.get_service("Spark")
    }
