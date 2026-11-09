from ml_system.ann import HNSWIndex


def retrain(system):

    artifacts = system["offline_pipeline"].run(
        system["datalake"].events
    )

    hnsw_index = HNSWIndex(seed=system.get("seed", 0))
    hnsw_index.build(
        artifacts["item_embeddings"]
    )

    system["retriever"].user_tower = artifacts["user_tower"]
    system["retriever"].item_tower = artifacts["item_tower"]
    system["retriever"].ann_index = hnsw_index

    # The new model serves from the next tick on: log retrain + deploy.
    logger = system.get("metrics_logger")
    if logger is not None:
        logger.log_retrain(
            system["sim"].clock.now,
            training_examples=artifacts.get("training_examples")
        )

    return artifacts
