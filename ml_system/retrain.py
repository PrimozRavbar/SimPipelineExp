from ml_system.ann import HNSWIndex


def retrain(system):

    artifacts = system["offline_pipeline"].run(
        system["datalake"].events
    )

    hnsw_index = HNSWIndex()
    hnsw_index.build(
        artifacts["item_embeddings"]
    )

    system["retriever"].user_tower = artifacts["user_tower"]
    system["retriever"].item_tower = artifacts["item_tower"]
    system["retriever"].ann_index = hnsw_index

    return artifacts
