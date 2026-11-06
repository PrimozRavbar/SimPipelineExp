from ml_system.ann import HNSWIndex


class BootstrapFeaturePipeline:

    def __init__(
        self,
        movie_objects,
        user_objects,
        feature_store
    ):
        self.movie_objects = movie_objects
        self.user_objects = user_objects
        self.feature_store = feature_store

    def run(self):

        for movie_id, movie in self.movie_objects.items():

            self.feature_store.write_item_features(
                movie_id,
                {
                    "movie_id": movie_id,
                    "genres": movie.genres,
                    "average_rating": 0,
                    "popularity": 0,
                    "interactions": 0
                }
            )


class BootstrapPipeline:

    def __init__(
        self,
        feature_pipeline,
        embedding_generator
    ):
        self.feature_pipeline = feature_pipeline
        self.embedding_generator = embedding_generator

    def run(self, item_tower):

        self.feature_pipeline.run()

        item_embeddings = (
            self.embedding_generator.generate(
                item_tower
            )
        )

        hnsw_index = HNSWIndex()
        hnsw_index.build(item_embeddings)

        return {
            "item_tower": item_tower,
            "item_embeddings": item_embeddings,
            "hnsw_index": hnsw_index
        }
