
from ml_system.ann import HNSWIndex
from ml_system.offline_feature_store import OfflineFeatureStore
from ml_system.offline_pipeline import OfflineFeaturePipeline
from ml_system.training_dataset import TrainingDatasetGenerator
from ml_system.training import TwoTowerTrainer
from ml_system.two_tower import SimpleUserTower, SimpleItemTower


class RetrievalEvaluator:

    def __init__(
        self,
        movie_objects,
        split_ratio=0.8,
        k=20
    ):
        self.movie_objects = movie_objects
        self.split_ratio = split_ratio
        self.k = k


    def evaluate(self, events):

        split_index = int(
            len(events) * self.split_ratio
        )

        train_events = events[:split_index]
        validation_events = events[split_index:]


        # ---------------------------------------------
        # Training-period feature store
        # ---------------------------------------------

        feature_store = OfflineFeatureStore()

        feature_pipeline = OfflineFeaturePipeline(
            self.movie_objects,
            feature_store
        )

        feature_pipeline.process(
            train_events
        )


        # ---------------------------------------------
        # Training dataset
        # ---------------------------------------------

        dataset_generator = TrainingDatasetGenerator(
            feature_store
        )

        for event_order, event in enumerate(train_events):

            dataset_generator.add_event(
                event,
                event_order
            )

        training_data = dataset_generator.build()


        # ---------------------------------------------
        # Train fresh Two-Tower
        # ---------------------------------------------

        user_tower = SimpleUserTower()

        item_tower = SimpleItemTower(
            num_movies=len(self.movie_objects)
        )

        trainer = TwoTowerTrainer(
            user_tower=user_tower,
            item_tower=item_tower
        )

        trainer.train(
            training_data
        )


        # ---------------------------------------------
        # Build ANN from training-period item features
        # ---------------------------------------------

        item_embeddings = {}

        train_cutoff_timestamp = train_events[-1].timestamp
        train_cutoff_order = len(train_events) - 1

        for movie_id in self.movie_objects:

            features = (
                feature_store.get_item_features_as_of(
                    movie_id,
                    train_cutoff_timestamp,
                    train_cutoff_order
                )
            )

            if features is not None:

                item_embeddings[movie_id] = (
                    item_tower.embed(features)
                )


        ann_index = HNSWIndex()

        ann_index.build(
            item_embeddings
        )


        # ---------------------------------------------
        # Evaluate future interactions
        # ---------------------------------------------

        hits = 0
        evaluated = 0
        validation_population = []

        for test_order, event in enumerate(validation_events):

            if not hasattr(event, "movie_id"):
                continue

            event_name = type(event).__name__

            if event_name not in (
                "RecommendationClickedEvent",
                "WatchEvent"
            ):
                continue


            global_event_order = (
                split_index + test_order
            )

            user_features = (
                feature_store.get_user_features_as_of(
                    event.user_id,
                    event.timestamp,
                    global_event_order
                )
            )

            # Users first appearing in validation are cold-start
            # and therefore cannot be evaluated with historical
            # training-period user features.
            if user_features is None:
                continue


            user_embedding = user_tower.embed(
                user_features
            )

            recommendations = ann_index.search(
                user_embedding,
                self.k
            )

            evaluated += 1

            validation_population.append({
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "timestamp": event.timestamp,
                "event_order": global_event_order
            })

            if event.movie_id in recommendations:
                hits += 1


        recall = (
            hits / evaluated
            if evaluated > 0
            else 0.0
        )


        return {
            "train_events": len(train_events),
            "validation_events": len(validation_events),
            "evaluated": evaluated,
            "hits": hits,
            "recall_at_k": recall,
            "k": self.k,
            "validation_population": validation_population
        }


def evaluate_popularity(
    events,
    validation_population,
    split_ratio=0.8,
    k=20
):
    split_index = int(len(events) * split_ratio)
    train_events = events[:split_index]

    popularity = {}

    for event in train_events:
        if not hasattr(event, "movie_id"):
            continue

        event_name = type(event).__name__

        if event_name not in (
            "RecommendationClickedEvent",
            "WatchEvent"
        ):
            continue

        movie_id = event.movie_id
        popularity[movie_id] = popularity.get(movie_id, 0) + 1

    recommendations = [
        movie_id
        for movie_id, _ in sorted(
            popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    # Include unseen training-period movies after the known popular
    # movies so the recommendation list can still contain k items.
    for movie_id in sorted(
        set(interaction["movie_id"] for interaction in validation_population)
        | set(popularity.keys())
    ):
        if movie_id not in recommendations:
            recommendations.append(movie_id)

    recommendations = recommendations[:k]

    hits = sum(
        1
        for interaction in validation_population
        if interaction["movie_id"] in recommendations
    )

    evaluated = len(validation_population)

    recall = (
        hits / evaluated
        if evaluated > 0
        else 0.0
    )

    return {
        "evaluated": evaluated,
        "hits": hits,
        "recall_at_k": recall,
        "k": k
    }


def evaluate_random(
    validation_population,
    movie_objects,
    k=20,
    seed=42
):
    import random

    rng = random.Random(seed)
    movie_ids = list(movie_objects.keys())

    hits = 0

    for interaction in validation_population:
        recommendations = rng.sample(
            movie_ids,
            min(k, len(movie_ids))
        )

        if interaction["movie_id"] in recommendations:
            hits += 1

    evaluated = len(validation_population)

    recall = (
        hits / evaluated
        if evaluated > 0
        else 0.0
    )

    return {
        "evaluated": evaluated,
        "hits": hits,
        "recall_at_k": recall,
        "k": k,
        "seed": seed
    }
