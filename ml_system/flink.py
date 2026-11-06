from simulation.service import Service
from simulation.events import (
    RecommendationClickedEvent,
    WatchEvent,
    RatingEvent,
)
from .features import (
    RatingFeatureComputer,
    ItemRatingFeatureComputer,
)

class Flink(Service):

    def __init__(self):
        super().__init__()

        self.consumer_name = "Flink"

        self.topics = [
            "RecommendationClickedEvent",
            "WatchEvent",
            "RatingEvent"
        ]

        self.state = {
            "activity_level": {},
            "item_popularity": {}
        }

        self.output = []

        # Shared feature computation
        self.rating_features = RatingFeatureComputer()
        self.item_rating_features = ItemRatingFeatureComputer()

    def start(self, sim):
        super().start(sim)

        kafka = sim.get_service("Kafka")

        for topic in self.topics:
            kafka.subscribe(
                self.consumer_name,
                topic
            )

    def process(self, sim):

        kafka = sim.get_service("Kafka")

        for topic in self.topics:

            events = kafka.poll(
                self.consumer_name,
                topic
            )

            for event in events:

                if isinstance(event, RecommendationClickedEvent):
                    self.handle_recommendation_clicked(sim, event)

                elif isinstance(event, WatchEvent):
                    self.handle_watch(sim, event)

                elif isinstance(event, RatingEvent):
                    self.handle_rating(sim, event)


    def update_activity_level(self, sim, user_id):

        self.state["activity_level"][user_id] = (
            self.state["activity_level"].get(user_id, 0) + 1
        )

        redis = sim.get_service("Redis")

        redis.receive({
            "feature": "activity_level",
            "user_id": user_id,
            "value": self.state["activity_level"][user_id]
        })


    def update_item_popularity(self, sim, movie_id):

        self.state["item_popularity"][movie_id] = (
            self.state["item_popularity"].get(movie_id, 0) + 1
        )

        redis = sim.get_service("Redis")

        redis.receive({
            "feature": "item_popularity",
            "movie_id": movie_id,
            "value": self.state["item_popularity"][movie_id]
        })


    def handle_recommendation_clicked(self, sim, event):

        user_id = event.user_id

        self.update_activity_level(
            sim,
            user_id
        )

        self.state.setdefault("user_click_counts", {})

        self.state["user_click_counts"][user_id] = (
            self.state["user_click_counts"].get(user_id, 0) + 1
        )

        redis = sim.get_service("Redis")

        redis.receive({
            "feature": "user_click_counts",
            "user_id": user_id,
            "value": self.state["user_click_counts"][user_id]
        })


    def handle_rating(self, sim, event):

        self.update_activity_level(
            sim,
            event.user_id
        )

        redis = sim.get_service("Redis")

        # User average rating
        average_rating = self.rating_features.update(
            event.user_id,
            event.rating
        )

        redis.receive({
            "feature": "average_rating",
            "user_id": event.user_id,
            "value": average_rating
        })


        # Item average rating
        item_average_rating = self.item_rating_features.update(
            event.movie_id,
            event.rating
        )

        redis.receive({
            "feature": "item_average_rating",
            "movie_id": event.movie_id,
            "value": item_average_rating
        })

    def handle_watch(self, sim, event):

        user_id = event.user_id
        movie_id = event.movie_id

        redis = sim.get_service("Redis")


        # -------------------------
        # Activity
        # -------------------------

        self.update_activity_level(
            sim,
            user_id
        )


        # -------------------------
        # Item popularity
        # -------------------------

        self.update_item_popularity(
            sim,
            movie_id
        )


        # -------------------------
        # Watch count
        # -------------------------

        self.state.setdefault(
            "watch_count",
            {}
        )

        self.state["watch_count"][user_id] = (
            self.state["watch_count"].get(user_id, 0) + 1
        )

        redis.receive({
            "feature": "watch_count",
            "user_id": user_id,
            "value": self.state["watch_count"][user_id]
        })


        # -------------------------
        # Genre preference update
        # -------------------------

        world = sim.get_service("World")
        movie = world.movie_objects[movie_id]


        self.state.setdefault(
            "genre_counts",
            {}
        )


        # New users start with empty online state
        if user_id not in self.state["genre_counts"]:

            self.state["genre_counts"][user_id] = {}


        counts = self.state["genre_counts"][user_id]


        # Add new watch signal
        for genre in movie.genres:

            counts[genre] = (
                counts.get(genre, 0) + 1
            )


        total = sum(
            counts.values()
        )


        if total > 0:

            genre_preferences = {
                genre: count / total
                for genre, count in counts.items()
            }


            redis.receive({
                "feature": "genre_preferences",
                "user_id": user_id,
                "value": genre_preferences
            })




