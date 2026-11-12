from simulation.service import Service
from simulation.events import (
    RecommendationShownEvent,
    RecommendationClickedEvent,
    WatchEvent,
    RatingEvent,
)
from simulation.events import OnboardingEvent
from .features import update_genre_counts, update_engagement, engagement_features, new_engagement_state
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

        # Operations knobs (defaults = the original behaviour):
        # - paused: process nothing; events wait in Kafka (outage).
        # - max_events_per_tick: processing capacity; the rest waits, so
        #   consumer lag builds up under load. None = unlimited.
        # - track_watched: also publish each user's watched movies to Redis
        #   ("watched_movies"), for filtering already-watched items.
        self.paused = False
        self.max_events_per_tick = None
        self.track_watched = False
        self.processed_last_tick = 0

        # track_recent: publish each user's recent watches and shown slates
        # ("recent_watches", "recent_slates") for the ranker. Subscribes to
        # shown slates from the current end of the topic when switched on.
        self.track_recent = False
        self.recent = None

        # track_item_signals: per-movie behaviour statistics and co-watching
        # (ml_system.item_signals), published to Redis as
        # ("item_signals", "all"). Needs shown slates (switches track_recent on).
        self.track_item_signals = False
        self.signals = None

        # track_engagement: per-genre click-through and profile strength
        # (features.update_engagement), published as "genre_engagement" and
        # "profile_strength". Needs shown slates (switches track_recent on).
        self.track_engagement = False
        self.engagement = {}

        # profile_mode: how user genre profiles are built ("watches" = the
        # original, or "rich"; see features.update_genre_counts; the
        # offline pipeline uses the same). track_onboarding: also consume
        # OnboardingEvent (declared favourite genres).
        self.profile_mode = "watches"
        self.track_onboarding = False
        self._onboarding_subscribed = False

        # preferences: a live ml_system.online_model.PreferenceModel (set by
        # its owner; None = off). Learns online from every event Flink
        # consumes (slates, clicks, watches, ratings, onboarding) and is
        # published to Redis as ("preference_model", "all").
        self.preferences = None

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
        self.processed_last_tick = 0

        if self.track_item_signals and self.signals is None:
            from ml_system.item_signals import ItemSignals
            self.signals = ItemSignals()
            sim.get_service("Redis").set("item_signals", "all", self.signals)
            self.track_recent = True

        if self.track_engagement:
            self.track_recent = True

        if self.preferences is not None:
            self.track_recent = True       # needs the shown slates
            redis = sim.get_service("Redis")
            if redis.get("preference_model", "all") is not self.preferences:
                redis.set("preference_model", "all", self.preferences)

        if self.track_recent and self.recent is None:
            from ml_system.ranking import RecentHistory
            self.recent = RecentHistory()
            # From the start of the current tick: the World has already
            # published this tick's slates before Flink runs.
            kafka.subscribe(self.consumer_name, "RecommendationShownEvent", from_tick=sim.clock.now)
            self.topics = self.topics + ["RecommendationShownEvent"]

        if (self.track_onboarding or self.preferences is not None) and not self._onboarding_subscribed:
            kafka.subscribe(self.consumer_name, "OnboardingEvent")
            self.topics = self.topics + ["OnboardingEvent"]
            self._onboarding_subscribed = True

        if self.paused:
            return

        budget = self.max_events_per_tick
        topics = self.topics

        if budget is not None:
            # Rotate the starting topic so no topic starves under load.
            start = sim.clock.now % len(topics)
            topics = topics[start:] + topics[:start]

        signal_events = []

        for topic in topics:

            if budget is not None and budget <= 0:
                break

            events = kafka.poll(
                self.consumer_name,
                topic,
                max_records=budget
            )

            if budget is not None:
                budget -= len(events)

            self.processed_last_tick += len(events)

            if self.signals is not None or self.preferences is not None:
                signal_events.extend(events)

            for event in events:

                if self.track_engagement:
                    self.update_engagement(sim, event)

                if isinstance(event, RecommendationClickedEvent):
                    self.handle_recommendation_clicked(sim, event)

                elif isinstance(event, WatchEvent):
                    self.handle_watch(sim, event)

                elif isinstance(event, RatingEvent):
                    self.handle_rating(sim, event)

                elif isinstance(event, OnboardingEvent) and self.track_onboarding:
                    self.update_profile(sim, event.user_id, "onboarding", event.genres)

                elif isinstance(event, RecommendationShownEvent):
                    if self.recent is not None:
                        sim.get_service("Redis").set(
                            "recent_slates", event.user_id,
                            self.recent.add_slate(event.user_id, event.movie_ids))


        if self.signals is not None and signal_events:
            self.apply_signals(signal_events)
        if self.preferences is not None:
            self.preferences.observe_tick(signal_events)
            # Every reaction to this tick's slates is in: learn from them now,
            # before the next tick is served.
            self.preferences.flush(sim.clock.now + 1)

    # Movie signals depend on event order (a click needs its slate first),
    # so each tick's events are applied in the data lake's topic order,
    # the order training replays them in.
    SIGNAL_ORDER = {"RecommendationShownEvent": 0, "RecommendationClickedEvent": 1,
                    "WatchEvent": 2, "RatingEvent": 3}

    def apply_signals(self, events):
        for event in sorted(events, key=lambda e: (e.timestamp, self.SIGNAL_ORDER.get(type(e).__name__, 4))):
            self.signals.observe(event)

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

        # Offline, `interactions` counts the same events as activity_level
        # (clicks, watches, ratings); publish it so serving sees it too.
        redis.receive({
            "feature": "interactions",
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


    def update_profile(self, sim, user_id, kind, genres, value=None):
        """Apply one profile signal and publish the normalised profile
        (onboarding in any mode; clicks and ratings in rich mode)."""
        counts = self.state.setdefault("genre_counts", {}).setdefault(user_id, {})
        if not update_genre_counts(counts, kind, genres, self.profile_mode, value):
            return
        total = sum(counts.values())
        if total > 0:
            sim.get_service("Redis").receive({
                "feature": "genre_preferences",
                "user_id": user_id,
                "value": {genre: count / total for genre, count in counts.items()},
            })

    def update_engagement(self, sim, event):
        kind = {"RecommendationShownEvent": "shown", "RecommendationClickedEvent": "click",
                "WatchEvent": "watch", "OnboardingEvent": "onboarding"}.get(type(event).__name__)
        if kind is None:
            return
        state = self.engagement.setdefault(event.user_id, new_engagement_state())
        if kind == "shown":
            update_engagement(state, kind, [self.movie_genres(sim, m) for m in event.movie_ids])
        elif kind == "onboarding":
            update_engagement(state, kind, event.genres)
        else:
            update_engagement(state, kind, self.movie_genres(sim, event.movie_id))
        rates, strength = engagement_features(state)
        redis = sim.get_service("Redis")
        redis.set("genre_engagement", event.user_id, rates)
        redis.set("profile_strength", event.user_id, strength)

    def movie_genres(self, sim, movie_id):
        return sim.get_service("World").movie_objects[movie_id].genres

    def handle_recommendation_clicked(self, sim, event):

        if self.profile_mode != "watches":
            self.update_profile(sim, event.user_id, "click", self.movie_genres(sim, event.movie_id))

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

        if self.profile_mode != "watches":
            self.update_profile(sim, event.user_id, "rating",
                                self.movie_genres(sim, event.movie_id), event.rating)

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


        if self.recent is not None:
            redis.set("recent_watches", user_id, self.recent.add_watch(user_id, movie_id))

        if self.track_watched:
            watched = redis.store.setdefault(("watched_movies", user_id), set())
            watched.add(movie_id)


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
        if self.profile_mode == "watches":
            for genre in movie.genres:

                counts[genre] = (
                    counts.get(genre, 0) + 1
                )
        else:
            update_genre_counts(counts, "watch", movie.genres, self.profile_mode, event.completion)


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




