import random
from simulation.events import (
    RecommendationShownEvent,
    RecommendationClickedEvent,
    WatchEvent,
    RatingEvent,
)

class UserSimulator:

    def __init__(self, world, rng=None):
        self.world = world

        # Seeded RNG so a simulation run is reproducible.
        self.rng = rng if rng is not None else random.Random(0)
        #self.click_model = RandomClickModel()
        #self.watch_model = RandomWatchModel()
        #self.rating_model = RandomRatingModel()

    def react(
        self,
        sim,
        user,
        recommendation_id,
        movie_ids
    ):

        events = []

        events.append(
            self.show(
                sim,
                user,
                recommendation_id,
                movie_ids
            )
        )

        #if self.should_click():
        if self.should_click(user, movie_ids):

            clicked = self.click(
                sim,
                user,
                recommendation_id,
                movie_ids
            )

            events.append(clicked)

            if self.should_watch():

                events.append(
                    self.watch(
                        sim,
                        user,
                        clicked.movie_id
                    )
                )

                if self.should_rate():

                    events.append(
                        self.rate(
                            sim,
                            user,
                            clicked.movie_id
                        )
                    )

        return events


    def show(
        self,
        sim,
        user,
        recommendation_id,
        movie_ids
    ):

        return RecommendationShownEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            recommendation_id=recommendation_id,
            movie_ids=movie_ids
        )


    #def should_click(self):
        #return self.rng.random() < 0.2

    def should_click(self, user, movie_ids):

        user_state = self.world.user_states[user.user_id]

        for movie_id in movie_ids:

            movie = self.world.movie_objects[movie_id]

            if any(
                genre in user_state.genre_preferences
                for genre in movie.genres
            ):
                return self.rng.random() < 0.6

        return self.rng.random() < 0.1


    def click(
        self,
        sim,
        user,
        recommendation_id,
        movie_ids
    ):

        user_state = self.world.user_states[user.user_id]

        scores = []

        for movie_id in movie_ids:

            movie = self.world.movie_objects[movie_id]

            score = sum(
                user_state.genre_preferences.get(genre, 0)
                for genre in movie.genres
            )

            scores.append(max(score, 0))

        total_score = sum(scores)

        if total_score > 0:
            probabilities = [
                score / total_score
                for score in scores
            ]
        else:
            probabilities = [
                1.0 / len(movie_ids)
                for _ in movie_ids
            ]

        selected_movie = self.rng.choices(
            movie_ids,
            weights=probabilities,
            k=1
        )[0]

        return RecommendationClickedEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            recommendation_id=recommendation_id,
            movie_id=selected_movie
        )


    def should_watch(self):
        return self.rng.random() < 0.8


    def watch(
        self,
        sim,
        user,
        movie_id
    ):

        return WatchEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            movie_id=movie_id,
            watch_time=self.rng.uniform(5, 120),
            completion=self.rng.uniform(0.1, 1.0)
        )


    def should_rate(self):
        return self.rng.random() < 0.1


    def rate(
        self,
        sim,
        user,
        movie_id
    ):

        return RatingEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            movie_id=movie_id,
            rating=self.rng.randint(1, 5)
        )