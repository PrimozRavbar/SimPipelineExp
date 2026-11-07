import math
import random

from world.relevance import genre_share, relevance_above_threshold
from simulation.events import (
    RecommendationShownEvent,
    RecommendationClickedEvent,
    WatchEvent,
    RatingEvent,
)

class UserSimulator:

    def __init__(
        self,
        world,
        rng=None,
        max_examined=10,
        p_noise=0.02,
        p_max=0.5
    ):
        self.world = world

        # Seeded RNG so a simulation run is reproducible.
        self.rng = rng if rng is not None else random.Random(0)

        # Cascade click model (see CLAUDE.md, "Relevance").
        self.max_examined = max_examined
        self.p_noise = p_noise
        self.p_max = p_max

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

        clicked_movie_id = self.choose_click(user, movie_ids)

        if clicked_movie_id is not None:

            clicked = self.click(
                sim,
                user,
                recommendation_id,
                clicked_movie_id
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


    def examine_probability(self, rank):
        # Position bias: rank 0 is always examined, later ranks less often.
        return 1.0 / math.log2(rank + 2)


    def click_probability(self, user_state, movie, share):
        # Non-relevant movies get only p_noise; above the relevance
        # threshold the click probability rises linearly to p_max.
        lift = relevance_above_threshold(user_state, movie, share=share)
        return self.p_noise + (self.p_max - self.p_noise) * lift


    def choose_click(self, user, movie_ids):
        """Cascade model: scan the slate in order, stop at the first click.
        Returns the clicked movie_id, or None if nothing was clicked."""

        user_state = self.world.user_states[user.user_id]
        share = genre_share(user_state)

        for rank, movie_id in enumerate(list(movie_ids)[:self.max_examined]):

            if self.rng.random() >= self.examine_probability(rank):
                continue

            movie = self.world.movie_objects[movie_id]

            if self.rng.random() < self.click_probability(
                user_state, movie, share
            ):
                return movie_id

        return None


    def click(
        self,
        sim,
        user,
        recommendation_id,
        movie_id
    ):

        return RecommendationClickedEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            recommendation_id=recommendation_id,
            movie_id=movie_id
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