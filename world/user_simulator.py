import math
import random

from world.relevance import genre_share, relevance, relevance_above_threshold
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
        p_max=0.5,
        outcome_model="relevance",
        rating_noise=0.75,
        completion_noise=0.15
    ):
        self.world = world

        # Seeded RNG so a simulation run is reproducible.
        self.rng = rng if rng is not None else random.Random(0)

        # Cascade click model (see CLAUDE.md, "Relevance").
        self.max_examined = max_examined
        self.p_noise = p_noise
        self.p_max = p_max

        # What happens after a click. "relevance": watch probability,
        # completion and rating all rise with rel(u, m), plus Gaussian
        # noise. "random": the old behaviour (uniform completion, uniform
        # 1-5 rating, fixed watch probability), kept for comparison.
        if outcome_model not in ("relevance", "random"):
            raise ValueError("outcome_model must be 'relevance' or 'random'")

        self.outcome_model = outcome_model
        self.rating_noise = rating_noise
        self.completion_noise = completion_noise

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

            rel = self.relevance_of(user, clicked.movie_id)

            if self.should_watch(rel):

                events.append(
                    self.watch(
                        sim,
                        user,
                        clicked.movie_id,
                        rel
                    )
                )

                if self.should_rate():

                    events.append(
                        self.rate(
                            sim,
                            user,
                            clicked.movie_id,
                            rel
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


    def relevance_of(self, user, movie_id):
        return relevance(
            self.world.user_states[user.user_id],
            self.world.movie_objects[movie_id]
        )


    def should_watch(self, rel=0.0):
        if self.outcome_model == "random":
            return self.rng.random() < 0.8

        # A clicked relevant movie is usually watched (rel 1 -> 95%),
        # a noise click on an off-taste movie less often (rel 0 -> 40%).
        return self.rng.random() < 0.4 + 0.55 * rel


    def watch(
        self,
        sim,
        user,
        movie_id,
        rel=0.0
    ):

        if self.outcome_model == "random":
            watch_time = self.rng.uniform(5, 120)
            completion = self.rng.uniform(0.1, 1.0)
        else:
            completion = min(1.0, max(
                0.05,
                0.15 + 0.8 * rel + self.rng.gauss(0.0, self.completion_noise)
            ))
            watch_time = 120 * completion

        return WatchEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            movie_id=movie_id,
            watch_time=watch_time,
            completion=completion
        )


    def should_rate(self):
        return self.rng.random() < 0.1


    def rate(
        self,
        sim,
        user,
        movie_id,
        rel=0.0
    ):

        if self.outcome_model == "random":
            rating = self.rng.randint(1, 5)
        else:
            # rel 0 -> ~1 star, rel 1 -> ~5 stars, noise sd `rating_noise`.
            rating = int(min(5, max(
                1,
                round(1 + 4 * rel + self.rng.gauss(0.0, self.rating_noise))
            )))

        return RatingEvent(
            timestamp=sim.clock.now,
            user_id=user.user_id,
            movie_id=movie_id,
            rating=rating
        )
