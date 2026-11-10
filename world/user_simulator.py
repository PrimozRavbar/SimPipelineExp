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
        completion_noise=0.15,
        fatigue=0.0,
        rewatch=1.0
    ):
        self.world = world

        # Exposure effects (defaults = none, the original behaviour):
        # - fatigue: each earlier impression of the same movie to the same
        #   user multiplies its click probability by (1 - fatigue).
        # - rewatch: click-probability multiplier for a movie the user
        #   already watched during the simulation (0 = never rewatch).
        if not 0.0 <= fatigue < 1.0:
            raise ValueError("fatigue must be in [0, 1)")
        if not 0.0 <= rewatch <= 1.0:
            raise ValueError("rewatch must be in [0, 1]")
        self.fatigue = fatigue
        self.rewatch = rewatch
        self.impressions = {}       # (user_id, movie_id) -> times shown
        self.watched = set()        # (user_id, movie_id) watched in the simulation

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

        if self.fatigue > 0:
            for movie_id in movie_ids:
                key = (user.user_id, movie_id)
                self.impressions[key] = self.impressions.get(key, 0) + 1

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


    def exposure_factor(self, user_id, movie_id):
        """Click-probability multiplier from fatigue and rewatching (1.0
        with the defaults). Uses no RNG."""
        factor = 1.0
        if self.fatigue > 0:
            factor *= (1.0 - self.fatigue) ** self.impressions.get((user_id, movie_id), 0)
        if self.rewatch < 1.0 and (user_id, movie_id) in self.watched:
            factor *= self.rewatch
        return factor


    def _exposure_active(self):
        return self.fatigue > 0 or self.rewatch < 1.0


    def expected_click_rate(self, user_id, movie_ids):
        """Probability that this user clicks anything on the slate under
        the cascade model (exact, no sampling). Uses no RNG."""

        user_state = self.world.user_states[user_id]
        share = genre_share(user_state)
        p_no_click = 1.0

        exposure = self._exposure_active()

        for rank, movie_id in enumerate(list(movie_ids)[:self.max_examined]):
            p_click = self.click_probability(
                user_state, self.world.movie_objects[movie_id], share
            )
            if exposure:
                p_click *= self.exposure_factor(user_id, movie_id)
            p_no_click *= 1.0 - self.examine_probability(rank) * p_click

        return 1.0 - p_no_click


    def choose_click(self, user, movie_ids):
        """Cascade model: scan the slate in order, stop at the first click.
        Returns the clicked movie_id, or None if nothing was clicked."""

        user_state = self.world.user_states[user.user_id]
        share = genre_share(user_state)
        exposure = self._exposure_active()

        for rank, movie_id in enumerate(list(movie_ids)[:self.max_examined]):

            if self.rng.random() >= self.examine_probability(rank):
                continue

            movie = self.world.movie_objects[movie_id]

            p_click = self.click_probability(user_state, movie, share)
            if exposure:
                p_click *= self.exposure_factor(user.user_id, movie_id)

            if self.rng.random() < p_click:
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

        if self.rewatch < 1.0:
            self.watched.add((user.user_id, movie_id))

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
