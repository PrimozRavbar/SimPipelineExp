from .user_simulator import UserSimulator
from simulation.service import Service
from .entities import User, Movie, Rating, UserState, MovieState

from uuid import uuid4


class World(Service):

    def __init__(
        self,
        user_objects,
        movie_objects,
        user_states,
        movie_states,
        rng=None,
        users_per_tick=1,
        outcome_model="relevance",
        user_simulator_kwargs=None
    ):
        super().__init__()

        self.user_objects = user_objects
        self.movie_objects = movie_objects

        self.user_states = user_states
        self.movie_states = movie_states

        self.active_sessions = {}

        if users_per_tick < 1:
            raise ValueError("users_per_tick must be >= 1")

        # Distinct users served per tick. With 1 (the old behaviour) the
        # per-tick CTR is 0 or 1; with more it is a real rate.
        self.users_per_tick = users_per_tick

        # Unique id per shown slate. It used to be the tick, which collides
        # once several users are active in the same tick.
        self.recommendation_count = 0

        # Users who can be active. None = everyone (the original behaviour);
        # WorldDynamics sets it when users join over time.
        self.active_user_ids = None

        self.user_simulator = UserSimulator(
            self,
            rng=rng,
            outcome_model=outcome_model,
            **(user_simulator_kwargs or {})
        )


    def process(self, sim):

        active_users = self.sample_active_users(sim)

        for user in active_users:
            self.simulate_user(user, sim)


    def sample_active_users(self, sim):
        """Users active this tick: `users_per_tick` distinct users drawn
        uniformly with the simulator's seeded RNG."""

        if self.active_user_ids is None:
            users = list(self.user_objects.values())
        else:
            users = [self.user_objects[user_id] for user_id in self.active_user_ids]

        if self.users_per_tick == 1:
            # Same RNG call as before, so 1-user runs are unchanged.
            return [sim.random.choice(users)]

        return sim.random.sample(users, min(self.users_per_tick, len(users)))


    def simulate_user(self, user, sim):

        model_server = sim.get_service("ModelServer")

        movie_ids = model_server.recommend(user)

        self.recommendation_count += 1

        events = self.user_simulator.react(
            sim=sim,
            user=user,
            recommendation_id=self.recommendation_count,
            movie_ids=movie_ids
        )

        self.outbox.extend(events)

    def process_events(self, events):

        for event in events:

            self.process_event(event)

            self.outbox.append(event)


    def process_event(self, event):

        if isinstance(event, RecommendationShownEvent):
            self.process_shown(event)

        elif isinstance(event, RecommendationClickedEvent):
            self.process_click(event)

        elif isinstance(event, WatchEvent):
            self.process_watch(event)

        elif isinstance(event, RatingEvent):
            self.process_rating(event)


    def process_shown(self, event):

        movie_state = self.movie_states[event.movie_ids[0]]  # Placeholder

        movie_state.impressions += 1


    def process_click(self, event):

        user_state = self.user_states[event.user_id]
        movie_state = self.movie_states[event.movie_id]

        user_state.click_count += 1
        movie_state.click_count += 1


    def process_watch(self, event):

        user_state = self.user_states[event.user_id]
        movie_state = self.movie_states[event.movie_id]

        user_state.watch_count += 1
        movie_state.watch_count += 1


    def process_rating(self, event):

        user_state = self.user_states[event.user_id]
        movie_state = self.movie_states[event.movie_id]

        user_state.rating_count += 1

        user_state.ratings[event.movie_id] = event.rating

        movie_state.ratings.append(event.rating)
        movie_state.average_rating = (
            sum(movie_state.ratings) / len(movie_state.ratings)
        )