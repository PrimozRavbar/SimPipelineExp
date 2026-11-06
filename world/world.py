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
        movie_states
    ):
        super().__init__()

        self.user_objects = user_objects
        self.movie_objects = movie_objects

        self.user_states = user_states
        self.movie_states = movie_states

        self.active_sessions = {}

        self.user_simulator = UserSimulator(self)


    def process(self, sim):

        active_users = self.sample_active_users(sim)

        for user in active_users:
            self.simulate_user(user, sim)


    def sample_active_users(self, sim):
        # Decide which users are active this tick.
        # For now, activate one random user.

        return [sim.random.choice(list(self.user_objects.values()))]


    def simulate_user(self, user, sim):

        model_server = sim.get_service("ModelServer")

        movie_ids = model_server.recommend(user)

        events = self.user_simulator.react(
            sim=sim,
            user=user,
            recommendation_id=sim.clock.now,
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