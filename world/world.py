from .user_simulator import UserSimulator
from simulation.service import Service
from .entities import User, Movie, Rating, UserState, MovieState

from uuid import uuid4

from simulation.events import BrowseEvent, OnboardingEvent


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
        user_simulator_kwargs=None,
        onboarding=None,
        onboarding_rng=None,
        taste=None,
        browsing=None,
        browsing_rng=None
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

        # Hidden tastes beyond genres (world.taste.TasteModel) in a "rich"
        # world; None = the genre-only world. World-side only.
        self.taste = taste

        # Onboarding (opt-in): the first time a user is active they pick
        # favourite genres (an OnboardingEvent, published before their first
        # slate). Dict {"genres": k, "noise": p}; None = off. Own RNG.
        self.onboarding = onboarding
        self.onboarding_rng = onboarding_rng
        # browsing: {"rate": p}: on each visit, with probability p, the user
        # browses a genre first (BrowseEvent). None = off (the original).
        self.browsing = browsing
        self.browsing_rng = browsing_rng
        self.onboarded = set()

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

        if self.onboarding and user.user_id not in self.onboarded:
            self.onboarded.add(user.user_id)
            self.outbox.append(OnboardingEvent(
                timestamp=sim.clock.now,
                user_id=user.user_id,
                genres=self.declared_genres(user.user_id),
            ))

        if self.browsing and self.browsing.get("rate", 0) > 0 \
                and self.browsing_rng.random() < self.browsing["rate"]:
            self.outbox.append(BrowseEvent(
                timestamp=sim.clock.now,
                user_id=user.user_id,
                genre=self.browsed_genre(user.user_id),
            ))

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

    def browsed_genre(self, user_id):
        """The genre a user browses: drawn from their current true genre
        share (tastes that drifted show up here at once)."""
        from world.dynamics import DRIFT_GENRES
        from world.relevance import genre_share

        share = genre_share(self.user_states[user_id])
        genres = [g for g in DRIFT_GENRES if share.get(g, 0) > 0]
        if not genres:
            return self.browsing_rng.choice(DRIFT_GENRES)
        pick = self.browsing_rng.random() * sum(share[g] for g in genres)
        for genre in genres:
            pick -= share[genre]
            if pick <= 0:
                return genre
        return genres[-1]

    def declared_genres(self, user_id):
        """The genres a user ticks at sign-up: their top `genres` genres by
        true taste, each replaced by a random genre with probability
        `noise` (people describe their taste imperfectly)."""
        from world.dynamics import DRIFT_GENRES
        from world.relevance import genre_share

        count = int(self.onboarding.get("genres", 3))
        noise = float(self.onboarding.get("noise", 0.3))
        share = genre_share(self.user_states[user_id])
        top = sorted((g for g in share if g in DRIFT_GENRES), key=lambda g: (-share[g], DRIFT_GENRES.index(g)))
        picks = []
        for genre in top[:count] + [None] * max(0, count - len(top)):
            if genre is None or self.onboarding_rng.random() < noise:
                genre = self.onboarding_rng.choice(DRIFT_GENRES)
            if genre not in picks:
                picks.append(genre)
        return picks

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