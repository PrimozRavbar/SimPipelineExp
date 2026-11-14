from dataclasses import dataclass

@dataclass
class Event:
    timestamp: int
    user_id: int

@dataclass
class RecommendationShownEvent(Event):
    recommendation_id: int
    movie_ids: list[int]


@dataclass
class ClickEvent(Event):
    movie_id: int

@dataclass
class RecommendationClickedEvent(Event):
    recommendation_id: int
    movie_id: int


@dataclass
class WatchEvent(Event):
    movie_id: int
    watch_time: float      # seconds
    completion: float      # 0-1


@dataclass
class RatingEvent(Event):
    movie_id: int
    rating: float

@dataclass
class OnboardingEvent(Event):
    """A new user picked favourite genres at sign-up (opt-in feature)."""
    genres: list[str]

@dataclass
class BrowseEvent(Event):
    """A user browsed or searched a genre (opt-in world feature): explicit
    evidence of current taste, beyond the slates they were shown."""
    genre: str

@dataclass
class InterestEvent(Event):
    """A signal short of a click on a movie the user examined (opt-in world
    feature): kind "save" (added to their list: interested) or "dismiss"
    ("not interested")."""
    movie_id: int
    kind: str

@dataclass
class Session:
    session_id: int
    user_id: int
    started_at: int

@dataclass
class RetrievalRequest:
    user_id: int
    k: int